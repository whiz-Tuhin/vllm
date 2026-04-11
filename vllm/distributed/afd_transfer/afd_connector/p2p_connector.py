# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import re
from datetime import timedelta
import pickle

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    TensorMetadata,
    init_afd_process_group,
)
# --- OLD CODE (imported _world for _fix_pg_group_ranks — no longer needed) ---
# from torch.distributed.distributed_c10d import _world
# --- END OLD CODE ---
from vllm.logger import init_logger
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    ncclUniqueId,
)
from vllm.utils.torch_utils import direct_register_custom_op
from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

logger = init_logger(__name__)

# --- OLD CODE (_fix_pg_group_ranks — attempted to fix pg_group_ranks for standalone
#     pair groups so dist.get_rank(group) works for DP2+ ranks. But this breaks
#     dist.send(dst=0) because dst is also a global rank lookup — 0 is no longer
#     a key in the fixed mapping. Replaced by bypassing torch.distributed.send/recv
#     entirely and calling gloo_pg.send/recv directly.) ---
# _PAIR_PLACEHOLDER_COUNTER = 0
#
# def _fix_pg_group_ranks(pg, rank_in_pair: int) -> None:
#     global _PAIR_PLACEHOLDER_COUNTER
#     _PAIR_PLACEHOLDER_COUNTER += 1
#     caller_global_rank = torch.distributed.get_rank()
#     partner_rank_in_pair = 1 - rank_in_pair
#     partner_placeholder = -(1000 + _PAIR_PLACEHOLDER_COUNTER)
#     _world.pg_group_ranks[pg] = {
#         caller_global_rank: rank_in_pair,
#         partner_placeholder: partner_rank_in_pair,
#     }
# --- END OLD CODE ---


def _create_pynccl_comm_for_pair(
    gloo_pg: torch.distributed.ProcessGroup,
    rank_in_pair: int,
    device: int,
) -> PyNcclCommunicator:
    """Create a PyNcclCommunicator using a Gloo pair group for ID exchange.

    PyNcclCommunicator's built-in broadcast uses dist.broadcast(src=global_rank),
    which fails for standalone groups created via init_afd_process_group because
    the pg_group_ranks mapping uses pair-local ranks (0,1) that don't include the
    caller's default PG global rank (e.g., DP2 has default rank 2).

    This helper exchanges the ncclUniqueId via direct pg.send/pg.recv calls on the
    Gloo ProcessGroup object, bypassing torch.distributed.send/recv entirely.
    This avoids the c10d_logger's dist.get_rank(group) call which triggers
    get_group_rank(group, default_pg.rank()) → ValueError for ranks >= 2.
    """
    nccl = NCCLLibrary()

    # --- OLD CODE (used torch.distributed.send/recv which goes through c10d_logger
    #     that calls dist.get_rank(group) → get_group_rank(group, default_pg.rank())
    #     → ValueError for ATTN DP2+ whose default PG rank isn't in pg_group_ranks) ---
    # if rank_in_pair == 0:
    #     unique_id = nccl.ncclGetUniqueId()
    #     tensor = torch.ByteTensor(list(unique_id.internal))
    #     torch.distributed.send(tensor, dst=1, group=gloo_pg)
    # else:
    #     unique_id = ncclUniqueId()
    #     tensor = torch.ByteTensor(list(unique_id.internal))
    #     torch.distributed.recv(tensor, src=0, group=gloo_pg)
    #     for idx, byte in enumerate(tensor.tolist()):
    #         unique_id.internal[idx] = byte
    # --- END OLD CODE ---

    # --- NEW CODE (call gloo_pg.send/recv directly — dst/src are group-local ranks,
    #     no c10d_logger, no pg_group_ranks lookup) ---
    if rank_in_pair == 0:
        unique_id = nccl.ncclGetUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.send([tensor], 1, 0).wait()  # send to group rank 1 (ATTN)
    else:
        unique_id = ncclUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.recv([tensor], 0, 0).wait()  # recv from group rank 0 (FFN)
        for idx, byte in enumerate(tensor.tolist()):
            unique_id.internal[idx] = byte
    # --- END NEW CODE ---

    device_obj = torch.device(f"cuda:{device}")
    with torch.cuda.device(device_obj):
        comm = nccl.ncclCommInitRank(2, unique_id, rank_in_pair)

    # Build a PyNcclCommunicator shell with our manually-created comm
    pynccl = object.__new__(PyNcclCommunicator)
    pynccl.rank = rank_in_pair
    pynccl.world_size = 2
    pynccl.group = gloo_pg
    pynccl.available = True
    pynccl.disabled = False
    pynccl.nccl = nccl
    pynccl.nccl_version = nccl.ncclGetRawVersion()
    pynccl.unique_id = unique_id
    pynccl.device = device_obj
    pynccl.comm = comm

    return pynccl

# -------------------------------------------------------------------------
# Custom Ops Registration for P2P Communication
# -------------------------------------------------------------------------

# Global registry to map integer IDs to PyNcclCommunicator objects
# because we cannot pass complex Python objects to custom ops.
_AFD_COMMUNICATORS: dict[int, PyNcclCommunicator] = {}
_AFD_COMM_ID_COUNTER = 0

def _register_comm(comm: PyNcclCommunicator) -> int:
    global _AFD_COMM_ID_COUNTER
    comm_id = _AFD_COMM_ID_COUNTER
    _AFD_COMMUNICATORS[comm_id] = comm
    _AFD_COMM_ID_COUNTER += 1
    return comm_id


def _unregister_comm(comm_id: int) -> None:
    _AFD_COMMUNICATORS.pop(comm_id, None)

# --- Send Op ---

def afd_p2p_send_impl(tensor: torch.Tensor, dst: int, comm_id: int) -> None:
    comm = _AFD_COMMUNICATORS.get(comm_id)
    if comm is None:
        raise RuntimeError(f"Communicator with ID {comm_id} not found/registered.")
    comm.send(tensor, dst)

def afd_p2p_send_fake(tensor: torch.Tensor, dst: int, comm_id: int) -> None:
    return None

direct_register_custom_op(
    op_name="afd_p2p_send",
    op_func=afd_p2p_send_impl,
    mutates_args=["tensor"],
    fake_impl=afd_p2p_send_fake,
)

# --- Recv Op ---

def afd_p2p_recv_impl(
    out: torch.Tensor,
    src: int,
    comm_id: int,
) -> None:
    comm = _AFD_COMMUNICATORS.get(comm_id)
    if comm is None:
        raise RuntimeError(f"Communicator with ID {comm_id} not found/registered.")
    comm.recv(out, src)


def afd_p2p_recv_fake(
    out: torch.Tensor,
    src: int,
    comm_id: int,
) -> None:
    return None

direct_register_custom_op(
    op_name="afd_p2p_recv",
    op_func=afd_p2p_recv_impl,
    mutates_args=["out"],
    fake_impl=afd_p2p_recv_fake,
)

@dataclasses.dataclass
class PairGroup:
    """Lightweight replacement for GroupCoordinator in AFD pair groups.
    Provides rank_in_group, world_size, and unique_name — the only attributes
    used by _send_hidden_states and _recv_hidden_states."""
    rank_in_group: int
    world_size: int = 2
    unique_name: str = ""


# --- OLD CODE (DefaultProcessGroupSwitcher — no longer needed since we bypass
#     torch.distributed.new_group entirely via init_afd_process_group) ---
# class DefaultProcessGroupSwitcher:
#     def __init__(self, default_group, new_default_group):
#         self.default_group = default_group
#         self.new_default_group = new_default_group
#     def __enter__(self):
#         _update_default_pg(self.new_default_group)
#     def __exit__(self, exc_type, exc_value, traceback):
#         _update_default_pg(self.default_group)
# --- END OLD CODE ---


class P2PAFDConnector(AFDConnectorBase):
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self._initialized: bool = False
        self._tensor_metadata_list: dict[int, TensorMetadata] = {}
        if getattr(self.config.model_config.hf_config, "text_config", None) is not None:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.text_config.num_hidden_layers
            )
        else:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.num_hidden_layers
            )

        # --- OLD CODE (symmetric-only, single group/comm) ---
        # self.a2e_pynccl: PyNcclCommunicator | None = None
        # self.e2a_pynccl: PyNcclCommunicator | None = None
        # self.a2e_comm_id: int | None = None
        # self.e2a_comm_id: int | None = None
        # self.ffn_size: int = 0
        # self.min_size: int = 0
        # self.dst_list = []
        # --- END OLD CODE ---

        # --- NEW CODE (asymmetric support: lists of groups/comms) ---
        # Lists of groups and comm_ids — one per partner for asymmetric configs,
        # exactly one entry for symmetric configs (1A1F, 2A2F).
        self.a2e_groups: list[PairGroup] = []
        self.e2a_groups: list[PairGroup] = []
        self.a2e_comm_ids: list[int] = []
        self.e2a_comm_ids: list[int] = []
        # Gloo ProcessGroup objects for each pair — used for metadata transfer
        # (send_dp_metadata_list / recv_dp_metadata_list). One per partner.
        self.a2e_gloo_pgs: list[torch.distributed.ProcessGroup] = []
        self.attn_size: int = 0
        self.ffn_size: int = 0
        self.min_size: int = 0
        # --- OLD CODE (dst_list — used with p2p_pg for metadata routing) ---
        # self.dst_list = []
        # --- END OLD CODE ---
        # --- END NEW CODE ---

        # Fixed recv buffers for FFN side when graph capturing; key = (stage_idx, size)
        self._recv_attn_buffers: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    def close(self) -> None:
        """Close the connector and release resources."""
        # --- OLD CODE (single comm_id) ---
        # if self.a2e_comm_id is not None:
        #     _unregister_comm(self.a2e_comm_id)
        #     self.a2e_comm_id = None
        # if self.e2a_comm_id is not None:
        #     _unregister_comm(self.e2a_comm_id)
        #     self.e2a_comm_id = None
        # --- END OLD CODE ---

        # --- NEW CODE (unregister all comm_ids in lists) ---
        for comm_id in self.a2e_comm_ids:
            _unregister_comm(comm_id)
        self.a2e_comm_ids.clear()
        for comm_id in self.e2a_comm_ids:
            _unregister_comm(comm_id)
        self.e2a_comm_ids.clear()
        # --- END NEW CODE ---

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        logger.info("jcz init_afd_connector begin")
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(int, re.match(r"(\d+)\D+(\d+)", afd_size).groups())

        # --- NEW CODE (asymmetric validation) ---
        assert attn_size == ffn_size or min(attn_size, ffn_size) == 1, (
            f"Asymmetric AFD requires one side to be 1 GPU. Got {attn_size}A{ffn_size}F. "
            f"Supported: 1AxF, xA1F, or NAxNF (symmetric)."
        )
        # --- END NEW CODE ---

        self.world_rank = self.rank if role == "ffn" else self.rank + ffn_size
        self.ffn_size = ffn_size
        self.attn_size = attn_size
        self.min_size = min(ffn_size, attn_size)
        self.max_size = max(ffn_size, attn_size)
        # For 1AxF, FFN uses TP across multiple GPUs. ATTN must broadcast
        # (not chunk) to all FFN workers, and recv from only FFN TP rank 0.
        # For xA1F, ATTN uses DP — each ATTN sends independently to the single FFN.
        self.is_tp_ffn = (attn_size == 1 and ffn_size > 1)
        # --- OLD CODE (p2p_rank — used for the multi-rank p2p_pg group, replaced by pair groups) ---
        # self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        # --- END OLD CODE ---
        # --- OLD CODE (backend="nccl" — NCCL barrier hangs with heterogeneous
        #     CUDA_VISIBLE_DEVICES because NCCL guesses device ID from global rank) ---
        # afd_pg = init_afd_process_group(
        #     backend="nccl",
        # --- END OLD CODE ---
        # --- NEW CODE (backend="gloo" — Gloo barriers are CPU-based, no GPU mapping issues.
        #     afd_pg is only used for barriers and as parent group for new_group.
        #     Actual NCCL P2P communication uses PyNcclCommunicator which creates
        #     its own NCCL communicator independently.) ---
        afd_pg = init_afd_process_group(
            backend="gloo",
            init_method=(
                f"tcp://{self.config.afd_config.afd_host}"
                f":{self.config.afd_config.afd_port}"
            ),
            world_size=ffn_size + attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=10),
        )
        logger.info(f"jcz afd_pg initialized world_rank:{self.world_rank}")

        # --- OLD CODE (symmetric-only sub-group creation) ---
        # # Construct rank lists for sub groups.
        # # Each group contains one attention and one ffn rank.
        # ffn_ranks = [i for i in range(ffn_size)]
        # attn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        # assert len(ffn_ranks) == len(attn_ranks), (
        #     "ffn_ranks and attn_ranks must have the same length"
        # )
        # default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), afd_pg)
        # with default_pg_switcher:
        #     sub_group_ranks = []
        #     for i in range(len(ffn_ranks)):
        #         ranks = [ffn_ranks[i], attn_ranks[i]]
        #         sub_group_ranks.append(ranks)
        #     self.a2e_group = init_model_parallel_group(
        #         sub_group_ranks, self.local_rank, backend="nccl", group_name="a2e",
        #     )
        #     self.e2a_group = init_model_parallel_group(
        #         sub_group_ranks, self.local_rank, backend="nccl", group_name="e2a",
        #     )
        #     self.a2e_pynccl = PyNcclCommunicator(
        #         group=self.a2e_group.cpu_group, device=self.local_rank,
        #     )
        #     self.a2e_comm_id = _register_comm(self.a2e_pynccl)
        #     self.e2a_pynccl = PyNcclCommunicator(
        #         group=self.e2a_group.cpu_group, device=self.local_rank,
        #     )
        #     self.e2a_comm_id = _register_comm(self.e2a_pynccl)
        # --- END OLD CODE ---

        # --- OLD CODE (two-phase GroupCoordinator + dummy groups approach — hangs due to
        #     torch.distributed.new_group desync across ranks in asymmetric configs) ---
        # ffn_ranks = list(range(ffn_size))
        # attn_ranks = list(range(ffn_size, ffn_size + attn_size))
        # all_world_ranks = set(range(ffn_size + attn_size))
        # default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), afd_pg)
        # with default_pg_switcher:
        #     all_a2e_groups, all_e2a_groups, all_pair_ranks = [], [], []
        #     for i in range(self.max_size):
        #         ffn_rank_i = ffn_ranks[i % ffn_size]
        #         attn_rank_i = attn_ranks[i % attn_size]
        #         pair_ranks = [ffn_rank_i, attn_rank_i]
        #         remaining = sorted(all_world_ranks - set(pair_ranks))
        #         sub_group_ranks = [pair_ranks, remaining] if remaining else [pair_ranks]
        #         a2e_group = init_model_parallel_group(sub_group_ranks, self.local_rank,
        #             backend="gloo", use_device_communicator=False, group_name=f"a2e_{i}")
        #         e2a_group = init_model_parallel_group(sub_group_ranks, self.local_rank,
        #             backend="gloo", use_device_communicator=False, group_name=f"e2a_{i}")
        #         all_a2e_groups.append(a2e_group)
        #         all_e2a_groups.append(e2a_group)
        #         all_pair_ranks.append(pair_ranks)
        #     for i, (a2e_group, e2a_group, pair_ranks) in enumerate(
        #         zip(all_a2e_groups, all_e2a_groups, all_pair_ranks)):
        #         if self.world_rank in pair_ranks:
        #             a2e_pynccl = PyNcclCommunicator(group=a2e_group.cpu_group, device=self.local_rank)
        #             self.a2e_groups.append(a2e_group)
        #             self.a2e_comm_ids.append(_register_comm(a2e_pynccl))
        #             e2a_pynccl = PyNcclCommunicator(group=e2a_group.cpu_group, device=self.local_rank)
        #             self.e2a_groups.append(e2a_group)
        #             self.e2a_comm_ids.append(_register_comm(e2a_pynccl))
        # --- END OLD CODE ---

        # --- OLD CODE (GroupCoordinator + barrier-based pair creation — hangs due to
        #     torch.distributed.new_group global counter desync. FFN (DP=1) and ATTN (DP=3)
        #     have different _group_count values from initialize_model_parallel, so new_group
        #     calls generate mismatched group IDs across ranks. No amount of barriers can fix
        #     this because the counters diverged before AFD init.) ---
        # ffn_ranks = list(range(ffn_size))
        # attn_ranks = list(range(ffn_size, ffn_size + attn_size))
        # all_world_ranks = set(range(ffn_size + attn_size))
        # default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), afd_pg)
        # with default_pg_switcher:
        #     ... barrier + init_model_parallel_group per pair ...
        #     ... PyNcclCommunicator creation for pair members ...
        # --- END OLD CODE ---

        # --- NEW CODE (Direct init_afd_process_group per pair — bypasses new_group entirely.
        #     Each pair gets its own TCP store at a unique port. Only pair members participate.
        #     No dummy groups, no barriers, no global counter involvement.
        #     PyNcclCommunicator uses the Gloo group for ncclUniqueId exchange,
        #     then creates its own NCCL communicator independently.) ---
        ffn_ranks = list(range(ffn_size))
        attn_ranks = list(range(ffn_size, ffn_size + attn_size))
        afd_host = self.config.afd_config.afd_host
        afd_base_port = int(self.config.afd_config.afd_port)

        for i in range(self.max_size):
            ffn_rank_i = ffn_ranks[i % ffn_size]
            attn_rank_i = attn_ranks[i % attn_size]
            pair_ranks = [ffn_rank_i, attn_rank_i]

            if self.world_rank not in pair_ranks:
                continue  # Non-pair ranks don't participate at all

            # rank_in_pair: 0 = FFN, 1 = ATTN (matches pair_ranks ordering)
            rank_in_pair = pair_ranks.index(self.world_rank)

            # Unique port per pair per direction (offset from afd_port)
            a2e_port = afd_base_port + 100 + i * 2
            e2a_port = afd_base_port + 100 + i * 2 + 1

            logger.info(
                f"jcz creating a2e pair {i}: pair={pair_ranks}, "
                f"port={a2e_port}, rank_in_pair={rank_in_pair}"
            )
            a2e_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{afd_host}:{a2e_port}",
                world_size=2,
                rank=rank_in_pair,
                group_name=f"a2e_{i}",
                timeout=timedelta(minutes=10),
            )
            # --- OLD CODE (_fix_pg_group_ranks — no longer needed since we bypass
            #     torch.distributed.send/recv and call gloo_pg.send/recv directly) ---
            # _fix_pg_group_ranks(a2e_pg, rank_in_pair)
            # --- END OLD CODE ---
            a2e_pynccl = _create_pynccl_comm_for_pair(
                a2e_pg, rank_in_pair, self.local_rank,
            )
            self.a2e_groups.append(
                PairGroup(rank_in_group=rank_in_pair, unique_name=f"a2e_{i}")
            )
            self.a2e_comm_ids.append(_register_comm(a2e_pynccl))
            # Store the Gloo PG for metadata transfer (replaces p2p_pg)
            self.a2e_gloo_pgs.append(a2e_pg)

            logger.info(
                f"jcz creating e2a pair {i}: pair={pair_ranks}, "
                f"port={e2a_port}, rank_in_pair={rank_in_pair}"
            )
            e2a_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{afd_host}:{e2a_port}",
                world_size=2,
                rank=rank_in_pair,
                group_name=f"e2a_{i}",
                timeout=timedelta(minutes=10),
            )
            # --- OLD CODE (_fix_pg_group_ranks — no longer needed) ---
            # _fix_pg_group_ranks(e2a_pg, rank_in_pair)
            # --- END OLD CODE ---
            e2a_pynccl = _create_pynccl_comm_for_pair(
                e2a_pg, rank_in_pair, self.local_rank,
            )
            self.e2a_groups.append(
                PairGroup(rank_in_group=rank_in_pair, unique_name=f"e2a_{i}")
            )
            self.e2a_comm_ids.append(_register_comm(e2a_pynccl))

        logger.info(
            f"jcz created {len(self.a2e_groups)} a2e groups and "
            f"{len(self.e2a_groups)} e2a groups for world_rank={self.world_rank}"
        )
        # --- END NEW CODE ---

        # --- OLD CODE (p2p_pg — multi-rank Gloo group for metadata transfer.
        #     Fails for 1AxF because p2p_rank collides: FFN TP rank 1 and ATTN both get p2p_rank=1.
        #     Replaced by using existing per-pair a2e Gloo groups for metadata transfer.) ---
        # if self.is_vaild_rank_for_inequal_AF(self.world_rank):
        #     self.p2p_pg = init_afd_process_group(
        #         backend="gloo",
        #         init_method=(
        #             f"tcp://{self.config.afd_config.afd_host}"
        #             f":{self.config.afd_config.afd_port}"
        #         ),
        #         world_size=self.ffn_size + self.min_size,
        #         rank=self.p2p_rank,
        #         group_name="p2p",
        #         timeout=timedelta(minutes=30),
        #     )
        #
        # # The first min_size Attention sends metadata to multiple FFNs (1-to-many mapping).
        # # Each attn_i sends to all ffn_j where (j % min_size == i)
        # if self.is_attn_top_min_size_rank(self.world_rank):
        #     local_attn_rank = self.world_rank - self.ffn_size
        #     dst = local_attn_rank
        #     while dst < self.ffn_size:
        #         self.dst_list.append(dst)
        #         dst += self.min_size
        # --- END OLD CODE ---

        logger.info(
            f"[P2P] world_rank={self.world_rank}, min_size={self.min_size}, "
            f"num_pairs={len(self.a2e_groups)}, p2p connector initialized"
        )

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    # --- OLD CODE (_send_hidden_states: looked up comm_id via object comparison
    #     against self.a2e_group / self.e2a_group. With lists of groups, object
    #     comparison no longer works, so callers now pass comm_id directly.) ---
    # def _send_hidden_states(
    #     self,
    #     hidden_states: torch.Tensor,
    #     dst: int,
    #     process_group: GroupCoordinator,
    # ) -> None:
    #     if not torch.distributed.is_initialized() or process_group.world_size == 1:
    #         return []
    #     assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
    #     assert not hidden_states.is_cpu, "Hidden states must be on GPU"
    #
    #     # Try to use PyNCCL first
    #     comm_id = None
    #     if process_group == self.a2e_group:
    #         comm_id = self.a2e_comm_id
    #     elif process_group == self.e2a_group:
    #         comm_id = self.e2a_comm_id
    #
    #     if comm_id is not None:
    #         # PyNCCL uses rank in group
    #         logger.info(
    #             f"[AFD_DIAG] SEND shape={hidden_states.shape} dtype={hidden_states.dtype} "
    #             f"sum={hidden_states.float().sum().item():.4f} "
    #             f"mean={hidden_states.float().mean().item():.6f} dst={dst}"
    #         )
    #         direction = "attn->ffn" if process_group == self.a2e_group else "ffn->attn"
    #         nvtx_msg = (
    #             f"afd_p2p_send"
    #             f"|direction={direction}"
    #             f"|shape={list(hidden_states.shape)}"
    #             f"|dtype={hidden_states.dtype}"
    #             f"|pg={process_group.unique_name}"
    #             f"|dst={dst}"
    #             f"|bytes={hidden_states.numel() * hidden_states.element_size()}"
    #         )
    #         with torch.profiler.record_function("afd_p2p_send", args=nvtx_msg), \
    #              torch.cuda.nvtx.range(nvtx_msg):
    #             torch.ops.vllm.afd_p2p_send(hidden_states, dst, comm_id)
    #     else:
    #         raise RuntimeError("PyNCCL communicator is required but not available.")
    # --- END OLD CODE ---

    # --- NEW CODE (_send_hidden_states: accepts comm_id and direction as params
    #     instead of looking them up via object comparison) ---
    def _send_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: PairGroup,
        comm_id: int,
        direction: str = "",
    ) -> None:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
        assert not hidden_states.is_cpu, "Hidden states must be on GPU"

        # --- OLD CODE (AFD_DIAG .item() forces CUDA sync — ~3s/layer with TP=2) ---
        # logger.info(
        #     f"[AFD_DIAG] SEND shape={hidden_states.shape} dtype={hidden_states.dtype} "
        #     f"sum={hidden_states.float().sum().item():.4f} "
        #     f"mean={hidden_states.float().mean().item():.6f} dst={dst}"
        # )
        # --- END OLD CODE ---
        nvtx_msg = (
            f"afd_p2p_send"
            f"|direction={direction}"
            f"|shape={list(hidden_states.shape)}"
            f"|dtype={hidden_states.dtype}"
            f"|pg={process_group.unique_name}"
            f"|dst={dst}"
            f"|bytes={hidden_states.numel() * hidden_states.element_size()}"
        )
        with torch.profiler.record_function("afd_p2p_send", args=nvtx_msg), \
             torch.cuda.nvtx.range(nvtx_msg):
            torch.ops.vllm.afd_p2p_send(hidden_states, dst, comm_id)
    # --- END NEW CODE ---

    # --- OLD CODE (_recv_hidden_states: looked up comm_id via object comparison
    #     against self.a2e_group / self.e2a_group) ---
    # def _recv_hidden_states(
    #     self,
    #     src: int,
    #     process_group: GroupCoordinator,
    #     tensor_metadata: TensorMetadata,
    #     ref_tensor: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     if not torch.distributed.is_initialized() or process_group.world_size == 1:
    #         return {}, []
    #     assert src < process_group.world_size, f"Invalid src rank ({src})"
    #
    #     comm_id = None
    #     if process_group == self.a2e_group:
    #         comm_id = self.a2e_comm_id
    #     elif process_group == self.e2a_group:
    #         comm_id = self.e2a_comm_id
    #
    #     if comm_id is not None:
    #         size = list(tensor_metadata.size)
    #         if ref_tensor is not None:
    #             size[0] = ref_tensor.shape[0]
    #         if (ref_tensor is not None and ref_tensor.shape == tuple(size)
    #                 and ref_tensor.dtype == tensor_metadata.dtype
    #                 and ref_tensor.device == tensor_metadata.device):
    #             hidden_states = ref_tensor
    #         else:
    #             hidden_states = torch.empty(tuple(size), dtype=tensor_metadata.dtype,
    #                                        device=tensor_metadata.device)
    #         direction = "ffn<-attn" if process_group == self.a2e_group else "attn<-ffn"
    #         nvtx_msg = (f"afd_p2p_recv|direction={direction}|shape={size}|...")
    #         with torch.profiler.record_function("afd_p2p_recv", args=nvtx_msg), \
    #              torch.cuda.nvtx.range(nvtx_msg):
    #             torch.ops.vllm.afd_p2p_recv(hidden_states, src, comm_id)
    #     else:
    #         raise RuntimeError("PyNCCL communicator is required but not available.")
    #     return hidden_states
    # --- END OLD CODE ---

    # --- NEW CODE (_recv_hidden_states: accepts comm_id and direction as params) ---
    def _recv_hidden_states(
        self,
        src: int,
        process_group: PairGroup,
        comm_id: int,
        tensor_metadata: TensorMetadata,
        direction: str = "",
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return torch.empty(0)
        assert src < process_group.world_size, f"Invalid src rank ({src})"

        # Use ref_tensor to capture dynamic shapes (e.g. batch size) if provided
        size = list(tensor_metadata.size)
        if ref_tensor is not None:
            # Assume dimension 0 is the dynamic batch/seq_len dimension
            size[0] = ref_tensor.shape[0]

        if (
            ref_tensor is not None
            and ref_tensor.shape == tuple(size)
            and ref_tensor.dtype == tensor_metadata.dtype
            and ref_tensor.device == tensor_metadata.device
        ):
            hidden_states = ref_tensor
        else:
            # Note: If using cudagraph, this branch should not be taken
            hidden_states = torch.empty(
                tuple(size),
                dtype=tensor_metadata.dtype,
                device=tensor_metadata.device,
            )
        nvtx_msg = (
            f"afd_p2p_recv"
            f"|direction={direction}"
            f"|shape={size}"
            f"|dtype={tensor_metadata.dtype}"
            f"|pg={process_group.unique_name}"
            f"|src={src}"
            f"|bytes={hidden_states.numel() * hidden_states.element_size()}"
        )
        with torch.profiler.record_function("afd_p2p_recv", args=nvtx_msg), \
             torch.cuda.nvtx.range(nvtx_msg):
            torch.ops.vllm.afd_p2p_recv(hidden_states, src, comm_id)
        # --- OLD CODE (AFD_DIAG .item() forces CUDA sync — ~3s/layer with TP=2) ---
        # logger.info(
        #     f"[AFD_DIAG] RECV shape={hidden_states.shape} dtype={hidden_states.dtype} "
        #     f"sum={hidden_states.float().sum().item():.4f} "
        #     f"mean={hidden_states.float().mean().item():.6f} src={src}"
        # )
        # --- END OLD CODE ---
        return hidden_states
    # --- END NEW CODE ---
    
    # --- OLD CODE (update_state_from_dp_metadata: used full num_tokens, no chunking) ---
    # def update_state_from_dp_metadata(self, dp_metadata_list, is_graph_capturing=False):
    #     self.dp_metadata_list = dp_metadata_list
    #     self.is_graph_capturing = is_graph_capturing
    #     num_of_stages = len(dp_metadata_list)
    #     self._tensor_metadata_list = {}
    #     for stage_idx in range(num_of_stages):
    #         dp_metadata = dp_metadata_list[stage_idx]
    #         dp_rank = self.config.parallel_config.data_parallel_rank
    #         num_tokens = dp_metadata.num_tokens_across_dp_cpu[dp_rank].item()
    #         self._tensor_metadata_list[stage_idx] = TensorMetadata(
    #             torch.device(f"cuda:{self.local_rank}"),
    #             self.config.model_config.dtype,
    #             torch.Size([num_tokens, self.config.model_config.hf_config.hidden_size]),
    #         )
    #     if self.config.afd_config.afd_role == "ffn":
    #         for stage_idx in range(num_of_stages):
    #             meta = self._tensor_metadata_list[stage_idx]
    #             buffer_key = (stage_idx, tuple(meta.size))
    #             existing = self._recv_attn_buffers.get(buffer_key)
    #             if (existing is not None and existing.shape == meta.size
    #                     and existing.dtype == meta.dtype and existing.device == meta.device):
    #                 continue
    #             self._recv_attn_buffers[buffer_key] = torch.empty(
    #                 tuple(meta.size), dtype=meta.dtype, device=meta.device)
    # --- END OLD CODE ---

    # --- NEW CODE (update_state_from_dp_metadata: computes per-recv chunk sizes
    #     for asymmetric configs. For symmetric, num_tokens is unchanged.) ---
    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, DPMetadata],
        is_graph_capturing: bool = False,
    ) -> None:
        """Update the connector state based on the received DPMetadata list.

        For asymmetric configs, the tensor metadata reflects the per-recv chunk
        size rather than the total token count:
        - 1AxF: attention chunks N tokens into ffn_size parts. Each recv is ~N/ffn_size.
        - xA1F: each attention sends its own token count. FFN recvs per-partner sizes.
        - Symmetric: unchanged (full N).
        """
        self.dp_metadata_list = dp_metadata_list
        self.is_graph_capturing = is_graph_capturing
        num_of_stages = len(dp_metadata_list)
        role = self.config.afd_config.afd_role
        hidden_size = self.config.model_config.hf_config.hidden_size
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.config.model_config.dtype

        # Build tensor metadata list for each stage
        self._tensor_metadata_list = {}

        for stage_idx in range(num_of_stages):
            dp_metadata = dp_metadata_list[stage_idx]
            dp_rank = self.config.parallel_config.data_parallel_rank
            num_tokens = dp_metadata.num_tokens_across_dp_cpu[dp_rank].item()

            # --- OLD CODE (asymmetric chunk size calculation — wrong for 1AxF TP FFN
            #     because we now broadcast full tensors instead of chunking.
            #     With TP FFN, all workers process the SAME tokens with different
            #     weight slices, so each needs the full num_tokens.) ---
            # if self.attn_size != self.ffn_size:
            #     if role == "ffn" and self.attn_size < self.ffn_size:
            #         k = self.ffn_size
            #         chunk_idx = self.rank
            #         remainder = num_tokens % k
            #         if chunk_idx < remainder:
            #             num_tokens = (num_tokens + k - 1) // k  # ceil
            #         else:
            #             num_tokens = num_tokens // k  # floor
            #     elif role == "attention" and self.attn_size < self.ffn_size:
            #         k = self.ffn_size
            #         num_tokens = num_tokens // k
            #     elif role == "ffn" and self.attn_size > self.ffn_size:
            #         pass
            #     elif role == "attention" and self.attn_size > self.ffn_size:
            #         pass
            # --- END OLD CODE ---
            # --- NEW CODE (no chunk size adjustment — all configs send/recv full tensors.
            #     1AxF: ATTN broadcasts full tensor to all FFN TP workers (TP splits
            #           weights, not tokens). xA1F: each ATTN sends full tensor to FFN.
            #     num_tokens stays as-is for all roles and configs.) ---
            # (no adjustment needed)
            # --- END NEW CODE ---

            self._tensor_metadata_list[stage_idx] = TensorMetadata(
                device,
                # TODO(jcz): use dtype from dp_metadata
                dtype,
                torch.Size([num_tokens, hidden_size]),
            )

        # Pre-allocate fixed recv buffers (FFN side only) so each recv writes into
        # the same buffer (required for CUDA graph capture; also used in eager).
        # Key is (stage_idx, meta.size) so different shapes get separate buffers.
        if role == "ffn":
            for stage_idx in range(num_of_stages):
                meta = self._tensor_metadata_list[stage_idx]
                buffer_key = (stage_idx, tuple(meta.size))
                existing = self._recv_attn_buffers.get(buffer_key)
                if (
                    existing is not None
                    and existing.shape == meta.size
                    and existing.dtype == meta.dtype
                    and existing.device == meta.device
                ):
                    continue
                self._recv_attn_buffers[buffer_key] = torch.empty(
                    tuple(meta.size),
                    dtype=meta.dtype,
                    device=meta.device,
                )
        # We do not clear _recv_attn_buffers so that replayed graphs still have
        # valid buffer addresses to write into.
    # --- END NEW CODE ---

    # -------------------------------------------------------------------------
    #                                attn -> ffn
    # -------------------------------------------------------------------------

    # --- OLD CODE (send_attn_output: single group, no chunking) ---
    # def send_attn_output(self, hidden_states, metadata):
    #     try:
    #         dst = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
    #         self._send_hidden_states(hidden_states, dst, self.a2e_group)
    #     except Exception as e:
    #         raise RuntimeError(f"Communication error: {e}")
    # --- END OLD CODE ---

    # --- OLD CODE (send_attn_output: asymmetric chunking — wrong for 1AxF with TP FFN
    #     because TP FFN workers need the SAME tokens, not different subsets) ---
    # def send_attn_output(self, hidden_states, metadata):
    #     n = len(self.a2e_groups)
    #     if n == 1:
    #         group = self.a2e_groups[0]
    #         cid = self.a2e_comm_ids[0]
    #         dst = (group.rank_in_group - 1) % group.world_size
    #         self._send_hidden_states(hidden_states, dst, group, cid, direction="attn->ffn")
    #     else:
    #         chunks = torch.chunk(hidden_states, n, dim=0)
    #         for chunk, group, cid in zip(chunks, self.a2e_groups, self.a2e_comm_ids):
    #             dst = (group.rank_in_group - 1) % group.world_size
    #             self._send_hidden_states(chunk, dst, group, cid, direction="attn->ffn")
    # --- END OLD CODE ---

    # --- NEW CODE (send_attn_output: broadcast for 1AxF TP FFN, chunk for xA1F DP FFN) ---
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """
        Called by ATTN side to send intermediate tensors to FFN.
        Symmetric (n==1): sends full tensor via single group.
        1AxF (TP FFN): broadcasts full tensor to ALL FFN TP workers (they need same input).
        xA1F (DP ATTN): not called here (each ATTN rank has n==1 to single FFN).
        """
        try:
            n = len(self.a2e_groups)
            if n == 1:
                # Symmetric or xA1F fast path — single partner
                group = self.a2e_groups[0]
                cid = self.a2e_comm_ids[0]
                dst = (group.rank_in_group - 1) % group.world_size
                self._send_hidden_states(hidden_states, dst, group, cid, direction="attn->ffn")
            elif self.is_tp_ffn:
                # 1AxF: broadcast full tensor to ALL FFN TP workers
                for group, cid in zip(self.a2e_groups, self.a2e_comm_ids):
                    dst = (group.rank_in_group - 1) % group.world_size
                    self._send_hidden_states(hidden_states, dst, group, cid, direction="attn->ffn")
            else:
                # Asymmetric DP: chunk along dim=0, send shard_i to partner_i
                chunks = torch.chunk(hidden_states, n, dim=0)
                for chunk, group, cid in zip(chunks, self.a2e_groups, self.a2e_comm_ids):
                    dst = (group.rank_in_group - 1) % group.world_size
                    self._send_hidden_states(chunk, dst, group, cid, direction="attn->ffn")
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")
    # --- END NEW CODE ---

    # --- OLD CODE (recv_ffn_output: single group, no concat) ---
    # def recv_ffn_output(self, ref_tensor=None):
    #     ubatch_idx = get_forward_context().afd_metadata.afd_stage_idx
    #     src = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
    #     hidden_states = self._recv_hidden_states(
    #         src, self.e2a_group, self._tensor_metadata_list[ubatch_idx], ref_tensor=ref_tensor
    #     )
    #     return hidden_states
    # --- END OLD CODE ---

    # --- OLD CODE (recv_ffn_output: chunked recv + concat — wrong for 1AxF TP FFN) ---
    # def recv_ffn_output(self, ref_tensor=None):
    #     ubatch_idx = get_forward_context().afd_metadata.afd_stage_idx
    #     n = len(self.e2a_groups)
    #     if n == 1:
    #         ...  # symmetric fast path
    #     else:
    #         # recv chunk from each FFN partner, concat
    #         parts = []
    #         for group, cid in zip(self.e2a_groups, self.e2a_comm_ids):
    #             ...
    #         return torch.cat(parts, dim=0)
    # --- END OLD CODE ---

    # --- NEW CODE (recv_ffn_output: for 1AxF TP FFN, recv from FFN TP rank 0 only) ---
    def recv_ffn_output(self, ref_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """
        Called by the ATTN side to receive MoE output from FFN.
        Symmetric (n==1): single recv.
        1AxF (TP FFN): recv from FFN TP rank 0 only (first e2a group).
            After TP allreduce, all FFN workers have the same output,
            so only rank 0 sends back.
        xA1F (DP ATTN): n==1 (single FFN partner), takes fast path.
        """
        ubatch_idx = get_forward_context().afd_metadata.afd_stage_idx
        n = len(self.e2a_groups)
        if n == 1 or self.is_tp_ffn:
            # Symmetric, xA1F, or 1AxF: recv from single partner (FFN TP rank 0)
            group = self.e2a_groups[0]
            cid = self.e2a_comm_ids[0]
            src = (group.rank_in_group + 1) % group.world_size
            hidden_states = self._recv_hidden_states(
                src, group, cid,
                self._tensor_metadata_list[ubatch_idx],
                direction="attn<-ffn",
                ref_tensor=ref_tensor,
            )
            return hidden_states
        else:
            # Asymmetric DP (xA1F with multiple ATTN, but this ATTN has n>1 partners):
            # recv chunk from each partner, concat
            parts = []
            for group, cid in zip(self.e2a_groups, self.e2a_comm_ids):
                src = (group.rank_in_group + 1) % group.world_size
                hs = self._recv_hidden_states(
                    src, group, cid,
                    self._tensor_metadata_list[ubatch_idx],
                    direction="attn<-ffn",
                )
                parts.append(hs)
            return torch.cat(parts, dim=0)
    # --- END NEW CODE ---


    # -------------------------------------------------------------------------
    #                                ffn -> attn
    # -------------------------------------------------------------------------

    # --- OLD CODE (send_ffn_output: single group, no chunking) ---
    # def send_ffn_output(self, hidden_states, metadata):
    #     dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
    #     self._send_hidden_states(hidden_states, dst, self.e2a_group)
    # --- END OLD CODE ---

    # --- OLD CODE (send_ffn_output: all FFN workers send — wrong for 1AxF TP FFN) ---
    # def send_ffn_output(self, hidden_states, metadata):
    #     n = len(self.e2a_groups)
    #     if n == 1:
    #         ...  # symmetric
    #     else:
    #         chunks = torch.chunk(hidden_states, n, dim=0)
    #         for chunk, group, cid in zip(chunks, ...):
    #             self._send_hidden_states(chunk, ...)
    # --- END OLD CODE ---

    # --- NEW CODE (send_ffn_output: only TP rank 0 sends for 1AxF) ---
    def send_ffn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """
        Called by FFN side to send results back to attention.
        Symmetric (n==1): sends full tensor via single group.
        1AxF (TP FFN): only TP rank 0 (self.rank==0) sends. Other TP ranks skip.
            After TP allreduce, all workers have same output, so only one needs to send.
        xA1F (DP ATTN): n>1, chunks output and sends one chunk per ATTN partner.
        """
        n = len(self.e2a_groups)
        if n == 1:
            if self.is_tp_ffn and self.rank != 0:
                # 1AxF: only FFN TP rank 0 sends back, other TP ranks skip
                return
            # Symmetric or 1AxF TP rank 0: send full tensor
            group = self.e2a_groups[0]
            cid = self.e2a_comm_ids[0]
            dst = (group.rank_in_group + 1) % group.world_size
            self._send_hidden_states(hidden_states, dst, group, cid, direction="ffn->attn")
        else:
            # Asymmetric (xA1F): split output, send chunk_i to ATTN_i
            chunks = torch.chunk(hidden_states, n, dim=0)
            for chunk, group, cid in zip(chunks, self.e2a_groups, self.e2a_comm_ids):
                dst = (group.rank_in_group + 1) % group.world_size
                self._send_hidden_states(chunk, dst, group, cid, direction="ffn->attn")
    # --- END NEW CODE ---

    # --- OLD CODE (recv_attn_output: single group, no concat) ---
    # def recv_attn_output(self, ubatch_idx=0):
    #     src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
    #     ref_tensor = None
    #     if not self.config.model_config.enforce_eager:
    #         meta = self._tensor_metadata_list[ubatch_idx]
    #         buffer_key = (ubatch_idx, tuple(meta.size))
    #         ref_tensor = self._recv_attn_buffers.get(buffer_key)
    #     hidden_states = self._recv_hidden_states(
    #         src, self.a2e_group, self._tensor_metadata_list[ubatch_idx],
    #         ref_tensor=ref_tensor,
    #     )
    #     from types import SimpleNamespace
    #     metadata = SimpleNamespace(stage_idx=ubatch_idx, recv_handle_list=None)
    #     return hidden_states, metadata
    # --- END OLD CODE ---

    # --- OLD CODE (recv_attn_output: concat for xA1F — doesn't handle 1AxF TP FFN) ---
    # def recv_attn_output(self, ubatch_idx=0):
    #     n = len(self.a2e_groups)
    #     if n == 1:
    #         ...  # symmetric
    #     else:
    #         parts = []
    #         for group, cid in zip(self.a2e_groups, ...):
    #             ...
    #         hidden_states = torch.cat(parts, dim=0)
    # --- END OLD CODE ---

    # --- NEW CODE (recv_attn_output: each FFN TP worker recvs full tensor for 1AxF) ---
    def recv_attn_output(
        self, ubatch_idx: int = 0
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """
        Called by FFN side to receive hidden states from ATTN.
        Symmetric (n==1): single recv from paired ATTN.
        1AxF (TP FFN): each FFN TP worker has n==1 pair, recvs FULL tensor from ATTN
            (ATTN broadcasts same data to all FFN workers).
        xA1F (DP ATTN): n>1, recv from each ATTN partner, concat along dim=0.
        """
        n = len(self.a2e_groups)
        if n == 1:
            # Symmetric, 1AxF TP FFN, or xA1F single-FFN: recv from single partner
            group = self.a2e_groups[0]
            cid = self.a2e_comm_ids[0]
            src = (group.rank_in_group - 1) % group.world_size
            ref_tensor = None
            if not self.config.model_config.enforce_eager:
                meta = self._tensor_metadata_list[ubatch_idx]
                buffer_key = (ubatch_idx, tuple(meta.size))
                ref_tensor = self._recv_attn_buffers.get(buffer_key)
            hidden_states = self._recv_hidden_states(
                src, group, cid,
                self._tensor_metadata_list[ubatch_idx],
                direction="ffn<-attn",
                ref_tensor=ref_tensor,
            )
        else:
            # Asymmetric (xA1F): recv from each ATTN partner, concat
            parts = []
            for group, cid in zip(self.a2e_groups, self.a2e_comm_ids):
                src = (group.rank_in_group - 1) % group.world_size
                hs = self._recv_hidden_states(
                    src, group, cid,
                    self._tensor_metadata_list[ubatch_idx],
                    direction="ffn<-attn",
                )
                parts.append(hs)
            hidden_states = torch.cat(parts, dim=0)

        # TODO(jcz): remove this after.
        from types import SimpleNamespace
        metadata = SimpleNamespace(
            stage_idx=ubatch_idx,
            recv_handle_list=None,
        )
        return hidden_states, metadata
    # --- END NEW CODE ---

    # --- OLD CODE (send_dp_metadata_list — used p2p_pg multi-rank group and dst_list) ---
    # def send_dp_metadata_list(self, data, is_graph_capturing: bool = False):
    #     self.update_state_from_dp_metadata(data, is_graph_capturing)
    #     send_data = (data, is_graph_capturing)
    #     for dst in self.dst_list:
    #         object_bytes = pickle.dumps(send_data)
    #         object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    #         size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)
    #         logger.info(f"jcz send_dp_metadata_list dst:{dst} self.p2p_rank:{self.p2p_rank} is_graph_capturing:{is_graph_capturing}")
    #         torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
    #         torch.distributed.send(object_tensor, dst=dst, group=self.p2p_pg)
    # --- END OLD CODE ---

    # --- OLD CODE (send_dp_metadata_list — used torch.distributed.send which goes
    #     through c10d_logger → dist.get_rank(group) → ValueError for ATTN DP2+) ---
    # def send_dp_metadata_list(self, data, is_graph_capturing: bool = False):
    #     self.update_state_from_dp_metadata(data, is_graph_capturing)
    #     send_data = (data, is_graph_capturing)
    #     object_bytes = pickle.dumps(send_data)
    #     object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    #     size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)
    #     for i, gloo_pg in enumerate(self.a2e_gloo_pgs):
    #         torch.distributed.send(size_tensor, dst=0, group=gloo_pg)
    #         torch.distributed.send(object_tensor, dst=0, group=gloo_pg)
    # --- END OLD CODE ---

    # --- NEW CODE (send_dp_metadata_list — calls gloo_pg.send directly, bypassing
    #     torch.distributed.send and its c10d_logger that triggers the ValueError.
    #     In the pair Gloo group: FFN=group rank 0, ATTN=group rank 1.
    #     Only called by ATTN DP ranks where is_attn_top_min_size_rank is True.
    #     For xA1F, only DP0 sends; for symmetric, each ATTN sends to its paired FFN.) ---
    def send_dp_metadata_list(self, data, is_graph_capturing: bool = False):
        self.update_state_from_dp_metadata(data, is_graph_capturing)
        send_data = (data, is_graph_capturing)
        object_bytes = pickle.dumps(send_data)
        object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)

        for i, gloo_pg in enumerate(self.a2e_gloo_pgs):
            logger.info(
                f"jcz send_dp_metadata_list pair={i}, "
                f"is_graph_capturing={is_graph_capturing}"
            )
            # Send to FFN (group rank 0). Use pg.send directly — dst is group-local rank.
            gloo_pg.send([size_tensor], 0, 0).wait()
            gloo_pg.send([object_tensor], 0, 0).wait()
    # --- END NEW CODE ---

    # --- OLD CODE (recv_dp_metadata_list — used p2p_pg multi-rank group) ---
    # def recv_dp_metadata_list(self):
    #     src = self.p2p_rank % self.min_size + self.ffn_size
    #     logger.info(f"jcz recv_dp_metadata_list src:{src} self.p2p_rank:{self.p2p_rank}")
    #     size_tensor = torch.empty(1, dtype=torch.long)
    #     rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
    #     object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
    #     rank_object = torch.distributed.recv(object_tensor, src=src, group=self.p2p_pg)
    #     assert rank_object == rank_size
    #     data, is_graph_capturing = pickle.loads(object_tensor.numpy().tobytes())
    #     logger.info(f"jcz recv_dp_metadata_list is_graph_capturing:{is_graph_capturing}")
    #     return data, is_graph_capturing
    # --- END OLD CODE ---

    # --- OLD CODE (recv_dp_metadata_list — used torch.distributed.recv which goes
    #     through c10d_logger → same ValueError issue as send) ---
    # def recv_dp_metadata_list(self):
    #     gloo_pg = self.a2e_gloo_pgs[0]
    #     size_tensor = torch.empty(1, dtype=torch.long)
    #     torch.distributed.recv(size_tensor, src=1, group=gloo_pg)
    #     object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
    #     torch.distributed.recv(object_tensor, src=1, group=gloo_pg)
    #     data, is_graph_capturing = pickle.loads(object_tensor.numpy().tobytes())
    #     return data, is_graph_capturing
    # --- END OLD CODE ---

    # --- NEW CODE (recv_dp_metadata_list — calls gloo_pg.recv directly, bypassing
    #     torch.distributed.recv and its c10d_logger.
    #     FFN receives from pair 0's Gloo group (ATTN DP0 is the only metadata sender
    #     for xA1F; for symmetric, each FFN has exactly 1 pair so index 0 is correct).
    #     In the pair group: FFN=group rank 0, ATTN=group rank 1.) ---
    def recv_dp_metadata_list(self):
        gloo_pg = self.a2e_gloo_pgs[0]
        logger.info(f"jcz recv_dp_metadata_list waiting for metadata from ATTN")

        size_tensor = torch.empty(1, dtype=torch.long)
        # Recv from ATTN (group rank 1). Use pg.recv directly — src is group-local rank.
        gloo_pg.recv([size_tensor], 1, 0).wait()

        object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
        gloo_pg.recv([object_tensor], 1, 0).wait()

        data, is_graph_capturing = pickle.loads(object_tensor.numpy().tobytes())
        logger.info(f"jcz recv_dp_metadata_list is_graph_capturing={is_graph_capturing}")
        return data, is_graph_capturing
    # --- END NEW CODE ---

    # --- OLD CODE (p2p_pg helpers — no longer needed with per-pair metadata transfer) ---
    # def is_vaild_rank_for_inequal_AF(self,rank):
    #     # Only support ffn rank < attn rank
    #     return ((rank >= self.ffn_size and rank < self.ffn_size + self.min_size) or rank < self.ffn_size)
    #
    # def is_attn_top_min_size_rank(self,rank):
    #     # Only support ffn rank < attn rank
    #     return (rank >= self.ffn_size and rank < self.ffn_size + self.min_size)
    # --- END OLD CODE ---

    # --- OLD CODE (is_attn_top_min_size_rank — returned True for ALL ATTN ranks.
    #     Wrong for xA1F: all 3 ATTN ranks send metadata but FFN only reads from
    #     pair 0's Gloo group. DP1/DP2 sends hang forever → 600s timeout.) ---
    # def is_attn_top_min_size_rank(self, rank):
    #     return self.config.afd_config.afd_role == "attention"
    # --- END OLD CODE ---

    # --- NEW CODE (is_attn_top_min_size_rank — only the first min_size ATTN ranks
    #     send metadata. For xA1F (min_size=1), only DP0 sends. For symmetric
    #     (min_size=N), all N ATTN ranks send to their paired FFN. For 1AxF
    #     (min_size=1), the single ATTN (DP0) sends.) ---
    def is_attn_top_min_size_rank(self, rank):
        """Returns True if this ATTN rank should send dp_metadata to FFN.

        Only the first min_size DP ranks send metadata, because FFN's
        recv_dp_metadata_list reads from a2e_gloo_pgs[0] only.
        For symmetric: min_size == attn_size, so all ATTN ranks send (each to its own FFN).
        For xA1F: min_size == 1, so only DP0 sends (FFN reads from pair 0).
        For 1AxF: min_size == 1, single ATTN is DP0, always sends.
        """
        if self.config.afd_config.afd_role != "attention":
            return False
        dp_rank = self.config.parallel_config.data_parallel_rank
        return dp_rank < self.min_size
    # --- END NEW CODE ---