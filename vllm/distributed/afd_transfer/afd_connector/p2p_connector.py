# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P AFD connector with M×N bipartite pair topology and pre-routing.

Design:
  - Every ATTN rank is paired with every FFN rank via a dedicated NCCL P2P
    communicator (full bipartite M×N topology). No collectives between FFN
    workers — the EP all-gather/reduce-scatter is replaced by direct
    point-to-point transfers.
  - ATTN runs the MoE router (gate) and shared experts locally. For each
    MoE layer, it computes topk_ids and routes each token only to the FFN
    worker(s) that hold its target experts. Tokens whose top-k spans
    multiple FFN workers are duplicated (sent to each relevant worker).
  - FFN runs only the routed expert compute on the tokens it received.
    fused_experts with expert_map correctly produces partial contributions
    for the local experts (non-local experts contribute 0).
  - ATTN combines partials from all FFN partners via an index_add on a
    zero-init'd output buffer, then adds the shared-expert output it
    computed locally. This is mathematically equivalent to the old EP
    all-gather + reduce-scatter flow.

  - For dense layers (e.g. DeepSeek layer 0 which is DeepseekV2MLP, not
    MoE), ATTN broadcasts the full tensor to every FFN partner; each FFN
    partner receives all sources, concatenates, runs the internally
    TP-sharded dense MLP (whose all-reduce unifies the partial TP results
    within the FFN TP group), splits the output by source count, and sends
    each slice back. ATTN picks the first partner's result (they're all
    identical after the FFN-side TP all-reduce).
"""

import dataclasses
import os
import pickle
import re
import time
from datetime import timedelta

import torch

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    ncclUniqueId,
)
from vllm.distributed.parallel_state import (
    TensorMetadata,
    init_afd_process_group,
)
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# NCCL pair helpers
# ----------------------------------------------------------------------

def _create_pynccl_comm_for_pair(
    gloo_pg: torch.distributed.ProcessGroup,
    rank_in_pair: int,
    device: int,
) -> PyNcclCommunicator:
    """Build a PyNcclCommunicator for a 2-rank pair.

    vLLM's built-in PyNcclCommunicator init uses ``dist.broadcast`` to share
    the ncclUniqueId, which fails on standalone pair groups because its
    ``pg_group_ranks`` doesn't match the caller's default-PG global rank.
    We exchange the unique id via the Gloo pair's ``send``/``recv`` directly
    (they take group-local ranks, no rank translation).
    """
    nccl = NCCLLibrary()

    if rank_in_pair == 0:
        unique_id = nccl.ncclGetUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.send([tensor], 1, 0).wait()  # → group rank 1 (ATTN)
    else:
        unique_id = ncclUniqueId()
        tensor = torch.ByteTensor(list(unique_id.internal))
        gloo_pg.recv([tensor], 0, 0).wait()  # ← group rank 0 (FFN)
        for idx, byte in enumerate(tensor.tolist()):
            unique_id.internal[idx] = byte

    device_obj = torch.device(f"cuda:{device}")
    with torch.cuda.device(device_obj):
        comm = nccl.ncclCommInitRank(2, unique_id, rank_in_pair)

    # Build a PyNcclCommunicator shell bound to the manually-created comm.
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


# ----------------------------------------------------------------------
# Custom ops for NCCL send/recv (so they can be traced by the profiler)
# ----------------------------------------------------------------------

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
    """Lightweight handle for an AFD pair.

    In every pair: FFN is rank 0, ATTN is rank 1. ``rank_in_group`` tells
    the local process which side it is.
    """
    rank_in_group: int
    world_size: int = 2
    unique_name: str = ""


class _TimingProfiler:
    """Per-process running totals of wall-clock time spent in each section
    of the AFD hot path. Enabled by env var AFD_TIMING=1.

    Not a CUDA profiler — it only captures the time spent in Python / host
    code, so it's biased toward surfacing syncs (where the host waits for
    the device) and dispatcher overhead. That's exactly what we're trying
    to localize right now.
    """
    def __init__(self) -> None:
        self.enabled: bool = os.environ.get("AFD_TIMING", "0") == "1"
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._forward_count: int = 0
        self._dump_every: int = int(os.environ.get("AFD_TIMING_EVERY", "64"))

    def add(self, section: str, dt: float) -> None:
        if not self.enabled:
            return
        self.totals[section] = self.totals.get(section, 0.0) + dt
        self.calls[section] = self.calls.get(section, 0) + 1

    def mark_forward_pass_end(self, role: str, world_rank: int) -> None:
        if not self.enabled:
            return
        self._forward_count += 1
        if self._forward_count % self._dump_every != 0:
            return
        lines = [f"[AFD_TIMING role={role} rank={world_rank} "
                 f"forwards={self._forward_count}]"]
        for k in sorted(self.totals.keys()):
            total = self.totals[k]
            calls = self.calls[k]
            avg_ms = (total / calls) * 1000 if calls else 0.0
            lines.append(
                f"  {k:42s} total={total*1000:9.2f}ms "
                f"calls={calls:6d} avg={avg_ms:6.3f}ms"
            )
        logger.info("\n".join(lines))


_timing = _TimingProfiler()


# ----------------------------------------------------------------------
# Connector
# ----------------------------------------------------------------------


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

        # Infer hidden layer count (for DeepSeek V2 and models that hide the
        # text config behind ``text_config``).
        hf_config = self.config.model_config.hf_config
        text_cfg = getattr(hf_config, "text_config", None) or hf_config
        self.num_hidden_layers: int = text_cfg.num_hidden_layers

        # MoE layer detection — both sides compute this identically.
        self.first_k_dense_replace: int = getattr(text_cfg, "first_k_dense_replace", 0)
        self.moe_layer_freq: int = getattr(text_cfg, "moe_layer_freq", 1)
        self.n_routed_experts: int = getattr(text_cfg, "n_routed_experts", 0) or 0

        # Populated in ``init_afd_connector``.
        self.role: str = ""
        self.world_rank: int = -1
        self.attn_size: int = 0
        self.ffn_size: int = 0
        self.min_size: int = 0
        self.max_size: int = 0
        self.experts_per_ffn_worker: int = 0

        # Per-partner state. Length == ``ffn_size`` for ATTN, ``attn_size`` for FFN.
        # For ATTN: index j is the pair to FFN_j.
        # For FFN:  index i is the pair to ATTN_i.
        self.a2e_groups: list[PairGroup] = []
        self.e2a_groups: list[PairGroup] = []
        self.a2e_comm_ids: list[int] = []
        self.e2a_comm_ids: list[int] = []
        self.a2e_gloo_pgs: list[torch.distributed.ProcessGroup] = []

        # ATTN-side send→recv lifecycle state. Populated in send_attn_output;
        # consumed and cleared in recv_ffn_output.
        # For MoE layers, _pending_masks is a list of GPU index tensors
        # (one per partner, result of mask.nonzero()). _pending_counts is
        # the Python-int counts read via a single .cpu() sync at send
        # time, letting recv_ffn_output skip any further data-dependent
        # operations that would force syncs behind pending NCCL/compute.
        self._pending_masks: list[torch.Tensor] | None = None
        self._pending_counts: list[int] | None = None
        self._pending_shared_output: torch.Tensor | None = None
        self._pending_shape: tuple[int, ...] | None = None
        self._pending_dtype: torch.dtype | None = None
        self._pending_device: torch.device | None = None

        # FFN-side recv→send lifecycle state. Populated in recv_attn_output;
        # consumed and cleared in send_ffn_output.
        self._pending_source_counts: list[int] | None = None

        # Tensor metadata cache (from dp_metadata). Used for stage-wise lookups.
        self._tensor_metadata_list: dict[int, TensorMetadata] = {}
        self.dp_metadata_list: dict[int, DPMetadata] | None = None
        self.is_graph_capturing: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        for comm_id in self.a2e_comm_ids:
            _unregister_comm(comm_id)
        self.a2e_comm_ids.clear()
        for comm_id in self.e2a_comm_ids:
            _unregister_comm(comm_id)
        self.e2a_comm_ids.clear()

    def init_afd_connector(self) -> None:
        """Initialize the connector with a full M×N bipartite pair topology.

        Every ATTN world rank (ffn_size..ffn_size+attn_size-1) is paired
        with every FFN world rank (0..ffn_size-1). Pair ``(attn_i, ffn_j)``
        has deterministic ``pair_id = i * ffn_size + j`` which maps to
        unique TCP ports for the Gloo rendezvous (a2e = base+100+2*pair_id,
        e2a = base+100+2*pair_id+1). Creation is deadlock-free because each
        pair has its own port and every process iterates (i, j) in the same
        order — only pair members participate.
        """
        logger.info("init_afd_connector begin")
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        self.role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(
            int, re.match(r"(\d+)\D+(\d+)", afd_size).groups()
        )

        self.attn_size = attn_size
        self.ffn_size = ffn_size
        self.min_size = min(ffn_size, attn_size)
        self.max_size = max(ffn_size, attn_size)
        self.world_rank = self.rank if self.role == "ffn" else self.rank + ffn_size

        if ffn_size > 0 and self.n_routed_experts > 0:
            # Linear expert placement: FFN_j holds experts
            # [j*experts_per_worker, (j+1)*experts_per_worker)
            assert self.n_routed_experts % ffn_size == 0, (
                f"n_routed_experts ({self.n_routed_experts}) must be divisible "
                f"by ffn_size ({ffn_size}) for linear expert placement"
            )
            self.experts_per_ffn_worker = self.n_routed_experts // ffn_size

        # Global Gloo rendezvous (used only for the initial handshake).
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
        logger.info(f"afd_pg initialized world_rank={self.world_rank}")

        afd_host = self.config.afd_config.afd_host
        afd_base_port = int(self.config.afd_config.afd_port)

        for i in range(attn_size):
            for j in range(ffn_size):
                ffn_world_rank = j
                attn_world_rank = ffn_size + i
                pair_ranks = [ffn_world_rank, attn_world_rank]

                if self.world_rank not in pair_ranks:
                    continue

                rank_in_pair = pair_ranks.index(self.world_rank)
                pair_id = i * ffn_size + j
                a2e_port = afd_base_port + 100 + pair_id * 2
                e2a_port = afd_base_port + 100 + pair_id * 2 + 1

                logger.info(
                    f"creating pair attn={i} ffn={j} pair_id={pair_id} "
                    f"rank_in_pair={rank_in_pair}"
                )

                a2e_pg = init_afd_process_group(
                    backend="gloo",
                    init_method=f"tcp://{afd_host}:{a2e_port}",
                    world_size=2,
                    rank=rank_in_pair,
                    group_name=f"a2e_{pair_id}",
                    timeout=timedelta(minutes=10),
                )
                a2e_pynccl = _create_pynccl_comm_for_pair(
                    a2e_pg, rank_in_pair, self.local_rank,
                )
                self.a2e_groups.append(
                    PairGroup(rank_in_group=rank_in_pair,
                              unique_name=f"a2e_{pair_id}")
                )
                self.a2e_comm_ids.append(_register_comm(a2e_pynccl))
                self.a2e_gloo_pgs.append(a2e_pg)

                e2a_pg = init_afd_process_group(
                    backend="gloo",
                    init_method=f"tcp://{afd_host}:{e2a_port}",
                    world_size=2,
                    rank=rank_in_pair,
                    group_name=f"e2a_{pair_id}",
                    timeout=timedelta(minutes=10),
                )
                e2a_pynccl = _create_pynccl_comm_for_pair(
                    e2a_pg, rank_in_pair, self.local_rank,
                )
                self.e2a_groups.append(
                    PairGroup(rank_in_group=rank_in_pair,
                              unique_name=f"e2a_{pair_id}")
                )
                self.e2a_comm_ids.append(_register_comm(e2a_pynccl))

        expected_partners = ffn_size if self.role == "attention" else attn_size
        assert len(self.a2e_groups) == expected_partners, (
            f"Expected {expected_partners} partner pairs for role={self.role} "
            f"in {attn_size}A{ffn_size}F, got {len(self.a2e_groups)}"
        )
        logger.info(
            f"[P2P] world_rank={self.world_rank} role={self.role} "
            f"created {len(self.a2e_groups)} pair groups "
            f"(M×N bipartite {attn_size}A{ffn_size}F)"
        )

        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Low-level NCCL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _partner_rank_in_pair(rank_in_group: int) -> int:
        """In a 2-rank pair, partner rank is the other one."""
        return 1 - rank_in_group

    def _nccl_send(
        self,
        tensor: torch.Tensor,
        dst: int,
        comm_id: int,
        nvtx_label: str = "",
    ) -> None:
        assert not tensor.is_cpu, "tensor must be on GPU"
        nvtx_msg = (
            f"afd_p2p_send|{nvtx_label}|shape={list(tensor.shape)}"
            f"|dtype={tensor.dtype}|dst={dst}"
            f"|bytes={tensor.numel() * tensor.element_size()}"
        )
        with torch.profiler.record_function("afd_p2p_send", args=nvtx_msg), \
             torch.cuda.nvtx.range(nvtx_msg):
            torch.ops.vllm.afd_p2p_send(tensor, dst, comm_id)

    def _nccl_recv_into(
        self,
        tensor: torch.Tensor,
        src: int,
        comm_id: int,
        nvtx_label: str = "",
    ) -> None:
        assert not tensor.is_cpu, "tensor must be on GPU"
        nvtx_msg = (
            f"afd_p2p_recv|{nvtx_label}|shape={list(tensor.shape)}"
            f"|dtype={tensor.dtype}|src={src}"
            f"|bytes={tensor.numel() * tensor.element_size()}"
        )
        with torch.profiler.record_function("afd_p2p_recv", args=nvtx_msg), \
             torch.cuda.nvtx.range(nvtx_msg):
            torch.ops.vllm.afd_p2p_recv(tensor, src, comm_id)

    # ------------------------------------------------------------------
    # dp_metadata (control plane, one-shot per forward pass)
    # ------------------------------------------------------------------

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, DPMetadata],
        is_graph_capturing: bool = False,
    ) -> None:
        self.dp_metadata_list = dp_metadata_list
        self.is_graph_capturing = is_graph_capturing
        num_of_stages = len(dp_metadata_list)
        hidden_size = self.config.model_config.hf_config.hidden_size
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.config.model_config.dtype

        self._tensor_metadata_list = {}
        for stage_idx in range(num_of_stages):
            dp_metadata = dp_metadata_list[stage_idx]
            dp_rank = self.config.parallel_config.data_parallel_rank
            num_tokens = dp_metadata.num_tokens_across_dp_cpu[dp_rank].item()
            self._tensor_metadata_list[stage_idx] = TensorMetadata(
                device, dtype,
                torch.Size([num_tokens, hidden_size]),
            )

    def send_dp_metadata_list(self, data, is_graph_capturing: bool = False):
        """ATTN DP0 → all FFN workers: broadcast the dp_metadata dict over Gloo.

        Every FFN worker receives on its first a2e Gloo pair (which is the
        pair from ATTN DP0). Only ATTN DP0 calls this — gated by
        ``is_attn_top_min_size_rank``.
        """
        self.update_state_from_dp_metadata(data, is_graph_capturing)
        send_data = (data, is_graph_capturing)
        object_bytes = pickle.dumps(send_data)
        object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)

        for gloo_pg in self.a2e_gloo_pgs:
            gloo_pg.send([size_tensor], 0, 0).wait()
            gloo_pg.send([object_tensor], 0, 0).wait()

    def recv_dp_metadata_list(self):
        """FFN → recv from its ATTN DP0 pair (index 0 of a2e_gloo_pgs)."""
        gloo_pg = self.a2e_gloo_pgs[0]

        size_tensor = torch.empty(1, dtype=torch.long)
        gloo_pg.recv([size_tensor], 1, 0).wait()
        object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
        gloo_pg.recv([object_tensor], 1, 0).wait()

        data, is_graph_capturing = pickle.loads(object_tensor.numpy().tobytes())
        return data, is_graph_capturing

    def is_attn_top_min_size_rank(self, rank) -> bool:
        """Only ATTN DP0 sends dp_metadata (and broadcasts to all FFN pairs)."""
        if self.config.afd_config.afd_role != "attention":
            return False
        dp_rank = self.config.parallel_config.data_parallel_rank
        return dp_rank == 0

    # ------------------------------------------------------------------
    # ATTN → FFN (hot path)
    # ------------------------------------------------------------------

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
        topk_ids: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        shared_output: torch.Tensor | None = None,
    ) -> None:
        """Send tokens to every FFN partner, optionally with pre-routing.

        Arguments:
          hidden_states: [N, H] post-attention tensor.
          metadata:      AFDConnectorMetadata carrying per-layer info (layer
                         index, stage, etc.). Sent implicitly by position.
          topk_ids:      [N, K] int32 (global expert indices) for MoE layers.
          topk_weights:  [N, K] float32 (routing weights) for MoE layers.
          shared_output: [N, H] shared-expert output computed on the ATTN
                         side; added after the partials are combined.

        Protocol per FFN partner j:
          1. Send count header ``[count_j, topk_k]`` (int64 tensor, shape 2)
             — the FFN side peeks at this to know what follows.
          2. If ``count_j > 0``: send ``hs_j [count_j, H]``.
          3. If ``count_j > 0`` and ``topk_k > 0``: send topk_ids_j and
             topk_weights_j for the masked subset.

        For dense layers (topk_ids=None), the full hidden_states is
        broadcast to all FFN partners and the count header uses topk_k=0.

        Masks + shared_output are stashed for the matching ``recv_ffn_output``.
        """
        n_partners = len(self.a2e_groups)
        assert n_partners > 0, "No FFN partners configured"

        # Stash the shape so recv_ffn_output can allocate the output buffer.
        self._pending_shape = tuple(hidden_states.shape)
        self._pending_dtype = hidden_states.dtype
        self._pending_device = hidden_states.device
        self._pending_shared_output = shared_output

        _t_send_total = time.perf_counter()

        if topk_ids is None:
            # Dense path: broadcast full tensor to every FFN partner.
            self._pending_masks = None
            for j in range(n_partners):
                group = self.a2e_groups[j]
                comm_id = self.a2e_comm_ids[j]
                dst = self._partner_rank_in_pair(group.rank_in_group)

                count_hdr = torch.tensor(
                    [hidden_states.shape[0], 0],
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                self._nccl_send(
                    count_hdr, dst, comm_id,
                    nvtx_label=f"attn->ffn[count_hdr,j={j}]",
                )
                if hidden_states.shape[0] > 0:
                    self._nccl_send(
                        hidden_states, dst, comm_id,
                        nvtx_label=f"attn->ffn[hs,j={j}]",
                    )
            _timing.add("send_attn.dense_total", time.perf_counter() - _t_send_total)
            return

        # MoE pre-routing path — fast path: batch all data-dependent work,
        # sync once to read counts + indices on CPU, then do all NCCL sends
        # without any further syncs.
        assert topk_weights is not None
        assert self.experts_per_ffn_worker > 0
        assert topk_ids.shape[0] == hidden_states.shape[0]
        topk_k = topk_ids.shape[1]

        # ---- Stage 1: batched mask computation on GPU (async). ----
        _t = time.perf_counter()
        starts = torch.arange(n_partners, device=hidden_states.device) * \
            self.experts_per_ffn_worker
        ends = starts + self.experts_per_ffn_worker
        # in_range: [P, N, K] where P=n_partners.
        in_range = (topk_ids.unsqueeze(0) >= starts[:, None, None]) & (
            topk_ids.unsqueeze(0) < ends[:, None, None]
        )
        all_masks = in_range.any(dim=-1)  # [P, N]
        counts = all_masks.sum(dim=1)  # [P]
        _timing.add("send_attn.moe.mask_compute", time.perf_counter() - _t)

        # ---- Stage 2: ONE sync to read counts, and precompute nonzero
        # indices per partner now (while we're paying the sync cost
        # anyway — any later .item() would pay the same). ----
        _t = time.perf_counter()
        counts_cpu = counts.tolist()  # single CPU sync for all partners
        # Precompute per-partner nonzero indices so recv_ffn_output can
        # skip nonzero() at recv time (which would otherwise force a
        # second, much more expensive sync behind pending NCCL+compute).
        per_partner_indices: list[torch.Tensor] = []
        for j in range(n_partners):
            idx_j = all_masks[j].nonzero(as_tuple=True)[0]  # [count_j]
            per_partner_indices.append(idx_j)
        _timing.add("send_attn.moe.slice_subset", time.perf_counter() - _t)

        # ---- Stage 3: per-partner NCCL sends using known counts. ----
        masks_for_recv: list[torch.Tensor] = []
        for j in range(n_partners):
            count_j = counts_cpu[j]
            idx_j = per_partner_indices[j]
            masks_for_recv.append(idx_j)  # store indices, not the bool mask

            group = self.a2e_groups[j]
            comm_id = self.a2e_comm_ids[j]
            dst = self._partner_rank_in_pair(group.rank_in_group)

            _t = time.perf_counter()
            count_hdr = torch.tensor(
                [count_j, topk_k],
                dtype=torch.int64,
                device=hidden_states.device,
            )
            self._nccl_send(
                count_hdr, dst, comm_id,
                nvtx_label=f"attn->ffn[count_hdr,j={j}]",
            )
            _timing.add("send_attn.moe.count_hdr_send", time.perf_counter() - _t)

            if count_j > 0:
                _t = time.perf_counter()
                # Use index_select with the precomputed indices. This gives
                # known-shape outputs (no hidden sync). index_select is
                # queued on the stream and runs in parallel with the NCCL
                # sends issued right after.
                hs_j = hidden_states.index_select(0, idx_j)
                topk_ids_j = topk_ids.index_select(0, idx_j)
                topk_weights_j = topk_weights.index_select(0, idx_j)
                self._nccl_send(
                    hs_j, dst, comm_id,
                    nvtx_label=f"attn->ffn[hs,j={j}]",
                )
                self._nccl_send(
                    topk_ids_j, dst, comm_id,
                    nvtx_label=f"attn->ffn[topk_ids,j={j}]",
                )
                self._nccl_send(
                    topk_weights_j, dst, comm_id,
                    nvtx_label=f"attn->ffn[topk_weights,j={j}]",
                )
                _timing.add("send_attn.moe.tensor_sends", time.perf_counter() - _t)

        self._pending_masks = masks_for_recv  # now: list of index tensors
        self._pending_counts = counts_cpu  # stash counts to avoid re-sync
        _timing.add("send_attn.moe_total", time.perf_counter() - _t_send_total)

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Receive partial FFN outputs and combine into the final tensor.

        Two modes:
          - MoE (masks stashed by send_attn_output): receive one partial per
            FFN partner, ``index_add_`` into a zero-init'd output using the
            stored mask, then add the pending shared-expert output.
          - Dense (no masks): receive from each partner (all identical after
            the FFN-side TP all-reduce) and return the first one.

        The ``ref_tensor`` parameter is accepted for signature compatibility
        with the old connector; its only use is to inherit shape/dtype/device
        when pending metadata is somehow missing.
        """
        n_partners = len(self.e2a_groups)

        if self._pending_shape is None:
            # Fallback: recv_ffn_output called without a matching send. Use
            # ref_tensor if available.
            assert ref_tensor is not None
            shape = tuple(ref_tensor.shape)
            dtype = ref_tensor.dtype
            device = ref_tensor.device
        else:
            shape = self._pending_shape
            dtype = self._pending_dtype
            device = self._pending_device

        _t_recv_total = time.perf_counter()

        if self._pending_masks is None:
            # Dense: recv from each partner, keep the first result.
            final: torch.Tensor | None = None
            for j in range(n_partners):
                group = self.e2a_groups[j]
                comm_id = self.e2a_comm_ids[j]
                src = self._partner_rank_in_pair(group.rank_in_group)
                buf = torch.empty(shape, dtype=dtype, device=device)
                self._nccl_recv_into(
                    buf, src, comm_id,
                    nvtx_label=f"attn<-ffn[hs,j={j}]",
                )
                if final is None:
                    final = buf
            if final is None:
                final = torch.zeros(shape, dtype=dtype, device=device)
            self._clear_attn_pending()
            _timing.add("recv_ffn.dense_total", time.perf_counter() - _t_recv_total)
            return final

        # MoE: recv partials and scatter-add.
        # Counts and per-partner index tensors were stashed by
        # send_attn_output, so we do not need to call .sum().item() or
        # .nonzero() here (which would force a sync behind the NCCL
        # recvs and compute on the stream).
        hidden_size = shape[-1]
        _t = time.perf_counter()
        final_hidden = torch.zeros(shape, dtype=dtype, device=device)
        _timing.add("recv_ffn.moe.alloc_final", time.perf_counter() - _t)
        assert self._pending_counts is not None
        for j in range(n_partners):
            indices_j = self._pending_masks[j]  # GPU index tensor [count_j]
            count_j = self._pending_counts[j]  # Python int (from send sync)
            if count_j == 0:
                continue
            group = self.e2a_groups[j]
            comm_id = self.e2a_comm_ids[j]
            src = self._partner_rank_in_pair(group.rank_in_group)
            _t = time.perf_counter()
            partial = torch.empty(
                (count_j, hidden_size), dtype=dtype, device=device,
            )
            self._nccl_recv_into(
                partial, src, comm_id,
                nvtx_label=f"attn<-ffn[partial,j={j}]",
            )
            _timing.add("recv_ffn.moe.recv_partial", time.perf_counter() - _t)
            _t = time.perf_counter()
            final_hidden.index_add_(0, indices_j, partial.to(final_hidden.dtype))
            _timing.add("recv_ffn.moe.scatter_add", time.perf_counter() - _t)

        if self._pending_shared_output is not None:
            final_hidden = final_hidden + self._pending_shared_output.to(
                final_hidden.dtype
            )

        self._clear_attn_pending()
        _timing.add("recv_ffn.moe_total", time.perf_counter() - _t_recv_total)
        # One forward pass on ATTN is made of many send_attn/recv_ffn
        # layer calls. Use the final recv (from forward_with_afd after the
        # layer loop) as the "end of forward pass" marker. The layer-0
        # case skips this since there's an extra recv before/after, but
        # the totals are still useful.
        _timing.mark_forward_pass_end(role=self.role, world_rank=self.world_rank)
        return final_hidden

    def _clear_attn_pending(self) -> None:
        self._pending_masks = None
        self._pending_counts = None
        self._pending_shared_output = None
        self._pending_shape = None
        self._pending_dtype = None
        self._pending_device = None

    # ------------------------------------------------------------------
    # FFN side (hot path)
    # ------------------------------------------------------------------

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """Receive tokens from every ATTN partner and concatenate.

        Returns ``(hidden_states, metadata)``. Metadata carries:
          - ``seq_lens``: per-source token counts (how many tokens came from
            each ATTN partner). Used by ``send_ffn_output`` to split the
            compute output for the return trip.
          - ``topk_ids``, ``topk_weights``: concatenated across sources when
            this is a MoE layer; ``None`` for dense layers.
        """
        _t_recv_total = time.perf_counter()
        n_partners = len(self.a2e_groups)
        hidden_size = self.config.model_config.hf_config.hidden_size
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.config.model_config.dtype

        # ---- Stage 1: post all count_hdr recvs first (all async), then
        # sync ONCE to read every partner's count. This avoids N separate
        # .cpu() calls, each of which would drain the whole CUDA stream
        # backlog. ----
        _t = time.perf_counter()
        count_hdrs: list[torch.Tensor] = []
        for i in range(n_partners):
            group = self.a2e_groups[i]
            comm_id = self.a2e_comm_ids[i]
            src = self._partner_rank_in_pair(group.rank_in_group)
            count_hdr = torch.empty((2,), dtype=torch.int64, device=device)
            self._nccl_recv_into(
                count_hdr, src, comm_id,
                nvtx_label=f"ffn<-attn[count_hdr,i={i}]",
            )
            count_hdrs.append(count_hdr)
        # Stack into one [n_partners, 2] tensor and do a single CPU sync.
        stacked = torch.stack(count_hdrs, dim=0)
        stacked_cpu = stacked.cpu().tolist()
        source_counts: list[int] = [row[0] for row in stacked_cpu]
        topk_k_per_partner: list[int] = [row[1] for row in stacked_cpu]
        any_topk = any(k > 0 for k in topk_k_per_partner)
        _timing.add("recv_attn.count_hdr_sync", time.perf_counter() - _t)

        # ---- Stage 2: now that counts are known, post all tensor recvs. ----
        hs_parts: list[torch.Tensor] = []
        topk_ids_parts: list[torch.Tensor] = []
        topk_weights_parts: list[torch.Tensor] = []

        _t = time.perf_counter()
        for i in range(n_partners):
            count_i = source_counts[i]
            topk_k_i = topk_k_per_partner[i]
            if count_i == 0:
                continue

            group = self.a2e_groups[i]
            comm_id = self.a2e_comm_ids[i]
            src = self._partner_rank_in_pair(group.rank_in_group)

            hs_buf = torch.empty(
                (count_i, hidden_size), dtype=dtype, device=device,
            )
            self._nccl_recv_into(
                hs_buf, src, comm_id,
                nvtx_label=f"ffn<-attn[hs,i={i}]",
            )
            hs_parts.append(hs_buf)

            if topk_k_i > 0:
                topk_ids_buf = torch.empty(
                    (count_i, topk_k_i), dtype=torch.int32, device=device,
                )
                self._nccl_recv_into(
                    topk_ids_buf, src, comm_id,
                    nvtx_label=f"ffn<-attn[topk_ids,i={i}]",
                )
                topk_weights_buf = torch.empty(
                    (count_i, topk_k_i), dtype=torch.float32, device=device,
                )
                self._nccl_recv_into(
                    topk_weights_buf, src, comm_id,
                    nvtx_label=f"ffn<-attn[topk_weights,i={i}]",
                )
                topk_ids_parts.append(topk_ids_buf)
                topk_weights_parts.append(topk_weights_buf)
        _timing.add("recv_attn.tensor_recvs", time.perf_counter() - _t)

        self._pending_source_counts = source_counts

        _t = time.perf_counter()
        total_tokens = sum(source_counts)
        if total_tokens == 0:
            hs = torch.empty((0, hidden_size), dtype=dtype, device=device)
        elif len(hs_parts) == 1:
            hs = hs_parts[0]
        else:
            hs = torch.cat(hs_parts, dim=0)
        _timing.add("recv_attn.cat", time.perf_counter() - _t)

        if any_topk:
            topk_ids = (
                topk_ids_parts[0] if len(topk_ids_parts) == 1
                else torch.cat(topk_ids_parts, dim=0)
            )
            topk_weights = (
                topk_weights_parts[0] if len(topk_weights_parts) == 1
                else torch.cat(topk_weights_parts, dim=0)
            )
        else:
            topk_ids = None
            topk_weights = None

        meta = AFDConnectorMetadata(
            layer_idx=0,  # not consumed by FFN
            stage_idx=ubatch_idx,
            # AFDConnectorMetadata.__post_init__ rejects empty seq_lens; use
            # [1] as a placeholder when we received nothing.
            seq_lens=source_counts if total_tokens > 0 else [1],
            dtype=dtype,
            device=device,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        _timing.add("recv_attn.total", time.perf_counter() - _t_recv_total)
        return hs, meta

    def send_ffn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """Split the FFN output by source partner and return each slice.

        Relies on ``self._pending_source_counts`` set by the matching
        ``recv_attn_output``. The order matches the ATTN partner order
        (i = 0, 1, ..., attn_size-1).
        """
        _t_send_total = time.perf_counter()
        assert self._pending_source_counts is not None
        source_counts = self._pending_source_counts
        n_partners = len(self.e2a_groups)
        assert len(source_counts) == n_partners, (
            f"source_counts length {len(source_counts)} != {n_partners} partners"
        )

        offset = 0
        for i in range(n_partners):
            count_i = source_counts[i]
            if count_i == 0:
                continue
            slice_i = hidden_states[offset:offset + count_i]
            offset += count_i

            group = self.e2a_groups[i]
            comm_id = self.e2a_comm_ids[i]
            dst = self._partner_rank_in_pair(group.rank_in_group)
            self._nccl_send(
                slice_i, dst, comm_id,
                nvtx_label=f"ffn->attn[partial,i={i}]",
            )

        self._pending_source_counts = None
        _timing.add("send_ffn.total", time.perf_counter() - _t_send_total)
        # FFN forward pass end marker: send_ffn_output is called once per
        # layer, and the final layer's send is followed by the FFN worker
        # loop's cuda.synchronize(). Treat this as "one unit of work".
        _timing.mark_forward_pass_end(role=self.role, world_rank=self.world_rank)
