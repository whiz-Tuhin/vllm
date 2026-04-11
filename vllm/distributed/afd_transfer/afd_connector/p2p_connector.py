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

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Whether the given layer index is an MoE layer vs a dense MLP
        (layer 0 in DeepSeek-V2)."""
        if self.n_routed_experts <= 0:
            return False
        if layer_idx < self.first_k_dense_replace:
            return False
        return (layer_idx - self.first_k_dense_replace) % self.moe_layer_freq == 0

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

        # Option B — fixed-size protocol: we always broadcast the full
        # ``hidden_states`` tensor to every FFN partner. For MoE layers
        # we additionally broadcast ``topk_ids`` and ``topk_weights``
        # so the FFN side can run the pre-computed routing without
        # re-running the gate. There is no masking, no count header,
        # no CPU sync on the hot path — all shapes are known statically
        # from the dp_metadata that was broadcast once per forward pass.
        #
        # Bandwidth-wise this is equivalent to the old EP all-gather +
        # reduce-scatter pattern. The win is structural: (1) no
        # collective coordination between FFN workers, so the M×N
        # topology works for asymmetric configs; (2) zero per-layer
        # CPU/GPU syncs, so NCCL ops can pipeline without draining
        # stalls; (3) correct for EP=2 where the bandwidth savings
        # of "real" pre-routing were negligible anyway (98% of tokens
        # need both workers with top-k=6).
        self._pending_masks = None  # signals recv_ffn_output to use the
                                    # element-wise sum combine path.

        is_moe = topk_ids is not None
        for j in range(n_partners):
            group = self.a2e_groups[j]
            comm_id = self.a2e_comm_ids[j]
            dst = self._partner_rank_in_pair(group.rank_in_group)

            _t = time.perf_counter()
            self._nccl_send(
                hidden_states, dst, comm_id,
                nvtx_label=f"attn->ffn[hs,j={j}]",
            )
            if is_moe:
                assert topk_weights is not None
                self._nccl_send(
                    topk_ids, dst, comm_id,
                    nvtx_label=f"attn->ffn[topk_ids,j={j}]",
                )
                self._nccl_send(
                    topk_weights, dst, comm_id,
                    nvtx_label=f"attn->ffn[topk_weights,j={j}]",
                )
            _timing.add("send_attn.tensor_sends", time.perf_counter() - _t)

        label = "send_attn.moe_total" if is_moe else "send_attn.dense_total"
        _timing.add(label, time.perf_counter() - _t_send_total)

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

        # All paths are now fixed-size: each FFN partner returns a tensor of
        # the same shape as the ATTN hidden_states we sent. The combine rule
        # differs:
        #   - Dense layer (no shared output, no expert routing): pick one
        #     partner's result; they're identical after the FFN's internal
        #     TP all-reduce.
        #   - MoE layer: every partner returns a partial where only its
        #     local experts contributed; sum across partners to get the
        #     full expert weighted sum, then add the shared-expert output
        #     computed locally.
        is_dense = self._pending_shared_output is None

        final_hidden: torch.Tensor | None = None
        for j in range(n_partners):
            group = self.e2a_groups[j]
            comm_id = self.e2a_comm_ids[j]
            src = self._partner_rank_in_pair(group.rank_in_group)
            buf = torch.empty(shape, dtype=dtype, device=device)
            _t = time.perf_counter()
            self._nccl_recv_into(
                buf, src, comm_id,
                nvtx_label=f"attn<-ffn[hs,j={j}]",
            )
            _timing.add("recv_ffn.recv_partial", time.perf_counter() - _t)

            if is_dense:
                # First partner wins; later partners' tensors are still
                # drained off the stream but their data is ignored.
                if final_hidden is None:
                    final_hidden = buf
            else:
                _t = time.perf_counter()
                if final_hidden is None:
                    final_hidden = buf
                else:
                    final_hidden = final_hidden + buf
                _timing.add("recv_ffn.combine_add", time.perf_counter() - _t)

        if final_hidden is None:
            final_hidden = torch.zeros(shape, dtype=dtype, device=device)

        if self._pending_shared_output is not None:
            _t = time.perf_counter()
            final_hidden = final_hidden + self._pending_shared_output.to(
                final_hidden.dtype
            )
            _timing.add("recv_ffn.add_shared", time.perf_counter() - _t)

        self._clear_attn_pending()
        _timing.add("recv_ffn.total", time.perf_counter() - _t_recv_total)
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
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """Receive hidden_states (and optionally topk) from every ATTN partner.

        Option B fixed-size protocol:
          - Per-partner token count is read from ``dp_metadata_list`` for the
            current stage. Every ATTN partner sends the same shape, so recv
            buffers are statically sized — no count handshake, no CPU sync.
          - For MoE layers, ATTN broadcasts the full ``(hidden_states,
            topk_ids, topk_weights)``. For dense layer 0, only ``hidden_states``.
            The caller drives this via ``layer_idx`` and the connector's cached
            MoE layer predicate.

        Returns ``(hs_concat, metadata)`` where ``hs_concat`` is the
        per-partner tensors concatenated along dim=0 and ``metadata`` carries
        the per-partner source counts (for ``send_ffn_output`` to split the
        compute output) plus the concatenated topk tensors on MoE layers.
        """
        _t_recv_total = time.perf_counter()
        n_partners = len(self.a2e_groups)
        hidden_size = self.config.model_config.hf_config.hidden_size
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.config.model_config.dtype
        is_moe = self._is_moe_layer(layer_idx)

        assert self.dp_metadata_list is not None, (
            "recv_attn_output called before dp_metadata_list was populated"
        )
        dp_metadata = self.dp_metadata_list[ubatch_idx]
        num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.tolist()
        # For xA1F (e.g. 3A1F) each ATTN DP rank has its own token count, so
        # num_tokens_across_dp has length == attn_size == n_partners (for the
        # single FFN worker). For 1AxF the single ATTN always sends the same
        # count to each FFN partner.
        if len(num_tokens_across_dp) == n_partners:
            per_partner_count = num_tokens_across_dp
        else:
            per_partner_count = [num_tokens_across_dp[0]] * n_partners

        topk_k = 0
        if is_moe:
            cfg = self.config.model_config.hf_config
            if hasattr(cfg, "text_config"):
                cfg = cfg.text_config
            topk_k = getattr(cfg, "num_experts_per_tok", 0)

        hs_parts: list[torch.Tensor] = []
        topk_ids_parts: list[torch.Tensor] = []
        topk_weights_parts: list[torch.Tensor] = []

        for i in range(n_partners):
            count_i = per_partner_count[i]
            group = self.a2e_groups[i]
            comm_id = self.a2e_comm_ids[i]
            src = self._partner_rank_in_pair(group.rank_in_group)

            hs_buf = torch.empty(
                (count_i, hidden_size), dtype=dtype, device=device,
            )
            _t = time.perf_counter()
            self._nccl_recv_into(
                hs_buf, src, comm_id,
                nvtx_label=f"ffn<-attn[hs,i={i}]",
            )
            hs_parts.append(hs_buf)

            if is_moe and topk_k > 0:
                topk_ids_buf = torch.empty(
                    (count_i, topk_k), dtype=torch.int32, device=device,
                )
                self._nccl_recv_into(
                    topk_ids_buf, src, comm_id,
                    nvtx_label=f"ffn<-attn[topk_ids,i={i}]",
                )
                topk_weights_buf = torch.empty(
                    (count_i, topk_k), dtype=torch.float32, device=device,
                )
                self._nccl_recv_into(
                    topk_weights_buf, src, comm_id,
                    nvtx_label=f"ffn<-attn[topk_weights,i={i}]",
                )
                topk_ids_parts.append(topk_ids_buf)
                topk_weights_parts.append(topk_weights_buf)
            _timing.add("recv_attn.tensor_recvs", time.perf_counter() - _t)

        self._pending_source_counts = per_partner_count

        _t = time.perf_counter()
        hs = hs_parts[0] if len(hs_parts) == 1 else torch.cat(hs_parts, dim=0)
        if is_moe and topk_ids_parts:
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
        _timing.add("recv_attn.cat", time.perf_counter() - _t)

        meta = AFDConnectorMetadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=per_partner_count,
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
        _timing.mark_forward_pass_end(role=self.role, world_rank=self.world_rank)
