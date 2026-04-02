# SPDX-License-Identifier: Apache-2.0
"""DryRunModelRunner — replaces GPUModelRunner with sleep-based simulation.

Instead of running real GPU kernels, this model runner:
1. Looks up per-operator runtimes from a RuntimeDB
2. Sums them into a total forward-pass duration
3. Calls time.sleep() for that duration
4. Returns a valid ModelRunnerOutput with dummy sampled tokens
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import the real vLLM ModelRunnerOutput. If unavailable
# (e.g. in lightweight tests), fall back to a local replica.
try:
    from vllm.v1.outputs import ModelRunnerOutput as _VllmModelRunnerOutput

    def _make_model_runner_output(
        req_ids: list[str],
        req_id_to_index: dict[str, int],
        sampled_token_ids: list[list[int]],
    ) -> Any:
        return _VllmModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )

except ImportError:
    pass


@dataclass
class ModelRunnerOutput:
    """Minimal replica of vllm.v1.outputs.ModelRunnerOutput.

    Used when the real vLLM output class is unavailable (e.g. in tests).
    """

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: Any = None
    prompt_logprobs_dict: dict = field(default_factory=dict)
    pooler_output: Any = None
    kv_connector_output: Any = None
    ec_connector_output: Any = None
    num_nans_in_logits: Any = None
    cudagraph_stats: Any = None


class DryRunModelRunner:
    """Drop-in replacement for GPUModelRunner that simulates execution.

    Args:
        vllm_config: VllmConfig (or compatible fake) with model/cache/parallel
            config attributes.
        hw_profile: HWProfile (or mock) providing tflops_fp16 and bandwidth.
        runtime_db: RuntimeDB (or mock) for operator latency lookups.
        device: torch.device — ignored, kept for API compatibility.
        parallelism_shim: ParallelismShim (or None) for TP/PP overhead.
    """

    def __init__(
        self,
        vllm_config: Any,
        hw_profile: Any,
        runtime_db: Any,
        device: Any,
        parallelism_shim: Any = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.hw_profile = hw_profile
        self.runtime_db = runtime_db
        self.device = device
        self.parallelism_shim = parallelism_shim

        # One-shot diagnostic logging flags
        self._logged_gemm = False
        self._logged_ctx_attn = False
        self._logged_gen_attn = False
        self._logged_forward_pass = False

        # Extract model architecture parameters
        hf = vllm_config.model_config.hf_config
        self.num_layers: int = hf.num_hidden_layers
        self.hidden_size: int = hf.hidden_size
        self.num_heads: int = hf.num_attention_heads
        self.num_kv_heads: int = hf.num_key_value_heads
        self.head_dim: int = self.hidden_size // self.num_heads
        self.intermediate_size: int = hf.intermediate_size
        self.vocab_size: int = hf.vocab_size

        # MoE configuration — only active when VLLM_DRY_RUN_MOE=1
        self.moe_enabled: bool = (
            os.environ.get("VLLM_DRY_RUN_MOE", "0") == "1"
        )
        if self.moe_enabled:
            self.num_experts: int = (
                getattr(hf, "num_local_experts", 0)
                or getattr(hf, "num_experts", 0)
                or getattr(hf, "n_routed_experts", 0)
                or getattr(hf, "moe_num_experts", 0)
            )
            self.num_experts_per_tok: int = getattr(
                hf, "num_experts_per_tok", 1
            )
            self.moe_layer_step: int = getattr(
                hf, "interleave_moe_layer_step", 0
            )
            self.is_moe_model: bool = self.num_experts > 0
            self.moe_layer_mask: list[bool] = self._build_moe_layer_mask()
        else:
            self.num_experts = 0
            self.num_experts_per_tok = 1
            self.moe_layer_step = 0
            self.is_moe_model = False
            self.moe_layer_mask = [False] * self.num_layers
        self._logged_moe = False
        self._logged_comm = False

        # Parallelism
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.ep_size: int = int(
            os.environ.get("VLLM_DRY_RUN_EP_SIZE", "1")
        )

        # Cache config
        self.block_size: int = vllm_config.cache_config.block_size

        # Cached output for sample_tokens()
        self._last_output: Optional[Any] = None

    # ------------------------------------------------------------------
    # Model execution (the core simulation)
    # ------------------------------------------------------------------

    def execute_model(self, scheduler_output: Any) -> Optional[Any]:
        """Simulate a forward pass.

        Args:
            scheduler_output: Must have ``num_scheduled_tokens`` (dict) and
                ``total_num_scheduled_tokens`` (int).

        Returns:
            ModelRunnerOutput with dummy sampled tokens, or None when no
            tokens are scheduled.
        """
        num_scheduled = scheduler_output.total_num_scheduled_tokens
        if num_scheduled == 0:
            # Return an empty (but valid) output — None would signal an error
            # to the engine's batch queue processing.
            try:
                from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
                return EMPTY_MODEL_RUNNER_OUTPUT
            except ImportError:
                return ModelRunnerOutput(req_ids=[], req_id_to_index={})

        per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(per_req)

        # Classify prefill vs decode tokens
        prefill_tokens = sum(t for t in per_req.values() if t > 1)
        decode_tokens = sum(1 for t in per_req.values() if t == 1)
        total_tokens = num_scheduled

        # Estimate forward-pass latency and sleep
        total_us = self._estimate_forward_pass_us(
            num_reqs=num_reqs,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            total_tokens=total_tokens,
        )
        if not self._logged_forward_pass:
            logger.info(
                "Forward pass estimate: reqs=%d prefill=%d decode=%d "
                "total_tokens=%d → %.1f μs (%.3f ms)",
                num_reqs, prefill_tokens, decode_tokens,
                total_tokens, total_us, total_us / 1000,
            )
            self._logged_forward_pass = True
        time.sleep(total_us / 1_000_000)

        # Build output
        output = self._make_output(per_req)
        self._last_output = output
        return output

    def sample_tokens(self) -> Any:
        """Return the cached output from the last execute_model call."""
        assert self._last_output is not None, (
            "sample_tokens called before execute_model"
        )
        out = self._last_output
        self._last_output = None
        return out

    # ------------------------------------------------------------------
    # Forward-pass latency estimation
    # ------------------------------------------------------------------

    def _estimate_forward_pass_us(
        self,
        num_reqs: int,
        prefill_tokens: int,
        decode_tokens: int,
        total_tokens: int,
    ) -> float:
        """Estimate total forward-pass latency in microseconds."""
        total_us = 0.0
        m = max(total_tokens, 1)

        num_moe_layers = 0
        for layer_idx in range(self.num_layers):
            qkv_us = self._lookup_gemm_us(
                m=m,
                n=(self.num_heads + 2 * self.num_kv_heads)
                * self.head_dim
                // self.tp_size,
                k=self.hidden_size,
            )
            total_us += qkv_us

            if prefill_tokens > 0:
                avg_prefill_seq = max(
                    prefill_tokens // max(num_reqs, 1), 1
                )
                ctx_us = self._lookup_context_attention_us(
                    batch_size=num_reqs,
                    seq_len=avg_prefill_seq,
                )
                total_us += ctx_us

            if decode_tokens > 0:
                gen_us = self._lookup_generation_attention_us(
                    batch_size=decode_tokens,
                    seq_len=128,
                )
                total_us += gen_us

            out_proj_us = self._lookup_gemm_us(
                m=m,
                n=self.hidden_size,
                k=self.num_heads * self.head_dim // self.tp_size,
            )
            total_us += out_proj_us

            # FFN: MoE layer (when enabled) or dense MLP
            if self.moe_layer_mask[layer_idx]:
                total_us += self._lookup_moe_us(num_tokens=m)
                num_moe_layers += 1
            else:
                mlp_up_us = self._lookup_gemm_us(
                    m=m,
                    n=self.intermediate_size // self.tp_size,
                    k=self.hidden_size,
                )
                total_us += mlp_up_us

                mlp_down_us = self._lookup_gemm_us(
                    m=m,
                    n=self.hidden_size,
                    k=self.intermediate_size // self.tp_size,
                )
                total_us += mlp_down_us

        # Add TP/PP/EP communication overhead
        if self.parallelism_shim is not None:
            num_reqs_for_comm = max(num_reqs, 1)
            avg_seq_len = max(total_tokens // num_reqs_for_comm, 1)
            comm_overhead = self.parallelism_shim.get_overhead(
                batch_size=num_reqs_for_comm,
                seq_len=avg_seq_len,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_moe_layers=num_moe_layers,
                num_experts=self.num_experts,
                topk=self.num_experts_per_tok,
            )
            total_us += comm_overhead.total_us
            if not self._logged_comm:
                logger.info(
                    "Communication overhead: tp_allreduce=%.1f μs, "
                    "ep_all2all=%.1f μs, total=%.1f μs",
                    comm_overhead.tp_allreduce_us,
                    comm_overhead.ep_all2all_us,
                    comm_overhead.total_us,
                )
                self._logged_comm = True

        return total_us

    # ------------------------------------------------------------------
    # MoE helpers
    # ------------------------------------------------------------------

    def _build_moe_layer_mask(self) -> list[bool]:
        """Build per-layer mask: True = MoE layer, False = dense."""
        if not self.is_moe_model:
            return [False] * self.num_layers
        if self.moe_layer_step <= 0:
            # All layers are MoE (Mixtral, Qwen3-MoE)
            return [True] * self.num_layers
        # Interleaved: layer i is MoE when (i % step) == (step - 1)
        # step=2 → layers 1, 3, 5, ... are MoE (Llama-4-Maverick)
        return [
            (i % self.moe_layer_step) == (self.moe_layer_step - 1)
            for i in range(self.num_layers)
        ]

    def _lookup_moe_us(self, num_tokens: int) -> float:
        """Look up MoE layer latency, with GEMM fallback."""
        try:
            from aiconfigurator.sdk.common import MoEQuantMode

            timing = self.runtime_db.query_moe(
                quant_mode=MoEQuantMode.float16,
                topk=self.num_experts_per_tok,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                inter_size=self.intermediate_size,
                moe_tp=max(self.tp_size, 1),
                moe_ep=self.ep_size,
                num_tokens=max(num_tokens, 1),
                workload="power_law_1.01",
            )
            if not self._logged_moe:
                logger.info(
                    "MoE lookup OK: tokens=%d experts=%d topk=%d → %.1f μs",
                    num_tokens, self.num_experts,
                    self.num_experts_per_tok, timing.latency_us,
                )
                self._logged_moe = True
            return timing.latency_us
        except Exception as e:
            # Fallback: topk * (up_proj + down_proj) GEMMs
            fallback = self.num_experts_per_tok * (
                self._lookup_gemm_us(
                    m=max(num_tokens, 1),
                    n=self.intermediate_size // max(self.tp_size, 1),
                    k=self.hidden_size,
                )
                + self._lookup_gemm_us(
                    m=max(num_tokens, 1),
                    n=self.hidden_size,
                    k=self.intermediate_size // max(self.tp_size, 1),
                )
            )
            if not self._logged_moe:
                logger.warning(
                    "MoE lookup FALLBACK: tokens=%d → %.1f μs (error: %s)",
                    num_tokens, fallback, e,
                )
                self._logged_moe = True
            return fallback

    # ------------------------------------------------------------------
    # Operator lookups with fallback
    # ------------------------------------------------------------------

    def _lookup_gemm_us(self, m: int, n: int, k: int) -> float:
        """Look up GEMM latency from RuntimeDB, falling back to FLOPs."""
        try:
            timing = self.runtime_db.query_gemm(m=m, n=n, k=k)
            if not self._logged_gemm:
                logger.info(
                    "GEMM lookup OK: m=%d n=%d k=%d → %.1f μs (source=%s)",
                    m, n, k, timing.latency_us, timing.source,
                )
                self._logged_gemm = True
            return timing.latency_us
        except Exception as e:
            fallback = self._fallback_gemm_us(m, n, k)
            if not self._logged_gemm:
                logger.warning(
                    "GEMM lookup FALLBACK: m=%d n=%d k=%d → %.1f μs "
                    "(error: %s)", m, n, k, fallback, e,
                )
                self._logged_gemm = True
            return fallback

    def _fallback_gemm_us(self, m: int, n: int, k: int) -> float:
        """Roofline-model estimate: 2*M*N*K FLOPs / peak_TFLOPS."""
        flops = 2.0 * m * n * k
        peak_flops = self.hw_profile.tflops_fp16 * 1e12
        if peak_flops <= 0:
            return 1.0
        return (flops / peak_flops) * 1_000_000

    def _lookup_context_attention_us(
        self, batch_size: int, seq_len: int
    ) -> float:
        """Look up prefill attention latency."""
        try:
            timing = self.runtime_db.query_context_attention(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=self.num_heads // self.tp_size,
                num_kv_heads=self.num_kv_heads // self.tp_size,
                head_dim=self.head_dim,
            )
            if not self._logged_ctx_attn:
                logger.info(
                    "Context attention lookup OK: bs=%d seq=%d → %.1f μs "
                    "(source=%s)", batch_size, seq_len, timing.latency_us,
                    timing.source,
                )
                self._logged_ctx_attn = True
            return timing.latency_us
        except Exception as e:
            if not self._logged_ctx_attn:
                logger.warning(
                    "Context attention FALLBACK: bs=%d seq=%d → 10.0 μs "
                    "(error: %s)", batch_size, seq_len, e,
                )
                self._logged_ctx_attn = True
            return 10.0

    def _lookup_generation_attention_us(
        self, batch_size: int, seq_len: int
    ) -> float:
        """Look up decode attention latency."""
        try:
            timing = self.runtime_db.query_generation_attention(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=self.num_heads // self.tp_size,
                num_kv_heads=self.num_kv_heads // self.tp_size,
                head_dim=self.head_dim,
            )
            if not self._logged_gen_attn:
                logger.info(
                    "Generation attention lookup OK: bs=%d seq=%d → %.1f μs "
                    "(source=%s)", batch_size, seq_len, timing.latency_us,
                    timing.source,
                )
                self._logged_gen_attn = True
            return timing.latency_us
        except Exception as e:
            if not self._logged_gen_attn:
                logger.warning(
                    "Generation attention FALLBACK: bs=%d seq=%d → 5.0 μs "
                    "(error: %s)", batch_size, seq_len, e,
                )
                self._logged_gen_attn = True
            return 5.0

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _make_output(self, per_req_tokens: dict[str, int]) -> Any:
        """Build a valid ModelRunnerOutput with dummy sampled tokens."""
        req_ids = list(per_req_tokens.keys())
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        sampled_token_ids = [
            [random.randint(1, self.vocab_size - 1)] for _ in req_ids
        ]

        # Try real vLLM output first, fall back to local dataclass
        try:
            return _make_model_runner_output(
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
                sampled_token_ids=sampled_token_ids,
            )
        except NameError:
            return ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
                sampled_token_ids=sampled_token_ids,
            )
