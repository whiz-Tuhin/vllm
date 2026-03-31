# SPDX-License-Identifier: Apache-2.0
"""DryRunWorker — GPU-free worker that bypasses all CUDA initialization.

Replaces vLLM's gpu_worker.Worker. All GPU-dependent lifecycle methods
(init_device, load_model, profile_run, etc.) become no-ops. Memory
calculations use HWProfile values instead of querying the GPU.

Usage:
    vllm serve <model> \
        --worker-cls vllm.v1.worker.dry_run.worker.DryRunWorker \
        --load-format dummy \
        --enforce-eager \
        --dtype float16

    Set VLLM_DRY_RUN_HARDWARE=h100_sxm (default), a100_sxm, or l40s
    to choose the simulated GPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)

_DEFAULT_HW_KEY = "h100_sxm"


def _estimate_num_params(hf_config: Any) -> int:
    """Estimate total parameter count from a HuggingFace model config."""
    if hasattr(hf_config, "num_parameters") and hf_config.num_parameters:
        return int(hf_config.num_parameters)

    H = hf_config.hidden_size
    L = hf_config.num_hidden_layers
    V = hf_config.vocab_size
    n_heads = hf_config.num_attention_heads
    n_kv = getattr(hf_config, "num_key_value_heads", n_heads)
    head_dim = H // n_heads
    intermediate = getattr(hf_config, "intermediate_size", 4 * H)

    embed_params = 2 * V * H
    qkv = H * (n_heads + 2 * n_kv) * head_dim
    out_proj = n_heads * head_dim * H
    mlp = 3 * H * intermediate
    norms = 4 * H
    per_layer = qkv + out_proj + mlp + norms

    return embed_params + L * per_layer


def _dtype_to_bytes(dtype: Any) -> int:
    """Convert a dtype (torch dtype or string) to bytes per element."""
    _torch_map = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }
    if dtype in _torch_map:
        return _torch_map[dtype]
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return 1
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return 1

    _str_map = {
        "float16": 2, "half": 2, "bfloat16": 2,
        "float32": 4, "float": 4, "float64": 8, "double": 8,
        "int8": 1, "uint8": 1, "fp8": 1, "float8": 1,
        "int16": 2, "int32": 4, "int64": 8, "auto": 2,
    }
    dtype_str = str(dtype).lower().replace("torch.", "")
    return _str_map.get(dtype_str, 2)


class DryRunWorker(WorkerBase):
    """GPU-free worker that simulates vLLM's Worker lifecycle.

    All GPU-dependent operations are no-ops. The hardware key used for
    simulation can be set via the ``VLLM_DRY_RUN_HARDWARE`` environment
    variable (default: ``h100_sxm``).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str = "",
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self._hw_key = os.environ.get(
            "VLLM_DRY_RUN_HARDWARE", _DEFAULT_HW_KEY
        )

        # Lazily initialized in _ensure_components()
        self._hw_profile: Any = None
        self._runtime_db: Any = None
        self._parallelism_shim: Any = None
        self._dry_run_model_runner: Any = None

        # Set device to CPU — no GPU needed
        self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # GPU lifecycle no-ops
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        """No-op: skip torch.cuda.set_device / NCCL init."""
        logger.info(
            "DryRunWorker.init_device() — skipping GPU init (rank=%d, hw=%s)",
            self.rank,
            self._hw_key,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """No-op: skip model weight loading."""
        logger.info("DryRunWorker.load_model() — skipping weight load")

    def compile_or_warm_up_model(self) -> float:
        """No-op: skip CUDA graph capture / torch.compile warmup."""
        return 0.0

    def shutdown(self) -> None:
        """No-op: nothing to tear down."""
        pass

    # ------------------------------------------------------------------
    # Component wiring (lazy init)
    # ------------------------------------------------------------------

    def _ensure_components(self) -> None:
        """Lazily initialize HWProfile, RuntimeDB, ParallelismShim, and
        DryRunModelRunner.
        """
        if self._hw_profile is not None:
            return

        from simulator.hw_profile.loader import load_profile
        from simulator.runtime_db.db import RuntimeDB
        from simulator.parallelism.shim import (
            ParallelismConfig,
            ParallelismShim,
        )

        self._hw_profile = load_profile(self._hw_key)
        self._runtime_db = RuntimeDB(hardware=self._hw_key)

        tp = self.parallel_config.tensor_parallel_size
        pp = getattr(self.parallel_config, "pipeline_parallel_size", 1)
        ep = int(os.environ.get("VLLM_DRY_RUN_EP_SIZE", "1"))
        self._parallelism_shim = ParallelismShim(
            runtime_db=self._runtime_db,
            hw_profile=self._hw_profile,
            config=ParallelismConfig(tp_size=tp, pp_size=pp, ep_size=ep),
        )

        from vllm.v1.worker.dry_run.model_runner import DryRunModelRunner

        self._dry_run_model_runner = DryRunModelRunner(
            vllm_config=self.vllm_config,
            hw_profile=self._hw_profile,
            runtime_db=self._runtime_db,
            device=self.device,
            parallelism_shim=self._parallelism_shim,
        )

        logger.info(
            "DryRunWorker components initialized: hw=%s, tp=%d, pp=%d, ep=%d",
            self._hw_key, tp, pp, ep,
        )

    # ------------------------------------------------------------------
    # KV cache specification (real vLLM types)
    # ------------------------------------------------------------------

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Return per-layer KV cache specs using real FullAttentionSpec."""
        self._ensure_components()

        hf = self.model_config.hf_config
        num_layers = hf.num_hidden_layers
        num_kv_heads = getattr(
            hf, "num_key_value_heads", hf.num_attention_heads
        )
        head_dim = hf.hidden_size // hf.num_attention_heads
        tp = self.parallel_config.tensor_parallel_size
        kv_heads_per_tp = max(num_kv_heads // tp, 1)
        block_size = self.cache_config.block_size

        # Use the model's dtype for KV cache
        dtype = self.model_config.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float16)

        specs: dict[str, KVCacheSpec] = {}
        for i in range(num_layers):
            key = f"model.layers.{i}.self_attn"
            specs[key] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=kv_heads_per_tp,
                head_size=head_dim,
                dtype=dtype,
            )
        return specs

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------

    def determine_available_memory(self) -> int:
        """Report available KV cache memory using HWProfile."""
        self._ensure_components()

        from simulator.hw_profile.memory_calculator import (
            KVCacheConfig as SimKVCacheConfig,
            ModelMemoryConfig,
            calculate_kv_cache_blocks,
        )

        hf = self.model_config.hf_config
        num_params = _estimate_num_params(hf)
        dtype_bytes = _dtype_to_bytes(self.model_config.dtype)

        model_mem = ModelMemoryConfig(
            num_params=num_params,
            num_hidden_layers=hf.num_hidden_layers,
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(
                hf, "num_key_value_heads", hf.num_attention_heads
            ),
            dtype_bytes=dtype_bytes,
            intermediate_size=getattr(
                hf, "intermediate_size", 4 * hf.hidden_size
            ),
        )

        cache_cfg = self.cache_config
        kv_dtype_bytes = 2
        if (
            hasattr(cache_cfg, "cache_dtype")
            and cache_cfg.cache_dtype in ("fp8", "float8")
        ):
            kv_dtype_bytes = 1

        kv_cfg = SimKVCacheConfig(
            block_size=cache_cfg.block_size,
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            kv_cache_dtype_bytes=kv_dtype_bytes,
        )

        result = calculate_kv_cache_blocks(
            self._hw_profile, model_mem, kv_cfg
        )

        logger.info(
            "DryRunWorker.determine_available_memory(): "
            "vram=%.1fGB, model=%.1fGB, kv_available=%.1fGB, blocks=%d",
            result.total_vram_bytes / (1024**3),
            result.model_weight_bytes / (1024**3),
            result.kv_cache_gb,
            result.num_blocks,
        )

        return result.available_kv_bytes

    # ------------------------------------------------------------------
    # KV cache initialization
    # ------------------------------------------------------------------

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        """Store KV cache config. No GPU allocation needed."""
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        logger.info(
            "DryRunWorker.initialize_from_config(): num_blocks=%d",
            kv_cache_config.num_blocks,
        )

    # ------------------------------------------------------------------
    # Model execution delegation
    # ------------------------------------------------------------------

    def execute_model(self, scheduler_output: Any) -> Any:
        """Delegate model execution to DryRunModelRunner."""
        self._ensure_components()
        return self._dry_run_model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        """Delegate sampling to DryRunModelRunner."""
        self._ensure_components()
        return self._dry_run_model_runner.sample_tokens()

    # ------------------------------------------------------------------
    # Model access stubs
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        """Return a dummy module (no real model loaded)."""
        return nn.Module()

    def get_cache_block_size_bytes(self) -> int:
        """Estimate cache block size from model config."""
        hf = self.model_config.hf_config
        num_kv_heads = getattr(
            hf, "num_key_value_heads", hf.num_attention_heads
        )
        head_dim = hf.hidden_size // hf.num_attention_heads
        tp = self.parallel_config.tensor_parallel_size
        kv_heads_per_tp = max(num_kv_heads // tp, 1)
        dtype_bytes = _dtype_to_bytes(self.model_config.dtype)
        block_size = self.cache_config.block_size

        # 2 (K+V) * block_size * kv_heads * head_dim * dtype * num_layers
        return (
            2
            * block_size
            * kv_heads_per_tp
            * head_dim
            * dtype_bytes
            * hf.num_hidden_layers
        )

    # ------------------------------------------------------------------
    # LoRA stubs (not supported in dry-run mode)
    # ------------------------------------------------------------------

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    # ------------------------------------------------------------------
    # Task support
    # ------------------------------------------------------------------

    def get_supported_tasks(self) -> tuple:
        """Report that we support text generation."""
        return ("generate",)
