# AFD M×N Pre-Routing Architecture

## Overview

This document describes the M×N bipartite pre-routing architecture for AFD (Attention-FFN Disaggregation) in vLLM. This replaces the original 1:1 ATTN↔FFN pairing + EP all-gather/reduce-scatter design with a topology where every ATTN rank communicates directly with every FFN rank via NCCL P2P.

**Key files modified:**
- `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` — connector rewrite
- `vllm/model_executor/models/deepseek_v2.py` — ATTN-side gate + shared experts
- `vllm/model_executor/layers/fused_moe/layer.py` — `forward_pre_routed` bypass
- `vllm/v1/worker/gpu_ffn_model_runner.py` — FFN forward loop changes
- `vllm/config/vllm.py` — Gloo→NCCL DP sync fix

---

## Architecture Comparison

### Before: 1:1 Pairing + EP All-to-All

```
ATTN_DP0 ──[N/2 tokens]──► FFN_TP0 (experts 0-31)
ATTN_DP1 ──[N/2 tokens]──► FFN_TP1 (experts 32-63)
                                 ↕
                     EP all-gather: both FFN workers get ALL N tokens
                     Router + fused_experts: each FFN computes its 32 local experts
                     Reduce-scatter: results sent back to originating FFN worker
                                 ↕
ATTN_DP0 ◄──[N/2 tokens]── FFN_TP0
ATTN_DP1 ◄──[N/2 tokens]── FFN_TP1
```

**Communication per layer:** 2x P2P (attn→ffn) + all-gather + reduce-scatter + 2x P2P (ffn→attn) = ~5·N·H bytes

### After: M×N Bipartite + Pre-Routing (Option B — Broadcast)

```
ATTN_DP0 ──[N/2, H] + topk──► FFN_TP0 (experts 0-31)
ATTN_DP0 ──[N/2, H] + topk──► FFN_TP1 (experts 32-63)
ATTN_DP1 ──[N/2, H] + topk──► FFN_TP0
ATTN_DP1 ──[N/2, H] + topk──► FFN_TP1
                                 │
                     No EP collectives. Each FFN runs only its local experts
                     via raw fused_experts() with expert_map filtering.
                                 │
ATTN_DP0 ◄──[N/2, H] partial── FFN_TP0
ATTN_DP0 ◄──[N/2, H] partial── FFN_TP1
ATTN_DP1 ◄──[N/2, H] partial── FFN_TP0
ATTN_DP1 ◄──[N/2, H] partial── FFN_TP1

ATTN: final = sum(partials) + shared_expert_output
```

**Communication per layer:** 4x P2P (attn→ffn) + 4x P2P (ffn→attn) = ~4·N·H bytes (no collectives)

---

## What Changed in p2p_connector.py

### 1. M×N Pair Topology (init_afd_connector)

**Before:** Created `min(attn_size, ffn_size)` pairs by zipping ATTN and FFN rank lists 1:1.

**After:** Full bipartite — every ATTN rank paired with every FFN rank. For `aA×fF`, creates `a×f` pairs. Each pair gets its own Gloo process group (for metadata) and NCCL communicator (for tensor data), created via `init_afd_process_group` with unique TCP ports per pair.

```python
for i in range(attn_size):
    for j in range(ffn_size):
        pair_id = i * ffn_size + j
        a2e_port = afd_base_port + 100 + pair_id * 2
        e2a_port = afd_base_port + 100 + pair_id * 2 + 1
        # Only the 2 members of this pair participate
        # Creates Gloo pg + PyNcclCommunicator per direction
```

Pair count by config: 1A1F=1, 1A2F=2, 2A2F=4, 3A1F=3, 2A4F=8.

### 2. Option B: Fixed-Size Broadcast (send_attn_output)

**Before (original pre-routing attempt):** Per-partner boolean mask → `hs[mask_j]` (data-dependent shape, forces CPU/GPU sync) → send subset.

**After:** Broadcast full `hidden_states` + `topk_ids` + `topk_weights` to every FFN partner. Shapes are statically known from `dp_metadata` broadcast at forward-pass start. Zero CPU/GPU syncs on the hot path.

```python
# Option B — no masking, no count_hdr, no CPU sync
for j in range(n_partners):
    self._nccl_send(hidden_states, dst, comm_id)  # full [N, H]
    if is_moe:
        self._nccl_send(topk_ids, dst, comm_id)    # full [N, K]
        self._nccl_send(topk_weights, dst, comm_id) # full [N, K]
```

### 3. Fixed-Size Receive (recv_attn_output on FFN side)

**Before:** Received a count header, then variable-size tensors. Required `.cpu().tolist()` sync.

**After:** Token counts come from `dp_metadata` (broadcast once per forward). Recv buffers pre-allocated at known sizes. No per-layer sync.

### 4. Element-Wise Partial Combine (recv_ffn_output on ATTN side)

**Before:** `scatter_add_` with mask indices (required `nonzero()` → CPU/GPU sync).

**After:** Each FFN partner returns a full `[N, H]` tensor where non-local experts contribute zero (via `expert_map`). ATTN simply sums: `final = partial_0 + partial_1 + ... + shared_output`.

### 5. MoE Layer Detection

Added `_is_moe_layer(layer_idx)` using `first_k_dense_replace` and `moe_layer_freq` from model config, so FFN's `recv_attn_output` knows whether to expect topk tensors.

---

## What Changed in deepseek_v2.py

### DeepseekV2MoEAttentionStub (new class)

Lightweight module that lives on the ATTN side for MoE layers. Contains:
- `gate`: `ReplicatedLinear(hidden_size, n_routed_experts)` — the MoE router
- `shared_experts`: `DeepseekV2MLP` — shared expert computation
- `compute_route_and_shared(hidden_states)` → `(topk_ids, topk_weights, shared_output)`

Weight names match the original `DeepseekV2MoE` layout, so checkpoint loading works unchanged.

### forward_with_afd (modified loop)

Per MoE layer, ATTN now:
1. Runs attention + layernorms (layer.forward with early return for `afd_role=="attention"`)
2. Calls `layer.mlp.compute_route_and_shared(hidden_states)` → topk_ids, topk_weights, shared_output
3. Calls `send_attn_output(hidden_states, metadata, topk_ids, topk_weights, shared_output)`
4. On recv: `final = sum(partials) + shared_output`

### Weight loading filter

ATTN side now loads gate + shared_experts weights (previously skipped by `is_moe_weight` filter). Routed expert weights still skipped on ATTN.

---

## What Changed in fused_moe/layer.py

### forward_pre_routed (critical fix)

**The bug:** The original `forward_pre_routed` called `self.quant_method.apply()` which dispatched into `FusedMoEModularKernel.forward`. That kernel's `_prepare()` / `_finalize()` hooks ran EP all-gather + reduce-scatter — the exact collectives we were trying to eliminate. This added ~170 ms/layer.

**The fix:** `forward_pre_routed` now calls raw `fused_experts()` directly, bypassing the modular kernel entirely:

```python
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

return fused_experts(
    hidden_states=hidden_states,
    w1=self.w13_weight,
    w2=self.w2_weight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    activation=self.activation,
    global_num_experts=self.global_num_experts,
    expert_map=self.expert_map,  # marks non-local experts as -1
    quant_config=quant_config,
)
```

This runs only the per-expert Triton gemm kernels — no dispatch, no combine, no NCCL collectives.

---

## What Changed in config/vllm.py

### Gloo → NCCL DP Synchronization Fix

**The bug:** vLLM auto-enables `disable_nccl_for_dp_synchronization = True` when `async_scheduling + DP > 1 + MoE model`. This forces `coordinate_batch_across_dp` (called once per forward pass) onto Gloo CPU, adding ~1500 ms/forward because the CPU all-reduce drains the entire GPU stream before it can proceed.

**The fix:** When AFD is active, skip the Gloo override:

```python
if (self.scheduler_config.async_scheduling
        and not (self.afd_config and self.afd_config.afd_connector != "dummy")):
    self.parallel_config.disable_nccl_for_dp_synchronization = True
```

AFD already serializes the forward pass (ATTN↔FFN round trips), so there's no async pipelining to protect. NCCL GPU allreduce is microseconds vs Gloo CPU's hundreds of ms.

---

## Bugs Found and Fixed

### Bug 1: CPU/GPU sync in boolean mask indexing (~400 ms/layer)

**Symptom:** 1A2F TPOT 1687 ms (should be ~55 ms like 1A1F).

**Root cause:** `hs[mask_j]` in `send_attn_output` produces a tensor with data-dependent shape. PyTorch must sync CPU↔GPU to learn the output shape before allocating. This sync drains the NCCL stream backlog (pending sends + FFN compute + FFN sends from previous layers), adding ~400 ms per MoE layer.

**Fix:** Option B — broadcast full tensors with statically-known shapes. Zero CPU/GPU syncs on the hot path.

### Bug 2: EP dispatch inside forward_pre_routed (~170 ms/layer)

**Symptom:** After Option B fix, FFN `compute_ffn.total` still 170-250 ms/layer despite raw NCCL being microseconds.

**Root cause:** `forward_pre_routed` → `quant_method.apply` → `FusedMoEModularKernel.forward` → `_prepare()` runs `NaiveEP` all-gather, `_finalize()` runs reduce-scatter. The entire EP collective pipeline was still running inside the kernel, even though pre-routing made it unnecessary.

**Fix:** Bypass the modular kernel. Call raw `fused_experts()` directly with `expert_map` for filtering.

### Bug 3: Gloo CPU DP synchronization (~1500 ms/forward, 2A2F only)

**Symptom:** 2A2F TPOT 1640 ms even after bugs 1+2 fixed. 1A2F was 138 ms on same hardware.

**Root cause:** vLLM auto-sets `disable_nccl_for_dp_synchronization = True` for async-scheduled MoE models with DP > 1. This forces `coordinate_batch_across_dp` onto Gloo CPU. The CPU all-reduce drains the GPU stream, serializing everything.

**Fix:** Exception for AFD — keep NCCL GPU path. Result: 2A2F dropped from 1640 ms → 311 ms.

### Bug 4: Triton kernel JIT warmup (~20 sec for first 8 decode steps)

**Symptom:** First benchmark run is 10-100× slower than second run on same server.

**Root cause:** With `--enforce-eager`, CUDA graph capture is disabled, so FFN's `_dummy_run` never fires at startup. Triton kernels for `fused_experts` at the decode batch shape are JIT-compiled on the first real request.

**Workaround:** Use the second benchmark run as the real measurement. For production: add explicit warmup pass at FFN server startup.

---

## Performance Results

### Validated warm-state TPOT (H200, NVSwitch, DeepSeek-V2-Lite)

| Config | TPOT (ms) | vs 1A1F | vs Original |
|--------|----------:|:-------:|:-----------:|
| 1A1F | 54 | 1.0× | — |
| 1A2F | 138 | 2.6× | 12.2× faster (was 1687 ms) |
| 2A2F | 311 | 5.8× | 6.7× faster (was 2076 ms on L40S) |
| 3A1F | 62 | 1.1× | — |
| 2A1F | 55 | 1.0× | — |

### Measurement methodology

- Benchmark: `vllm bench serve --request-rate inf --num-prompts 20 --random-input-len 128 --random-output-len 32`
- Run benchmark twice on same server. Run 1 = Triton JIT warmup (discard). Run 2 = real number.
- All configs use `--enforce-eager` (no CUDA graphs)
- Model: DeepSeek-V2-Lite (16B MoE, 64 routed experts, top-6, 26 MoE layers + 1 dense)
