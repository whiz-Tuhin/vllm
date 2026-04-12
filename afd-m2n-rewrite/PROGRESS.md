# AFD mAnF Pre-Routing Implementation Progress

**Branch**: `tk/pr-29772-afd-pre-routing` (off `tk/pr-29772-afd-fix`)
**Repo**: `/storage/scratch1/0/hwu419/tkhare7/vllm-afd`
**Plan**: `/storage/home/hcoda1/0/hwu419/.claude/plans/whimsical-wishing-koala.md`
**Started**: 2026-04-11

---

## Goal

Replace the EP all-gather/reduce-scatter between FFN workers with direct ATTN→expert pre-routing. ATTN runs the MoE gate locally, computes `topk_ids`, sends each token only to the FFN worker(s) holding its target experts. FFN skips EP collectives and returns partial results. ATTN combines partials.

Shared experts and gate move to the ATTN side. Target configs: EP=4, EP=8.

---

## Phase Status

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 0 | Branch + tracker setup | ✅ Done | Branch created off `tk/pr-29772-afd-fix` at commit `400c4ece7` |
| 1 | Move gate + shared experts to ATTN side | ✅ Done | Commit `9085e2ed8` — DeepseekV2MoEAttentionStub + weight loading filter |
| 1b | Phase 1 regression test | ✅ Done | 1A1F on L40S: 20/20 req OK, ATTN mem 2.51 GiB (+890 MB vs baseline = stub loaded), TPOT 55ms |
| 2 | Extend AFDConnectorMetadata | ✅ Done | Commit `3ce3ecf3c` — added `topk_ids`, `topk_weights`, `source_attn_rank` |
| 3 | Atomic pre-routing rewrite | ✅ Done | Commit `ac5deeb69` — M×N init + pre-routing send/recv + deepseek_v2 forward + FFN bypass |
| 3b | Phase 3 end-to-end tests | ✅ Done | 1A1F=54ms, 3A1F=62ms, 2A1F=55ms, **1A2F=1687ms**, **2A2F=10876ms** |
| 3c | Diagnose 1A2F/2A2F slowdown | ✅ Done | Root cause: CPU/GPU syncs in mask ops + EP dispatch inside modular kernel |
| 3d | Option B: broadcast full tensors | ✅ Done | Eliminated all per-layer CPU/GPU syncs. Commit `9aac72697` |
| 3e | Bypass FusedMoEModularKernel | ✅ Done | `forward_pre_routed` calls raw `fused_experts()`, skipping EP all-gather/reduce-scatter |
| 3f | Fix Gloo CPU DP sync for 2A2F | ✅ Done | Force NCCL (not Gloo CPU) for DP allreduce when AFD active. Modified `config/vllm.py` |
| 3g | Warm-state validation | ✅ Done | **1A2F=138ms, 2A2F=311ms** (warm). Cold runs dominated by Triton JIT warmup |
| 4 | Scale up to 1A4F / 2A4F | ⏳ Pending | Need 8-GPU node |
| 5 | PyTorch profiler traces | ⏳ Pending | Clean traces without AFD_TIMING for publication |

---

### Phase 3: Atomic pre-routing rewrite — ✅ Done

**Commit**: `ac5deeb69` — "Phase 3: Atomic pre-routing rewrite across connector, model, and runners"

**Files changed**:
- `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` — full rewrite (~1600 → ~800 lines)
- `vllm/model_executor/models/deepseek_v2.py` — forward path wiring + `compute_route_and_shared` on the stub
- `vllm/model_executor/layers/fused_moe/layer.py` — new `FusedMoE.forward_pre_routed` method
- `vllm/v1/worker/gpu_ffn_model_runner.py` — pass topk through, drop TP all-gather

**Data flow** (per MoE layer):

```
ATTN:                                   FFN:
  gate(hs) → router_logits                
  grouped_topk → topk_ids, topk_weights
  shared_experts(hs) → shared_out
  for each FFN partner j:
    mask_j = any(topk_ids ∈ j's experts)
    send mask_j subset + topk      ───►   recv from each ATTN partner, concat
                                          run forward_pre_routed (local experts only,
                                          expert_map skips non-local)
                                          split output by source counts
  recv from each FFN partner         ◄───  send partial to each ATTN partner
  scatter-add partials into final
  final += shared_out
```

**Known caveats / risks**:
- `routed_scaling_factor` is treated as 1.0 (fine for V2-Lite, breaks V3)
- FFN-side dense layer 0 requires all ATTN ranks to broadcast the same tensor — my code does this via the "topk_ids=None" dense path in `send_attn_output`
- TP all-gather inside `DeepseekV2MLP` still happens for dense layers (FFN-side TP wiring is unchanged)
- We're eager-only — no CUDA graph capture

**Phase 3 validation commands** (L40S node):

```bash
# Kill any stale processes first
pkill -9 -f "vllm fserver|vllm serve" 2>/dev/null; sleep 1

# Terminal 1 — FFN (GPU 1, 1A1F)
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase3-1A1F-ffn.log

# Terminal 2 — ATTN (GPU 0, 1A1F)
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase3-1A1F-attn.log

# Terminal 3 — benchmark (after ATTN shows "Application startup complete")
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd && source .venv/bin/activate
vllm bench serve --backend vllm --model deepseek-ai/DeepSeek-V2-Lite \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate 2 \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase3-1A1F-bench.log
```

**Success criteria**:
- ATTN + FFN both start and complete init without weight/NCCL errors
- `vllm bench` finishes with 20/20 successful requests
- TTFT and TPOT in the same ballpark as Phase 1 1A1F (TTFT ~1.2s, TPOT ~55ms on L40S)
- A sanity curl should return coherent English text (non-garbled)

**If it fails**:
- Check ATTN log for weight loading errors (missing gate / shared_experts)
- Check FFN log for NCCL errors in `recv_attn_output` — probably a send/recv shape mismatch
- Check for AssertionError in `forward_with_afd` or `compute_ffn_output`
- Check for CUDA OOM (pre-routing buffers are allocated per-layer in eager)

---

## Results Summary

### Baselines (before refactor, Phase 3 first run)

| Config | Cluster | TPOT (ms) | Notes |
|--------|---------|----------:|-------|
| 1A1F | H200 | 54 | Single ATTN, single FFN. Baseline. |
| 3A1F | H200 | 62 | 3 ATTN DP, 1 FFN. Fast — no EP. |
| 2A1F | H200 | 55 | 2 ATTN DP, 1 FFN. Isolates DP overhead. |
| 1A2F | H200 | 1,687 | 1 ATTN, 2 FFN TP/EP. **Slow** — EP dispatch + CPU syncs. |
| 2A2F | L40S | 2,076 | 2 ATTN DP, 2 FFN TP/EP. Original impl with .item() fix. |

### After refactor (warm-state, second benchmark run on same server)

| Config | Cluster | TPOT (ms) | Speedup vs baseline | Key fix |
|--------|---------|----------:|:-------------------:|---------|
| **1A2F** | H200 | **138** | **12.2×** | Modular kernel bypass |
| **2A2F** | H200 | **311** | **6.7×** (vs L40S baseline) | + Gloo→NCCL DP sync |

### Bottlenecks found and fixed

| Bottleneck | Impact | Fix |
|------------|--------|-----|
| `hs[mask]` / `nonzero()` forcing CPU/GPU sync per layer | ~400 ms/layer | Option B: broadcast full tensors, zero masking |
| `FusedMoEModularKernel` running EP all-gather/reduce-scatter inside `forward_pre_routed` | ~170 ms/layer | Bypass modular kernel, call raw `fused_experts()` directly |
| `coordinate_batch_across_dp` using Gloo CPU all-reduce (auto-enabled for MoE + DP + async scheduling) | ~1500 ms/forward | Force NCCL GPU path when AFD active (`config/vllm.py`) |
| Triton kernel JIT compilation on first decode | ~20s per unique shape | Warmup artifact — second benchmark run is the real number |

### Remaining gap to 1A1F

| | 1A1F | 1A2F | 2A2F |
|--|-----:|-----:|-----:|
| TPOT | 54 ms | 138 ms | 311 ms |
| Per-layer overhead | ~2 ms | ~5 ms | ~12 ms |

The gap is structural: more NCCL pairs per layer (2 sends+recvs for 1A2F, 4 for 2A2F), DP coordination for 2A2F, and serialized ATTN↔FFN round trip. Closing it further requires DBO (dual-batch overlap) or CUDA graph capture.

---

## Phase Log

### Phase 0: Setup — 2026-04-11

- ✅ Committed `.item()` logging fix on `tk/pr-29772-afd-fix` (commit `400c4ece7`)
- ✅ Created `tk/pr-29772-afd-pre-routing` branched off `tk/pr-29772-afd-fix`
- ✅ Created progress tracker at this file

### Phase 1: Move gate + shared experts to ATTN side — ✅ Done

**Commit**: `9085e2ed8` — "Phase 1: Move MoE gate + shared experts to ATTN side"

**Changes**:
- Added `DeepseekV2MoEAttentionStub` class (new lightweight module with `gate` + `shared_experts`)
- `DeepseekV2DecoderLayer.__init__`: when `afd_role == "attention"` AND layer is MoE, create `self.mlp = DeepseekV2MoEAttentionStub(...)`. Submodule names match the original `DeepseekV2MoE.mlp.gate` / `mlp.shared_experts` layout, so checkpoint weight names work without remapping.
- `load_weights` filter: allow `.gate.weight`, `.gate.e_score_correction_bias`, and `shared_experts` weights through on the ATTN side. Routed expert weights still skipped.
- Forward path **unchanged** — stub is loaded but not yet called. This keeps Phase 1 correctness-only: if it passes the regression, we know weight loading is working and we haven't broken anything.

**Phase 1 validation commands** (run these on the L40S node):

```bash
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd && source .venv/bin/activate

# Terminal 1 — FFN (GPU 1, 1A1F)
CUDA_VISIBLE_DEVICES=1 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase1-1A1F-ffn.log

# Terminal 2 — ATTN (GPU 0, 1A1F)
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase1-1A1F-attn.log

# Terminal 3 — benchmark
vllm bench serve --backend vllm --model deepseek-ai/DeepSeek-V2-Lite \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate 2 \
  2>&1 | tee /storage/scratch1/0/hwu419/tkhare7/scratchpad/phase1-1A1F-bench.log
```

**Success criteria**:
- ATTN server starts without weight loading errors (no "unexpected key", no "missing key")
- Benchmark completes, returns coherent output
- TTFT/TPOT comparable to baseline 1A1F on H200 (TTFT ~1.4s, TPOT ~68ms on H200; L40S will be slower but within 2-3x)
- ATTN log shows the extra shared_experts weights loaded (~23MB/layer × 26 layers ≈ 600MB extra weight memory on ATTN GPU)

---

## Open Issues / Decisions

- Shared experts go on **ATTN side** (confirmed)
- Token duplication policy: currently **broadcast full tensor to all FFN workers** (Option B). True pre-routing (send subset) saves bandwidth at EP≥4 but requires sync-free implementation (pre-allocated buffers or CUDA graph capture).
- Graph capture: **eager-only for this branch** (confirmed). Enabling CUDA graphs is the biggest remaining perf lever.
- `routed_scaling_factor` treated as 1.0 (fine for V2-Lite, would break V3)
- DBO (dual-batch overlap): connector's `_pending_*` state is not thread-safe for 2 ubatches. Needs per-stage indexing before enabling.
- Warmup: FFN `_dummy_run` not called with `--enforce-eager`, causing Triton JIT on first real request. Add explicit warmup for production.

---

## Run Commands (Clean, No Instrumentation)

All commands assume:
- `cd /path/to/vllm-afd && source .venv/bin/activate`
- 4 H200 GPUs on the same node
- Port 29500 free (or change to 29510 etc.)
- Kill stale processes first: `pkill -9 -u $USER -f 'vllm' ; sleep 5`

### 1A1F (baseline)

```bash
# Terminal 1: FFN (GPU 1)
CUDA_VISIBLE_DEVICES=1 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}'

# Terminal 2: ATTN (GPU 0)
CUDA_VISIBLE_DEVICES=0 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}'

# Terminal 3: Sanity + benchmark (run 2x, use run2 as real number)
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 1A2F

```bash
# Terminal 1: FFN (GPUs 1,2 — TP=2, EP=2)
CUDA_VISIBLE_DEVICES=1,2 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}'

# Terminal 2: ATTN (GPU 0)
CUDA_VISIBLE_DEVICES=0 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}'

# Terminal 3: Sanity + 2 benchmarks
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 2A2F

```bash
# Terminal 1: FFN (GPUs 2,3 — TP=2, EP=2)
CUDA_VISIBLE_DEVICES=2,3 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 2: ATTN (GPUs 0,1 — DP=2)
CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 2 \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 3: Sanity + 2 benchmarks
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 3A1F

```bash
# Terminal 1: FFN (GPU 3 — single GPU)
CUDA_VISIBLE_DEVICES=3 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}'

# Terminal 2: ATTN (GPUs 0,1,2 — DP=3)
CUDA_VISIBLE_DEVICES=0,1,2 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 3 \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}'

# Terminal 3: same sanity + bench pattern
```

---

## PyTorch Profiler Trace Commands

Add `--profiler-config.profiler torch` and `--profiler-config.torch_profiler_dir <dir>` to BOTH server commands. Traces go to the specified directory as Chrome trace JSON files. Open in `chrome://tracing` or Perfetto UI.

**Important**: run profiler on the SECOND benchmark (warm state) to avoid capturing Triton JIT compilation in the trace.

### Example: 1A2F with profiler

```bash
LOGDIR=/path/to/profiler-traces/1A2F

# Terminal 1: FFN with profiler
CUDA_VISIBLE_DEVICES=1,2 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir $LOGDIR/ffn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}'

# Terminal 2: ATTN with profiler
CUDA_VISIBLE_DEVICES=0 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir $LOGDIR/attn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}'

# Terminal 3: warmup run (discard), then profiled run
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
sleep 5

# This run gets profiled (server captures trace on step boundary)
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
```

### Example: 2A2F with profiler

```bash
LOGDIR=/path/to/profiler-traces/2A2F

# Terminal 1: FFN with profiler
CUDA_VISIBLE_DEVICES=2,3 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir $LOGDIR/ffn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 2: ATTN with profiler
CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 2 \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir $LOGDIR/attn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 3: warmup then profiled run
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
sleep 5

vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
```

### What to verify in the profiler traces

In the ATTN trace, look for:
- **No `AgRsAll2AllManager.dispatch`** or `AgRsAll2AllManager.combine` (EP collectives eliminated)
- **`afd_p2p_send`** / **`afd_p2p_recv`** NCCL ops at each layer boundary
- **`fused_moe_kernel`** should NOT appear on ATTN side (gate + shared only)
- **Gate compute**: small `ReplicatedLinear` per MoE layer

In the FFN trace, look for:
- **`afd_p2p_recv`** per layer (receiving from ATTN)
- **`fused_moe_kernel`** (the raw Triton experts, NOT inside modular kernel prepare/finalize)
- **No all-gather or reduce-scatter** between FFN workers
- **`afd_p2p_send`** per layer (sending partial back)

---

## GPU Monitoring During Benchmarks

Run alongside any benchmark to capture temperature, clocks, and utilization:

```bash
nvidia-smi dmon -s pcut -d 1 2>&1 | tee gpu-monitor.log
```

Columns: power (W), GPU temp (C), mem clock (MHz), GPU clock (MHz), SM util (%), mem util (%).

---

## Reference Commits

- `400c4ece7` — perf: Disable AFD_DIAG logging with .item() CUDA sync
- `004c761a5` — feat: Asymmetric AFD support (3A1F, 1A2F) with per-pair Gloo+NCCL groups
- `ac539d421` — Merge branch 'tk/pr-29772-local-fixes'
- `9085e2ed8` — Phase 1: Move MoE gate + shared experts to ATTN side
- `3ce3ecf3c` — Phase 2: Extend AFDConnectorMetadata with topk fields
- `ac5deeb69` — Phase 3: Atomic pre-routing rewrite
- `9aac72697` — perf: Switch MoE path to Option B (full broadcast)
- *(uncommitted)* — perf: Bypass FusedMoEModularKernel in forward_pre_routed
- *(uncommitted)* — perf: Force NCCL for DP sync when AFD active
