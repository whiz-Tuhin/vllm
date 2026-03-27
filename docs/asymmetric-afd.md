# Asymmetric AFD (Attention-FFN Disaggregation) for vLLM

Extends [PR #29772](https://github.com/vllm-project/vllm/pull/29772) to support asymmetric GPU configurations for MoE model inference disaggregation.

## Overview

AFD splits MoE model inference so attention layers run on one GPU group and FFN/MoE layers run on another, communicating via NCCL P2P. The original PR only supports symmetric configs (1A1F, 2A2F). This work adds support for:

- **xA1F** (e.g., 3A1F): Multiple attention GPUs (DP) + 1 FFN GPU
- **1AxF** (e.g., 1A2F): 1 attention GPU + multiple FFN GPUs (TP/EP)

Only MoE models are supported: DeepSeek-V2 (`deepseek_v2.py`) and Step3 (`step3_text.py`).

## Supported Configurations

| Config | Attention | FFN | ATTN parallelism | FFN parallelism | Status |
|--------|-----------|-----|-------------------|-----------------|--------|
| 1A1F   | 1 GPU     | 1 GPU | None            | None            | Working |
| 2A2F   | 2 GPUs    | 2 GPUs | DP=2           | TP=2 + EP=2     | Working |
| 3A1F   | 3 GPUs    | 1 GPU  | DP=3           | None            | In progress |
| 1A2F   | 1 GPU     | 2 GPUs | None           | TP=2 + EP=2     | Working |

## Architecture

### Data Flow per Decoder Layer

```
ATTN Process                              FFN Process
============                              ===========

input → RMSNorm → MLA Attention
              │
              ├── [Gloo] send dp_metadata (token counts, shapes)
              │
              ├── [NCCL] send hidden_states [N, hidden_dim]
              │                                    │
              │                              RMSNorm → MoE Router
              │                              → Expert Compute (64 experts)
              │                              → Weighted sum
              │                                    │
              ◄── [NCCL] recv ffn_output [N, hidden_dim]
              │
     residual += ffn_output → next layer
```

### 3A1F (xA1F) — DP Attention, Single FFN

```
ATTN0 (GPU 0, DP0) ──NCCL pair 0──►
ATTN1 (GPU 1, DP1) ──NCCL pair 1──► FFN0 (GPU 3) → concat → MoE → split → send back
ATTN2 (GPU 2, DP2) ──NCCL pair 2──►

- Each ATTN sends its own tokens independently to the single FFN
- FFN concatenates all received tensors, runs MoE, splits output back
- Only ATTN DP0 sends dp_metadata (same global data for all ranks)
```

### 1A2F (1AxF) — Single Attention, TP FFN

```
                    ┌──► FFN0 (GPU 1, TP0, experts 0-31)
ATTN0 (GPU 0) ─────┤    ↕ all-to-all (EP)
  broadcasts same   └──► FFN1 (GPU 2, TP1, experts 32-63)
  tensor to BOTH              │
                    ◄─────────┘ only FFN TP0 sends result back

- ATTN broadcasts FULL tensor to ALL FFN workers (TP splits weights, not tokens)
- FFN workers cooperate via TP all-reduce + EP all-to-all
- Only FFN TP rank 0 sends result back (all have same output after all-reduce)
```

## Process Group Architecture

### Two-Layer Communication

| Layer | Backend | Purpose | Data size |
|-------|---------|---------|-----------|
| Gloo (CPU) | Control plane | Rendezvous, ncclUniqueId exchange, dp_metadata transfer | ~100 bytes |
| NCCL (GPU) | Data plane | Hidden state tensors | ~2MB per layer |

### Group Creation Flow

```
Step 1: Global Gloo rendezvous (all ranks, port 29500)
  FFN0 ──┐
  ATTN0 ─┤── afd_pg (Gloo, world_size=4 for 3A1F)
  ATTN1 ─┤
  ATTN2 ──┘

Step 2: Per-pair standalone Gloo groups (only pair members)
  Pair 0: FFN0 ↔ ATTN0   port 29600 (a2e), 29601 (e2a)
  Pair 1: FFN0 ↔ ATTN1   port 29602 (a2e), 29603 (e2a)
  Pair 2: FFN0 ↔ ATTN2   port 29604 (a2e), 29605 (e2a)

Step 3: NCCL comm init per pair (ncclUniqueId exchanged over Gloo)
  Each pair → independent PyNcclCommunicator (world_size=2)
```

### Why Per-Pair Standalone Groups (Not `torch.distributed.new_group`)

`new_group` uses a global counter (`_group_count`) that ALL ranks must increment in lockstep. FFN (DP=1) makes ~12 `new_group` calls during `initialize_model_parallel`; ATTN (DP=3) makes ~36. By the time AFD init runs, their counters are permanently out of sync. No amount of barriers can fix this.

Solution: Use `_new_process_group_helper` (wrapped as `init_afd_process_group`) to create standalone groups with their own TCP store. Only pair members participate.

### Why Gloo for Control, NCCL for Data

- **afd_pg uses Gloo**: NCCL barriers hang with heterogeneous `CUDA_VISIBLE_DEVICES` because NCCL guesses device ID from global rank. Gloo barriers are CPU-based.
- **Pair metadata uses `pg.send/recv` directly**: `torch.distributed.send/recv` goes through `c10d_logger` which calls `dist.get_rank(group)` → `get_group_rank(group, default_pg.rank())`. For ATTN DP2+ (default PG rank >= 2), this fails because standalone pair groups have `pg_group_ranks = {0:0, 1:1}`. Calling `gloo_pg.send([tensor], dst_group_rank, tag).wait()` bypasses this entirely.
- **NCCL for tensors**: GPU-direct memory transfer for the hot path (hidden states every layer).

## Files Modified

| File | Changes |
|------|---------|
| `vllm/distributed/afd_transfer/afd_connector/p2p_connector.py` | Core rewrite: per-pair Gloo groups, `_create_pynccl_comm_for_pair`, asymmetric send/recv logic, `PairGroup` dataclass |
| `vllm/v1/worker/gpu_model_runner.py` | `elif` branch for non-sending ATTN ranks to call `update_state_from_dp_metadata` |
| `vllm/v1/worker/gpu_worker.py` | Graceful skip when profiler not configured (instead of RuntimeError) |
| `vllm/distributed/parallel_state.py` | `init_afd_process_group` function, PyTorch version comparison fix |
| `vllm/model_executor/layers/quantization/bitsandbytes.py` | fp16/bf16 dtype cast fix for fused MoE |

## Environment Setup

### Requirements

- GPUs: Ampere or newer (SM >= 8.0) — FlashAttention/MLA requirement
- CUDA toolkit: 12.0+
- PyTorch: 2.6+
- vLLM: v0.16.0rc2 on `pr-29772` branch
- Model: DeepSeek-V2-Lite (or any supported MoE model)

### Install

```bash
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

### SLURM GPU Allocation (Georgia Tech PACE cluster)

```bash
# 4x L40S (for 3A1F / 1A2F testing)
salloc -A gts-tkrishna3-ece -q inferno -p gpu-l40s --nodes=1 --gres=gpu:L40S:4 --mem 128G -t 2:00:00

# 4x H100
salloc -A gts-tkrishna3-ece -q inferno -p gpu-h100 --nodes=1 --gres=gpu:h100:4 --mem 128G -t 2:00:00

# 4x A100
salloc -A gts-tkrishna3-ece -q inferno -p gpu-a100 --nodes=1 --gres=gpu:a100:4 --mem 128G -t 2:00:00
```

## Launch Commands

**Important:** Always start FFN server first, then ATTN server. FFN creates the TCP store for rendezvous.

### 1A1F (Symmetric Baseline)

**FFN server (GPU 1):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee test-logs/ffn.log
```

**ATTN server (GPU 0):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A1F"}}' \
  2>&1 | tee test-logs/attn.log
```

### 2A2F (Symmetric)

**FFN server (GPUs 2,3):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}' \
  2>&1 | tee test-logs/ffn.log
```

**ATTN server (GPUs 0,1):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}' \
  2>&1 | tee test-logs/attn.log
```

### 3A1F (3 Attention + 1 FFN)

**FFN server (GPU 3):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --enable-expert-parallel --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}' \
  2>&1 | tee test-logs/ffn.log
```

**ATTN server (GPUs 0,1,2):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_ENGINE_READY_TIMEOUT_S=1800 \
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --data-parallel-size 3 --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}' \
  2>&1 | tee test-logs/attn.log
```

### 1A2F (1 Attention + 2 FFN)

**FFN server (GPUs 2,3):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}' \
  2>&1 | tee test-logs/ffn.log
```

**ATTN server (GPU 0):**
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --enable-expert-parallel \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"1A2F"}}' \
  2>&1 | tee test-logs/attn.log
```

## Test Request

```bash
# Single completion request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"Hello, how are you","max_tokens":32}'

# Benchmark (vLLM built-in)
vllm bench serve \
  --backend vllm \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random \
  --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate 2
```

## Cleanup

After killing servers, stale GPU processes may linger:

```bash
# Check for leftover processes
nvidia-smi

# Kill all vllm processes
ps aux | grep "vllm fserver\|vllm serve" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
```

## Bugs Fixed (Worklog)

### 1. `torch.distributed.new_group` counter desync deadlock
**Problem:** `new_group` uses a global counter. FFN (DP=1) and ATTN (DP=3) diverge during `initialize_model_parallel` → deadlock in AFD group creation.
**Fix:** Replaced with per-pair `init_afd_process_group` (standalone TCP store per pair).

### 2. NCCL barrier hang with heterogeneous CUDA_VISIBLE_DEVICES
**Problem:** `afd_pg` with NCCL backend hangs because NCCL guesses device ID from global rank.
**Fix:** Changed `afd_pg` to Gloo backend.

### 3. `ncclCommInitRank` failure from corrupted unique ID
**Problem:** `PyNcclCommunicator` uses `dist.broadcast(src=ranks[0])` which fails for standalone groups.
**Fix:** Created `_create_pynccl_comm_for_pair` — exchanges ncclUniqueId via point-to-point Gloo send/recv.

### 4. `ValueError: Global rank 2 is not part of group`
**Problem:** `torch.distributed.send/recv` goes through `c10d_logger` which calls `dist.get_rank(group)` → `get_group_rank(group, default_pg.rank())`. ATTN DP2 has default PG rank 2, not in standalone group's `{0:0, 1:1}`.
**Fix:** Bypass `torch.distributed.send/recv` entirely — call `gloo_pg.send([tensor], dst_group_rank, tag).wait()` directly.

### 5. Metadata send/recv mismatch in xA1F
**Problem:** All ATTN ranks send metadata, but FFN only reads from pair 0's Gloo group. DP1/DP2 sends hang forever.
**Fix:** `is_attn_top_min_size_rank` now returns True only for `dp_rank < min_size`.

### 6. 1A2F token chunking corruption
**Problem:** ATTN chunked tokens between FFN TP workers. But TP requires ALL workers to process SAME tokens.
**Fix:** ATTN broadcasts full tensor to all FFN workers. Only FFN TP rank 0 sends result back.

### 7. p2p_pg rank collision in 1A2F
**Problem:** FFN TP1 and ATTN both claimed p2p_rank=1 in the metadata transfer group.
**Fix:** Replaced p2p_pg with per-pair Gloo metadata transfer.

### 8. NoneType dp_metadata during profile/execute
**Problem:** With `data_parallel_size=1`, `DPMetadata` is never created, causing AttributeError.
**Fix:** Synthetic DPMetadata when forward context returns None.

### 9. fp16/bf16 dtype mismatch in fused MoE
**Problem:** bitsandbytes dequantizes to bf16 but activations are fp16.
**Fix:** Cast weights to match activation dtype.

### 10. PyTorch version comparison bug
**Problem:** String comparison `"2.10.0" < "2.6"` (lexicographic).
**Fix:** Numeric tuple comparison.

## Related Links

- [PR #29772](https://github.com/vllm-project/vllm/pull/29772) — AFD with P2P NCCL connector (this branch)
- [PR #25162](https://github.com/vllm-project/vllm/pull/25162) — Original AFD framework
- [Issue #22799](https://github.com/vllm-project/vllm/issues/22799) — AFD tracking issue
