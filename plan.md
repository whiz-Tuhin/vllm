# Plan: Set up vLLM with AFD (Attention-FFN Disaggregation) from PR #29772

## Context
The user wants to benchmark attention vs FFN timings using vLLM's experimental AFD feature from PR #29772. This PR splits model inference so attention runs on one GPU and FFN/MoE layers run on another, communicating via NCCL P2P. Target: DeepSeek-V2-Lite on 2x TITAN RTX (24GB each), set up at `/home/tuhin/work/vllm-afd/`.

## Setup Steps

### Step 1: Clone vLLM and checkout PR #29772
```bash
cd /home/tuhin/work/vllm-afd
git clone https://github.com/vllm-project/vllm.git .
git fetch origin pull/29772/head:pr-29772
git checkout pr-29772
```

### Step 2: Create Python virtual environment with uv
```bash
cd /home/tuhin/work/vllm-afd
uv venv .venv --python 3.12
source .venv/bin/activate
```

### Step 3: Install vLLM from source (fast method)
Since PR #29772's changes are **all Python** (configs, model code, entrypoints, workers), we can use `VLLM_USE_PRECOMPILED=1` to skip CUDA kernel compilation and use prebuilt binaries:
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```
This is much faster than a full source build (minutes vs. 30-60 min).

**Fallback** if CUDA 12.0 causes issues: install a specific vLLM wheel first, then overlay the PR's Python changes.

### Step 4: Download DeepSeek-V2-Lite
```bash
# Using huggingface-cli (installed with vllm dependencies)
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite --local-dir /home/tuhin/work/vllm-afd/models/DeepSeek-V2-Lite
```

### Step 5: Run AFD benchmark (two terminals)

**Important notes for TITAN RTX:**
- TITAN RTX (Turing) does **not** support BF16 — must use `--dtype float16`
- Use `CUDA_VISIBLE_DEVICES` to pin each server to its GPU

**Terminal 1 — Attention server (GPU 0):**
```bash
cd /home/tuhin/work/vllm-afd
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve "/home/tuhin/work/vllm-afd/models/DeepSeek-V2-Lite" \
  --dtype float16 \
  --enforce_eager \
  --enable-dbo \
  --dbo-prefill-token-threshold 12 \
  --dbo-decode-token-threshold 2 \
  --afd-config '{"afd_connector":"p2pconnector", "afd_role":"attention", "afd_host":"127.0.0.1", "afd_port":"29500", "num_afd_stages":"2", "afd_extra_config":{"afd_size":"1A1F"}}'
```

**Terminal 2 — FFN server (GPU 1):**
```bash
cd /home/tuhin/work/vllm-afd
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm fserver "/home/tuhin/work/vllm-afd/models/DeepSeek-V2-Lite" \
  --dtype float16 \
  --enforce_eager \
  --afd-config '{"afd_connector":"p2pconnector", "afd_role":"ffn", "afd_host":"127.0.0.1", "afd_port":"29500", "num_afd_stages":"2", "afd_extra_config":{"afd_size":"1A1F"}}'
```

## Key Risks / Notes

1. **CUDA 12.0 vs 12.1+**: The PR targets vLLM main which may need CUDA >= 12.1. If `VLLM_USE_PRECOMPILED=1` install fails, we may need to upgrade the CUDA toolkit or pin a compatible vLLM version.
2. **PR needs rebase**: PR #29772 is behind main and labeled `needs-rebase`. There may be merge conflicts when checking out. If the PR branch is too stale, we can try cherry-picking its commits onto a recent vLLM release tag instead.
3. **Memory**: DeepSeek-V2-Lite is ~16B params (31GB FP16). With AFD, each side loads only its portion of weights (attention vs FFN/MoE), so it should fit on 24GB per GPU, but could be tight.
4. **NCCL P2P across GPUs**: The P2PConnector uses `torch.distributed` send/recv. Both TITAN RTX GPUs need PCIe P2P support (should work on the same machine).
5. **`--enable_expert_parallel`**: The PR README uses this flag with `data_parallel_size=2`. With only 1 GPU per role (1A1F), it may not be needed. We'll omit it initially and add if required.

## Verification
1. Both servers start without errors and log successful NCCL connection
2. Send a test request to the attention server's API endpoint: `curl http://localhost:8000/v1/completions -d '{"model":"DeepSeek-V2-Lite","prompt":"Hello","max_tokens":16}'`
3. Check GPU utilization with `nvidia-smi` — both GPUs should show activity
4. Review vLLM logs for attention/FFN timing breakdowns
