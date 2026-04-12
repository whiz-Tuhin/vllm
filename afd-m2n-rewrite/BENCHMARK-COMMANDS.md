# AFD M×N Benchmark & Profiler Commands

## Prerequisites

```bash
cd /path/to/vllm-afd && source .venv/bin/activate

# Kill stale processes and verify GPUs are free
pkill -9 -u $USER -f 'vllm' ; sleep 5
nvidia-smi  # should show no Python processes
```

**Important notes:**
- Always run benchmarks **twice** on the same server. Run 1 = Triton JIT warmup (discard). Run 2 = real number.
- Port 29500 may linger after killing servers. Either wait 30 sec or use a different port (29510, 29520, etc.).
- All commands use `--enforce-eager` (no CUDA graphs).
- Model: `deepseek-ai/DeepSeek-V2-Lite`

---

## Benchmark Commands

### 1A1F (Baseline — 2 GPUs)

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

# Terminal 3: Sanity check + benchmarks
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  echo "=== Run $i at $(date) ==="
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 1A2F (3 GPUs — 1 ATTN + 2 FFN TP/EP)

```bash
# Terminal 1: FFN (GPUs 1,2)
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

# Terminal 3: Sanity + benchmarks
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  echo "=== Run $i at $(date) ==="
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 2A2F (4 GPUs — 2 ATTN DP + 2 FFN TP/EP)

```bash
# Terminal 1: FFN (GPUs 2,3)
CUDA_VISIBLE_DEVICES=2,3 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 2: ATTN (GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 2 \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 3: Sanity + benchmarks
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":16}'

for i in 1 2; do
  echo "=== Run $i at $(date) ==="
  vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
    --dataset-name random --random-input-len 128 --random-output-len 32 \
    --num-prompts 20 --request-rate inf
  sleep 5
done
```

### 3A1F (4 GPUs — 3 ATTN DP + 1 FFN)

```bash
# Terminal 1: FFN (GPU 3)
CUDA_VISIBLE_DEVICES=3 \
  vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --max-model-len 2048 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}'

# Terminal 2: ATTN (GPUs 0,1,2)
CUDA_VISIBLE_DEVICES=0,1,2 \
  vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --data-parallel-size 3 \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"3A1F"}}'

# Terminal 3: Sanity + benchmarks (same pattern)
```

---

## Profile with PyTorch Profiler

Add `--profiler-config.profiler torch` and `--profiler-config.torch_profiler_dir <dir>` to **both** server commands. Run the benchmark **twice** — the first run warms up Triton JIT, the second run captures the clean profiled trace.

### 1A2F Profiler Trace

```bash
LOGDIR=/path/to/profiler-traces/1A2F
mkdir -p $LOGDIR/ffn $LOGDIR/attn

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

# Terminal 3: Warmup + profiled run
echo "=== Warmup run (discard) ==="
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
sleep 5

echo "=== Profiled run ==="
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
```

### 2A2F Profiler Trace

```bash
LOGDIR=/path/to/profiler-traces/2A2F
mkdir -p $LOGDIR/ffn $LOGDIR/attn

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

# Terminal 3: Warmup + profiled run
echo "=== Warmup run (discard) ==="
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
sleep 5

echo "=== Profiled run ==="
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 20 --request-rate inf
```

### Viewing Traces

Traces are saved as Chrome trace JSON files in `$LOGDIR/attn/` and `$LOGDIR/ffn/`. Open with:
- **Chrome**: Navigate to `chrome://tracing`, click "Load", select the `.json` file
- **Perfetto UI**: https://ui.perfetto.dev — drag and drop the trace file

### What to Verify in Traces

**ATTN trace — should see:**
- `afd_p2p_send` NCCL ops at each layer boundary
- `afd_p2p_recv` NCCL ops for receiving FFN partials
- Gate compute (`ReplicatedLinear`) per MoE layer
- Shared expert MLP per MoE layer
- MLA attention kernels

**ATTN trace — should NOT see:**
- `AgRsAll2AllManager.dispatch` or `.combine` (EP collectives)
- `fused_moe_kernel` (routed experts live on FFN side only)

**FFN trace — should see:**
- `afd_p2p_recv` per layer
- `fused_moe_kernel` / `triton_` kernels (raw expert compute)
- `afd_p2p_send` per layer

**FFN trace — should NOT see:**
- EP all-gather or reduce-scatter between FFN workers
- `FusedMoEModularKernel._prepare` / `._finalize` (modular kernel bypassed)

---

## GPU Monitoring

Run alongside benchmarks to capture temperature, clocks, power, and utilization:

```bash
# In a separate terminal, start before benchmarks
nvidia-smi dmon -s pcut -d 1 2>&1 | tee gpu-monitor.log

# Ctrl+C when done
```

Check GPU topology:
```bash
nvidia-smi topo -m
```

Expected for H200 NVSwitch node: `NV18` between all GPU pairs.
