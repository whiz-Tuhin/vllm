# Testing AFD pre-routing + DBO + FFN HTTP control on a new machine

This guide is for testing the four capabilities added in this branch. For a
description of *what* changed and *why*, read [PR-SUMMARY.md](PR-SUMMARY.md)
first. This document is the operator's manual.

The four toggles are independent; you can mix and match.

| `ENABLE_PREROUTING` | `ENABLE_DBO` | What runs |
|---:|---:|---|
| 0 (default) | 0 (default) | Option B (broadcast), no DBO. Existing baseline. |
| 1 | 0 | Option A (true pre-routing) |
| 0 | 1 | Option B + DBO |
| 1 | 1 | Option A + DBO — **target configuration** |

The `--ffn-control-port` flag is also independent — it just controls whether
the FFN process exposes HTTP profiler endpoints. Pass it (or not) under any
of the four combinations above.

---

## 0. Prerequisites on the new machine

```bash
# Clone this branch
git clone -b tk/pr-29772-afd-pre-routing <repo-url>
cd vllm-afd

# Create venv + editable install (precompiled CUDA kernels)
python -m venv .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Sanity
python -c "import vllm; print(vllm.__version__)"
which vllm                       # should be inside .venv/bin
vllm fserver --help | grep ffn-control-port   # confirms the new flag is wired
```

**GPU requirement.** AFD splits attention and FFN across separate GPUs.
Minimum 2 GPUs for `1A1F`. Larger configs need M+N GPUs total
(e.g. `4A4F` needs 8). Each GPU should have ≥24 GB for V2-Lite at fp16
without quantization, or ≥12 GB with bitsandbytes 4-bit.

**FlashAttention requirement.** DeepSeek V2 uses MLA, which requires
**Ampere or newer** (A100/H100/RTX 30xx+). Turing GPUs (TITAN RTX, T4)
will crash on prefill — see CLAUDE.md worklog item 7.

**Optional: Python.h headers** (only if Triton CUDA-utils compile fails):

```bash
export CPATH=/path/to/.venv/include/python3.12
```

---

## 1. Recommended path: the sweep script

The fastest way to exercise everything is the bundled sweep script. It
launches both servers in coordinated configurations, runs `vllm bench serve`,
and writes per-config logs.

```bash
cd vllm-afd
source .venv/bin/activate

# Edit afd-m2n-rewrite/run-sweep.sh once for your machine:
#   - MODEL=...                  (path to V2-Lite checkout, or HF id)
#   - SCRATCH=...                (where to write results)
#   - CUDA_VISIBLE_DEVICES_*     (which GPUs to use)
#   - configs in CONFIGS=()      (e.g. only "1A1F" for the first smoke test)

# Default — Option B, no DBO. Confirms baseline is intact.
./afd-m2n-rewrite/run-sweep.sh

# Option A only
ENABLE_PREROUTING=1 ./afd-m2n-rewrite/run-sweep.sh

# DBO only (Option B + DBO)
ENABLE_DBO=1 ./afd-m2n-rewrite/run-sweep.sh

# Both — what we care about for the paper
ENABLE_PREROUTING=1 ENABLE_DBO=1 ./afd-m2n-rewrite/run-sweep.sh
```

**Workload shape** (override defaults: 20 prompts / 128 input / 32 output / inf rate):

```bash
# Sustained-load throughput sweep
NUM_PROMPTS=1000 OUTPUT_LEN=128 REQ_RATE=50 \
  ENABLE_PREROUTING=1 ENABLE_DBO=1 \
  ./afd-m2n-rewrite/run-sweep.sh
```

Results land in `$SCRATCH/afd-sweep/<timestamp>/configs/<cfg>/<phase>/`.
The sweep banner and per-config `.cmd` files log both toggle states so you
can confirm the run was what you intended.

---

## 2. Manual launch (when you want full control)

This is also what you do when using the **FFN HTTP control port** for
measurement-quality torch profiler traces.

### Terminal 1: FFN server with control port

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 2048 \
  --ffn-control-port 8002 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /tmp/traces/ffn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'
```

### Terminal 2: ATTN server

For **Option A**: prefix the launch with `VLLM_AFD_USE_PREROUTING=1`.
For **DBO**: pass `--enable-dbo --dbo-decode-token-threshold=2 --dbo-prefill-token-threshold=10`
and set `num_afd_stages=2` in the JSON.

```bash
source .venv/bin/activate
# Option A + DBO example:
VLLM_AFD_USE_PREROUTING=1 \
CUDA_VISIBLE_DEVICES=0,1 vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 2048 \
  --enable-dbo --dbo-decode-token-threshold=2 --dbo-prefill-token-threshold=10 \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}}'
```

### Terminal 3: client (smoke test)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"What is the capital of France?","max_tokens":32}'
```

Coherent text → all four moving parts work for that toggle combination.

---

## 3. FFN HTTP control port workflow

Use this when you want measurement-quality torch traces and need to discard
warmup runs without restarting the FFN server.

```bash
# 1. Confirm the control port is up
curl http://localhost:8002/health
# → {"status":"ok"}

# 2. End the auto-started startup trace (so warmup data is dropped)
curl -X POST http://localhost:8002/stop_profile

# 3. Run a warmup benchmark (no profiler running)
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite \
  --num-prompts 20 --random-input-len 128 --random-output-len 32 \
  --request-rate inf --port 8000

# 4. Start a fresh profiler trace for the measured run
curl -X POST http://localhost:8002/start_profile

# 5. Run the measured benchmark
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite ...

# 6. Stop and flush the trace JSON to disk
curl -X POST http://localhost:8002/stop_profile

# 7. Repeat steps 4–6 for additional measurements without restarting servers.
```

Trace files appear in the directory passed to `--profiler-config.torch_profiler_dir`.

**Security note:** the HTTP server has no authentication. Bind only on
trusted networks; default `0.0.0.0` is fine for localhost-only access on a
single-tenant machine but **don't expose it to the internet**.

---

## 4. Verifying each capability took effect

### Option A active

```bash
grep "Pre-routing (Option A) ENABLED" <attn-log>
# Should appear once at startup if VLLM_AFD_USE_PREROUTING=1.
```

If grep returns nothing → the env var didn't propagate to the ATTN process.
Re-export it before the `vllm serve` command, not after.

### DBO active

```bash
grep "Asynchronous scheduling is enabled" <attn-log>
grep "Disabling NCCL for DP synchronization" <attn-log>   # should be absent
```

Also confirm the AFD config JSON for both processes shows
`num_afd_stages: 2`. With `num_afd_stages: 1` the connector's per-stage
state still works correctly (Change 2 is backward-compatible) but DBO will
not actually split the batch.

### Bandwidth saving (Option A vs B)

Run the same workload under both toggles with `AFD_TIMING=1`:

```bash
AFD_TIMING=1 ENABLE_PREROUTING=0 ./afd-m2n-rewrite/run-sweep.sh   # Option B baseline
AFD_TIMING=1 ENABLE_PREROUTING=1 ./afd-m2n-rewrite/run-sweep.sh   # Option A
grep "send_attn.bytes_total" <attn-log>
```

Expect Option A's value to be ~80% of Option B's at EP=4 (16 experts/FFN).
~57% at EP=8. ~98% at EP=2 (essentially no saving — Option A on EP=2 is not
worth the complexity, fall back to B).

### FFN control port reachable

```bash
curl -fsS http://localhost:8002/health && echo OK
# OK
```

A 200 response confirms the daemon thread started. If it returns connection
refused, check the FFN log for `FFN control HTTP server listening on port`
— if absent, the `--ffn-control-port` flag wasn't passed to `vllm fserver`.

---

## 5. Recommended experiment order on a new machine

Step through this list before any large sweep:

1. **Smoke test 1A1F under default** (Option B, no DBO). Curl returns
   coherent text. Confirms the per-stage state refactor (Change 2) didn't
   break the existing path.
2. **Smoke test 1A1F under all four toggle combinations.** 4 curls, all
   coherent. Verifies Option A combine and DBO don't corrupt state.
3. **Smoke test 4A4F under all four toggle combinations.** Larger EP
   exercises more partner pairs, more masks per layer.
4. **Bandwidth ablation at 4A4F** with `AFD_TIMING=1`. Compare
   `send_attn.bytes_total` for A vs B on the same workload.
5. **TPOT 2×2 matrix at 4A4F**, sustained `request_rate=20`. All four
   toggle combinations. Expect A+DBO best.
6. **Saturation sweep** under A+DBO, configs 1A1F → 7A1F. Find the
   AFD-scaling crossover (or confirm V2-Lite can't expose it at this scale).

V2 (236B) experiments should wait until V2-Lite passes step 5 cleanly.

---

## 6. Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `connection refused` on port 8002 | `--ffn-control-port` not passed | Add it to `vllm fserver` command |
| Coherent output under B but garbled under A | Mask/index_add bug | Reproduce on 1A1F first; check `_pending_masks` lengths match `n_partners` |
| Hang during DBO warmup at startup | `num_afd_stages` mismatch between ATTN/FFN | Both must be 2 |
| `bytes_total` identical between A and B | `VLLM_AFD_USE_PREROUTING=1` didn't reach the ATTN process | Confirm with the "Option A active" grep above |
| Profiler trace JSON empty / missing | Worker exited before flush | Use `POST /stop_profile` (or sweep script's two-stage SIGTERM-then-SIGKILL kill_vllm) |
| FlashAttention crash on prefill | GPU is Turing (SM 7.5) | Run on Ampere or newer |
| Triton compile error: `Python.h not found` | Missing Python dev headers | `export CPATH=/path/to/python-include` |

---

## 7. What files to look at if something breaks

| Symptom area | File |
|---|---|
| Option A send/recv path | [`p2p_connector.py`](../vllm/distributed/afd_transfer/afd_connector/p2p_connector.py) — `_send_attn_output_option_a`, `_recv_ffn_output_option_a` |
| DBO state across stages | Same file — `_pending_*` dicts, `stage_idx` plumbing |
| FFN HTTP control | [`afd_ffn_server.py`](../vllm/entrypoints/afd_ffn_server.py) — `_start_control_server`, `_start_profiler_rpc`, `_stop_profiler_rpc` |
| `--ffn-control-port` flag | [`fserver.py`](../vllm/entrypoints/cli/fserver.py) |
| stage_idx flow into connector | [`deepseek_v2.py`](../vllm/model_executor/models/deepseek_v2.py) — `forward_with_afd` |
| Profiler RPC routing | [`gpu_worker.py`](../vllm/v1/worker/gpu_worker.py) — `profile()` method |
| Sweep harness | [`run-sweep.sh`](run-sweep.sh) |
