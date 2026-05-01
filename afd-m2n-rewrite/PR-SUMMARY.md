# PR Summary — Option A pre-routing, DBO thread-safety, FFN HTTP control

This PR layers four independent capabilities onto the existing M×N AFD
implementation. All four are off by default — current behavior is preserved
unless the new toggles are explicitly enabled.

---

## What's new

### 1. Option A: true pre-routing on the ATTN side

`P2PAFDConnector.send_attn_output` and `recv_ffn_output` now have a
sync-free Option A path that sends each FFN partner only the subset of
tokens whose top-k experts include any of that partner's local experts.

- Per-partner mask is computed on GPU.
- A single `.cpu()` per layer (with `non_blocking=True` async copy + one
  stream sync) reads all per-partner counts in one shot — does not drain
  the NCCL queue.
- ATTN sends a count header followed by the variable-size subset
  (hidden_states, topk_ids, topk_weights) per partner.
- FFN reads the count header, sizes its recv buffers accordingly, runs
  `forward_pre_routed`, and sends back a `[count_j, H]` partial.
- ATTN combines partials via `index_add_` using the stored masks.

Estimated bandwidth saving on the attn→ffn link, with top-6 of 64 experts:

| Config | Experts/FFN | P(token skips a partner) | Bandwidth saving |
|--------|------------:|-------------------------:|-----------------:|
| EP=2 (1A2F, 2A2F) | 32 | 1.9% | ~2% |
| EP=4 (1A4F, 2A4F, 4A4F) | 16 | 19.6% | ~20% |
| EP=8 | 8 | 43% | ~43% |

Auto-falls-back to Option B for dense layers (no routing) and for
single-FFN-partner configs (xA1F where Option A is degenerate).

**Default: OFF** — set env var `VLLM_AFD_USE_PREROUTING=1` to enable.

### 2. DBO thread-safety

All seven `_pending_*` fields in `P2PAFDConnector` are now keyed by
`stage_idx` (per-ubatch slot). Required for `--enable-dbo` to work
correctly: under DBO, two Python threads call `send_attn_output` and
`recv_ffn_output` concurrently for the same MoE layer with different
microbatches. Single-slot state would clobber across threads. With
per-stage dicts, each ubatch has its own slot.

`stage_idx` is plumbed through via `metadata.stage_idx` (already populated
by `gpu_ubatch_wrapper._make_ubatch_metadata` per ubatch).

**Default: stage_idx defaults to 0** — non-DBO single-stage code path
is unchanged behaviorally.

### 3. FFN HTTP control server

`vllm fserver` now accepts `--ffn-control-port N`. When set, spawns a
small FastAPI/uvicorn server in a daemon thread on port `N` that exposes:

- `POST /start_profile` → starts torch profiler on all FFN workers
- `POST /stop_profile` → stops profiler and **flushes traces to disk**
- `GET /health` → 200 OK liveness check

Eliminates the SIGTERM-to-flush workaround for measurement-quality traces.
The server stays up across multiple bench runs — no model reload between
measurements.

**Default: not started** — pass `--ffn-control-port 8002` to enable.

### 4. Sweep script env-var toggles

`afd-m2n-rewrite/run-sweep.sh` now honors:

- `ENABLE_DBO=1` → adds `--enable-dbo --dbo-decode-token-threshold=2
  --dbo-prefill-token-threshold=10` and bumps `num_afd_stages` to 2 in
  the AFD config JSON.
- `ENABLE_PREROUTING=1` → exports `VLLM_AFD_USE_PREROUTING=1` for the
  connector.
- `NUM_PROMPTS / INPUT_LEN / OUTPUT_LEN / REQ_RATE` env-overridable for
  flexible workload shaping (default 20 / 128 / 32 / inf as before).

Sweep banner and per-config `.cmd` files log both toggle states.

### 5. Bandwidth-validation instrumentation

`_nccl_send` and `_nccl_recv_into` now accumulate `send_attn.bytes_total`
and `recv.bytes_total` in the existing `_TimingProfiler` (gated by
`AFD_TIMING=1`). Lets you compare wire bytes between Option A and Option B
on the same workload to validate the bandwidth saving.

---

## Files modified

| File | Changes |
|---|---|
| [`vllm/distributed/afd_transfer/afd_connector/p2p_connector.py`](vllm-afd/vllm/distributed/afd_transfer/afd_connector/p2p_connector.py) | Per-stage `_pending_*` dicts; new Option A send/recv helper methods; `VLLM_AFD_USE_PREROUTING` env-var gate; bytes_total instrumentation |
| [`vllm/model_executor/models/deepseek_v2.py`](vllm-afd/vllm/model_executor/models/deepseek_v2.py) | Plumb `stage_idx` from `afd_metadata` into both `recv_ffn_output` calls |
| [`vllm/entrypoints/afd_ffn_server.py`](vllm-afd/vllm/entrypoints/afd_ffn_server.py) | `_start_control_server` + `/start_profile`/`/stop_profile`/`/health` endpoints |
| [`vllm/entrypoints/cli/fserver.py`](vllm-afd/vllm/entrypoints/cli/fserver.py) | New `--ffn-control-port` CLI flag |
| [`vllm/v1/worker/gpu_ffn_model_runner.py`](vllm-afd/vllm/v1/worker/gpu_ffn_model_runner.py) | `start_profile` / `stop_profile` driver methods; `compute_ffn.total` timing instrumentation |
| [`vllm/v1/worker/gpu_worker.py`](vllm-afd/vllm/v1/worker/gpu_worker.py) | Route `profile(is_start)` RPC to `model_runner.start_profile()` for FFN role so the FFN-side trace handler actually flushes (was previously dormant) |
| [`vllm/v1/worker/gpu_model_runner.py`](vllm-afd/vllm/v1/worker/gpu_model_runner.py) | Hot-path debug log lines commented out (per-forward `nums_reqs` / `input_ids` printouts) |
| [`vllm/v1/worker/gpu_ubatch_wrapper.py`](vllm-afd/vllm/v1/worker/gpu_ubatch_wrapper.py) | Same — `jcz` debug logs commented |
| [`afd-m2n-rewrite/run-sweep.sh`](vllm-afd/afd-m2n-rewrite/run-sweep.sh) | `ENABLE_DBO` + `ENABLE_PREROUTING` toggles; workload-shape env vars; SIGTERM-then-SIGKILL kill_vllm |

---

## How to experiment

### Toggle matrix

| `ENABLE_PREROUTING` | `ENABLE_DBO` | What runs |
|---:|---:|---|
| 0 (default) | 0 (default) | Option B (broadcast), no DBO. Current published baseline. |
| 1 | 0 | Option A (true pre-routing, ~20% bandwidth saving at EP=4), no DBO. |
| 0 | 1 | Option B + DBO (overlap-only experiment). |
| 1 | 1 | Both — the target configuration. |

### Quickstart

```bash
cd /storage/scratch1/0/hwu419/tkhare7/vllm-afd && source .venv/bin/activate

# Default — Option B, no DBO (what we've been running)
./afd-m2n-rewrite/run-sweep.sh

# Option A only
ENABLE_PREROUTING=1 ./afd-m2n-rewrite/run-sweep.sh

# DBO only
ENABLE_DBO=1 ./afd-m2n-rewrite/run-sweep.sh

# Both — this is what we want to measure
ENABLE_PREROUTING=1 ENABLE_DBO=1 ./afd-m2n-rewrite/run-sweep.sh
```

Each invocation writes to a fresh timestamped directory under
`/storage/scratch1/0/hwu419/tkhare7/scratchpad/afd-sweep/<timestamp>/`.
The first 200 lines of `<timestamp>/configs/<cfg>/<phase>/attn.log` show
the launch flags and confirm both toggles took effect.

### Workload shaping

```bash
# Sustained-load throughput sweep (overrides defaults)
NUM_PROMPTS=1000 OUTPUT_LEN=128 REQ_RATE=50 \
  ENABLE_PREROUTING=1 ENABLE_DBO=1 \
  ./afd-m2n-rewrite/run-sweep.sh

# Single-shot latency floor (default workload)
ENABLE_PREROUTING=1 ENABLE_DBO=1 ./afd-m2n-rewrite/run-sweep.sh
```

### Manual launch with FFN HTTP control port

For measurement-quality traces without restarting the FFN server between
runs:

```bash
# Terminal 1: FFN with control port
CUDA_VISIBLE_DEVICES=2,3 vllm fserver deepseek-ai/DeepSeek-V2-Lite \
  --dtype float16 --enforce-eager --tensor-parallel-size 2 \
  --enable-expert-parallel --max-model-len 2048 \
  --ffn-control-port 8002 \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /path/to/traces/ffn \
  --afd-config '{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"29500","num_afd_stages":"1","afd_extra_config":{"afd_size":"2A2F"}}'

# Terminal 2: ATTN as usual

# Terminal 3: workflow per measurement
# 1. Stop the auto-started warmup trace
curl -X POST http://localhost:8002/stop_profile

# 2. Run a warmup benchmark (no profiler trace)
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite ...

# 3. Start fresh trace for measurement
curl -X POST http://localhost:8002/start_profile

# 4. Run measured benchmark
vllm bench serve --model deepseek-ai/DeepSeek-V2-Lite ...

# 5. Stop and flush
curl -X POST http://localhost:8002/stop_profile

# 6. Repeat 3-5 for next measurement without restarting servers
```

### Verification per change

| Capability | How to verify it took effect |
|---|---|
| Option A active | grep `clean-attn.log` for `Pre-routing (Option A) ENABLED via VLLM_AFD_USE_PREROUTING=1` |
| DBO active | grep ATTN log for `Asynchronous scheduling is enabled` AND no `Disabling NCCL for DP synchronization` banner; AFD config JSON shows `num_afd_stages: 2` |
| Bandwidth saving | grep `clean-attn.log` for `send_attn.bytes_total` (requires `AFD_TIMING=1` in the launch env) — Option A's value should be ~80% of Option B's at EP=4 |
| FFN control port | `curl http://localhost:8002/health` returns `{"status":"ok"}` |

---

## Recommended experiment order

1. **Smoke test** — launch 1A1F under default (Option B, no DBO), curl a
   sanity prompt, confirm coherent output. Confirms the per-stage state
   refactor didn't break anything.
2. **Smoke test** — same on 4A4F under all four toggle combinations.
   Verifies Option A's mask/index_add combine produces correct output
   and DBO doesn't cause silent state corruption.
3. **Bandwidth ablation** — same workload, A vs B at 4A4F, with
   `AFD_TIMING=1`. Compare `send_attn.bytes_total` numbers.
4. **TPOT 2×2 matrix** — 4A4F, sustained `request_rate=20`, all four
   toggle combinations. Expect A+DBO best.
5. **Saturation sweep** — Option A + DBO, configs from 1A1F up through
   7A1F, sweep `request_rate`. Find the AFD-scaling crossover (or
   confirm it doesn't appear on V2-Lite at this scale).

V2 (236B) experiments are deferred — sanity numbers on V2-Lite first.

---

## Known limitations

- DBO with Option A has not yet been smoke-tested. The connector state
  refactor and Option A path are independent, but their interaction
  needs end-to-end verification.
- Dense layer 0 still does TP all-reduce in `DeepseekV2MLP.down_proj`.
  Out of scope for this PR; mentioned as future work in the
  AFD-communication-structure doc.
- For DeepSeek-V2 full / V3, the `routed_scaling_factor` fix
  (`DeepseekV2MoEAttentionStub.compute_route_and_shared` line ~317)
  has not yet been applied. V2-Lite uses 1.0 so this is a no-op for
  the current experiments. See plan §"Change 3 (deferred)".
- The `--ffn-control-port` HTTP server has no auth. Don't expose it
  beyond localhost.
