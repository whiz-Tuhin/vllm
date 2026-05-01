#!/bin/bash
# AFD M×N experimental sweep
# ---------------------------
# Runs 4 AFD configurations (1A2F, 2A1F, 2A2F, 3A1F) through two phases:
#   Phase A: server WITHOUT profiler, 1 warmup + 3 clean benchmarks
#   Phase B: server WITH PyTorch profiler, 1 warmup + 1 profiled benchmark
#
# Expected wall time: ~50-80 min for all 4 configs on a dedicated H200 node.
#
# Usage:
#   ./afd-m2n-rewrite/run-sweep.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: /storage/scratch1/0/hwu419/tkhare7/scratchpad/afd-sweep/<timestamp>
#
# Requires:
#   - 4 H200 GPUs visible to the node (GPUs 0-3)
#   - Python venv at /storage/scratch1/0/hwu419/tkhare7/vllm-afd/.venv
#   - Model pre-cached: deepseek-ai/DeepSeek-V2-Lite
#   - Port 29500 free (or pass AFD_PORT=xxxxx in env)
# ------------------------------------------------------------------

set -euo pipefail

# -------------------- Configuration --------------------

REPO=/storage/scratch1/0/hwu419/tkhare7/vllm-afd
VENV_ACTIVATE=$REPO/.venv/bin/activate
MODEL=deepseek-ai/DeepSeek-V2-Lite
AFD_PORT=${AFD_PORT:-29500}
NUM_PROMPTS=${NUM_PROMPTS:-20}
INPUT_LEN=${INPUT_LEN:-128}
OUTPUT_LEN=${OUTPUT_LEN:-32}
REQ_RATE=${REQ_RATE:-inf}

# Boot/warmup timeouts (seconds)
SERVER_READY_TIMEOUT=300   # max wait for both servers to be ready
BENCH_SLEEP_BETWEEN=5      # pause between consecutive benchmarks on same server
CONFIG_SLEEP_BETWEEN=15    # pause between config switches (lets TCP sockets release)

# DBO + Pre-routing toggles
# ENABLE_DBO=1     -> turn on dual-batch overlap on the ATTN side and bump
#                     num_afd_stages from 1 to 2 in the AFD config JSON.
#                     Requires connector state to be thread-safe (Change 2).
# ENABLE_PREROUTING=1 -> turn on Option A (true pre-routing: send only the
#                     subset of tokens each FFN partner needs). Read by the
#                     connector via VLLM_AFD_USE_PREROUTING env var.
#                     Default off keeps Option B (broadcast) behavior.
ENABLE_DBO=${ENABLE_DBO:-0}
ENABLE_PREROUTING=${ENABLE_PREROUTING:-0}

if [ "$ENABLE_DBO" = "1" ]; then
  ATTN_DBO_FLAGS="--enable-dbo --dbo-decode-token-threshold=2 --dbo-prefill-token-threshold=10"
  AFD_NUM_STAGES="2"
else
  ATTN_DBO_FLAGS=""
  AFD_NUM_STAGES="1"
fi

if [ "$ENABLE_PREROUTING" = "1" ]; then
  export VLLM_AFD_USE_PREROUTING=1
else
  unset VLLM_AFD_USE_PREROUTING || true
fi

# Output directory
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_ROOT=${1:-/storage/scratch1/0/hwu419/tkhare7/scratchpad/afd-sweep/$STAMP}
mkdir -p "$OUT_ROOT"

# -------------------- Utilities --------------------

log() { echo "[$(date +%H:%M:%S)] $*"; }

kill_vllm() {
  # Two-stage shutdown: SIGTERM first so profiler traces flush, SIGKILL only
  # if the process refuses to exit. Without this, profiler trace JSON is lost
  # because torch.profiler's trace handler only fires on profiler.stop().
  pkill -TERM -u "$USER" -f 'vllm (serve|fserver)' 2>/dev/null || true
  # Give the SIGTERM handler in afd_ffn_server.py up to 30s to flush traces
  for i in $(seq 1 30); do
    if ! pgrep -u "$USER" -f 'vllm (serve|fserver)' >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  # Hard kill anything that didn't exit cleanly
  pkill -9 -u "$USER" -f 'vllm (serve|fserver)' 2>/dev/null || true
  sleep 2
  # Wait for AFD port to be released (up to 60s)
  for i in $(seq 1 60); do
    if ! ss -tln 2>/dev/null | grep -q ":$AFD_PORT "; then
      return 0
    fi
    sleep 1
  done
  log "WARN: port $AFD_PORT still in use after 60s — TIME_WAIT will eventually clear"
}

wait_for_ready() {
  # Waits until both ATTN and FFN servers log their ready indicators.
  # Args: $1=attn_log $2=ffn_log
  local attn_log=$1 ffn_log=$2
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  local attn_ready=0 ffn_ready=0
  while [ $SECONDS -lt $deadline ]; do
    if [ $attn_ready -eq 0 ] && grep -q "Application startup complete" "$attn_log" 2>/dev/null; then
      attn_ready=1
      log "  ATTN ready"
    fi
    if [ $ffn_ready -eq 0 ] && grep -q "FFN server loop started" "$ffn_log" 2>/dev/null; then
      ffn_ready=1
      log "  FFN ready"
    fi
    if [ $attn_ready -eq 1 ] && [ $ffn_ready -eq 1 ]; then
      return 0
    fi
    # Bail early if any server already errored out
    if grep -qE "EngineCore.*ERROR|ImportError|CUDA out of memory" "$attn_log" "$ffn_log" 2>/dev/null; then
      log "ERROR: server reported a fatal error — aborting this config"
      return 1
    fi
    sleep 2
  done
  log "ERROR: servers not ready after ${SERVER_READY_TIMEOUT}s"
  return 1
}

run_bench() {
  # Runs one benchmark and tees output to the given log file.
  local out=$1
  vllm bench serve --model "$MODEL" \
    --dataset-name random \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS --request-rate $REQ_RATE \
    >"$out" 2>&1 || { log "WARN: bench failed (exit $?)"; return 1; }
  local tpot
  tpot=$(grep "Mean TPOT" "$out" | awk '{print $NF}')
  local dur
  dur=$(grep "Benchmark duration" "$out" | awk '{print $NF}')
  log "  TPOT=${tpot:-?}ms  duration=${dur:-?}s"
}

# -------------------- Per-config launch ----------------

# Each config specifies:
#   ATTN_GPUS, ATTN_DP    (and empty ATTN_DP means --data-parallel-size 1)
#   FFN_GPUS, FFN_TP, FFN_EP
# These are then used to build the command lines.

launch_servers() {
  # Args: $1=phase_name ("clean" or "profiled")
  #       $2=attn_log  $3=ffn_log
  #       $4=attn_gpus $5=attn_dp_flag
  #       $6=ffn_gpus  $7=ffn_tp_flag $8=ffn_ep_flag
  #       $9=profiler_flag (empty or --profiler-config.profiler=torch ...)
  #       $10=afd_size (e.g., 2A2F)
  local phase=$1 attn_log=$2 ffn_log=$3
  local attn_gpus=$4 attn_dp_flag=$5
  local ffn_gpus=$6 ffn_tp_flag=$7 ffn_ep_flag=$8
  local profiler_attn=${9:-}
  local profiler_ffn=${10:-}
  local afd_size=${11}

  local afd_cfg_attn='{"afd_connector":"p2pconnector","afd_role":"attention","afd_host":"127.0.0.1","afd_port":"'$AFD_PORT'","num_afd_stages":"'$AFD_NUM_STAGES'","afd_extra_config":{"afd_size":"'$afd_size'"}}'
  local afd_cfg_ffn='{"afd_connector":"p2pconnector","afd_role":"ffn","afd_host":"127.0.0.1","afd_port":"'$AFD_PORT'","num_afd_stages":"'$AFD_NUM_STAGES'","afd_extra_config":{"afd_size":"'$afd_size'"}}'

  # Save launch commands for reproducibility
  local cmd_file="${attn_log%.log}.cmd"
  {
    echo "# Phase: $phase"
    echo "# afd_size: $afd_size"
    echo "# attn_gpus: $attn_gpus (flags: $attn_dp_flag)"
    echo "# ffn_gpus:  $ffn_gpus (flags: $ffn_tp_flag $ffn_ep_flag)"
    echo "# DBO:       $ENABLE_DBO  (num_afd_stages=$AFD_NUM_STAGES; flags='$ATTN_DBO_FLAGS')"
    echo "# Pre-routing (Option A): $ENABLE_PREROUTING  (VLLM_AFD_USE_PREROUTING=${VLLM_AFD_USE_PREROUTING:-unset})"
    echo ""
    echo "# --- FFN server ---"
    echo "CUDA_VISIBLE_DEVICES=$ffn_gpus VLLM_AFD_USE_PREROUTING=${VLLM_AFD_USE_PREROUTING:-} vllm fserver $MODEL \\"
    echo "  --dtype float16 --enforce-eager \\"
    echo "  --max-model-len 2048 \\"
    echo "  $ffn_tp_flag $ffn_ep_flag $profiler_ffn \\"
    echo "  --afd-config '$afd_cfg_ffn'"
    echo ""
    echo "# --- ATTN server ---"
    echo "CUDA_VISIBLE_DEVICES=$attn_gpus VLLM_AFD_USE_PREROUTING=${VLLM_AFD_USE_PREROUTING:-} vllm serve $MODEL \\"
    echo "  --dtype float16 --enforce-eager \\"
    echo "  --max-model-len 2048 --gpu-memory-utilization 0.85 \\"
    echo "  $attn_dp_flag $ATTN_DBO_FLAGS $profiler_attn \\"
    echo "  --afd-config '$afd_cfg_attn'"
  } >"$cmd_file"

  log "Launching FFN (GPUs=$ffn_gpus) ..."
  (
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$ffn_gpus \
    vllm fserver "$MODEL" \
      --dtype float16 --enforce-eager \
      --max-model-len 2048 \
      $ffn_tp_flag $ffn_ep_flag $profiler_ffn \
      --afd-config "$afd_cfg_ffn" \
      >"$ffn_log" 2>&1
  ) &

  log "Launching ATTN (GPUs=$attn_gpus) ..."
  (
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$attn_gpus \
    vllm serve "$MODEL" \
      --dtype float16 --enforce-eager \
      --max-model-len 2048 --gpu-memory-utilization 0.85 \
      $attn_dp_flag $ATTN_DBO_FLAGS $profiler_attn \
      --afd-config "$afd_cfg_attn" \
      >"$attn_log" 2>&1
  ) &

  wait_for_ready "$attn_log" "$ffn_log" || return 1

  # Assert the NCCL DP sync fix is active (no Gloo CPU allreduce).
  # If the edit to config/vllm.py is missing or reverted, this banner appears
  # and every forward pass pays ~1.5s of CPU-stream-drain latency.
  if grep -q "Disabling NCCL for DP synchronization" "$attn_log" 2>/dev/null; then
    log "  ERROR: 'Disabling NCCL for DP synchronization' banner found in ATTN log."
    log "         The Gloo CPU DP sync fix in vllm/config/vllm.py is NOT active."
    log "         Check that the AFD connector override is in place."
    return 1
  else
    log "  OK: NCCL DP sync active (no Gloo CPU fallback)"
  fi

  return 0
}

run_config_phase() {
  # Runs phase A (clean) or phase B (profiled) for a single config.
  # Args: $1=afd_size  $2=attn_gpus  $3=attn_dp_flag
  #       $4=ffn_gpus  $5=ffn_tp_flag  $6=ffn_ep_flag
  #       $7=phase ("clean" or "profiled")
  local afd_size=$1 attn_gpus=$2 attn_dp_flag=$3
  local ffn_gpus=$4 ffn_tp_flag=$5 ffn_ep_flag=$6
  local phase=$7

  local dir=$OUT_ROOT/$afd_size
  mkdir -p "$dir"

  local attn_log=$dir/$phase-attn.log
  local ffn_log=$dir/$phase-ffn.log

  local profiler_attn="" profiler_ffn=""
  if [ "$phase" = "profiled" ]; then
    mkdir -p "$dir/traces/attn" "$dir/traces/ffn"
    profiler_attn="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$dir/traces/attn"
    profiler_ffn="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$dir/traces/ffn"
  fi

  log ""
  log "=== [$afd_size] PHASE: $phase ==="

  kill_vllm
  if ! launch_servers "$phase" "$attn_log" "$ffn_log" \
        "$attn_gpus" "$attn_dp_flag" \
        "$ffn_gpus" "$ffn_tp_flag" "$ffn_ep_flag" \
        "$profiler_attn" "$profiler_ffn" \
        "$afd_size"; then
    log "SKIP $afd_size/$phase due to server startup failure"
    kill_vllm
    return 1
  fi

  # Sanity check with curl (generation must be coherent)
  log "Sanity check..."
  curl -sS --max-time 60 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":16}' \
    >"$dir/$phase-curl.log" 2>&1 || log "  WARN: curl failed"
  log "  $(grep -o '\"text\":\"[^\"]*\"' "$dir/$phase-curl.log" | head -c 120)..."

  if [ "$phase" = "clean" ]; then
    log "Warmup run (discard)..."
    run_bench "$dir/$phase-bench-warmup.log" || true
    sleep $BENCH_SLEEP_BETWEEN

    for i in 1 2 3; do
      log "Clean run $i/3..."
      run_bench "$dir/$phase-bench-run$i.log" || true
      sleep $BENCH_SLEEP_BETWEEN
    done
  else
    log "Warmup run (discard)..."
    run_bench "$dir/$phase-bench-warmup.log" || true
    sleep $BENCH_SLEEP_BETWEEN

    log "Profiled run (TPOT not meaningful)..."
    run_bench "$dir/$phase-bench-profiled.log" || true
  fi

  log "Stopping servers for $afd_size/$phase..."
  kill_vllm
  sleep $CONFIG_SLEEP_BETWEEN
  return 0
}

# -------------------- Main sweep --------------------

log "AFD M×N sweep starting"
log "Output: $OUT_ROOT"
log "Port: $AFD_PORT"
log "Workload: NUM_PROMPTS=$NUM_PROMPTS INPUT_LEN=$INPUT_LEN OUTPUT_LEN=$OUTPUT_LEN REQ_RATE=$REQ_RATE"
log "DBO enabled: $ENABLE_DBO  (num_afd_stages=$AFD_NUM_STAGES)"
log "Pre-routing (Option A) enabled: $ENABLE_PREROUTING"
log ""

# Verify GPUs available
if ! nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not available"
  exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 4 ]; then
  log "ERROR: need at least 4 GPUs, found $GPU_COUNT"
  exit 1
fi

# Activate venv
if [ ! -f "$VENV_ACTIVATE" ]; then
  log "ERROR: venv not found at $VENV_ACTIVATE"
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# Trap to kill servers if the script is interrupted
trap 'log "Script interrupted, killing servers..."; kill_vllm; exit 130' INT TERM

# Record environment
{
  echo "timestamp: $STAMP"
  echo "host: $(hostname)"
  echo "gpus: $GPU_COUNT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "---"
  cd "$REPO" && git rev-parse HEAD && git log -1 --oneline
  echo ""
  echo "## NCCL DP sync fix check"
  if grep -q "and not (self.afd_config" "$REPO/vllm/config/vllm.py" 2>/dev/null; then
    echo "OK: AFD override for disable_nccl_for_dp_synchronization is present"
  else
    echo "WARNING: AFD override NOT found in vllm/config/vllm.py — 2A2F / 3A1F will be slow"
  fi
  echo ""
  echo "## FusedMoE modular kernel bypass check"
  if grep -q "forward_pre_routed" "$REPO/vllm/model_executor/layers/fused_moe/layer.py" 2>/dev/null; then
    echo "OK: forward_pre_routed method present in FusedMoE"
  else
    echo "WARNING: forward_pre_routed NOT found — pre-routing will run EP collectives"
  fi
  echo ""
  echo "## Gate + shared experts on ATTN check"
  if grep -q "DeepseekV2MoEAttentionStub" "$REPO/vllm/model_executor/models/deepseek_v2.py" 2>/dev/null; then
    echo "OK: DeepseekV2MoEAttentionStub present (ATTN-side router/shared)"
  else
    echo "WARNING: DeepseekV2MoEAttentionStub NOT found — router is on FFN side"
  fi
} >"$OUT_ROOT/_env.txt" 2>&1
cat "$OUT_ROOT/_env.txt"

# -------------------- Config matrix --------------------
#
# Layout (H200 node with GPUs 0-3):
#   1A2F: ATTN=GPU0                  | FFN TP=2 EP=2 = GPUs 1,2
#   2A1F: ATTN DP=2 = GPUs 0,1       | FFN=GPU 2
#   2A2F: ATTN DP=2 = GPUs 0,1       | FFN TP=2 EP=2 = GPUs 2,3
#   3A1F: ATTN DP=3 = GPUs 0,1,2     | FFN=GPU 3
#
# Order: 1A2F, 2A1F, 2A2F, 3A1F
# ------------------------------------------------------------------

declare -a CONFIGS=(
  # afd_size  attn_gpus  attn_dp_flag                 ffn_gpus  ffn_tp_flag                ffn_ep_flag
  "1A2F       0          --data-parallel-size=1       1,2       --tensor-parallel-size=2   --enable-expert-parallel"
  "2A1F       0,1        --data-parallel-size=2       2         --tensor-parallel-size=1   "
  "2A2F       0,1        --data-parallel-size=2       2,3       --tensor-parallel-size=2   --enable-expert-parallel"
  "3A1F       0,1,2      --data-parallel-size=3       3         --tensor-parallel-size=1   "
)

FAILED_CONFIGS=()

for cfg_line in "${CONFIGS[@]}"; do
  read -r afd_size attn_gpus attn_dp ffn_gpus ffn_tp ffn_ep <<<"$cfg_line"

  log ""
  log "################  CONFIG: $afd_size  ################"

  # Phase A: clean (no profiler)
  if ! run_config_phase "$afd_size" "$attn_gpus" "$attn_dp" \
                        "$ffn_gpus" "$ffn_tp" "$ffn_ep" "clean"; then
    FAILED_CONFIGS+=("$afd_size/clean")
  fi

  # Phase B: profiled
  if ! run_config_phase "$afd_size" "$attn_gpus" "$attn_dp" \
                        "$ffn_gpus" "$ffn_tp" "$ffn_ep" "profiled"; then
    FAILED_CONFIGS+=("$afd_size/profiled")
  fi
done

# -------------------- Summary --------------------

SUMMARY=$OUT_ROOT/SUMMARY.md
{
  echo "# AFD M×N Sweep Results"
  echo ""
  echo "**Timestamp**: $STAMP  "
  echo "**Host**: $(hostname)  "
  echo "**Commit**: $(cd "$REPO" && git rev-parse HEAD)"
  echo ""
  echo "## Clean benchmarks"
  echo ""
  echo "| Config | Run 1 TPOT (ms) | Run 2 TPOT (ms) | Run 3 TPOT (ms) | Run 1 TTFT (ms) | Run 2 TTFT (ms) | Run 3 TTFT (ms) |"
  echo "|--------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|"
  for cfg_line in "${CONFIGS[@]}"; do
    read -r afd_size _ <<<"$cfg_line"
    dir=$OUT_ROOT/$afd_size
    extract() {
      local f=$1 metric=$2
      [ -f "$f" ] && grep "Mean $metric" "$f" 2>/dev/null | awk '{print $NF}' || echo "-"
    }
    t1=$(extract "$dir/clean-bench-run1.log" TPOT)
    t2=$(extract "$dir/clean-bench-run2.log" TPOT)
    t3=$(extract "$dir/clean-bench-run3.log" TPOT)
    f1=$(extract "$dir/clean-bench-run1.log" TTFT)
    f2=$(extract "$dir/clean-bench-run2.log" TTFT)
    f3=$(extract "$dir/clean-bench-run3.log" TTFT)
    echo "| $afd_size | ${t1:--} | ${t2:--} | ${t3:--} | ${f1:--} | ${f2:--} | ${f3:--} |"
  done
  echo ""
  echo "## Profiled benchmarks (TPOT not meaningful — for trace inspection only)"
  echo ""
  echo "| Config | TPOT (ms) | TTFT (ms) | Trace dir |"
  echo "|--------|----------:|----------:|-----------|"
  for cfg_line in "${CONFIGS[@]}"; do
    read -r afd_size _ <<<"$cfg_line"
    dir=$OUT_ROOT/$afd_size
    tp=$(grep "Mean TPOT" "$dir/profiled-bench-profiled.log" 2>/dev/null | awk '{print $NF}' || echo "-")
    ft=$(grep "Mean TTFT" "$dir/profiled-bench-profiled.log" 2>/dev/null | awk '{print $NF}' || echo "-")
    echo "| $afd_size | ${tp:--} | ${ft:--} | \`$afd_size/traces/\` |"
  done
  echo ""
  if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo "## Failures"
    echo ""
    for f in "${FAILED_CONFIGS[@]}"; do echo "- $f"; done
  fi
} >"$SUMMARY"

log ""
log "=========================================="
log "DONE. Results: $OUT_ROOT"
log "Summary: $SUMMARY"
log "=========================================="
cat "$SUMMARY"
