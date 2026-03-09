"""
Forward-pass logger for vLLM profiling.
Runs in the EngineCore subprocess (GPU worker process).

Activated by env vars:
  VLLM_PROF_MODEL      — model name (used in output filename)
  VLLM_PROF_OUTPUT_DIR — directory to write forward pass JSONL

Each line written:
  {fwd_id, start_ts, end_ts, duration_ms, req_ids, num_tokens, total_tokens}
"""
import json
import os
import threading
from typing import Dict, List, Optional

_fp_logger: Optional["ForwardPassLogger"] = None
_fp_logger_lock = threading.Lock()


class ForwardPassLogger:
    def __init__(self, model: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        safe_model = model.replace("/", "_")
        self.path = os.path.join(output_dir, f"{safe_model}_fwd.jsonl")
        self._file = open(self.path, "a")
        self._lock = threading.Lock()
        self._counter = 0
        print(f"[profiling] ForwardPassLogger initialized: {self.path}", flush=True)

    def record(
        self,
        start_ts: float,
        end_ts: float,
        duration_ms: float,
        req_ids: List[str],
        num_tokens: Dict[str, int],
    ):
        with self._lock:
            fwd_id = self._counter
            self._counter += 1
            entry = {
                "fwd_id": fwd_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": duration_ms,
                "req_ids": req_ids,
                "num_tokens": num_tokens,
                "total_tokens": sum(num_tokens.values()),
            }
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()


def get_fp_logger() -> Optional["ForwardPassLogger"]:
    global _fp_logger
    if _fp_logger is not None:
        return _fp_logger
    # Lazy init from env vars (EngineCore subprocess inherits parent env)
    model = os.environ.get("VLLM_PROF_MODEL", "")
    output_dir = os.environ.get("VLLM_PROF_OUTPUT_DIR", "")
    if not model or not output_dir:
        return None
    with _fp_logger_lock:
        if _fp_logger is None:
            _fp_logger = ForwardPassLogger(model, output_dir)
    return _fp_logger
