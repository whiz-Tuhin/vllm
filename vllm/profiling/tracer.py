"""
Request-level latency tracer for vLLM profiling.
Runs in the API server process only (not EngineCore subprocess).
"""
import json
import os
import time
import threading
from typing import Dict, Optional

_tracer: Optional["RequestTracer"] = None
_tracer_lock = threading.Lock()


class RequestTracer:
    def __init__(self, model: str, output_dir: str):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        safe_model = model.replace("/", "_")
        self.trace_path = os.path.join(output_dir, f"{safe_model}.jsonl")
        self.token_timeline_path = os.path.join(
            output_dir, f"{safe_model}_per-token-timeline.jsonl")
        self._records: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._file = open(self.trace_path, "a")
        self._token_file = open(self.token_timeline_path, "a")
        print(f"[profiling] Tracer initialized: {self.trace_path}", flush=True)
        print(f"[profiling] Token timeline: {self.token_timeline_path}", flush=True)

    def record_api_receive(
        self,
        request_id: str,
        prompt_id: str = "",
        max_tokens: int = 0,
        prompt_tokens: int = 0,
    ):
        with self._lock:
            self._records[request_id] = {
                "request_id": request_id,
                "model": self.model,
                "prompt_id": prompt_id,
                "max_tokens": max_tokens,
                "prompt_tokens": prompt_tokens,
                "output_tokens": 0,
                "api_receive_ts": time.time(),
                "engine_add_request_ts": 0.0,
                "first_token_ts": 0.0,
                "completion_ts": 0.0,
                "token_timestamps": [],  # internal only — written to separate file
            }

    def record_engine_add_request(self, request_id: str):
        with self._lock:
            rec = self._records.get(request_id)
            if rec:
                rec["engine_add_request_ts"] = time.time()

    def record_first_token(self, request_id: str):
        with self._lock:
            rec = self._records.get(request_id)
            if rec and rec["first_token_ts"] == 0.0:
                rec["first_token_ts"] = time.time()

    def record_token(self, request_id: str):
        """Record the delivery timestamp of each streamed token (one call per yield)."""
        ts = time.time()
        with self._lock:
            rec = self._records.get(request_id)
            if rec is not None:
                rec["token_timestamps"].append(ts)

    def record_completion(self, request_id: str, output_tokens: int, prompt_tokens: int):
        with self._lock:
            rec = self._records.pop(request_id, None)
        if rec is None:
            return
        rec["completion_ts"] = time.time()
        rec["output_tokens"] = output_tokens
        if prompt_tokens > 0:
            rec["prompt_tokens"] = prompt_tokens

        # Compute derived metrics (ms)
        api_ts = rec["api_receive_ts"]
        eng_ts = rec["engine_add_request_ts"]
        first_ts = rec["first_token_ts"]
        done_ts = rec["completion_ts"]

        rec["ttft_ms"] = (first_ts - api_ts) * 1000 if first_ts > 0 else 0.0
        rec["tpot_ms"] = (
            (done_ts - first_ts) * 1000 / max(output_tokens - 1, 1)
            if output_tokens > 1 and first_ts > 0
            else 0.0
        )
        rec["total_latency_ms"] = (done_ts - api_ts) * 1000
        rec["scheduling_overhead_ms"] = (
            (eng_ts - api_ts) * 1000 if eng_ts > 0 else 0.0
        )

        # Write per-token timeline to separate file (one line per token)
        token_timestamps = rec.pop("token_timestamps", [])
        if token_timestamps and first_ts > 0:
            token_lines = []
            prev_ts = None
            for idx, ts in enumerate(token_timestamps):
                itl_ms = (ts - prev_ts) * 1000 if prev_ts is not None else 0.0
                token_lines.append(json.dumps({
                    "request_id": request_id,
                    "prompt_id": rec["prompt_id"],
                    "token_idx": idx,
                    "timestamp": ts,
                    "rel_ts_ms": (ts - first_ts) * 1000,
                    "itl_ms": itl_ms,
                }))
                prev_ts = ts
            with self._lock:
                self._token_file.write("\n".join(token_lines) + "\n")
                self._token_file.flush()

        # Write main request trace (no token_timestamps)
        line = json.dumps(rec)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()
            self._token_file.close()


def init_tracer(model: str, output_dir: str) -> "RequestTracer":
    global _tracer
    with _tracer_lock:
        if _tracer is None:
            _tracer = RequestTracer(model, output_dir)
    return _tracer


def get_tracer() -> Optional["RequestTracer"]:
    return _tracer
