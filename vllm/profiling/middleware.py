"""
ASGI middleware for request-level profiling.

Injected via vLLM's --middleware flag:
  --middleware vllm.profiling.middleware.ProfilingMiddleware

Reads VLLM_PROF_MODEL and VLLM_PROF_OUTPUT_DIR env vars.
Sets X-Request-Id header so vLLM uses our ID in request_id,
then records api_receive_ts in the tracer.
"""
import json
import os
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from vllm.profiling.tracer import get_tracer, init_tracer


class ProfilingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        model = os.environ.get("VLLM_PROF_MODEL", "unknown")
        output_dir = os.environ.get("VLLM_PROF_OUTPUT_DIR", "traces")
        init_tracer(model, output_dir)

    async def dispatch(self, request: Request, call_next):
        tracer = get_tracer()
        if tracer is None or request.method != "POST":
            return await call_next(request)

        path = request.url.path
        is_chat = "/chat/completions" in path
        is_completion = not is_chat and "/completions" in path
        if not (is_chat or is_completion):
            return await call_next(request)

        # Read body (Starlette caches it after first read)
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except Exception:
            return await call_next(request)

        # Determine our base ID: use body request_id if present, else uuid
        base_id = body.get("request_id") or uuid.uuid4().hex[:12]
        prompt_id = body.get("user", "")
        max_tokens = body.get("max_tokens", 0)

        # vLLM 0.7.3 request_id format (no random suffix added):
        #   /v1/completions:      cmpl-{X-Request-Id}
        #   /v1/chat/completions: chatcmpl-{X-Request-Id || body.request_id}
        prefix = "chatcmpl" if is_chat else "cmpl"
        tracer_key = f"{prefix}-{base_id}"

        tracer.record_api_receive(tracer_key, prompt_id=prompt_id, max_tokens=max_tokens)

        # Inject X-Request-Id so vLLM uses our base_id
        # Starlette headers are immutable, so we rebuild the scope headers
        headers = dict(request.headers)
        headers["x-request-id"] = base_id
        new_headers = [
            (k.lower().encode(), v.encode()) for k, v in headers.items()
        ]
        request.scope["headers"] = new_headers

        return await call_next(request)
