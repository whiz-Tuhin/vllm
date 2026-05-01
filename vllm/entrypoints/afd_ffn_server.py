#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM AFD FFN Server Entry Point

This script provides a standalone entry point for running FFN servers in an AFD
(Attention-FFN Disaggregation) setup. FFN servers handle remote FFN computation
for attention workers.

Usage:
    python -m vllm.entrypoints.afd_ffn_server /path/to/model \
        --tensor-parallel-size 8 \
        --afd-config '{"afd_connector": "dummy", "afd_role": "ffn"}' \

Control HTTP port (optional):
    Pass ``--ffn-control-port N`` to start a small HTTP server on port N
    that exposes:
        POST /start_profile     -> kicks off the torch profiler on FFN workers
        POST /stop_profile      -> stops the profiler and flushes traces to disk
        GET  /health            -> 200 OK
    Useful for taking measurement-quality traces without restarting the FFN
    server: discard warmup runs, then POST /start_profile, run the measured
    bench, POST /stop_profile to flush, repeat for the next run.
"""

import signal
import threading
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


class AFDFFNServer:
    """AFD FFN Server main class."""

    def __init__(self, args: Any):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.vllm_config = engine_args.create_engine_config()
        # Optional HTTP control port for on-demand profiler start/stop.
        self.ffn_control_port: int | None = getattr(args, "ffn_control_port", None)
        self._control_server_thread: threading.Thread | None = None
        logger.info("Start AFD FFN Server with vllm_config: %s", self.vllm_config)

    def start(self) -> None:
        """Start the AFD FFN server."""
        try:
            # Import here to avoid circular imports
            from vllm.v1.executor.abstract import Executor

            # Create configurations
            executor_class = Executor.get_class(self.vllm_config)
            self.model_executor = executor_class(vllm_config=self.vllm_config)
            # Start the FFN server loop
            self._run_server_loop()

        except Exception as e:
            logger.error("Failed to start AFD FFN server: %s", e)
            raise

    def _start_profiler_rpc(self) -> tuple[bool, str]:
        """Idempotent profiler start. Safe to call multiple times.

        Returns ``(ok, message)`` for the HTTP control endpoint.
        """
        try:
            self.model_executor.collective_rpc(
                "profile", kwargs={"is_start": True}
            )
            logger.info("Profiler started on FFN workers (via control port)")
            return True, "profiler started"
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)
            return False, f"failed to start profiler: {e}"

    def _stop_profiler_rpc(self) -> tuple[bool, str]:
        """Idempotent profiler stop + trace flush. Safe to call multiple times."""
        try:
            self.model_executor.collective_rpc(
                "profile", kwargs={"is_start": False}
            )
            logger.info(
                "Profiler stopped on FFN workers, traces flushed (via control port)"
            )
            return True, "profiler stopped, traces flushed"
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
            return False, f"failed to stop profiler: {e}"

    def _start_control_server(self) -> None:
        """Start a background FastAPI/uvicorn server for on-demand profiler
        control. Called once at server startup if ``--ffn-control-port`` was set.

        Routes:
          - POST /start_profile -> kicks off torch profiler on all FFN workers
          - POST /stop_profile  -> stops profiler, flushes trace JSONs to disk
          - GET  /health        -> 200 OK liveness check

        The server runs in a daemon thread so it dies with the FFN process.
        It uses uvicorn directly (no app reload, no extra workers) — this is
        intentional: a single thread on a side port, kept simple.
        """
        if self.ffn_control_port is None:
            return

        # Local imports so the FFN server doesn't pull FastAPI when control
        # port is not set.
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI()

        @app.post("/start_profile")
        async def start_profile():
            ok, msg = self._start_profiler_rpc()
            return JSONResponse(
                status_code=200 if ok else 500,
                content={"ok": ok, "message": msg},
            )

        @app.post("/stop_profile")
        async def stop_profile():
            ok, msg = self._stop_profiler_rpc()
            return JSONResponse(
                status_code=200 if ok else 500,
                content={"ok": ok, "message": msg},
            )

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.ffn_control_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        def _serve():
            try:
                server.run()
            except Exception as e:
                logger.warning(
                    "FFN control server exited: %s", e
                )

        thread = threading.Thread(target=_serve, daemon=True, name="ffn-ctrl-http")
        thread.start()
        self._control_server_thread = thread
        logger.info(
            "FFN control HTTP server listening on port %d "
            "(POST /start_profile, POST /stop_profile)",
            self.ffn_control_port,
        )

    def _stop_profiler_and_workers(self) -> None:
        """Idempotent shutdown: stop profiler (flush traces) then stop workers.

        Safe to call multiple times — each collective_rpc is wrapped in a
        try/except so a partial failure doesn't prevent later steps from
        running. Trace flushing must come before the worker stop, otherwise
        the worker process exits before torch.profiler writes its JSON.
        """
        try:
            self.model_executor.collective_rpc(
                "profile", kwargs={"is_start": False}
            )
            logger.info("Profiler stopped on FFN workers, traces flushed")
        except (RuntimeError, Exception) as e:
            logger.debug("Profiler stop skipped: %s", e)

        try:
            self.model_executor.collective_rpc("stop_ffn_server_loop")
        except (RuntimeError, Exception) as e:
            logger.debug("Worker stop skipped: %s", e)

    def _run_server_loop(self) -> None:
        """Start FFN workers and wait for completion"""
        logger.info("AFD FFN Server started, workers running...")

        # Install SIGTERM/SIGINT handlers so the sweep script's pkill flushes
        # profiler traces before the FFN process exits. Without this, traces
        # are only flushed on KeyboardInterrupt — pkill -TERM/SIGTERM would
        # kill the process before the trace handler runs.
        shutdown_event = threading.Event()

        def _signal_handler(signum, _frame):
            logger.info(
                "AFD FFN Server received signal %d — initiating graceful shutdown",
                signum,
            )
            shutdown_event.set()

        signal.signal(signal.SIGTERM, _signal_handler)
        # SIGINT still raises KeyboardInterrupt by default in the wait()
        # below; but if we registered SIGTERM only, ctrl+c would skip the
        # flush. Register both for symmetry.
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            # Start the optional HTTP control server (no-op if --ffn-control-port
            # was not passed). Started before the auto-profile-start so that
            # /stop_profile is available immediately if the user wants to skip
            # the automatic startup window.
            self._start_control_server()

            # Start profiler on all FFN workers if profiler is configured.
            # When --ffn-control-port is in use, the user can still POST
            # /stop_profile to end this initial trace window early and POST
            # /start_profile later for measurement runs.
            try:
                self.model_executor.collective_rpc("profile", kwargs={"is_start": True})
                logger.info("Profiler started on FFN workers")
            except RuntimeError:
                logger.info("Profiling not enabled for FFN workers "
                            "(use --profiler-config to enable)")

            # Tell workers to start FFN server loops (one-time call)
            self.model_executor.collective_rpc("start_ffn_server_loop")

            # Main thread waits without busy polling
            shutdown_event.wait()  # Block until SIGINT or SIGTERM
            logger.info("Server shutting down — flushing profiler traces...")
            self._stop_profiler_and_workers()
        except KeyboardInterrupt:
            logger.info("Server interrupted (KeyboardInterrupt) — flushing...")
            self._stop_profiler_and_workers()
        except Exception as e:
            logger.error("Server error: %s — attempting to flush traces", e)
            try:
                self._stop_profiler_and_workers()
            except Exception:
                pass
            raise


def main(args: Any) -> None:
    """Main entry point for AFD FFN server."""
    try:
        server = AFDFFNServer(args)
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Server error: %s", e)
        raise


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add model as positional argument (like vllm serve)
    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument(
        "--ffn-control-port",
        type=int,
        default=None,
        help=(
            "Optional port for an HTTP control server that exposes "
            "POST /start_profile and POST /stop_profile, plus GET /health. "
            "Useful for taking measurement-quality torch traces without "
            "restarting the FFN server. Default: not started."
        ),
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Set the model from positional argument
    args.model = args.model

    main(args)
