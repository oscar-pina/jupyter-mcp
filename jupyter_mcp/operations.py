"""Operation tracking for async jobs.

Defines OperationRecord dataclass and OperationManager, which manages
asynchronous operations with bounded concurrency using a ThreadPoolExecutor.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

from jupyter_mcp import _utc_now, _new_id, _tool_error, _OPERATION_TTL_SECONDS, ConflictError


@dataclasses.dataclass
class OperationRecord:
    op_id: str
    kind: str
    status: str
    submitted_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[dict] = None
    cancelled: bool = False
    cancel_callback: Optional[Any] = None
    progress: Optional[dict] = None


class OperationManager:
    """Tracks and executes asynchronous operations with bounded concurrency."""

    def __init__(self, max_workers: int = 4, max_inflight: int = 32) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="op-worker")
        self._max_inflight = max_inflight
        self._ops: dict[str, OperationRecord] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

        self._reaper_stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="op-reaper")
        self._reaper.start()

    def _snapshot(self, op: OperationRecord) -> dict:
        out: dict = {
            "op_id": op.op_id,
            "kind": op.kind,
            "status": op.status,
            "submitted_at": op.submitted_at,
        }
        if op.started_at is not None:
            out["started_at"] = op.started_at
        if op.ended_at is not None:
            out["ended_at"] = op.ended_at
        if op.result is not None:
            out["result"] = op.result
        if op.error is not None:
            out["error"] = op.error
        if op.progress is not None:
            out["progress"] = op.progress
        if op.started_at and op.ended_at:
            out["timings"] = {"duration_ms": round((op.ended_at - op.started_at) * 1000, 2)}
        return out

    def _reap_loop(self) -> None:
        while not self._reaper_stop.is_set():
            time.sleep(30)
            cutoff = _utc_now() - _OPERATION_TTL_SECONDS
            with self._lock:
                stale = [
                    op_id
                    for op_id, op in self._ops.items()
                    if op.ended_at is not None and op.ended_at < cutoff
                ]
                for op_id in stale:
                    self._ops.pop(op_id, None)
                    self._futures.pop(op_id, None)

    def submit(self, kind: str, fn: Any, cancel_callback: Optional[Any] = None) -> dict:
        # All mutation (op creation, future storage, snapshot) happens inside a single lock
        # acquisition so get() can never observe a missing future for a known op_id.
        with self._lock:
            inflight = sum(1 for op in self._ops.values() if op.status in {"queued", "running"})
            if inflight >= self._max_inflight:
                return _tool_error("Busy", "Too many in-flight operations", {"max_inflight": self._max_inflight})

            op_id = _new_id("op")
            rec = OperationRecord(op_id=op_id, kind=kind, status="queued", submitted_at=_utc_now())
            rec.cancel_callback = cancel_callback
            self._ops[op_id] = rec

            # _runner closes over op_id. executor.submit enqueues it; _runner immediately
            # tries to re-acquire self._lock, so it cannot run until we release below.
            def _runner() -> None:
                with self._lock:
                    op = self._ops[op_id]
                    if op.cancelled:
                        op.status = "cancelled"
                        op.ended_at = _utc_now()
                        return
                    op.status = "running"
                    op.started_at = _utc_now()

                try:
                    result = fn(op_id)
                    with self._lock:
                        op = self._ops[op_id]
                        if op.cancelled:
                            op.status = "cancelled"
                        else:
                            op.status = "completed"
                            op.result = result
                        op.ended_at = _utc_now()
                except ConflictError as exc:
                    with self._lock:
                        op = self._ops[op_id]
                        op.status = "failed"
                        op.error = {"code": "Conflict", "message": str(exc)}
                        op.ended_at = _utc_now()
                except Exception as exc:
                    with self._lock:
                        op = self._ops[op_id]
                        op.status = "failed"
                        op.error = {"code": "ExecutionError", "message": str(exc)}
                        op.ended_at = _utc_now()

            fut = self._executor.submit(_runner)
            self._futures[op_id] = fut
            return self._snapshot(rec)

    def get(self, op_id: str, wait_ms: int = 0) -> dict:
        with self._lock:
            op = self._ops.get(op_id)
            fut = self._futures.get(op_id)
        if op is None:
            return _tool_error("NotFound", f"Operation {op_id!r} not found")

        if wait_ms > 0 and fut is not None and not fut.done():
            try:
                fut.result(timeout=wait_ms / 1000)
            except Exception:
                # _runner records status/error into self._ops regardless of exceptions;
                # the snapshot below is authoritative. Also handles concurrent.futures.TimeoutError
                # on Python <3.11 where it's not a subclass of the builtin TimeoutError.
                pass

        with self._lock:
            return self._snapshot(self._ops[op_id])

    def cancel(self, op_id: str) -> dict:
        with self._lock:
            op = self._ops.get(op_id)
            fut = self._futures.get(op_id)
        if op is None:
            return _tool_error("NotFound", f"Operation {op_id!r} not found")

        with self._lock:
            op.cancelled = True

        if op.cancel_callback:
            try:
                op.cancel_callback()
            except Exception:
                pass

        if fut and fut.cancel():
            with self._lock:
                op.status = "cancelled"
                op.ended_at = _utc_now()

        with self._lock:
            return self._snapshot(op)

    def is_cancelled(self, op_id: str) -> bool:
        """Return True if the operation has been marked for cancellation."""
        with self._lock:
            rec = self._ops.get(op_id)
            return rec is not None and rec.cancelled

    def update_progress(self, op_id: str, progress: dict) -> None:
        """Update the progress field of a running operation."""
        with self._lock:
            rec = self._ops.get(op_id)
            if rec is not None:
                rec.progress = progress

    def shutdown(self) -> None:
        self._reaper_stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
