"""Kernel provider abstraction and local implementation.

Defines the KernelProvider ABC and LocalKernelProvider, which manages
local process-backed Jupyter kernels via jupyter_client.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

from jupyter_mcp import _utc_now, _new_id, _parse_iopub_messages

_SESSION_IDLE_TIMEOUT = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SessionRecord:
    session_id: str
    runtime: str
    isolation: str
    cwd: str
    created_at: float
    last_used_at: float
    state: str = "ready"
    active_op_id: Optional[str] = None


@dataclasses.dataclass
class _LocalSession:
    record: SessionRecord
    km: KernelManager
    kc: Any
    exec_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class KernelProvider(ABC):
    """Backend abstraction for kernel lifecycle and execution."""

    @abstractmethod
    def list_runtimes(self) -> list[dict]:
        pass

    @abstractmethod
    def create_session(
        self,
        runtime: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        isolation: str = "ephemeral",
        python_path: Optional[str] = None,
    ) -> SessionRecord:
        pass

    @abstractmethod
    def list_sessions(self) -> list[SessionRecord]:
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> SessionRecord:
        pass

    @abstractmethod
    def close_session(self, session_id: str, force: bool = False) -> None:
        pass

    @abstractmethod
    def restart(self, session_id: str) -> None:
        pass

    @abstractmethod
    def interrupt(self, session_id: str) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        session_id: str,
        code: str,
        timeout_s: int,
        on_timeout: str,
    ) -> dict:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------


class LocalKernelProvider(KernelProvider):
    """Local process-backed KernelProvider using jupyter_client."""

    def __init__(self, idle_timeout: int = _SESSION_IDLE_TIMEOUT) -> None:
        self._sessions: dict[str, _LocalSession] = {}
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout
        self._reaper_stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_idle_sessions, daemon=True, name="session-reaper"
        )
        self._reaper.start()

    def _reap_idle_sessions(self) -> None:
        """Background daemon: close ephemeral sessions idle longer than _idle_timeout."""
        while not self._reaper_stop.wait(timeout=60):
            cutoff = _utc_now() - self._idle_timeout
            with self._lock:
                stale = [
                    sid
                    for sid, entry in self._sessions.items()
                    if entry.record.isolation == "ephemeral" and entry.record.last_used_at < cutoff
                ]
            for sid in stale:
                try:
                    self.close_session(sid, force=True)
                except Exception:
                    pass

    def _get_session_entry(self, session_id: str) -> _LocalSession:
        with self._lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError(f"Session {session_id!r} not found")
        if not entry.km.is_alive():
            entry.record.state = "dead"
            raise RuntimeError(f"Session {session_id!r} kernel is not alive")
        return entry

    def list_runtimes(self) -> list[dict]:
        ksm = KernelSpecManager()
        specs = ksm.get_all_specs()
        out = []
        for name, spec_data in sorted(specs.items()):
            spec = spec_data.get("spec", {})
            out.append(
                {
                    "runtime": name,
                    "display_name": spec.get("display_name", name),
                    "language": spec.get("language", "unknown"),
                }
            )
        return out

    def create_session(
        self,
        runtime: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        isolation: str = "ephemeral",
        python_path: Optional[str] = None,
    ) -> SessionRecord:
        resolved_cwd = str(Path(cwd).expanduser().resolve()) if cwd else str(Path.cwd())
        if python_path:
            from jupyter_client.kernelspec import KernelSpec
            # expanduser but do NOT resolve() — resolving follows symlinks and
            # would replace a virtualenv's `python` symlink with the base
            # interpreter, losing the virtualenv's site-packages.
            py_path = Path(python_path).expanduser()
            if not py_path.exists():
                raise ValueError(f"python_path {python_path!r} does not exist")
            if not py_path.is_file():
                raise ValueError(f"python_path {python_path!r} is not a file")
            spec = KernelSpec(
                argv=[str(py_path), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name=f"Python ({py_path.parent.parent.name})",
                language="python",
            )
            km = KernelManager()
            km._kernel_spec = spec
        else:
            km = KernelManager(kernel_name=runtime)
        kw: dict = {"cwd": resolved_cwd}
        if env is not None:
            kw["env"] = env
        km.start_kernel(**kw)
        try:
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=30)
        except Exception:
            km.shutdown_kernel(now=True)
            raise

        sid = _new_id("sess")
        now = _utc_now()
        rec = SessionRecord(
            session_id=sid,
            runtime=runtime,
            isolation=isolation,
            cwd=resolved_cwd,
            created_at=now,
            last_used_at=now,
        )
        entry = _LocalSession(record=rec, km=km, kc=kc)
        with self._lock:
            self._sessions[sid] = entry
        return rec

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock:
            return [dataclasses.replace(entry.record) for entry in self._sessions.values()]

    def get_session(self, session_id: str) -> SessionRecord:
        entry = self._get_session_entry(session_id)
        return dataclasses.replace(entry.record)

    def close_session(self, session_id: str, force: bool = False) -> None:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None:
            raise KeyError(f"Session {session_id!r} not found")
        try:
            entry.kc.stop_channels()
        except Exception:
            pass
        try:
            entry.km.shutdown_kernel(now=force)
        except Exception:
            pass

    def restart(self, session_id: str) -> None:
        entry = self._get_session_entry(session_id)
        try:
            entry.kc.stop_channels()
        except Exception:
            pass
        entry.km.restart_kernel(now=True)
        kc = entry.km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=30)
        with self._lock:
            entry.kc = kc
            entry.record.last_used_at = _utc_now()
            entry.record.state = "ready"

    def interrupt(self, session_id: str) -> None:
        entry = self._get_session_entry(session_id)
        entry.km.interrupt_kernel()

    def execute(self, session_id: str, code: str, timeout_s: int, on_timeout: str) -> dict:
        entry = self._get_session_entry(session_id)

        with entry.exec_lock:
            entry.record.last_used_at = _utc_now()
            msg_id = entry.kc.execute(code)
            messages: list[dict] = []
            deadline = _utc_now() + timeout_s

            while True:
                remaining = deadline - _utc_now()
                if remaining <= 0:
                    if on_timeout == "interrupt":
                        try:
                            entry.km.interrupt_kernel()
                        except Exception:
                            pass
                    return {
                        "stdout": "",
                        "stderr": "",
                        "rich_outputs": [],
                        "error": {
                            "ename": "TimeoutError",
                            "evalue": f"Execution exceeded {timeout_s}s",
                            "traceback": "",
                        },
                        "truncated": False,
                        "execution_count": None,
                    }

                try:
                    msg = entry.kc.get_iopub_msg(timeout=min(remaining, 5.0))
                except queue.Empty:
                    continue

                # Best-effort auto-unblock stdin request.
                try:
                    stdin_msg = entry.kc.get_stdin_msg(timeout=0)
                    if stdin_msg.get("msg_type") == "input_request":
                        entry.kc.input("")
                        messages.append(
                            {
                                "msg_type": "stream",
                                "parent_header": {"msg_id": msg_id},
                                "content": {
                                    "name": "stderr",
                                    "text": "[Warning: input() is unsupported. Empty string sent.]\n",
                                },
                            }
                        )
                except queue.Empty:
                    pass

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                messages.append(msg)
                if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                    break

            shell_reply: Optional[dict] = None
            shell_deadline = _utc_now() + 10
            while _utc_now() < shell_deadline:
                try:
                    shell_msg = entry.kc.get_shell_msg(timeout=min(1.0, shell_deadline - _utc_now()))
                except queue.Empty:
                    continue
                if shell_msg.get("parent_header", {}).get("msg_id") == msg_id:
                    shell_reply = shell_msg
                    break

            parsed = _parse_iopub_messages(messages)
            parsed["execution_count"] = None
            if shell_reply:
                parsed["execution_count"] = shell_reply.get("content", {}).get("execution_count")
            # Internal field used by notebook execution to persist full-fidelity outputs.
            parsed["_raw_messages"] = messages
            return parsed

    def shutdown(self) -> None:
        self._reaper_stop.set()
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                self.close_session(sid, force=True)
            except Exception:
                pass
