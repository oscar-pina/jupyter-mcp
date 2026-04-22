"""Execution orchestrator for coordinating kernels and notebooks.

Defines ExecutionOrchestrator, which coordinates between KernelProvider,
NotebookStore, and OperationManager to run code and full notebooks.
"""

from __future__ import annotations

from typing import Callable, Optional

from jupyter_mcp import _parse_iopub_messages
from jupyter_mcp.kernel import KernelProvider
from jupyter_mcp.notebooks import NotebookStore, parse_cell_selector


class ExecutionOrchestrator:
    """Coordinates session execution, notebook I/O, and operation boundaries."""

    def __init__(self, provider: KernelProvider, notebooks: NotebookStore) -> None:
        self.provider = provider
        self.notebooks = notebooks

    def run_code(self, session_id: str, code: str, timeout_s: int, on_timeout: str, capture: str, include_images: bool = False) -> dict:
        result = self.provider.execute(session_id=session_id, code=code, timeout_s=timeout_s, on_timeout=on_timeout)
        raw_messages = result.pop("_raw_messages", None)
        if include_images and raw_messages is not None:
            parsed = _parse_iopub_messages(raw_messages, include_images=True)
            parsed["execution_count"] = result.get("execution_count")
            parsed["truncated"] = result.get("truncated", False)
            result = parsed
        if capture == "summary":
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "error": result.get("error"),
                "execution_count": result.get("execution_count"),
                "truncated": result.get("truncated", False),
                "rich_outputs": result.get("rich_outputs", [])[:10],
            }
        return result

    def run_notebook(
        self,
        path: str,
        mode: str,
        python_path: str = "python",
        target_session_id: Optional[str] = None,
        cell_selector: Optional[str] = None,
        timeout_s: int = 300,
        stop_on_error: bool = True,
        on_progress: Optional[Callable[[dict], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        on_session_ready: Optional[Callable[[str], None]] = None,
    ) -> dict:
        session_id: Optional[str] = None
        created_temp = False

        if mode == "session":
            if target_session_id is None:
                raise ValueError("mode='session' requires session_id")
            sessions = {s.session_id: s for s in self.provider.list_sessions()}
            if target_session_id not in sessions:
                raise KeyError(f"Session {target_session_id!r} not found")
            session_id = target_session_id
        elif mode == "fresh":
            rec = self.provider.create_session(python_path=python_path, isolation="ephemeral")
            session_id = rec.session_id
            created_temp = True
        else:
            raise ValueError("mode must be 'fresh' or 'session'")

        assert session_id is not None

        if on_session_ready is not None:
            on_session_ready(session_id)

        try:
            read_data = self.notebooks.read(path=path, cell_range=None, include_outputs=False, output_limit=20000)
            revision = read_data["revision"]
            cells = read_data["cells"]
            start, end = parse_cell_selector(cell_selector, len(cells))
            selected = list(range(start, end))

            total_code = sum(
                1 for idx in selected
                if cells[idx]["cell_type"] == "code" and str(cells[idx].get("source", "")).strip()
            )
            results: list[dict] = []
            executed_count = 0

            for idx in selected:
                if is_cancelled is not None and is_cancelled():
                    return {
                        "status": "cancelled",
                        "cells_executed": executed_count,
                        "results": results,
                        "revision": revision,
                    }

                cell = cells[idx]
                if cell["cell_type"] != "code" or not str(cell.get("source", "")).strip():
                    results.append({"index": idx, "skipped": True, "reason": "non-code or empty"})
                    continue

                run = self.provider.execute(
                    session_id=session_id,
                    code=cell["source"],
                    timeout_s=timeout_s,
                    on_timeout="interrupt",
                )
                executed_count += 1
                err = run.get("error")

                revision_write = self.notebooks.write_execution_cell(
                    path=path,
                    expected_revision=revision,
                    cell_index=idx,
                    expected_source=cell["source"],
                    execution_count=run.get("execution_count"),
                    raw_messages=run.get("_raw_messages", []),
                )
                revision = revision_write["revision"]

                if on_progress is not None:
                    try:
                        on_progress({"cells_completed": executed_count, "cells_total": total_code, "last_cell_index": idx})
                    except Exception:
                        pass

                results.append(
                    {
                        "index": idx,
                        "execution_count": run.get("execution_count"),
                        "has_error": err is not None,
                        "stdout_preview": (run.get("stdout") or "")[:300],
                    }
                )

                if err and stop_on_error:
                    return {
                        "status": "error",
                        "stopped_at_cell": idx,
                        "cells_executed": executed_count,
                        "error": err,
                        "results": results,
                        "revision": revision,
                    }

            return {
                "status": "completed",
                "cells_executed": executed_count,
                "results": results,
                "revision": revision,
            }
        finally:
            if created_temp:
                try:
                    self.provider.close_session(session_id, force=True)
                except Exception:
                    pass

