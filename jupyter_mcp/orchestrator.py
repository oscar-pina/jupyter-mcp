"""Execution orchestrator for coordinating kernels and notebooks.

Defines ExecutionOrchestrator, which coordinates between KernelProvider,
NotebookStore, and OperationManager to run code and full notebooks.
"""

from __future__ import annotations

from typing import Callable, Optional

from jupyter_mcp import _parse_iopub_messages
from jupyter_mcp.kernel import KernelProvider, SessionRecord
from jupyter_mcp.notebooks import NotebookStore


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
        runtime_or_session: str,
        mode: str,
        cell_selector: Optional[str],
        timeout_s: int,
        stop_on_error: bool,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        session_id: Optional[str] = None
        created_temp = False

        sessions = {s.session_id: s for s in self.provider.list_sessions()}
        if mode == "session":
            if runtime_or_session not in sessions:
                raise KeyError("mode='session' requires an existing session_id")
            session_id = runtime_or_session
        elif mode == "fresh":
            runtime = runtime_or_session
            if runtime_or_session in sessions:
                runtime = sessions[runtime_or_session].runtime
            rec = self.provider.create_session(runtime=runtime, isolation="ephemeral")
            session_id = rec.session_id
            created_temp = True
        else:
            raise ValueError("mode must be 'fresh' or 'session'")

        assert session_id is not None

        try:
            read_data = self.notebooks.read(path=path, cell_range=None, include_outputs=False, output_limit=20000)
            revision = read_data["revision"]
            cells = read_data["cells"]
            selected = self._select_cells(cells, cell_selector)

            total_code = sum(
                1 for idx in selected
                if cells[idx]["cell_type"] == "code" and str(cells[idx].get("source", "")).strip()
            )
            results: list[dict] = []
            executed_count = 0

            for idx in selected:
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

    def _select_cells(self, cells: list[dict], selector: Optional[str]) -> list[int]:
        """Parse a cell_selector string into a sorted list of cell indices.

        Supported syntax (combinable with commas):
          ``all``     — all cells (default when None or "all")
          ``5``       — single cell at index 5
          ``5:9``     — slice from index 5 up to (excluding) 9
          ``1,3,5:9`` — indices 1, 3, and 5 through 8
        """
        if not selector or selector == "all":
            return list(range(len(cells)))

        chosen: set[int] = set()
        parts = [p.strip() for p in selector.split(",") if p.strip()]
        for p in parts:
            if ":" in p:
                start_raw, end_raw = p.split(":", 1)
                start = int(start_raw) if start_raw else 0
                end = int(end_raw) if end_raw else len(cells)
                for i in range(start, end):
                    if 0 <= i < len(cells):
                        chosen.add(i)
            else:
                idx = int(p)
                if 0 <= idx < len(cells):
                    chosen.add(idx)
        return sorted(chosen)
