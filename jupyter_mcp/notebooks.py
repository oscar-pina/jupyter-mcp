"""Notebook store abstraction and filesystem implementation.

Defines the NotebookStore ABC and FileNotebookStore, which manages
.ipynb files with per-path locking and SHA-256 revision control.
"""

from __future__ import annotations

import hashlib
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import nbformat
from jupyter_client.kernelspec import KernelSpecManager

from jupyter_mcp import (
    ConflictError,
    _strip_ansi,
    _truncate_text,
    _format_display_dict,
    _build_nbformat_outputs,
)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class NotebookStore(ABC):
    """Notebook persistence abstraction for extensibility."""

    @abstractmethod
    def list_notebooks(self, directory: str) -> dict:
        pass

    @abstractmethod
    def create_notebook(self, path: str, kernel_name: str) -> dict:
        pass

    @abstractmethod
    def delete_notebook(self, path: str, expected_revision: str) -> dict:
        pass

    @abstractmethod
    def read(
        self,
        path: str,
        cell_range: Optional[str],
        include_outputs: bool,
        output_limit: int,
        include_images: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def insert_cell(
        self,
        path: str,
        base_revision: str,
        index: int,
        cell_type: str,
        source: str,
    ) -> dict:
        pass

    @abstractmethod
    def update_cell(
        self,
        path: str,
        base_revision: str,
        cell_index: int,
        source: str,
        reset_outputs: bool,
    ) -> dict:
        pass

    @abstractmethod
    def delete_cell(self, path: str, base_revision: str, cell_index: int) -> dict:
        pass

    @abstractmethod
    def move_cell(self, path: str, base_revision: str, from_index: int, to_index: int) -> dict:
        pass

    @abstractmethod
    def clear_outputs(self, path: str, base_revision: str, cell_index: Optional[int] = None) -> dict:
        """Clear outputs for one cell (cell_index given) or all code cells (cell_index=None)."""
        pass

    @abstractmethod
    def batch_cells(self, path: str, base_revision: str, operations: list[dict]) -> dict:
        """Apply multiple insert/update/delete operations atomically under one lock."""
        pass

    @abstractmethod
    def write_execution_cell(
        self,
        path: str,
        expected_revision: str,
        cell_index: int,
        expected_source: str,
        execution_count: Optional[int],
        raw_messages: list[dict],
    ) -> dict:
        pass


# ---------------------------------------------------------------------------
# Filesystem implementation
# ---------------------------------------------------------------------------


class FileNotebookStore(NotebookStore):
    """Filesystem notebook store with per-path locking and revision control."""

    _MAX_LOCKS = 256

    def __init__(self, allowed_roots: Optional[list[str]] = None) -> None:
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._locks_guard = threading.Lock()
        self._allowed_roots: Optional[list[Path]] = (
            [Path(r).expanduser().resolve() for r in allowed_roots] if allowed_roots else None
        )

    def _lock_for(self, path: Path) -> threading.Lock:
        key = str(path.resolve())
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                if len(self._locks) >= self._MAX_LOCKS:
                    # Only evict LRU if it is not currently held;
                    # otherwise allow temporary over-capacity.
                    _, lru_lock = next(iter(self._locks.items()))
                    if not lru_lock.locked():
                        self._locks.popitem(last=False)
                self._locks[key] = lock
            else:
                self._locks.move_to_end(key)
            return lock

    def _check_allowed(self, path: Path) -> None:
        if self._allowed_roots is None:
            return
        resolved = path.resolve()
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root)
                return
            except ValueError:
                continue
        raise PermissionError(f"Path {path!r} is outside allowed directories")

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        self._check_allowed(p)
        if not p.exists():
            raise FileNotFoundError(f"Notebook {path!r} not found")
        return p

    def _read_nb(self, path: Path) -> Any:
        return nbformat.read(str(path), as_version=4)

    def _write_atomic(self, nb: Any, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            nbformat.write(nb, str(tmp))
            os.replace(str(tmp), str(path))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _revision(self, path: Path) -> str:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _check_revision(self, path: Path, expected_revision: str) -> str:
        current_revision = self._revision(path)
        if expected_revision != current_revision:
            raise ConflictError(
                f"Revision conflict: expected {expected_revision}, current {current_revision}"
            )
        return current_revision

    def _new_cell(self, cell_type: str, source: str) -> Any:
        if cell_type == "code":
            return nbformat.v4.new_code_cell(source=source)
        if cell_type == "markdown":
            return nbformat.v4.new_markdown_cell(source=source)
        if cell_type == "raw":
            return nbformat.v4.new_raw_cell(source=source)
        raise ValueError(f"Unsupported cell_type {cell_type!r}")

    def _parse_cell_range(self, cell_range: Optional[str], count: int) -> tuple[int, int]:
        if not cell_range:
            return 0, count
        if ":" not in cell_range:
            raise ValueError("cell_range must be 'start:end'")
        start_raw, end_raw = cell_range.split(":", 1)
        start = int(start_raw) if start_raw else 0
        end = int(end_raw) if end_raw else count
        start = max(0, start)
        end = min(count, end)
        if start > end:
            raise ValueError("Invalid cell_range: start > end")
        return start, end

    def _format_saved_outputs(self, outputs: list, output_limit: int, include_images: bool = False) -> list:
        formatted: list[dict] = []
        for out in outputs:
            out_type = out.get("output_type", "")
            if out_type == "stream":
                text, _ = _truncate_text(out.get("text", ""), output_limit)
                formatted.append({"type": "stream", "name": out.get("name"), "text": text})
            elif out_type in ("execute_result", "display_data"):
                formatted.append({"type": out_type, **_format_display_dict(out.get("data", {}), include_images=include_images)})
            elif out_type == "error":
                formatted.append(
                    {
                        "type": "error",
                        "ename": out.get("ename", ""),
                        "evalue": out.get("evalue", ""),
                        "traceback": "\n".join(_strip_ansi(line) for line in out.get("traceback", [])),
                    }
                )
        return formatted

    def list_notebooks(self, directory: str) -> dict:
        path = Path(directory).expanduser().resolve()
        self._check_allowed(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory {directory!r} not found")
        if not path.is_dir():
            raise ValueError(f"{directory!r} is not a directory")

        notebooks = []
        for nb in sorted(path.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in nb.parts:
                continue
            notebooks.append(
                {
                    "path": str(nb),
                    "name": nb.name,
                    "size_kb": round(nb.stat().st_size / 1024, 1),
                }
            )
        return {"directory": str(path), "notebooks": notebooks}

    def create_notebook(self, path: str, kernel_name: str) -> dict:
        nb_path = Path(path).expanduser().resolve()
        self._check_allowed(nb_path)
        if nb_path.exists():
            raise FileExistsError(f"Notebook {path!r} already exists")
        nb_path.parent.mkdir(parents=True, exist_ok=True)

        language = "python"
        try:
            spec = KernelSpecManager().get_kernel_spec(kernel_name)
            language = spec.language
        except Exception:
            pass

        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": kernel_name,
            "language": language,
            "name": kernel_name,
        }
        nb.metadata["language_info"] = {"name": language, "version": ""}

        with self._lock_for(nb_path):
            self._write_atomic(nb, nb_path)
            revision = self._revision(nb_path)

        return {
            "status": "created",
            "path": str(nb_path),
            "kernel_name": kernel_name,
            "cell_count": len(nb.cells),
            "revision": revision,
        }

    def delete_notebook(self, path: str, expected_revision: str) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, expected_revision)
            nb_path.unlink()
        return {"status": "deleted", "path": str(nb_path), "base_revision": expected_revision}

    def read(self, path: str, cell_range: Optional[str], include_outputs: bool, output_limit: int, include_images: bool = False) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            nb = self._read_nb(nb_path)
            revision = self._revision(nb_path)

        start, end = self._parse_cell_range(cell_range, len(nb.cells))
        cells = []
        for i in range(start, end):
            cell = nb.cells[i]
            item: dict[str, Any] = {
                "index": i,
                "cell_type": cell.cell_type,
                "source": cell.source,
            }
            if cell.cell_type == "code":
                item["execution_count"] = cell.get("execution_count")
                if include_outputs:
                    item["outputs"] = self._format_saved_outputs(cell.get("outputs", []), output_limit, include_images=include_images)
            cells.append(item)

        return {
            "path": str(nb_path),
            "revision": revision,
            "cell_count": len(nb.cells),
            "range": {"start": start, "end": end},
            "kernel_name": nb.metadata.get("kernelspec", {}).get("name", "unknown"),
            "language": nb.metadata.get("kernelspec", {}).get("language", "unknown"),
            "cells": cells,
        }

    def insert_cell(
        self, path: str, base_revision: str, index: int, cell_type: str, source: str
    ) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            if not (0 <= index <= len(nb.cells)):
                raise IndexError(f"index {index} out of range")
            nb.cells.insert(index, self._new_cell(cell_type, source))
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "inserted",
            "base_revision": base_revision,
            "revision": new_revision,
            "index": index,
            "cell_count": len(nb.cells),
        }

    def update_cell(
        self, path: str, base_revision: str, cell_index: int, source: str, reset_outputs: bool
    ) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            if not (0 <= cell_index < len(nb.cells)):
                raise IndexError(f"cell_index {cell_index} out of range")
            cell = nb.cells[cell_index]
            cell.source = source
            if reset_outputs and cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "updated",
            "base_revision": base_revision,
            "revision": new_revision,
            "cell_index": cell_index,
            "cell_type": cell.cell_type,
            "cell_count": len(nb.cells),
        }

    def delete_cell(self, path: str, base_revision: str, cell_index: int) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            if not (0 <= cell_index < len(nb.cells)):
                raise IndexError(f"cell_index {cell_index} out of range")
            removed = nb.cells.pop(cell_index)
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "deleted",
            "base_revision": base_revision,
            "revision": new_revision,
            "cell_index": cell_index,
            "cell_type": removed.cell_type,
            "cell_count": len(nb.cells),
        }

    def move_cell(self, path: str, base_revision: str, from_index: int, to_index: int) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            if not (0 <= from_index < len(nb.cells)):
                raise IndexError(f"from_index {from_index} out of range")
            if not (0 <= to_index <= len(nb.cells)):
                raise IndexError(f"to_index {to_index} out of range")
            cell = nb.cells.pop(from_index)
            nb.cells.insert(to_index, cell)
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "moved",
            "base_revision": base_revision,
            "revision": new_revision,
            "from_index": from_index,
            "to_index": to_index,
            "cell_count": len(nb.cells),
        }

    def clear_outputs(self, path: str, base_revision: str, cell_index: Optional[int] = None) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            if cell_index is not None:
                if not (0 <= cell_index < len(nb.cells)):
                    raise IndexError(f"cell_index {cell_index} out of range")
                cell = nb.cells[cell_index]
                if cell.cell_type != "code":
                    raise ValueError(f"Cell {cell_index} is type {cell.cell_type!r}, not 'code'")
                cell.outputs = []
                cell.execution_count = None
                cleared = 1
            else:
                cleared = 0
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        if cell.get("outputs") or cell.get("execution_count") is not None:
                            cleared += 1
                        cell.outputs = []
                        cell.execution_count = None
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "outputs_cleared",
            "base_revision": base_revision,
            "revision": new_revision,
            "cleared_cells": cleared,
            "cell_index": cell_index,
            "cell_count": len(nb.cells),
        }

    def batch_cells(self, path: str, base_revision: str, operations: list[dict]) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            self._check_revision(nb_path, base_revision)
            nb = self._read_nb(nb_path)
            applied: list[dict] = []
            for op in operations:
                action = op.get("action", "")
                if action == "insert":
                    index = op["index"]
                    if not (0 <= index <= len(nb.cells)):
                        raise IndexError(f"insert index {index} out of range")
                    nb.cells.insert(index, self._new_cell(op.get("cell_type", "code"), op.get("source", "")))
                    applied.append({"action": "insert", "index": index})
                elif action == "update":
                    ci = op["cell_index"]
                    if not (0 <= ci < len(nb.cells)):
                        raise IndexError(f"update cell_index {ci} out of range")
                    nb.cells[ci].source = op["source"]
                    if op.get("reset_outputs", True) and nb.cells[ci].cell_type == "code":
                        nb.cells[ci].outputs = []
                        nb.cells[ci].execution_count = None
                    applied.append({"action": "update", "cell_index": ci})
                elif action == "delete":
                    ci = op["cell_index"]
                    if not (0 <= ci < len(nb.cells)):
                        raise IndexError(f"delete cell_index {ci} out of range")
                    nb.cells.pop(ci)
                    applied.append({"action": "delete", "cell_index": ci})
                else:
                    raise ValueError(f"Unknown action {action!r}; must be 'insert', 'update', or 'delete'")
            nbformat.validate(nb)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)
        return {
            "path": str(nb_path),
            "status": "batch_applied",
            "base_revision": base_revision,
            "revision": new_revision,
            "operations_applied": len(applied),
            "cell_count": len(nb.cells),
        }

    def write_execution_cell(
        self,
        path: str,
        expected_revision: str,
        cell_index: int,
        expected_source: str,
        execution_count: Optional[int],
        raw_messages: list[dict],
    ) -> dict:
        nb_path = self._resolve(path)
        with self._lock_for(nb_path):
            current_revision = self._revision(nb_path)
            if expected_revision != current_revision:
                raise ConflictError(
                    f"Revision conflict during execution write: expected {expected_revision}, current {current_revision}"
                )

            nb = self._read_nb(nb_path)
            if not (0 <= cell_index < len(nb.cells)):
                raise IndexError(f"cell_index {cell_index} out of range")
            cell = nb.cells[cell_index]
            if cell.cell_type != "code":
                raise ValueError(f"Cell {cell_index} is not code")
            if cell.source != expected_source:
                raise ConflictError(f"Cell {cell_index} source changed; refusing to write outputs")

            cell.execution_count = execution_count
            cell.outputs = _build_nbformat_outputs(raw_messages, execution_count)
            self._write_atomic(nb, nb_path)
            new_revision = self._revision(nb_path)

        return {"revision": new_revision, "cell_index": cell_index}
