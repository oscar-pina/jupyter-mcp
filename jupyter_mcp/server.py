"""FastMCP server entry point with all MCP tool definitions.

Start with: python -m jupyter_mcp.server
"""

from __future__ import annotations

import atexit
import dataclasses
import signal
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from jupyter_mcp.kernel import LocalKernelProvider
from jupyter_mcp.notebooks import FileNotebookStore
from jupyter_mcp.operations import OperationManager
from jupyter_mcp.orchestrator import ExecutionOrchestrator

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "jupyter-mcp",
    instructions=(
        "Agent-first Jupyter MCP server.\n"
        "\n"
        "No Jupyter server required. This MCP server launches and manages kernel\n"
        "processes directly — do not ask the user to start jupyter notebook or\n"
        "jupyter lab. Just call create_session to begin.\n"
        "\n"
        "If create_session fails with an ipykernel error, ipykernel is most likely\n"
        "not installed. Tell the user to run: pip install ipykernel\n"
        "\n"
        "Workflow for running code interactively:\n"
        "  1. create_session() → session_id  (uses system 'python' by default)\n"
        "     Or: create_session(python_path='/path/to/.venv/bin/python')\n"
        "  2. run_code(session_id, code, wait_ms=5000) → inline result when fast,\n"
        "     or operation descriptor to poll with get_operation(op_id, wait_ms=5000)\n"
        "  3. close_session(session_id) when done\n"
        "  Variables and imports persist between run_code calls in the same session.\n"
        "\n"
        "Workflow for executing a notebook file:\n"
        "  1. run_notebook(path) → uses default 'python' interpreter\n"
        "     Or: run_notebook(path, python_path='/path/to/.venv/bin/python')\n"
        "     Executes all cells and writes outputs back to the file.\n"
        "     Returns an operation descriptor (default wait_ms=0, fire-and-forget).\n"
        "  2. Poll with get_operation(op_id, wait_ms=5000) until status is\n"
        "     'completed' or 'failed'. Use cancel_operation(op_id) to abort.\n"
        "  mode='fresh': disposable isolated kernel (reproducible). Default.\n"
        "  mode='session': reuse an existing stateful session (pass session_id).\n"
        "\n"
        "Notebook editing uses optimistic concurrency: every mutation (edit_notebook,\n"
        "delete_notebook) requires a base_revision token from read_notebook.\n"
        "Re-read to get the new revision after any mutation — each mutation returns it.\n"
        "\n"
        "Errors are raised as tool errors. Error codes in the message prefix:\n"
        "  [Conflict]        — notebook changed externally; re-read and retry\n"
        "  [NotFound]        — verify session_id or file path\n"
        "  [ValidationError] — fix the parameter value\n"
        "  [ExecutionError]  — check code syntax or use restart_session to reset\n"
        "  [SecurityError]   — disallowed operation (e.g. env variable blocked)\n"
        "  [Busy]            — too many in-flight operations; wait and retry\n"
        "\n"
        "Operation statuses: queued → running → completed | failed | cancelled\n"
        "Kernel reset (keeps session_id): restart_session\n"
        "Interrupt without reset: interrupt_session"
    ),
)

# ---------------------------------------------------------------------------
# Global app state
# ---------------------------------------------------------------------------

provider = LocalKernelProvider()
notebooks = FileNotebookStore()
orchestrator = ExecutionOrchestrator(provider, notebooks)
ops = OperationManager(max_workers=4, max_inflight=32)

# ---------------------------------------------------------------------------
# Environment variable security
# ---------------------------------------------------------------------------

# Block OS-level loader variables that take effect before user code runs and
# cannot be overridden post-startup. PYTHONPATH, PATH, and HOME are intentionally
# not blocked because run_code already grants equivalent capability at runtime.
_ENV_DENIED: frozenset[str] = frozenset({"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "PYTHONSTARTUP"})
_ENV_DENIED_PREFIXES: tuple[str, ...] = ("LD_", "DYLD_")


def _validate_env(env: dict[str, str]) -> None:
    """Raise ToolError if env contains a disallowed variable."""
    for key in env:
        ku = key.upper()
        if ku in _ENV_DENIED or any(ku.startswith(p) for p in _ENV_DENIED_PREFIXES):
            raise ToolError(f"[SecurityError] Environment variable {key!r} is not permitted")


def _raise_on_op_error(result: dict) -> dict:
    """If an ops method returned an in-band error dict, raise ToolError."""
    if "error" in result:
        err = result["error"]
        raise ToolError(f"[{err['code']}] {err['message']}")
    return result


# ---------------------------------------------------------------------------
# MCP tools — Session management (5 tools)
# ---------------------------------------------------------------------------


@mcp.tool()
def create_session(
    python_path: str = "python",
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    isolation: str = "ephemeral",
) -> dict:
    """Create a kernel-backed execution session.

    Args:
        python_path: Python interpreter to use. Accepts a bare command name
            (e.g. 'python', 'python3', resolved via PATH) or an absolute path
            (e.g. '/path/to/.venv/bin/python'). Defaults to 'python'.
            ipykernel must be installed in that environment.
        cwd: Optional working directory for kernel startup.
        env: Optional environment variables for the kernel process.
            Dangerous variables (LD_*, DYLD_*, PYTHONSTARTUP) are rejected.
        isolation: `ephemeral` (auto-closed after 30 min idle; default) or
            `persistent` (remains until explicitly closed).

    Returns:
        Session descriptor with session_id, python_path, cwd, created_at.
    """
    if isolation not in {"ephemeral", "persistent"}:
        raise ToolError("[ValidationError] isolation must be 'ephemeral' or 'persistent'")
    if env:
        _validate_env(env)
    try:
        rec = provider.create_session(python_path=python_path, cwd=cwd, env=env, isolation=isolation)
        return dataclasses.asdict(rec)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def list_sessions() -> dict:
    """List all active sessions.

    Returns:
        Dict with `sessions` list of session descriptors.
    """
    try:
        return {"sessions": [dataclasses.asdict(s) for s in provider.list_sessions()]}
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def close_session(session_id: str, force: bool = False) -> dict:
    """Close a session and shut down its kernel.

    Args:
        session_id: Session identifier.
        force: Force immediate shutdown.

    Returns:
        Status payload.
    """
    try:
        provider.close_session(session_id, force=force)
        return {"status": "closed", "session_id": session_id, "force": force}
    except KeyError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def restart_session(session_id: str) -> dict:
    """Restart a session's kernel, clearing all variables and state.

    The session_id remains stable after restart. Use this to recover from errors
    or reset execution context without losing session configuration (cwd, env).

    Args:
        session_id: Session identifier from create_session.

    Returns:
        Status payload.
    """
    try:
        provider.restart(session_id)
        return {"status": "restarted", "session_id": session_id}
    except KeyError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def interrupt_session(session_id: str) -> dict:
    """Interrupt a running kernel execution without clearing session state.

    Sends an interrupt signal to the kernel, stopping any currently running
    cell. All previously defined variables and imports remain intact — the
    session stays fully usable after the interrupt. Use this instead of
    restart_session when you want to stop a long-running cell but keep
    the existing execution context.

    Args:
        session_id: Session identifier from create_session.

    Returns:
        Status payload.
    """
    try:
        provider.interrupt(session_id)
        return {"status": "interrupted", "session_id": session_id}
    except KeyError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


# ---------------------------------------------------------------------------
# MCP tools — Notebook operations (6 tools)
# ---------------------------------------------------------------------------


@mcp.tool()
def list_notebooks(directory: str = ".") -> dict:
    """List notebooks under a directory (recursive, checkpoint dirs excluded).

    Args:
        directory: Base directory to scan (default '.' resolves to the server
            process working directory; prefer absolute paths).

    Returns:
        Dict with canonical directory path and notebook entries.
    """
    try:
        return notebooks.list_notebooks(directory)
    except FileNotFoundError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except PermissionError as exc:
        raise ToolError(f"[SecurityError] {exc}") from exc
    except ValueError as exc:
        raise ToolError(f"[ValidationError] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def create_notebook(path: str, kernel_name: str = "python3") -> dict:
    """Create a new notebook file.

    Args:
        path: Destination notebook path.
        kernel_name: Kernelspec name to embed in metadata.

    Returns:
        Status payload with initial revision.
    """
    try:
        return notebooks.create_notebook(path, kernel_name)
    except FileExistsError as exc:
        raise ToolError(f"[Conflict] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def rename_notebook(path: str, new_path: str, base_revision: str) -> dict:
    """Rename or move a notebook file.

    Atomically renames the notebook file. The destination must not already exist.
    Requires a base_revision to guard against renaming a stale file.

    Args:
        path: Current notebook path.
        new_path: Destination path (can move across directories).
        base_revision: Latest revision from read_notebook.

    Returns:
        Status payload with old_path, new path, and new revision.
    """
    try:
        return notebooks.rename_notebook(path, new_path, base_revision)
    except FileNotFoundError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except FileExistsError as exc:
        raise ToolError(f"[Conflict] {exc}") from exc
    except RuntimeError as exc:
        raise ToolError(f"[Conflict] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def delete_notebook(path: str, base_revision: str) -> dict:
    """Delete a notebook, guarded by revision.

    Args:
        path: Notebook path.
        base_revision: Latest notebook revision from read_notebook.

    Returns:
        Deletion status.
    """
    try:
        return notebooks.delete_notebook(path, base_revision)
    except FileNotFoundError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except RuntimeError as exc:
        raise ToolError(f"[Conflict] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def read_notebook(
    path: str,
    cell_start: Optional[int] = None,
    cell_end: Optional[int] = None,
    include_outputs: bool = False,
    output_limit: int = 4_000,
    include_images: bool = False,
) -> dict:
    """Read notebook content with optional cell range and output inclusion.

    Returns a `revision` token needed for all subsequent notebook mutations.
    Re-read the notebook whenever you get a Conflict error to get the latest revision.

    Args:
        path: Notebook file path.
        cell_start: First cell index to include (default 0 = from the start).
        cell_end: Exclusive end index (default = cell_count = to the end).
            E.g. cell_start=0, cell_end=5 returns the first 5 cells.
        include_outputs: Include code cell outputs (default False).
            Set to True when you need to inspect execution results.
        output_limit: Max characters per textual output (default 4000).
            Increase up to 20000 for large outputs like DataFrames.
        include_images: When True, include raw base64 image data. When False
            (default), images are replaced with size placeholders like
            '[base64 PNG, 1234 chars]'.

    Returns:
        Notebook payload including `revision` for optimistic updates.
    """
    try:
        return notebooks.read(
            path,
            cell_start=cell_start,
            cell_end=cell_end,
            include_outputs=include_outputs,
            output_limit=output_limit,
            include_images=include_images,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except ValueError as exc:
        raise ToolError(f"[ValidationError] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def edit_notebook(path: str, base_revision: str, operations: list) -> dict:
    """Apply one or more cell operations atomically under a single revision.

    The sole tool for modifying notebook cells — handles single and bulk edits
    with the same API. All operations succeed or none are applied.

    Each operation dict must include an `action` key:
      - `{"action": "insert", "cell_index": <int>, "cell_type": <str>, "source": <str>}`
          Inserts a new cell at cell_index. cell_type: 'code', 'markdown', or 'raw'.
      - `{"action": "update", "cell_index": <int>, "source": <str>}`
          Replaces a cell's source. Also clears outputs/execution_count for code cells
          unless `"reset_outputs": false` is added.
      - `{"action": "delete", "cell_index": <int>}`
          Removes the cell at cell_index.

    IMPORTANT: Operations are applied sequentially to a live list. An insert
    shifts all subsequent cell indices up by 1; a delete shifts them down.
    Plan indices accordingly, or process in reverse index order when mixing
    inserts and deletes.

    Args:
        path: Notebook path.
        base_revision: Revision from read_notebook.
        operations: List of operation dicts (see above).

    Returns:
        Mutation status with new revision and count of applied operations.
    """
    try:
        return notebooks.batch_cells(path, base_revision, operations)
    except FileNotFoundError as exc:
        raise ToolError(f"[NotFound] {exc}") from exc
    except RuntimeError as exc:
        raise ToolError(f"[Conflict] {exc}") from exc
    except (ValueError, IndexError) as exc:
        raise ToolError(f"[ValidationError] {exc}") from exc
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


# ---------------------------------------------------------------------------
# MCP tools — Execution (2 tools)
# ---------------------------------------------------------------------------


@mcp.tool()
def run_code(
    session_id: str,
    code: str,
    timeout_s: int = 60,
    wait_ms: int = 5000,
    include_images: bool = False,
) -> dict:
    """Execute code in an existing session, optionally waiting for inline results.

    When wait_ms > 0 (default 5000ms), waits up to that duration for the result.
    If the code finishes in time, returns the completed operation with `result` inline
    (status='completed'). If it times out, returns the operation descriptor
    (status='queued' or 'running') — then poll with get_operation(op_id, wait_ms=5000).

    On timeout, the kernel receives an interrupt signal — execution stops but all
    previously defined variables and imports remain available.

    Args:
        session_id: Existing session ID from create_session.
        code: Source code to execute.
        timeout_s: Wall-clock execution timeout in seconds.
        wait_ms: Milliseconds to wait for inline result (0 = fire-and-forget).
        include_images: When True, include raw base64 image data. When False
            (default), images are replaced with size placeholders like
            '[base64 PNG, 1234 chars]'.

    Returns:
        Completed operation snapshot when fast, or operation descriptor to poll.
    """
    def _job(_: str) -> dict:
        return orchestrator.run_code(session_id, code, timeout_s, "summary", include_images)

    def _cancel_cb() -> None:
        provider.interrupt(session_id)

    snapshot = _raise_on_op_error(ops.submit(kind="run_code", fn=_job, cancel_callback=_cancel_cb))
    if wait_ms <= 0:
        return snapshot
    return _raise_on_op_error(ops.get(op_id=snapshot["op_id"], wait_ms=wait_ms))


@mcp.tool()
def run_notebook(
    path: str,
    python_path: str = "python",
    mode: str = "fresh",
    session_id: Optional[str] = None,
    cell_start: Optional[int] = None,
    cell_end: Optional[int] = None,
    timeout_s: int = 300,
    stop_on_error: bool = True,
    wait_ms: int = 0,
) -> dict:
    """Execute notebook cells asynchronously and persist outputs.

    Executes selected cells, writes outputs back to the notebook file, and
    returns an operation descriptor. For long runs, poll with
    get_operation(op_id, wait_ms=5000); the operation includes a `progress`
    field updated after each cell.

    Args:
        path: Notebook path.
        python_path: Python interpreter for fresh mode (default 'python',
            resolved via PATH). Ignored when mode='session'.
        mode: `fresh` (reproducible isolated run, default) or `session`
            (reuse an existing stateful session).
        session_id: Existing session ID. Required when mode='session',
            ignored when mode='fresh'.
        cell_start: First cell index to execute (default 0 = from the start).
        cell_end: Exclusive end index (default = cell_count = to the end).
        timeout_s: Timeout per cell in seconds.
        stop_on_error: Stop at first execution error.
        wait_ms: Milliseconds to wait for an inline result. Default 0 means
            fire-and-forget — the tool returns immediately with an operation
            descriptor. Poll with get_operation(op_id, wait_ms=5000).

    Returns:
        Completed operation snapshot when fast, or operation descriptor to poll.
    """
    if mode not in {"fresh", "session"}:
        raise ToolError("[ValidationError] mode must be 'fresh' or 'session'")
    if mode == "session" and session_id is None:
        raise ToolError("[ValidationError] mode='session' requires session_id")

    # Shared container: written by orchestrator once session is ready,
    # read by _cancel_cb so it can interrupt the kernel mid-cell.
    session_holder: list = [None]

    def _job(op_id: str) -> dict:
        def _on_progress(p: dict) -> None:
            ops.update_progress(op_id, p)

        def _is_cancelled() -> bool:
            return ops.is_cancelled(op_id)

        def _on_session_ready(sid: str) -> None:
            session_holder[0] = sid

        return orchestrator.run_notebook(
            path, mode, python_path, session_id, cell_start, cell_end, timeout_s, stop_on_error,
            _on_progress, _is_cancelled, _on_session_ready,
        )

    def _cancel_cb() -> None:
        sid = session_holder[0]
        if sid is not None:
            try:
                provider.interrupt(sid)
            except Exception:
                pass

    snapshot = _raise_on_op_error(ops.submit(kind="run_notebook", fn=_job, cancel_callback=_cancel_cb))
    if wait_ms <= 0:
        return snapshot
    return _raise_on_op_error(ops.get(op_id=snapshot["op_id"], wait_ms=wait_ms))


# ---------------------------------------------------------------------------
# MCP tools — Operation management (3 tools)
# ---------------------------------------------------------------------------


@mcp.tool()
def get_operation(op_id: str, wait_ms: int = 0) -> dict:
    """Get operation state and result.

    Args:
        op_id: Operation ID from run_code/run_notebook.
        wait_ms: Long-poll timeout in milliseconds. The server blocks up to this
            duration waiting for the operation to finish, then returns the
            current snapshot regardless. 0 = immediate non-blocking check.

    Returns:
        Operation snapshot.
    """
    try:
        return _raise_on_op_error(ops.get(op_id=op_id, wait_ms=wait_ms))
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def cancel_operation(op_id: str) -> dict:
    """Request cancellation for an operation.

    Args:
        op_id: Operation ID.

    Returns:
        Updated operation snapshot.
    """
    try:
        return _raise_on_op_error(ops.cancel(op_id))
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


@mcp.tool()
def list_operations() -> dict:
    """List all tracked operations.

    Returns all operations known to the server — queued, running, completed,
    failed, or cancelled. Useful for recovering op_ids after context truncation
    or checking the status of multiple concurrent operations.

    Returns:
        Dict with `operations` list, each entry being an operation snapshot.
    """
    try:
        return {"operations": ops.list()}
    except Exception as exc:
        raise ToolError(f"[ExecutionError] {exc}") from exc


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _shutdown() -> None:
    try:
        ops.shutdown()
    except Exception:
        pass
    try:
        provider.shutdown()
    except Exception:
        pass


def _signal_handler(signum: int, _frame: Any) -> None:
    _shutdown()
    raise SystemExit(128 + signum)


atexit.register(_shutdown)


def main() -> None:
    """Console-script entry point (e.g. `jupyter-mcp` or `uvx jupyter-mcp`)."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    mcp.run()


if __name__ == "__main__":
    main()
