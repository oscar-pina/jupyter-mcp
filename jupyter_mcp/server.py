"""FastMCP server entry point with all 22 MCP tool definitions.

Start with: python -m jupyter_mcp.server
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import signal
from typing import Any, Optional

from fastmcp import FastMCP

from jupyter_mcp import _tool_error
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
        "Typical workflow:\n"
        "  1. list_runtimes → pick a runtime name\n"
        "  2. create_session(runtime) → session_id\n"
        "  3. run_code(session_id, code, wait_ms=5000) → inline result when fast,\n"
        "     or operation descriptor to poll with get_operation(op_id, wait_ms=5000)\n"
        "  4. close_session(session_id) when done\n"
        "\n"
        "Notebook editing uses optimistic concurrency: every mutation (insert_cell,\n"
        "update_cell, delete_cell, move_cell, clear_outputs, batch_cells) requires a\n"
        "base_revision token from read_notebook. On conflict (file changed externally),\n"
        "the tool returns error code 'Conflict' — re-read and retry.\n"
        "\n"
        "Error format: {\"error\": {\"code\": \"...\", \"message\": \"...\"}}\n"
        "Operation statuses: queued → running → completed | failed | cancelled\n"
        "\n"
        "Variable inspection (synchronous): get_variable, list_variables\n"
        "Kernel reset (keeps session_id): restart_session"
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


def _validate_env(env: dict[str, str]) -> Optional[dict]:
    for key in env:
        ku = key.upper()
        if ku in _ENV_DENIED or any(ku.startswith(p) for p in _ENV_DENIED_PREFIXES):
            return _tool_error("SecurityError", f"Environment variable {key!r} is not permitted")
    return None


# ---------------------------------------------------------------------------
# Variable inspection code templates
# ---------------------------------------------------------------------------

_GET_VARIABLE_CODE = """\
import json as _json
_val = {name!r}
try:
    _v = eval(_val)
    print(_json.dumps({{"name": _val, "value": repr(_v), "type": type(_v).__name__}}))
except NameError:
    print(_json.dumps({{"name": _val, "error": "not defined"}}))
except Exception as _e:
    print(_json.dumps({{"name": _val, "error": str(_e)}}))
del _val, _json
"""

_LIST_VARIABLES_CODE = """\
import json as _json
_skip = {"__builtins__", "__name__", "__doc__", "__package__",
         "__loader__", "__spec__", "__annotations__", "__builtinms__"}
_rows = []
for _k, _v in list(vars().items()):
    if _k.startswith("_") or _k in _skip:
        continue
    _rows.append({"name": _k, "type": type(_v).__name__, "repr": repr(_v)[:200]})
print(_json.dumps(_rows))
del _json, _skip, _rows, _k, _v
"""

# ---------------------------------------------------------------------------
# MCP tools — Session management
# ---------------------------------------------------------------------------


@mcp.tool()
def list_runtimes() -> dict:
    """List available kernel runtimes.

    Returns:
        Dict with `runtimes`, where each item has runtime, display_name, language.
    """
    try:
        return {"runtimes": provider.list_runtimes()}
    except Exception as exc:
        return _tool_error("BackendUnavailable", str(exc))


@mcp.tool()
def create_session(
    runtime: str,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    isolation: str = "ephemeral",
    python_path: Optional[str] = None,
) -> dict:
    """Create a kernel-backed execution session.

    Args:
        runtime: Kernel runtime name from list_runtimes (e.g. 'python3').
        cwd: Optional working directory for kernel startup.
        env: Optional environment variables for the kernel process.
            Dangerous variables (LD_*, DYLD_*, PYTHONSTARTUP) are rejected.
        isolation: `ephemeral` (auto-closed after idle; default) or `persistent`.
        python_path: Path to a Python interpreter to use instead of the
            registered kernel spec (e.g. '/path/to/.venv/bin/python').
            ipykernel must be installed in that environment.

    Returns:
        Session descriptor with session_id, runtime, cwd, created_at.
    """
    if isolation not in {"ephemeral", "persistent"}:
        return _tool_error("ValidationError", "isolation must be 'ephemeral' or 'persistent'")
    if env:
        err = _validate_env(env)
        if err:
            return err
    try:
        rec = provider.create_session(runtime=runtime, cwd=cwd, env=env, isolation=isolation, python_path=python_path)
        return dataclasses.asdict(rec)
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def get_session(session_id: str) -> dict:
    """Get one session by ID.

    Args:
        session_id: Session identifier.

    Returns:
        Session descriptor.
    """
    try:
        rec = provider.get_session(session_id)
        return dataclasses.asdict(rec)
    except KeyError as exc:
        return _tool_error("NotFound", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def list_sessions() -> dict:
    """List all active sessions.

    Returns:
        Dict with `sessions` list.
    """
    try:
        return {"sessions": [dataclasses.asdict(s) for s in provider.list_sessions()]}
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


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
        return _tool_error("NotFound", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


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
        return _tool_error("NotFound", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


# ---------------------------------------------------------------------------
# MCP tools — Notebook operations
# ---------------------------------------------------------------------------


@mcp.tool()
def list_notebooks(directory: str = ".") -> dict:
    """List notebooks under a directory (recursive, checkpoint dirs excluded).

    Args:
        directory: Base directory to scan.

    Returns:
        Dict with canonical directory path and notebook entries.
    """
    try:
        return notebooks.list_notebooks(directory)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except ValueError as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


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
        return _tool_error("Conflict", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def delete_notebook(path: str, expected_revision: str) -> dict:
    """Delete a notebook, guarded by revision.

    Args:
        path: Notebook path.
        expected_revision: Latest notebook revision from read_notebook.

    Returns:
        Deletion status.
    """
    try:
        return notebooks.delete_notebook(path, expected_revision)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def read_notebook(
    path: str,
    cell_selector: Optional[str] = None,
    include_outputs: bool = True,
    output_limit: int = 20_000,
    include_images: bool = False,
) -> dict:
    """Read notebook content with optional cell slicing and output inclusion.

    Returns a `revision` token needed for all subsequent notebook mutations.
    Re-read the notebook whenever you get a 'Conflict' error to get the latest revision.

    Args:
        path: Notebook file path.
        cell_selector: Optional `start:end` slice to limit returned cells (exclusive end,
            e.g. '0:5' = first 5 cells). Use run_notebook's cell_selector for richer
            selection syntax when executing.
        include_outputs: Include code cell outputs (default True).
        output_limit: Max characters per textual output in response.
        include_images: Include raw base64 image data in outputs (default False).

    Returns:
        Notebook payload including `revision` for optimistic updates.
    """
    try:
        return notebooks.read(path, cell_selector, include_outputs, output_limit, include_images=include_images)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except ValueError as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def insert_cell(
    path: str,
    base_revision: str,
    index: int,
    cell_type: str,
    source: str,
) -> dict:
    """Insert a cell at a specific index.

    Args:
        path: Notebook path.
        base_revision: Revision returned by read_notebook.
        index: Insert position (0..cell_count).
        cell_type: One of `code`, `markdown`, `raw`.
        source: Cell source content.

    Returns:
        Mutation status with new revision.
    """
    try:
        return notebooks.insert_cell(path, base_revision, index, cell_type, source)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def update_cell(
    path: str,
    base_revision: str,
    cell_index: int,
    source: str,
    reset_outputs: bool = True,
) -> dict:
    """Update a cell source.

    Args:
        path: Notebook path.
        base_revision: Revision returned by read_notebook.
        cell_index: Target cell index.
        source: New source content.
        reset_outputs: For code cells, clear outputs and execution_count when true.

    Returns:
        Mutation status with new revision.
    """
    try:
        return notebooks.update_cell(path, base_revision, cell_index, source, reset_outputs)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def delete_cell(path: str, base_revision: str, cell_index: int) -> dict:
    """Delete a cell by index.

    Args:
        path: Notebook path.
        base_revision: Revision returned by read_notebook.
        cell_index: Cell index to remove.

    Returns:
        Mutation status with new revision.
    """
    try:
        return notebooks.delete_cell(path, base_revision, cell_index)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def move_cell(path: str, base_revision: str, from_index: int, to_index: int) -> dict:
    """Move a cell from one index to another.

    Args:
        path: Notebook path.
        base_revision: Revision returned by read_notebook.
        from_index: Current cell index.
        to_index: Destination index.

    Returns:
        Mutation status with new revision.
    """
    try:
        return notebooks.move_cell(path, base_revision, from_index, to_index)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def clear_outputs(path: str, base_revision: str, cell_index: Optional[int] = None) -> dict:
    """Clear outputs for one code cell or all code cells in a notebook.

    Args:
        path: Notebook path.
        base_revision: Revision from read_notebook.
        cell_index: If provided, clear only this cell's outputs.
            If omitted, clear all code cell outputs in the notebook.

    Returns:
        Mutation status with new revision and cleared_cells count.
    """
    try:
        return notebooks.clear_outputs(path, base_revision, cell_index)
    except FileNotFoundError as exc:
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def batch_cells(path: str, base_revision: str, operations: list) -> dict:
    """Apply multiple cell operations atomically under a single revision.

    Avoids the N sequential round-trips required when building notebooks with
    many insert_cell calls. All operations succeed or none are applied.

    Each operation dict must include an `action` key:
      - `{"action": "insert", "index": <int>, "cell_type": <str>, "source": <str>}`
      - `{"action": "update", "cell_index": <int>, "source": <str>, "reset_outputs": <bool>}`
      - `{"action": "delete", "cell_index": <int>}`

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
        return _tool_error("NotFound", str(exc))
    except RuntimeError as exc:
        return _tool_error("Conflict", str(exc))
    except (ValueError, IndexError) as exc:
        return _tool_error("ValidationError", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


# ---------------------------------------------------------------------------
# MCP tools — Execution
# ---------------------------------------------------------------------------


@mcp.tool()
def run_code(
    session_id: str,
    code: str,
    timeout_s: int = 60,
    on_timeout: str = "interrupt",
    capture: str = "summary",
    wait_ms: int = 5000,
    include_images: bool = False,
) -> dict:
    """Execute code in an existing session, optionally waiting for inline results.

    When wait_ms > 0 (default 5000ms), waits up to that duration for the result.
    If the code finishes in time, returns the completed operation with `result` inline
    (status='completed'). If it times out, returns the operation descriptor
    (status='queued' or 'running') — then poll with get_operation(op_id, wait_ms=5000).

    Args:
        session_id: Existing session ID from create_session.
        code: Source code to execute.
        timeout_s: Wall-clock execution timeout in seconds.
        on_timeout: Timeout policy (`interrupt` only).
        capture: `summary` (default, limited rich outputs) or `full`.
        wait_ms: Milliseconds to wait for inline result (0 = fire-and-forget).
        include_images: Include raw base64 image data in outputs (default False).

    Returns:
        Completed operation snapshot when fast, or operation descriptor to poll.
    """
    if on_timeout not in {"interrupt"}:
        return _tool_error("ValidationError", "on_timeout currently supports only 'interrupt'")
    if capture not in {"summary", "full"}:
        return _tool_error("ValidationError", "capture must be 'summary' or 'full'")

    def _job(_: str) -> dict:
        return orchestrator.run_code(session_id, code, timeout_s, on_timeout, capture, include_images)

    def _cancel_cb() -> None:
        provider.interrupt(session_id)

    snapshot = ops.submit(kind="run_code", fn=_job, cancel_callback=_cancel_cb)
    if "error" in snapshot or wait_ms <= 0:
        return snapshot
    return ops.get(op_id=snapshot["op_id"], wait_ms=wait_ms)


@mcp.tool()
def run_notebook(
    path: str,
    runtime_or_session: str,
    mode: str = "fresh",
    cell_selector: Optional[str] = None,
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
        runtime_or_session: Runtime name (fresh mode) or session_id (session mode).
        mode: `fresh` (reproducible isolated run) or `session` (reuse stateful session).
        cell_selector: Which cells to run. Supports: `all` (default), single index
            (`5`), slice (`5:9`, exclusive end), or comma-combinations (`1,3,5:9`).
        timeout_s: Timeout per cell in seconds.
        stop_on_error: Stop at first execution error.
        wait_ms: Milliseconds to wait for an inline result (0 = fire-and-forget).

    Returns:
        Completed operation snapshot when fast, or operation descriptor to poll.
    """
    if mode not in {"fresh", "session"}:
        return _tool_error("ValidationError", "mode must be 'fresh' or 'session'")

    def _job(op_id: str) -> dict:
        def _on_progress(p: dict) -> None:
            ops.update_progress(op_id, p)

        return orchestrator.run_notebook(
            path, runtime_or_session, mode, cell_selector, timeout_s, stop_on_error, _on_progress
        )

    snapshot = ops.submit(kind="run_notebook", fn=_job)
    if "error" in snapshot or wait_ms <= 0:
        return snapshot
    return ops.get(op_id=snapshot["op_id"], wait_ms=wait_ms)


# ---------------------------------------------------------------------------
# MCP tools — Variable inspection
# ---------------------------------------------------------------------------


@mcp.tool()
def get_variable(session_id: str, name: str) -> dict:
    """Inspect a single variable in a running session (synchronous).

    Args:
        session_id: Existing session ID.
        name: Variable name to inspect.

    Returns:
        Dict with `name`, `value` (repr), and `type`, or `error` if not defined.
    """
    import json as _json

    code = _GET_VARIABLE_CODE.format(name=name)
    try:
        run = provider.execute(session_id=session_id, code=code, timeout_s=15, on_timeout="interrupt")
    except KeyError as exc:
        return _tool_error("NotFound", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))

    stdout = (run.get("stdout") or "").strip()
    if not stdout:
        return _tool_error("ExecutionError", "No output from variable inspection")
    try:
        return _json.loads(stdout.splitlines()[-1])
    except Exception:
        return _tool_error("ExecutionError", f"Could not parse output: {stdout[:200]!r}")


@mcp.tool()
def list_variables(session_id: str) -> dict:
    """List all user-defined variables in a running session (synchronous).

    Returns a snapshot of the kernel's namespace, excluding private and
    built-in names.

    Args:
        session_id: Existing session ID.

    Returns:
        Dict with `variables` list, each entry having `name`, `type`, `repr`.
    """
    import json as _json

    try:
        run = provider.execute(session_id=session_id, code=_LIST_VARIABLES_CODE, timeout_s=15, on_timeout="interrupt")
    except KeyError as exc:
        return _tool_error("NotFound", str(exc))
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))

    stdout = (run.get("stdout") or "").strip()
    if not stdout:
        return {"variables": []}
    try:
        rows = _json.loads(stdout.splitlines()[-1])
        return {"variables": rows}
    except Exception:
        return _tool_error("ExecutionError", f"Could not parse output: {stdout[:200]!r}")


# ---------------------------------------------------------------------------
# MCP tools — Operation management
# ---------------------------------------------------------------------------


@mcp.tool()
def get_operation(op_id: str, wait_ms: int = 0) -> dict:
    """Get operation state and result.

    Args:
        op_id: Operation ID from run_code/run_notebook.
        wait_ms: Optional wait duration before returning.

    Returns:
        Operation snapshot.
    """
    try:
        return ops.get(op_id=op_id, wait_ms=wait_ms)
    except TimeoutError:
        return _tool_error("Timeout", "Operation wait timed out")
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


@mcp.tool()
def cancel_operation(op_id: str) -> dict:
    """Request cancellation for an operation.

    Args:
        op_id: Operation ID.

    Returns:
        Updated operation snapshot.
    """
    try:
        return ops.cancel(op_id)
    except Exception as exc:
        return _tool_error("ExecutionError", str(exc))


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
