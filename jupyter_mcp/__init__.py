"""jupyter_mcp — Agent-first MCP server for Jupyter execution and notebook operations.

Shared helpers, constants, and public re-exports for the package.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Optional

import nbformat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_TEXT_LIMIT = 50_000
_OPERATION_TTL_SECONDS = 3600

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\[.*?[@-~]")


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _utc_now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ConflictError(RuntimeError):
    """Raised when a notebook revision conflict is detected."""


def _tool_error(code: str, message: str, details: Optional[dict] = None) -> dict:
    payload = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return {"error": payload}


# ---------------------------------------------------------------------------
# Output shaping helpers
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [truncated — {len(text):,} total chars]", True


def _format_display_dict(data: dict, include_images: bool = False) -> dict:
    out: dict[str, Any] = {}
    if "text/plain" in data:
        out["text"] = data["text/plain"]
    if "text/html" in data:
        html = data["text/html"]
        out["html"] = html[:2000] + "... [truncated]" if len(html) > 2000 else html
    if "image/png" in data:
        out["image_png"] = data["image/png"] if include_images else f"[base64 PNG, {len(data['image/png'])} chars]"
    if "image/jpeg" in data:
        out["image_jpeg"] = data["image/jpeg"] if include_images else f"[base64 JPEG, {len(data['image/jpeg'])} chars]"
    if "image/svg+xml" in data:
        out["image_svg"] = data["image/svg+xml"] if include_images else f"[SVG, {len(data['image/svg+xml'])} chars]"
    if "text/latex" in data:
        out["latex"] = data["text/latex"]
    if "text/markdown" in data:
        out["markdown"] = data["text/markdown"]
    if "application/json" in data:
        out["json"] = data["application/json"]
    if "application/vnd.plotly.v1+json" in data:
        out["plotly"] = "[Plotly figure]"
    known = {
        "text/plain",
        "text/html",
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "text/latex",
        "text/markdown",
        "application/json",
        "application/vnd.plotly.v1+json",
    }
    other = sorted(set(data.keys()) - known)
    if other:
        out["other_mime_types"] = other
    return out


def _parse_iopub_messages(messages: list[dict], include_images: bool = False) -> dict:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    outputs: list[dict] = []
    error: Optional[dict] = None

    for msg in messages:
        msg_type = msg.get("msg_type", "")
        content = msg.get("content", {})

        if msg_type == "stream":
            if content.get("name") == "stdout":
                stdout_parts.append(content.get("text", ""))
            elif content.get("name") == "stderr":
                stderr_parts.append(content.get("text", ""))
        elif msg_type in ("execute_result", "display_data"):
            outputs.append({"type": msg_type, **_format_display_dict(content.get("data", {}), include_images=include_images)})
        elif msg_type == "error":
            error = {
                "ename": content.get("ename", ""),
                "evalue": content.get("evalue", ""),
                "traceback": "\n".join(_strip_ansi(line) for line in content.get("traceback", [])),
            }

    stdout, stdout_truncated = _truncate_text("".join(stdout_parts), _OUTPUT_TEXT_LIMIT)
    stderr, stderr_truncated = _truncate_text("".join(stderr_parts), _OUTPUT_TEXT_LIMIT)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "rich_outputs": outputs,
        "error": error,
        "truncated": stdout_truncated or stderr_truncated,
    }


def _build_nbformat_outputs(messages: list[dict], execution_count: Optional[int]) -> list:
    outputs: list = []
    pending_clear = False

    for msg in messages:
        msg_type = msg.get("msg_type", "")
        content = msg.get("content", {})

        if msg_type not in ("clear_output", "status") and pending_clear:
            outputs.clear()
            pending_clear = False

        if msg_type == "clear_output":
            if content.get("wait", False):
                pending_clear = True
            else:
                outputs.clear()
            continue

        if msg_type == "stream":
            name = content.get("name", "stdout")
            text = content.get("text", "")
            if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1].get("name") == name:
                outputs[-1]["text"] += text
            else:
                outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
        elif msg_type == "execute_result":
            outputs.append(
                nbformat.v4.new_output(
                    "execute_result",
                    data=content.get("data", {}),
                    metadata=content.get("metadata", {}),
                    execution_count=execution_count,
                )
            )
        elif msg_type == "display_data":
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data=content.get("data", {}),
                    metadata=content.get("metadata", {}),
                )
            )
        elif msg_type == "error":
            outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=content.get("ename", ""),
                    evalue=content.get("evalue", ""),
                    traceback=content.get("traceback", []),
                )
            )

    return outputs


# ---------------------------------------------------------------------------
# Public re-exports (imported after all helpers are defined to avoid circular
# imports — each sibling module only imports helpers defined above)
# ---------------------------------------------------------------------------

from jupyter_mcp.kernel import KernelProvider, LocalKernelProvider, SessionRecord  # noqa: E402
from jupyter_mcp.notebooks import NotebookStore, FileNotebookStore  # noqa: E402
from jupyter_mcp.operations import OperationRecord, OperationManager  # noqa: E402
from jupyter_mcp.orchestrator import ExecutionOrchestrator  # noqa: E402

__all__ = [
    # exceptions
    "ConflictError",
    # helpers
    "_utc_now",
    "_new_id",
    "_tool_error",
    "_strip_ansi",
    "_truncate_text",
    "_format_display_dict",
    "_parse_iopub_messages",
    "_build_nbformat_outputs",
    # constants
    "_OUTPUT_TEXT_LIMIT",
    "_OPERATION_TTL_SECONDS",
    "_ANSI_RE",
    # public classes
    "KernelProvider",
    "LocalKernelProvider",
    "SessionRecord",
    "NotebookStore",
    "FileNotebookStore",
    "OperationRecord",
    "OperationManager",
    "ExecutionOrchestrator",
]
