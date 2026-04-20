# jupyter-mcp

An agent-first MCP server for Jupyter execution and notebook operations.

## What it does

`jupyter-mcp` exposes 22 tools to an AI agent via the [Model Context Protocol](https://modelcontextprotocol.io):

- Create and manage kernel sessions (any installed Jupyter runtime)
- Execute code and get results synchronously or poll asynchronously
- Read and edit notebooks with optimistic concurrency (revision tokens)
- Inspect live variables without consuming an operation slot
- Run full notebooks and persist outputs back to the file

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/jupyter-mcp.git
cd jupyter-mcp
pip install -e .
```

Or with `uv` (recommended):

```bash
git clone https://github.com/YOUR_USERNAME/jupyter-mcp.git
cd jupyter-mcp
uv pip install -e .
```

### 2. Configure Claude Code

**Option A — project-level** (works automatically when you open the repo in Claude Code):

The repo ships a `.mcp.json` at the root. Claude Code picks it up automatically when you `cd` into the repo directory.

**Option B — global** (makes it available in every project):

```bash
claude mcp add jupyter-mcp -- python -m jupyter_mcp.server
```

Or add it manually to `~/.claude.json`:

```json
{
  "mcpServers": {
    "jupyter-mcp": {
      "command": "python",
      "args": ["-m", "jupyter_mcp.server"]
    }
  }
}
```

> Tip: if you installed with `uv` into a venv, use the venv's Python:
> ```json
> "command": "/path/to/jupyter-mcp/.venv/bin/python"
> ```

### 3. Verify

```
claude mcp list
# jupyter-mcp  python -m jupyter_mcp.server  connected
```

## Quick start (inside Claude Code)

```
1. list_runtimes                          → choose a runtime, e.g. "python3"
2. create_session(runtime="python3")      → { session_id: "sess_abc123" }
3. run_code(session_id, "1 + 1", wait_ms=5000)
   → { status: "completed", result: { stdout: "", execution_count: 1 } }
4. close_session(session_id)
```

Notebook editing uses optimistic concurrency — every mutation requires the `revision` token from `read_notebook`. On conflict (file changed externally) you get error code `"Conflict"`: re-read and retry.

## Tool reference

### Session management
| Tool | Description |
|------|-------------|
| `list_runtimes` | List installed kernel specs |
| `create_session` | Start a kernel session |
| `get_session` | Get session by ID |
| `list_sessions` | List all active sessions |
| `close_session` | Shut down a session |
| `restart_session` | Restart kernel, keeping session ID |

### Notebook operations
| Tool | Description |
|------|-------------|
| `list_notebooks` | Recursively list `.ipynb` files |
| `create_notebook` | Create a new empty notebook |
| `delete_notebook` | Delete (revision-guarded) |
| `read_notebook` | Read cells and outputs; returns revision |
| `insert_cell` | Insert a cell at index |
| `update_cell` | Update cell source |
| `delete_cell` | Delete cell by index |
| `move_cell` | Move cell to new index |
| `clear_outputs` | Clear one or all code cell outputs |
| `batch_cells` | Atomic multi-cell insert/update/delete |

### Execution
| Tool | Description |
|------|-------------|
| `run_code` | Execute code in a session (async + optional inline wait) |
| `run_notebook` | Execute notebook cells and persist outputs |

### Variable inspection
| Tool | Description |
|------|-------------|
| `get_variable` | Inspect a single variable (synchronous) |
| `list_variables` | List all user-defined variables (synchronous) |

### Operation management
| Tool | Description |
|------|-------------|
| `get_operation` | Poll operation status / wait for result |
| `cancel_operation` | Request cancellation |

## Architecture

```
jupyter_mcp/
├── __init__.py      Shared helpers and constants
├── server.py        FastMCP instance + all 22 tool definitions
├── kernel.py        KernelProvider ABC + LocalKernelProvider
├── notebooks.py     NotebookStore ABC + FileNotebookStore
├── operations.py    OperationRecord + OperationManager
└── orchestrator.py  ExecutionOrchestrator
```

`KernelProvider` and `NotebookStore` are abstract base classes — swap in custom backends for remote kernels or alternative storage.

## Development

```bash
pip install -e .
python -m pytest tests/ -v
```

## License

MIT
