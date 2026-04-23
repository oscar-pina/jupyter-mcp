# jupyter-mcp

An agent-first MCP server for Jupyter execution and notebook operations.

## What it does

`jupyter-mcp` exposes 16 tools to an AI agent via the [Model Context Protocol](https://modelcontextprotocol.io):

- Create and manage kernel sessions (any Python interpreter)
- Execute code and get results synchronously or poll asynchronously
- Read and edit notebooks with optimistic concurrency (revision tokens)
- Run full notebooks and persist outputs back to the file

## Install

### Option A — Global (recommended)

Available in every project, no cloning required. Run once:

```bash
claude mcp add jupyter-mcp -s user -- uvx --from "git+https://github.com/oscar-pina/jupyter-mcp" jupyter-mcp
```

`uvx` fetches the package from GitHub and installs it in an isolated cache it manages. Nothing lands in your project or system Python.

Verify:

```bash
claude mcp list
# jupyter-mcp  ...  connected
```

### Option B — Project-level

Registers the MCP server only for the current project (writes to `.mcp.json`):

```bash
claude mcp add jupyter-mcp -- uvx --from "git+https://github.com/oscar-pina/jupyter-mcp" jupyter-mcp
```

Colleagues who clone the project get the same configuration automatically when they open it in Claude Code.

### Option C — Development (contributing to jupyter-mcp)

```bash
git clone https://github.com/oscar-pina/jupyter-mcp.git
cd jupyter-mcp
uv sync
```

`uv sync` creates a `.venv` inside the repo and installs the package and its dependencies into it. Nothing touches system Python.

Then activate the venv and open the directory in Claude Code. The `.mcp.json` at the repo root uses `python`, which resolves to the active venv:

```bash
source .venv/bin/activate
claude  # or open Claude Code from this shell
```

Run tests:

```bash
uv run pytest tests/ -v
```

## Using a project venv

By default, the kernel runs using the `python` command on your PATH. To run code against a specific project venv instead, pass `python_path` when creating a session:

```
create_session(python_path="/path/to/your-project/.venv/bin/python")
```

**Requirement:** `ipykernel` must be installed in that venv:

```bash
pip install ipykernel
# or: uv pip install ipykernel
```

### Why this works this way

There are three separate processes involved:

```
Claude Code  ──MCP──►  jupyter-mcp server  ──spawns──►  Jupyter kernel
(client)                (manages sessions,               (runs your code,
                         notebook I/O)                    your project's venv)
```

The MCP server and the kernel are separate processes. `python_path` is the seam between them — it tells the server which Python interpreter to use when spawning the kernel. Your project packages (pandas, torch, etc.) only need to be in the kernel's environment, not the server's.

## Quick start

```
1. create_session()                       → { session_id: "sess_abc123" }
   Or: create_session(python_path="/path/to/.venv/bin/python")
2. run_code(session_id, "1 + 1", wait_ms=5000)
   → { status: "completed", result: { stdout: "", execution_count: 1 } }
3. close_session(session_id)
```

Notebook editing uses optimistic concurrency — every mutation requires the `revision` token from `read_notebook`. On conflict (file changed externally) you get error code `"Conflict"`: re-read and retry.

## Tool reference

### Session management
| Tool | Description |
|------|-------------|
| `create_session` | Start a kernel session |
| `list_sessions` | List all active sessions |
| `close_session` | Shut down a session |
| `restart_session` | Restart kernel, keeping session ID |
| `interrupt_session` | Interrupt current execution without resetting state |

### Notebook operations
| Tool | Description |
|------|-------------|
| `list_notebooks` | Recursively list `.ipynb` files |
| `create_notebook` | Create a new empty notebook |
| `rename_notebook` | Rename or move notebook (revision-guarded) |
| `delete_notebook` | Delete (revision-guarded) |
| `read_notebook` | Read cells and outputs; returns revision |
| `edit_notebook` | Atomic insert/update/delete cell operations |

### Execution
| Tool | Description |
|------|-------------|
| `run_code` | Execute code in a session (async + optional inline wait) |
| `run_notebook` | Execute notebook cells and persist outputs |

### Operation management
| Tool | Description |
|------|-------------|
| `get_operation` | Poll operation status / wait for result |
| `cancel_operation` | Request cancellation |
| `list_operations` | List all tracked operations |

## Architecture

```
jupyter_mcp/
├── __init__.py      Shared helpers and constants
├── server.py        FastMCP instance + all 16 tool definitions
├── kernel.py        KernelProvider ABC + LocalKernelProvider
├── notebooks.py     NotebookStore ABC + FileNotebookStore
├── operations.py    OperationRecord + OperationManager
└── orchestrator.py  ExecutionOrchestrator
```

`KernelProvider` and `NotebookStore` are abstract base classes — swap in custom backends for remote kernels or alternative storage.

## License

MIT — see [LICENSE](LICENSE).
