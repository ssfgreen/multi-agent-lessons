# Multi-Agent Lessons: Agentic Loop with MCP Scratchpad

A minimal, readable implementation of an **agentic loop** that supports tool
use and a **shared scratchpad** via the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/).
The LLM interface is provided by LiteLLM.

---

## Concepts

### Agentic loop

An agentic loop repeatedly calls the model until it produces a final answer:

```
User message
     │
     ▼
┌──────────────────────────┐
│  LLM via LiteLLM         │◄──────────────────────┐
└──────────────────────────┘                       │
         │ response                               │
         ├── final text ──► return text           │
         └── tool calls ──► execute tools ────────┘
```

### Shared scratchpad via MCP

Agents share state through an **MCP scratchpad server** running as a
subprocess.  Neither agent knows about the other; they communicate only through
named keys in the scratchpad.

```
┌─────────────────────────────────────────────────────┐
│  Main process                                       │
│                                                     │
│  MCPToolExecutor ──stdio──► mcp_scratchpad_server   │
│                               shared _scratchpad    │
│                                                     │
│  Researcher agent ──writes──► scratchpad            │
│  Writer agent     ──reads───► scratchpad            │
└─────────────────────────────────────────────────────┘
```

---

## Project layout

```
multi-agent-lessons/
├── mcp_scratchpad_server.py   # MCP server (scratchpad tools)
├── requirements.txt
├── src/
│   ├── agentic_loop.py        # Core loop + pipeline helper (LiteLLM)
│   └── tools.py               # Tool schemas, executors, registry, MCP client
└── examples/
    ├── single_agent.py        # One agent, built-in tools only
    └── multi_agent_scratchpad.py  # Two agents sharing a scratchpad via MCP
```

---

## Setup

```bash
pip install -r requirements.txt

# Set the API key for the provider you want to use (LiteLLM-compatible).
# Example for OpenAI:
export OPENAI_API_KEY="sk-..."
```

---

## Running the examples

### Single agent with built-in tools

```bash
python examples/single_agent.py
```

The agent uses a **calculator** and **clock** to answer a maths question that
requires several tool calls.

### Multi-agent pipeline with shared scratchpad

```bash
python examples/multi_agent_scratchpad.py
```

Two agents run sequentially:

| Agent | Task |
|---|---|
| **Researcher** | Calculates compound-interest figures and writes findings to `scratchpad["research_findings"]` |
| **Writer** | Reads from the scratchpad and produces a polished report saved to `scratchpad["final_report"]` |

---

## Key files explained

### `mcp_scratchpad_server.py`

A standalone MCP server that exposes five tools:

| Tool | Description |
|---|---|
| `scratchpad_write` | Write (overwrite) a value at a key |
| `scratchpad_append` | Append to an existing key |
| `scratchpad_read` | Read a key |
| `scratchpad_list` | List all keys |
| `scratchpad_clear` | Clear one key or all keys |

Run it directly (`python mcp_scratchpad_server.py`) for development / manual
testing with any MCP client, or let `MCPToolExecutor` launch it automatically.

### `src/agentic_loop.py`

- **`AgentConfig`** – dataclass holding name, system prompt, tool schemas, model, etc.
- **`run_agent`** – runs the loop for a single agent.
- **`run_pipeline`** – convenience wrapper for sequential multi-agent pipelines.

Model selection is driven by `AgentConfig.model` and uses LiteLLM model
strings (e.g. `gpt-4o-mini`, `anthropic/claude-3-5-sonnet-20240620`, etc.).

### `src/tools.py`

- **`create_builtin_registry()`** – returns a `ToolRegistry` pre-loaded with
  `calculator` and `get_current_time`.
- **`SCRATCHPAD_TOOLS`** – list of Anthropic-style tool schemas for the five
  scratchpad operations.  Pass these to `AgentConfig.tools` to let an agent
  use the scratchpad. They are normalized to OpenAI-style function tools in
  `run_agent`.
- **`MCPToolExecutor`** – context manager that launches the MCP server
  subprocess and routes tool calls to it synchronously.

---

## Extending

**Add a new built-in tool:**

```python
from src.tools import ToolRegistry

MY_TOOL = {
    "name": "my_tool",
    "description": "...",
    "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
}

def execute_my_tool(x: str) -> str:
    return f"processed: {x}"

registry = create_builtin_registry()
registry.register(MY_TOOL, execute_my_tool)
```

**Add more scratchpad keys / structure:** just agree on key names between
agents in their system prompts—no code changes needed.

**Run agents in parallel:** launch multiple `run_agent` calls in separate
threads, all sharing the same `MCPToolExecutor` instance (the MCP server is
single-process but handles sequential requests correctly).
