"""
Tool Definitions and Registry

Provides:
  - Built-in tool schemas and executors (calculator, get_current_time).
  - Scratchpad tool schemas to pass to the model (the actual execution is
    routed to the MCP scratchpad server via MCPToolExecutor).
  - ToolRegistry: a lightweight container that maps tool names to executors.
  - MCPToolExecutor: connects to the MCP scratchpad server and executes tools
    against it synchronously.
"""

from __future__ import annotations

import asyncio
import math
import sys
from datetime import datetime
from typing import Callable, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Built-in tool schemas (Anthropic-style tool format)
# ---------------------------------------------------------------------------

CALCULATOR_TOOL: dict = {
    "name": "calculator",
    "description": (
        "Evaluate a mathematical expression. "
        "Supports arithmetic operators and functions from Python's math module "
        "(e.g. sqrt, sin, cos, log, pi, e)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
            }
        },
        "required": ["expression"],
    },
}

GET_TIME_TOOL: dict = {
    "name": "get_current_time",
    "description": "Return the current date and time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": (
                    "strftime format string (optional). "
                    "Defaults to ISO-8601 if omitted."
                ),
            }
        },
    },
}

# ---------------------------------------------------------------------------
# Scratchpad tool schemas
# These are passed to the model so it knows how to invoke the MCP server.
# ---------------------------------------------------------------------------

SCRATCHPAD_TOOLS: list[dict] = [
    {
        "name": "scratchpad_write",
        "description": (
            "Write a value to the shared scratchpad under a given key. "
            "Overwrites any existing content. Use to save notes or results "
            "that other agents will read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key name (e.g. 'findings', 'summary')",
                },
                "value": {"type": "string", "description": "Content to store"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "scratchpad_append",
        "description": (
            "Append content to an existing scratchpad key. "
            "Creates the key if it does not exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to append to"},
                "value": {"type": "string", "description": "Content to append"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "scratchpad_read",
        "description": "Read the content stored at a specific scratchpad key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to read"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "scratchpad_list",
        "description": "List all keys currently stored in the shared scratchpad.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Built-in tool executors
# ---------------------------------------------------------------------------

def execute_calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_globals.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, safe_globals)  # noqa: S307
        return f"{result}"
    except Exception as exc:
        return f"Error: {exc}"


def execute_get_time(format: str | None = None) -> str:  # noqa: A002
    now = datetime.now()
    return now.strftime(format) if format else now.isoformat()


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Maps tool names to (schema, executor) pairs.

    Usage::

        registry = ToolRegistry()
        registry.register(CALCULATOR_TOOL, execute_calculator)
        schemas = registry.get_schemas()          # pass to run_agent config.tools
        result  = registry.execute("calculator", {"expression": "2+2"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict, Callable[..., str]]] = {}

    def register(self, schema: dict, executor: Callable[..., str]) -> None:
        self._tools[schema["name"]] = (schema, executor)

    def get_schemas(self) -> list[dict]:
        return [schema for schema, _ in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"Unknown tool: '{name}'"
        _, executor = self._tools[name]
        try:
            return executor(**arguments)
        except Exception as exc:
            return f"Error executing '{name}': {exc}"


def create_builtin_registry() -> ToolRegistry:
    """Return a registry pre-loaded with the built-in tools."""
    registry = ToolRegistry()
    registry.register(CALCULATOR_TOOL, execute_calculator)
    registry.register(GET_TIME_TOOL, execute_get_time)
    return registry


# ---------------------------------------------------------------------------
# MCP tool executor
# ---------------------------------------------------------------------------

class MCPToolExecutor:
    """
    Connects to the MCP scratchpad server (via stdio) and executes tool calls
    against it synchronously from normal (non-async) code.

    The server is launched as a subprocess running `mcp_scratchpad_server.py`.

    Usage::

        with MCPToolExecutor("path/to/mcp_scratchpad_server.py") as mcp:
            result = mcp.call("scratchpad_write", {"key": "notes", "value": "hello"})
    """

    def __init__(self, server_script: str) -> None:
        self.server_script = server_script
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._stdio_ctx: Any = None
        self._session_ctx: Any = None

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> MCPToolExecutor:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        return self

    def __exit__(self, *_: Any) -> None:
        if self._loop:
            self._loop.run_until_complete(self._disconnect())
            self._loop.close()

    # ── public API ─────────────────────────────────────────────────────────

    def call(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return the text result."""
        if not self._session or not self._loop:
            raise RuntimeError("MCPToolExecutor is not connected (use as context manager)")
        result = self._loop.run_until_complete(
            self._session.call_tool(tool_name, arguments)
        )
        return "\n".join(
            item.text for item in result.content if hasattr(item, "text")
        )

    # ── async internals ────────────────────────────────────────────────────

    async def _connect(self) -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)
