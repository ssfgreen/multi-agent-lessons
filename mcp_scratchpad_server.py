#!/usr/bin/env python3
"""
MCP Scratchpad Server

A Model Context Protocol server that exposes a shared scratchpad for agents
to read and write during an agentic loop. Multiple agents can share state by
reading and writing named keys.

Run this server in one process; agents connect to it via stdio transport.

Usage:
    python mcp_scratchpad_server.py
"""

import asyncio
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp import types
import mcp.server.stdio

# ---------------------------------------------------------------------------
# In-memory shared scratchpad state
# ---------------------------------------------------------------------------
_scratchpad: dict[str, str] = {}

server = Server("scratchpad-server")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="scratchpad_write",
            description=(
                "Write a value to the shared scratchpad under a given key. "
                "Overwrites any existing content at that key."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Scratchpad key (e.g. 'notes', 'summary', 'todos')",
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        types.Tool(
            name="scratchpad_append",
            description=(
                "Append content to an existing scratchpad key. "
                "Creates the key if it does not exist yet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Scratchpad key to append to",
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to append",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        types.Tool(
            name="scratchpad_read",
            description="Read the content stored at a specific scratchpad key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Scratchpad key to read",
                    },
                },
                "required": ["key"],
            },
        ),
        types.Tool(
            name="scratchpad_list",
            description="List all keys currently stored in the shared scratchpad.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="scratchpad_clear",
            description=(
                "Clear a specific key from the scratchpad. "
                "If no key is provided, clears the entire scratchpad."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to clear (optional; omit to clear all)",
                    },
                },
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    if name == "scratchpad_write":
        key = arguments["key"]
        value = arguments["value"]
        _scratchpad[key] = value
        return [types.TextContent(
            type="text",
            text=f"Wrote {len(value)} chars to scratchpad['{key}']",
        )]

    if name == "scratchpad_append":
        key = arguments["key"]
        value = arguments["value"]
        _scratchpad[key] = _scratchpad.get(key, "") + value
        total = len(_scratchpad[key])
        return [types.TextContent(
            type="text",
            text=f"Appended to scratchpad['{key}'] (total: {total} chars)",
        )]

    if name == "scratchpad_read":
        key = arguments["key"]
        if key in _scratchpad:
            return [types.TextContent(type="text", text=_scratchpad[key])]
        available = list(_scratchpad.keys())
        return [types.TextContent(
            type="text",
            text=f"Key '{key}' not found. Available keys: {available}",
        )]

    if name == "scratchpad_list":
        if not _scratchpad:
            return [types.TextContent(type="text", text="Scratchpad is empty")]
        lines = "\n".join(
            f"  '{k}': {len(v)} chars" for k, v in _scratchpad.items()
        )
        return [types.TextContent(type="text", text=f"Scratchpad keys:\n{lines}")]

    if name == "scratchpad_clear":
        key = arguments.get("key")
        if key:
            if key in _scratchpad:
                del _scratchpad[key]
                return [types.TextContent(
                    type="text", text=f"Cleared scratchpad['{key}']"
                )]
            return [types.TextContent(
                type="text", text=f"Key '{key}' not found"
            )]
        _scratchpad.clear()
        return [types.TextContent(type="text", text="Cleared all scratchpad entries")]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scratchpad-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
