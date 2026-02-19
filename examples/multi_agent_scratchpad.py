#!/usr/bin/env python3
"""
Multi-Agent Scratchpad Example

Demonstrates two agents sharing state through the MCP scratchpad server:

  Stage 1 – Researcher agent
    Analyses a topic using the calculator and writes structured findings to
    the shared scratchpad.

  Stage 2 – Writer agent
    Reads the Researcher's findings from the scratchpad and produces a
    polished summary report.

The key insight: agents never pass data to each other directly.  They
communicate exclusively through the MCP scratchpad—the same pattern you would
use in a real distributed multi-agent system.

Architecture:

    ┌────────────────────────────────────────────────────────┐
    │  Main process                                          │
    │                                                        │
    │  MCPToolExecutor (one shared instance)                 │
    │      │  stdio transport                                │
    │      ▼                                                 │
    │  mcp_scratchpad_server.py  (subprocess)                │
    │      shared _scratchpad dict                           │
    │                                                        │
    │  run_agent(ResearcherConfig) ──writes──▶ scratchpad    │
    │  run_agent(WriterConfig)    ──reads──▶  scratchpad     │
    └────────────────────────────────────────────────────────┘

Run:
    OPENAI_API_KEY=<key> python examples/multi_agent_scratchpad.py
"""

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agentic_loop import AgentConfig, run_agent
from src.tools import (
    SCRATCHPAD_TOOLS,
    create_builtin_registry,
    MCPToolExecutor,
)

# Path to the MCP server script (relative to project root)
MCP_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "mcp_scratchpad_server.py"
)


def make_combined_executor(
    builtin_registry,
    mcp_executor: MCPToolExecutor,
) -> callable:
    """
    Return a single executor function that routes tool calls to either the
    built-in registry or the MCP scratchpad server.
    """
    scratchpad_tool_names = {t["name"] for t in SCRATCHPAD_TOOLS}

    def execute(tool_name: str, arguments: dict) -> str:
        if tool_name in scratchpad_tool_names:
            return mcp_executor.call(tool_name, arguments)
        return builtin_registry.execute(tool_name, arguments)

    return execute


def main() -> None:
    builtin_registry = create_builtin_registry()

    # All tools available to agents that need both built-ins and scratchpad.
    all_tools = builtin_registry.get_schemas() + SCRATCHPAD_TOOLS

    # Open a single connection to the MCP scratchpad server for both agents.
    with MCPToolExecutor(MCP_SERVER_SCRIPT) as mcp:
        executor = make_combined_executor(builtin_registry, mcp)

        # ── Stage 1: Researcher ───────────────────────────────────────────
        researcher_config = AgentConfig(
            name="Researcher",
            system_prompt=(
                "You are a research analyst. Your job is to analyse a topic, "
                "perform any required calculations with the calculator tool, "
                "and save structured findings to the shared scratchpad so that "
                "a writer agent can use them later.\n\n"
                "Save your work under the key 'research_findings'. "
                "Structure your findings with clear sections: "
                "Background, Key Numbers, and Key Insights."
            ),
            tools=all_tools,
            model="gpt-4o-mini",
            max_iterations=10,
            verbose=True,
        )

        research_task = (
            "Research the topic: compound interest.\n"
            "1. Calculate how much £1,000 grows to after 10 years at 7% per year "
            "   compounded annually (use the formula: P * (1 + r)^n).\n"
            "2. Calculate the same for 20 years.\n"
            "3. Calculate the total interest earned in each case.\n"
            "Write your findings to the scratchpad key 'research_findings'."
        )

        print("\n" + "=" * 60)
        print("STAGE 1: Researcher agent")
        print("=" * 60)
        research_result = run_agent(
            config=researcher_config,
            initial_message=research_task,
            tool_executor=executor,
        )

        # ── Stage 2: Writer ───────────────────────────────────────────────
        writer_config = AgentConfig(
            name="Writer",
            system_prompt=(
                "You are a financial writer. Read the research findings from "
                "the shared scratchpad (key 'research_findings') and write a "
                "clear, engaging summary suitable for a general audience. "
                "After writing the summary, save it to the scratchpad under "
                "the key 'final_report'."
            ),
            tools=SCRATCHPAD_TOOLS,  # Writer only needs the scratchpad
            model="gpt-4o-mini",
            max_iterations=10,
            verbose=True,
        )

        writer_task = (
            "Read the research findings from the scratchpad and write a "
            "polished, reader-friendly summary report about compound interest. "
            "Save the final report to the scratchpad key 'final_report'."
        )

        print("\n" + "=" * 60)
        print("STAGE 2: Writer agent")
        print("=" * 60)
        writer_result = run_agent(
            config=writer_config,
            initial_message=writer_task,
            tool_executor=executor,
        )

        # ── Read back the final report from the scratchpad ────────────────
        final_report = mcp.call("scratchpad_read", {"key": "final_report"})

        print("\n" + "=" * 60)
        print("FINAL REPORT (from shared scratchpad)")
        print("=" * 60)
        print(final_report)

        # Show everything that ended up in the scratchpad.
        print("\n" + "=" * 60)
        print("SCRATCHPAD CONTENTS")
        print("=" * 60)
        print(mcp.call("scratchpad_list", {}))


if __name__ == "__main__":
    main()
