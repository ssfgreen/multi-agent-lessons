#!/usr/bin/env python3
"""
Single Agent Example

Demonstrates the core agentic loop with built-in tools (calculator and clock).
The agent is given a task that requires multiple tool calls before it can
produce a final answer.

Run:
    ANTHROPIC_API_KEY=<key> python examples/single_agent.py
"""

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from src.agentic_loop import AgentConfig, run_agent
from src.tools import create_builtin_registry


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build a registry that holds the built-in tools.
    registry = create_builtin_registry()

    config = AgentConfig(
        name="MathAgent",
        system_prompt=(
            "You are a helpful assistant with access to a calculator and a clock. "
            "Always use the calculator tool for any arithmetic rather than computing "
            "in your head, even for simple sums. Show your working step by step."
        ),
        tools=registry.get_schemas(),
        model="claude-opus-4-6",
        max_iterations=10,
        verbose=True,
    )

    task = (
        "What is (137 * 48) + sqrt(1764), and what is today's date? "
        "Please calculate both and give me a final combined answer."
    )

    result = run_agent(
        client=client,
        config=config,
        initial_message=task,
        tool_executor=registry.execute,
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
