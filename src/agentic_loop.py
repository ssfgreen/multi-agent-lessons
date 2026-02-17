"""
Agentic Loop

Core implementation of a tool-using agent loop powered by the Anthropic API.

The loop works as follows:
  1. Send the current message history to the model.
  2. If stop_reason == "end_turn"  → return the final text response.
  3. If stop_reason == "tool_use"  → execute each requested tool, append the
     results, and go back to step 1.
  4. Stop after `max_iterations` to prevent runaway loops.

Agents are independent instances of this loop. They share state only through
external mechanisms—in these examples, through the MCP scratchpad server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import anthropic


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single agent run."""

    name: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
    model: str = "claude-opus-4-6"
    max_tokens: int = 4096
    max_iterations: int = 10
    verbose: bool = True


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_agent(
    client: anthropic.Anthropic,
    config: AgentConfig,
    initial_message: str,
    tool_executor: Callable[[str, dict], str],
) -> str:
    """
    Run a single agent to completion.

    Args:
        client:           Anthropic API client.
        config:           Agent configuration (name, system prompt, tools, …).
        initial_message:  The first user message to start the conversation.
        tool_executor:    Callable(tool_name, arguments) → result_string.
                          Called for every tool_use block the model emits.

    Returns:
        The final text response produced by the model.
    """
    messages: list[dict] = [{"role": "user", "content": initial_message}]

    _log(config, f"Starting | task: {initial_message[:120]}")

    for iteration in range(1, config.max_iterations + 1):
        _log(config, f"Iteration {iteration}/{config.max_iterations}")

        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=config.system_prompt,
            tools=config.tools,
            messages=messages,
        )

        _log(config, f"stop_reason={response.stop_reason}")

        # Append the assistant turn to history.
        messages.append({"role": "assistant", "content": response.content})

        # ── Done ──────────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    _log(config, f"Final response: {block.text[:200]}")
                    return block.text
            return ""

        # ── Tool use ──────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    _log(config, f"Tool call: {block.name}({block.input})")
                    result = tool_executor(block.name, block.input)
                    _log(config, f"Tool result: {str(result)[:200]}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason – bail out.
        _log(config, f"Unexpected stop_reason: {response.stop_reason}. Stopping.")
        break

    _log(config, "Max iterations reached")
    return "Max iterations reached without completing the task."


# ---------------------------------------------------------------------------
# Multi-agent orchestration helper
# ---------------------------------------------------------------------------

def run_pipeline(
    client: anthropic.Anthropic,
    stages: list[tuple[AgentConfig, str, Callable[[str, dict], str]]],
) -> list[str]:
    """
    Run a sequential pipeline of agents.

    Each stage is a tuple of (config, initial_message, tool_executor).
    Agents run one after the other; they communicate via shared external state
    (e.g. the MCP scratchpad) rather than by passing return values directly.

    Returns:
        List of final responses, one per stage.
    """
    results: list[str] = []
    for i, (config, message, executor) in enumerate(stages, 1):
        print(f"\n{'='*60}")
        print(f"Pipeline stage {i}/{len(stages)}: {config.name}")
        print("=" * 60)
        result = run_agent(client, config, message, executor)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(config: AgentConfig, msg: str) -> None:
    if config.verbose:
        print(f"[{config.name}] {msg}")
