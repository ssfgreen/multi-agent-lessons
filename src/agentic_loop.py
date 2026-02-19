"""
Agentic Loop

Core implementation of a tool-using agent loop powered by LiteLLM.

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
from typing import Callable, Any
import json

from litellm import completion


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single agent run."""

    name: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    max_iterations: int = 10
    verbose: bool = True


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_agent(
    config: AgentConfig,
    initial_message: str,
    tool_executor: Callable[[str, dict], str],
) -> str:
    """
    Run a single agent to completion.

    Args:
        config:           Agent configuration (name, system prompt, tools, …).
        initial_message:  The first user message to start the conversation.
        tool_executor:    Callable(tool_name, arguments) → result_string.
                          Called for every tool_use block the model emits.

    Returns:
        The final text response produced by the model.
    """
    messages: list[dict] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": initial_message})

    _log(config, f"Starting | task: {initial_message[:120]}")

    for iteration in range(1, config.max_iterations + 1):
        _log(config, f"Iteration {iteration}/{config.max_iterations}")

        request: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
        }
        if config.tools:
            request["tools"] = _normalize_tools(config.tools)
            request["tool_choice"] = "auto"

        response = completion(**request)
        response_message = response.choices[0].message
        assistant_msg = _normalize_assistant_message(response_message)

        # Append the assistant turn to history.
        messages.append(assistant_msg)

        tool_calls = _extract_tool_calls(response_message)
        if tool_calls:
            for tool_call in tool_calls:
                tool_name, arguments, tool_call_id = _parse_tool_call(tool_call)
                _log(config, f"Tool call: {tool_name}({arguments})")
                result = tool_executor(tool_name, arguments)
                _log(config, f"Tool result: {str(result)[:200]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(result),
                    }
                )
            continue

        # ── Done ──────────────────────────────────────────────────────────
        final_text = _extract_text(response_message)
        _log(config, f"Final response: {final_text[:200]}")
        return final_text

    _log(config, "Max iterations reached")
    return "Max iterations reached without completing the task."


# ---------------------------------------------------------------------------
# Multi-agent orchestration helper
# ---------------------------------------------------------------------------

def run_pipeline(
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
        result = run_agent(config, message, executor)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(config: AgentConfig, msg: str) -> None:
    if config.verbose:
        print(f"[{config.name}] {msg}")


def _normalize_assistant_message(message: Any) -> dict:
    """Return a dict suitable to append to the messages list."""
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if isinstance(message, dict):
        return message
    # Fallback for unknown message types
    return {"role": "assistant", "content": getattr(message, "content", None)}


def _extract_tool_calls(message: Any) -> list:
    if hasattr(message, "tool_calls"):
        return message.tool_calls or []
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return tool_calls
        function_call = message.get("function_call")
        return [function_call] if function_call else []
    if hasattr(message, "function_call") and message.function_call:
        return [message.function_call]
    return []


def _parse_tool_call(tool_call: Any) -> tuple[str, dict, str]:
    # Supports both object-style and dict-style tool call payloads.
    if hasattr(tool_call, "function"):
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        tool_call_id = tool_call.id
    elif "function" in tool_call:
        tool_name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]
        tool_call_id = tool_call.get("id")
    elif "name" in tool_call and "arguments" in tool_call:
        tool_name = tool_call["name"]
        raw_args = tool_call["arguments"]
        tool_call_id = tool_call.get("id")
    else:
        raise ValueError(f"Unrecognized tool call payload: {tool_call}")
    if not tool_call_id:
        tool_call_id = f"tool_call_{id(tool_call)}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        arguments = {}
    return tool_name, arguments, tool_call_id


def _extract_text(message: Any) -> str:
    if hasattr(message, "content"):
        return message.content or ""
    if isinstance(message, dict):
        return message.get("content") or ""
    return ""


def _normalize_tools(tools: list[dict]) -> list[dict]:
    """
    Convert Anthropic-style tool schemas to OpenAI-style function tools.

    Supports:
      - {"name": ..., "description": ..., "input_schema": {...}}
      - already OpenAI-style {"type": "function", "function": {...}}
    """
    normalized: list[dict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            normalized.append(tool)
            continue
        if "name" in tool and "input_schema" in tool:
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            )
            continue
        normalized.append(tool)
    return normalized
