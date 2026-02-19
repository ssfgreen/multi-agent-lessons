import os

from src.agentic_loop import AgentConfig, run_agent


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeToolFunction(name, arguments)


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def test_run_agent_works_with_openai_key(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "sk-test-openai"

    calls = {"count": 0}

    def fake_completion(**kwargs):
        # Ensure tools are normalized into OpenAI-style function tools
        tools = kwargs.get("tools")
        assert tools and tools[0]["type"] == "function"
        assert "parameters" in tools[0]["function"]

        calls["count"] += 1
        if calls["count"] == 1:
            tool_call = _FakeToolCall("calculator", "{\"expression\": \"2+2\"}")
            return _FakeResponse(_FakeMessage(content=None, tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="Final answer"))

    monkeypatch.setattr("src.agentic_loop.completion", fake_completion)

    config = AgentConfig(
        name="TestAgent",
        system_prompt="System prompt",
        tools=[
            {
                "name": "calculator",
                "description": "calc",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ],
        model="gpt-4o-mini",
        max_iterations=3,
        verbose=False,
    )

    def tool_exec(name, arguments):
        assert name == "calculator"
        assert arguments == {"expression": "2+2"}
        return "4"

    result = run_agent(config, "do math", tool_exec)
    assert result == "Final answer"


def test_run_agent_works_with_anthropic_key(monkeypatch):
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-anthropic"

    def fake_completion(**kwargs):
        assert kwargs["model"].startswith("anthropic/")
        return _FakeResponse(_FakeMessage(content="Anthropic OK"))

    monkeypatch.setattr("src.agentic_loop.completion", fake_completion)

    config = AgentConfig(
        name="TestAgent",
        system_prompt="System prompt",
        tools=[],
        model="anthropic/claude-3-5-sonnet-20240620",
        max_iterations=1,
        verbose=False,
    )

    result = run_agent(config, "hello", lambda *_: "")
    assert result == "Anthropic OK"
