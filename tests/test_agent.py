"""Tests for Agent streaming response."""

import json
from unittest.mock import MagicMock, patch
import pytest
from agent.config import AppConfig
from agent.run_agent import Agent

@pytest.fixture
def agent_config():
    return AppConfig()

@pytest.fixture
def agent(agent_config):
    agent = Agent(agent_config)
    agent._system_prompt_snapshot = "System Prompt"
    return agent

class MockChunkChoiceDeltaFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments

class MockChunkChoiceDeltaToolCall:
    def __init__(self, index, id=None, name=None, arguments=None, type="function"):
        self.index = index
        self.id = id
        self.type = type
        self.function = MockChunkChoiceDeltaFunction(name, arguments)

class MockChunkChoiceDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class MockChunkChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason

class MockChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage

class TestAgentStreaming:
    @patch("openai.OpenAI")
    def test_run_turn_streaming_text(self, mock_openai, agent):
        # Setup mock chunks for text generation
        chunks = [
            MockChunk([MockChunkChoice(MockChunkChoiceDelta(content="Hello "))]),
            MockChunk([MockChunkChoice(MockChunkChoiceDelta(content="world!"))]),
            MockChunk([], usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        ]

        # Configure mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = chunks

        # Call run_turn with on_chunk callback
        streamed_chunks = []
        def on_chunk(chunk):
            streamed_chunks.append(chunk)

        response = agent.run_turn("Hi", on_chunk=on_chunk)

        assert response == "Hello world!"
        assert streamed_chunks == ["Hello ", "world!"]
        assert len(agent.messages) == 2
        assert agent.messages[0] == {"role": "user", "content": "Hi"}
        assert agent.messages[1] == {"role": "assistant", "content": "Hello world!"}

        # Verify usage was updated
        assert agent.compressor.last_total_tokens == 15

    @patch("openai.OpenAI")
    def test_run_turn_streaming_tool_call(self, mock_openai, agent):
        # Setup mock chunks for tool call first, then text response
        tool_chunks = [
            MockChunk([MockChunkChoice(MockChunkChoiceDelta(tool_calls=[
                MockChunkChoiceDeltaToolCall(index=0, id="call-1", name="dummy_tool", arguments='{"value":')
            ]))]),
            MockChunk([MockChunkChoice(MockChunkChoiceDelta(tool_calls=[
                MockChunkChoiceDeltaToolCall(index=0, arguments=' 42}')
            ]))])
        ]

        text_chunks = [
            MockChunk([MockChunkChoice(MockChunkChoiceDelta(content="Tool executed, answer is 42."))])
        ]

        # Mock handler
        mock_handler = MagicMock(return_value="Success")
        agent.register_tool(
            name="dummy_tool",
            description="Dummy description",
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
            handler=mock_handler
        )

        # Configure mock client behavior
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [tool_chunks, text_chunks]

        # Call run_turn
        streamed_chunks = []
        def on_chunk(chunk):
            streamed_chunks.append(chunk)

        response = agent.run_turn("Run dummy", on_chunk=on_chunk)

        assert response == "Tool executed, answer is 42."
        assert streamed_chunks == ["Tool executed, answer is 42."]

        # Check messages history structure
        # Message 0: User input
        # Message 1: Assistant tool call
        # Message 2: Tool result
        # Message 3: Assistant text response
        assert len(agent.messages) == 4
        assert agent.messages[0] == {"role": "user", "content": "Run dummy"}
        assert agent.messages[1]["role"] == "assistant"
        assert agent.messages[1]["tool_calls"][0]["function"]["name"] == "dummy_tool"
        assert json.loads(agent.messages[1]["tool_calls"][0]["function"]["arguments"]) == {"value": 42}
        assert agent.messages[2] == {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Success"
        }
        assert agent.messages[3] == {"role": "assistant", "content": "Tool executed, answer is 42."}

        # Verify tool was actually called with arguments parsed correctly
        mock_handler.assert_called_once_with(value=42)
