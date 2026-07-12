"""Tests for Session title/summary generation."""

import pytest
from unittest.mock import MagicMock, patch
from agent.config import AppConfig
from agent.run_agent import Agent

@pytest.fixture
def agent_config():
    return AppConfig()

@pytest.fixture
def agent(agent_config):
    return Agent(agent_config)

class TestSessionSummaryGeneration:
    def test_generate_summary_no_messages(self, agent):
        # Should return None when there are no messages
        assert agent.generate_summary() is None

    @patch("openai.resources.chat.completions.Completions.create")
    def test_generate_summary_with_messages(self, mock_create, agent):
        # Mock OpenAI client initialization to avoid missing credentials error in test
        agent._client = MagicMock()
        agent._client.chat.completions.create = mock_create

        # Mock chat response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Awesome Python Session"))
        ]
        mock_create.return_value = mock_response

        # Add message history
        agent.messages.append({"role": "user", "content": "How do I reverse a list in Python?"})
        agent.messages.append({"role": "assistant", "content": "You can use list.reverse() or slicing like [::-1]."})

        # Call generate summary
        summary = agent.generate_summary()

        assert summary == "Awesome Python Session"
        # Verify it passed messages to completions.create
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == agent.config.model.auxiliary
        assert len(call_kwargs["messages"]) == 2
        assert "summarize" in call_kwargs["messages"][0]["content"].lower()
