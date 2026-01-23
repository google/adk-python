"""
Tests for prompt building.
"""

from adk_rlm.prompts import build_rlm_system_prompt
from adk_rlm.prompts import build_user_prompt
from adk_rlm.prompts import RLM_SYSTEM_PROMPT
from adk_rlm.types import QueryMetadata
import pytest


class TestBuildRLMSystemPrompt:
  """Tests for build_rlm_system_prompt."""

  def test_build_system_prompt(self):
    """Build with metadata."""
    metadata = QueryMetadata("test context")
    messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, metadata)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "assistant"
    assert "context" in messages[1]["content"].lower()

  def test_includes_context_length(self):
    """Includes context length info."""
    metadata = QueryMetadata("x" * 1000)
    messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, metadata)

    assert "1000" in messages[1]["content"]

  def test_custom_system_prompt(self):
    """Uses custom system prompt."""
    custom_prompt = "You are a custom assistant."
    metadata = QueryMetadata("test")
    messages = build_rlm_system_prompt(custom_prompt, metadata)

    assert messages[0]["content"] == custom_prompt


class TestBuildUserPrompt:
  """Tests for build_user_prompt."""

  def test_first_iteration(self):
    """First iteration prompt."""
    prompt = build_user_prompt(root_prompt=None, iteration=0)

    assert prompt["role"] == "user"
    assert (
        "haven't seen" in prompt["content"].lower()
        or "not interacted" in prompt["content"].lower()
    )

  def test_subsequent_iteration(self):
    """Later iteration prompt."""
    prompt = build_user_prompt(root_prompt=None, iteration=1)

    assert prompt["role"] == "user"
    assert "history" in prompt["content"].lower()

  def test_with_root_prompt(self):
    """Include root prompt."""
    prompt = build_user_prompt(root_prompt="What is the answer?", iteration=0)

    assert "What is the answer?" in prompt["content"]

  def test_multiple_contexts(self):
    """Notes multiple contexts."""
    prompt = build_user_prompt(root_prompt=None, iteration=1, context_count=3)

    assert "3 contexts" in prompt["content"]
    assert "context_0" in prompt["content"]
    assert "context_2" in prompt["content"]

  def test_with_histories(self):
    """Notes prior histories."""
    prompt = build_user_prompt(
        root_prompt=None, iteration=1, context_count=1, history_count=2
    )

    assert "2 prior conversation histories" in prompt["content"]
    assert "history_0" in prompt["content"]

  def test_single_history(self):
    """Notes single history differently."""
    prompt = build_user_prompt(
        root_prompt=None, iteration=1, context_count=1, history_count=1
    )

    assert "1 prior conversation history" in prompt["content"]
    assert (
        "history_0" not in prompt["content"]
    )  # Just mentions `history` variable
