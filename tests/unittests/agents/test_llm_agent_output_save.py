# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LlmAgent output saving functionality."""

import logging
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from pydantic import BaseModel
import pytest


class MockOutputSchema(BaseModel):
  message: str
  confidence: float


def create_test_event(
    author: str = "test_agent",
    content_text: str = "Hello world",
    is_final: bool = True,
    invocation_id: str = "test_invocation",
) -> Event:
  """Helper to create test events."""
  # Create mock content
  parts = [types.Part.from_text(text=content_text)] if content_text else []
  content = types.Content(role="model", parts=parts) if parts else None

  # Create event
  event = Event(
      invocation_id=invocation_id,
      author=author,
      content=content,
      actions=EventActions(),
  )

  # Mock is_final_response if needed
  if not is_final:
    event.partial = True

  return event


class TestLlmAgentOutputSave:
  """Test suite for LlmAgent output saving functionality."""

  def test_maybe_save_output_to_state_skips_different_author(self, caplog):
    """Test that output is not saved when event author differs from agent name."""
    # Set the LlmAgent logger to DEBUG level
    llm_agent_logger = logging.getLogger(
        "google_adk.google.adk.agents.llm_agent"
    )
    original_level = llm_agent_logger.level
    llm_agent_logger.setLevel(logging.DEBUG)

    try:
      agent = LlmAgent(name="agent_a", output_key="result")
      event = create_test_event(
          author="agent_b", content_text="Response from B"
      )

      with caplog.at_level("DEBUG"):
        agent._LlmAgent__maybe_save_output_to_state(event)

      # Should not add anything to state_delta
      assert len(event.actions.state_delta) == 0

      # Should log the skip
      assert (
          "Skipping output save for agent agent_a: event authored by agent_b"
          in caplog.text
      )
    finally:
      # Restore original logger level
      llm_agent_logger.setLevel(original_level)

  def test_maybe_save_output_to_state_saves_same_author(self):
    """Test that output is saved when event author matches agent name."""
    agent = LlmAgent(name="test_agent", output_key="result")
    event = create_test_event(author="test_agent", content_text="Test response")

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should save to state_delta
    assert event.actions.state_delta["result"] == "Test response"

  def test_maybe_save_output_to_state_no_output_key(self):
    """Test that nothing is saved when output_key is not set."""
    agent = LlmAgent(name="test_agent")  # No output_key
    event = create_test_event(author="test_agent", content_text="Test response")

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should not save anything
    assert len(event.actions.state_delta) == 0

  def test_maybe_save_output_to_state_not_final_response(self):
    """Test that output is not saved for non-final responses."""
    agent = LlmAgent(name="test_agent", output_key="result")
    event = create_test_event(
        author="test_agent", content_text="Partial response", is_final=False
    )

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should not save partial responses
    assert len(event.actions.state_delta) == 0

  def test_maybe_save_output_to_state_no_content(self):
    """Test that nothing is saved when event has no content."""
    agent = LlmAgent(name="test_agent", output_key="result")
    event = create_test_event(author="test_agent", content_text="")

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should not save empty content
    assert len(event.actions.state_delta) == 0

  def test_maybe_save_output_to_state_with_output_schema(self):
    """Test that output is processed with schema when output_schema is set."""
    agent = LlmAgent(
        name="test_agent", output_key="result", output_schema=MockOutputSchema
    )

    # Create event with JSON content
    json_content = '{"message": "Hello", "confidence": 0.95}'
    event = create_test_event(author="test_agent", content_text=json_content)

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should save parsed and validated output
    expected_output = {"message": "Hello", "confidence": 0.95}
    assert event.actions.state_delta["result"] == expected_output

  def test_maybe_save_output_to_state_multiple_parts(self):
    """Test that multiple text parts are concatenated."""
    agent = LlmAgent(name="test_agent", output_key="result")

    # Create event with multiple text parts
    parts = [
        types.Part.from_text(text="Hello "),
        types.Part.from_text(text="world"),
        types.Part.from_text(text="!"),
    ]
    content = types.Content(role="model", parts=parts)

    event = Event(
        invocation_id="test_invocation",
        author="test_agent",
        content=content,
        actions=EventActions(),
    )

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should concatenate all text parts
    assert event.actions.state_delta["result"] == "Hello world!"

  def test_maybe_save_output_to_state_agent_transfer_scenario(self, caplog):
    """Test realistic agent transfer scenario."""
    # Scenario: Agent A transfers to Agent B, Agent B produces output
    # Agent A should not save Agent B's output

    # Set the LlmAgent logger to DEBUG level
    llm_agent_logger = logging.getLogger(
        "google_adk.google.adk.agents.llm_agent"
    )
    original_level = llm_agent_logger.level
    llm_agent_logger.setLevel(logging.DEBUG)

    try:
      agent_a = LlmAgent(name="support_agent", output_key="support_result")
      agent_b_event = create_test_event(
          author="billing_agent", content_text="Your bill is $100"
      )

      with caplog.at_level("DEBUG"):
        agent_a._LlmAgent__maybe_save_output_to_state(agent_b_event)

      # Agent A should not save Agent B's output
      assert len(agent_b_event.actions.state_delta) == 0
      assert (
          "Skipping output save for agent support_agent: event authored by"
          " billing_agent"
          in caplog.text
      )
    finally:
      # Restore original logger level
      llm_agent_logger.setLevel(original_level)

  def test_maybe_save_output_to_state_case_sensitive_names(self, caplog):
    """Test that agent name comparison is case-sensitive."""
    # Set the LlmAgent logger to DEBUG level
    llm_agent_logger = logging.getLogger(
        "google_adk.google.adk.agents.llm_agent"
    )
    original_level = llm_agent_logger.level
    llm_agent_logger.setLevel(logging.DEBUG)

    try:
      agent = LlmAgent(name="TestAgent", output_key="result")
      event = create_test_event(
          author="testagent", content_text="Test response"
      )

      with caplog.at_level("DEBUG"):
        agent._LlmAgent__maybe_save_output_to_state(event)

      # Should not save due to case mismatch
      assert len(event.actions.state_delta) == 0
      assert (
          "Skipping output save for agent TestAgent: event authored by"
          " testagent"
          in caplog.text
      )
    finally:
      # Restore original logger level
      llm_agent_logger.setLevel(original_level)

  @patch("google.adk.agents.llm_agent.logger")
  def test_maybe_save_output_to_state_logging(self, mock_logger):
    """Test that debug logging works correctly."""
    agent = LlmAgent(name="agent1", output_key="result")
    event = create_test_event(author="agent2", content_text="Test response")

    agent._LlmAgent__maybe_save_output_to_state(event)

    # Should call logger.debug with correct parameters
    mock_logger.debug.assert_called_once_with(
        "Skipping output save for agent %s: event authored by %s",
        "agent1",
        "agent2",
    )

  @pytest.mark.parametrize("empty_content", ["", "  ", "\n"])
  def test_maybe_save_output_to_state_handles_empty_final_chunk_with_schema(
      self, empty_content
  ):
    """Tests that the agent correctly handles an empty final streaming chunk

    when an output_schema is specified, preventing a crash.
    """
    # ARRANGE: Create an agent that expects a JSON output matching a schema.
    agent = LlmAgent(
        name="test_agent", output_key="result", output_schema=MockOutputSchema
    )

    # ARRANGE: Create a final event with empty or whitespace-only content.
    # This simulates the final, empty chunk from a model's streaming response.
    event = create_test_event(
        author="test_agent", content_text=empty_content, is_final=True
    )

    # ACT: Call the method. The test's primary goal is to ensure this
    # does NOT raise a pydantic.ValidationError, which it would have before the fix.
    try:
      agent._LlmAgent__maybe_save_output_to_state(event)
    except Exception as e:
      pytest.fail(f"The method unexpectedly raised an exception: {e}")

    # ASSERT: Because the method should return early, the state_delta
    # should remain empty.
    assert len(event.actions.state_delta) == 0
