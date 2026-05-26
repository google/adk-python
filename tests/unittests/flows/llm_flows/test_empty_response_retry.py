# Copyright 2026 Google LLC
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

"""Tests for empty model response retry logic in BaseLlmFlow.run_async."""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.base_llm_flow import _has_meaningful_content
from google.adk.flows.llm_flows.base_llm_flow import _MAX_EMPTY_RESPONSE_RETRIES
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class TestHasMeaningfulContent:
  """Tests for the _has_meaningful_content helper function."""

  def test_no_content(self):
    """Event with no content is not meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=None,
    )
    assert not _has_meaningful_content(event)

  def test_empty_parts(self):
    """Event with empty parts list is not meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(role="model", parts=[]),
    )
    assert not _has_meaningful_content(event)

  def test_only_empty_text_part(self):
    """Event with only an empty text part is not meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(
            role="model", parts=[types.Part.from_text(text="")]
        ),
    )
    assert not _has_meaningful_content(event)

  def test_only_whitespace_text_part(self):
    """Event with only whitespace text is not meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(
            role="model", parts=[types.Part.from_text(text="   \n  ")]
        ),
    )
    assert not _has_meaningful_content(event)

  def test_non_empty_text(self):
    """Event with actual text is meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Hello, world!")],
        ),
    )
    assert _has_meaningful_content(event)

  def test_function_call(self):
    """Event with a function call is meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="test_tool", args={"key": "value"}
                )
            ],
        ),
    )
    assert _has_meaningful_content(event)

  def test_function_response(self):
    """Event with a function response is meaningful."""
    event = Event(
        invocation_id="test",
        author="model",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_response(
                    name="test_tool", response={"result": "ok"}
                )
            ],
        ),
    )
    assert _has_meaningful_content(event)


class TestEmptyResponseRetry:
  """Tests for the agent loop retrying on empty model responses."""

  @pytest.mark.asyncio
  async def test_empty_response_retried_then_succeeds(self):
    """Agent loop retries when model returns empty content, then succeeds."""
    empty_response = LlmResponse(
        content=types.Content(role="model", parts=[]),
        partial=False,
    )
    good_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Here are the results.")],
        ),
        partial=False,
    )

    mock_model = testing_utils.MockModel.create(
        responses=[empty_response, good_response]
    )
    agent = Agent(
        name="test_agent",
        model=mock_model,
        instruction="You are a helpful assistant.",
    )

    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content="test"
    )

    events = []
    async for event in agent.run_async(invocation_context):
      events.append(event)

    # Should have events from both LLM calls (empty + good)
    non_partial_events = [e for e in events if not e.partial]
    final_texts = [
        part.text
        for e in non_partial_events
        if e.content and e.content.parts
        for part in e.content.parts
        if part.text
    ]
    assert any(
        "results" in t for t in final_texts
    ), "Expected the good response text after retry"

  @pytest.mark.asyncio
  async def test_empty_response_stops_after_max_retries(self):
    """Agent loop stops after max retries of empty responses."""
    empty_responses = [
        LlmResponse(
            content=types.Content(role="model", parts=[]),
            partial=False,
        )
        for _ in range(_MAX_EMPTY_RESPONSE_RETRIES + 1)
    ]

    mock_model = testing_utils.MockModel.create(responses=empty_responses)
    agent = Agent(
        name="test_agent",
        model=mock_model,
        instruction="You are a helpful assistant.",
    )

    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content="test"
    )

    events = []
    async for event in agent.run_async(invocation_context):
      events.append(event)

    # The model should have been called _MAX_EMPTY_RESPONSE_RETRIES + 1 times
    # (1 initial + N retries) and then the loop should stop.
    assert mock_model.response_index == _MAX_EMPTY_RESPONSE_RETRIES

  @pytest.mark.asyncio
  async def test_non_empty_response_not_retried(self):
    """A normal response with content is not retried."""
    good_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="All good.")],
        ),
        partial=False,
    )

    mock_model = testing_utils.MockModel.create(responses=[good_response])
    agent = Agent(
        name="test_agent",
        model=mock_model,
        instruction="You are a helpful assistant.",
    )

    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content="test"
    )

    events = []
    async for event in agent.run_async(invocation_context):
      events.append(event)

    # Model should only be called once
    assert mock_model.response_index == 0
