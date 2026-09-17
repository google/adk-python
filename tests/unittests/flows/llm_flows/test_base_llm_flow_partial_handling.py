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

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import _SingleAfterModelCallback
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _handle_after_model_callback
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.mark.asyncio
async def test_run_async_breaks_on_partial_event():
  """Test that run_async breaks when the last event is partial."""
  # Create a mock model that returns partial responses
  partial_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Partial response')]
      ),
      partial=True,
  )

  mock_model = testing_utils.MockModel.create(responses=[partial_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have one event (the partial response)
  assert len(events) == 1
  assert events[0].partial is True
  assert events[0].content.parts[0].text == 'Partial response'


@pytest.mark.asyncio
async def test_run_async_breaks_on_final_response():
  """Test that run_async breaks when the last event is a final response."""
  # Create a mock model that returns a final response
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Final response')]
      ),
      partial=False,
      error_code=types.FinishReason.STOP,
  )

  mock_model = testing_utils.MockModel.create(responses=[final_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have one event (the final response)
  assert len(events) == 1
  assert events[0].partial is False
  assert events[0].content.parts[0].text == 'Final response'


@pytest.mark.asyncio
async def test_run_async_breaks_on_no_last_event():
  """Test that run_async breaks when there is no last event."""
  # Create a mock model that returns an empty response (no content)
  empty_response = LlmResponse(content=None, partial=False)

  mock_model = testing_utils.MockModel.create(responses=[empty_response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have no events because empty responses are filtered out
  assert len(events) == 0


@pytest.mark.asyncio
async def test_run_async_breaks_on_first_partial_response():
  """Test run_async breaks on the first partial response."""
  # Create responses with mixed partial states
  partial_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Partial response')]
      ),
      partial=True,
  )

  # These won't be reached because the flow breaks on the first partial
  non_partial_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Non-partial response')],
      ),
      partial=False,
  )

  final_partial_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Final partial response')],
      ),
      partial=True,
  )

  mock_model = testing_utils.MockModel.create(
      responses=[partial_response, non_partial_response, final_partial_response]
  )

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []

  # Collect events from the flow
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # Should have only one event, breaking on the first partial response
  assert len(events) == 1
  assert events[0].partial is True
  assert events[0].content.parts[0].text == 'Partial response'


def _text_response(
    text: str,
    *,
    partial: bool | None = None,
    turn_complete: bool | None = None,
) -> LlmResponse:
  """Builds a single-part text response with the given streaming flags."""
  return LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text=text)]
      ),
      partial=partial,
      turn_complete=turn_complete,
  )


def _rebuilding_callback(
    replacement: LlmResponse,
) -> _SingleAfterModelCallback:
  """Returns a callback that rebuilds the response, dropping control fields."""

  async def _callback(
      callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse:
    del callback_context, llm_response
    return replacement

  return _callback


async def _run_handle_after_model_callback(
    agent: Agent, llm_response: LlmResponse
) -> LlmResponse | None:
  """Runs the after-model callback chain for the agent over one response."""
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )
  return await _handle_after_model_callback(
      invocation_context, llm_response, event
  )


@pytest.mark.asyncio
async def test_after_model_callback_replacement_inherits_partial():
  """A rebuilt replacement keeps the delta's partial flag."""
  replacement = _text_response('Hello ')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('Hello ', partial=True)
  )

  assert result.partial is True
  assert result.content.parts[0].text == 'Hello '
  # The callback's object is not mutated; inheritance returns a copy.
  assert replacement.partial is None


@pytest.mark.asyncio
async def test_after_model_callback_replacement_inherits_turn_complete():
  """A rebuilt replacement keeps turn_complete from the replaced response."""
  replacement = _text_response('done')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('done', turn_complete=True)
  )

  assert result.turn_complete is True


@pytest.mark.asyncio
async def test_after_model_callback_replacement_explicit_partial_respected():
  """An explicitly set partial=False on the replacement is not overridden."""
  agent = Agent(
      name='test_agent',
      after_model_callback=[
          _rebuilding_callback(_text_response('final', partial=False))
      ],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('final', partial=True)
  )

  assert result.partial is False


@pytest.mark.asyncio
async def test_after_model_callback_replacement_without_streaming_keeps_identity():
  """Non-streaming replacements pass through untouched (no copy)."""
  replacement = _text_response('final')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _text_response('final')
  )

  assert result is replacement


@pytest.mark.asyncio
async def test_after_model_callback_returning_its_argument_keeps_identity():
  """A callback that hands back what it was given has nothing to inherit."""
  llm_response = _text_response('Hello ', partial=True)
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(llm_response)],
  )

  result = await _run_handle_after_model_callback(agent, llm_response)

  assert result is llm_response


class _ResponseLookalike:
  """A response object without the streaming fields, as some models yield."""

  def __init__(self, text: str):
    self.content = types.Content(
        role='model', parts=[types.Part.from_text(text=text)]
    )
    self.partial = False


@pytest.mark.asyncio
async def test_after_model_callback_response_without_streaming_fields():
  """A response that is not an LlmResponse leaves the replacement alone."""
  replacement = _text_response('final')
  agent = Agent(
      name='test_agent',
      after_model_callback=[_rebuilding_callback(replacement)],
  )

  result = await _run_handle_after_model_callback(
      agent, _ResponseLookalike('final')
  )

  assert result is replacement


class _StreamingFakeModel(BaseLlm):
  """Yields two partial deltas then the aggregated final response."""

  model: str = 'fake-streaming'

  @classmethod
  def create(cls) -> _StreamingFakeModel:
    return cls(model='fake-streaming')

  @classmethod
  def supported_models(cls) -> list[str]:
    return ['.*']

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    deltas = ['Hello ', 'world.']
    if stream:
      for delta in deltas:
        yield _text_response(delta, partial=True)
    yield _text_response(''.join(deltas))


@pytest.mark.asyncio
async def test_run_async_sse_rebuilt_responses_stay_partial():
  """End-to-end: rebuilding callbacks must not flip SSE deltas to final."""
  agent = Agent(
      name='test_agent',
      model=_StreamingFakeModel.create(),
      after_model_callback=[_rebuilding_callback(_text_response('scrubbed'))],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test message',
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert [event.partial for event in events] == [True, True, None]
  # The callback ran on every response (content replaced), but the streaming
  # semantics of the originals survived.
  assert [event.content.parts[0].text for event in events] == [
      'scrubbed',
      'scrubbed',
      'scrubbed',
  ]
