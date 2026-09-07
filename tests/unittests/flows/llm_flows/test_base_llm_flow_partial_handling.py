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

from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.flows.llm_flows.base_llm_flow import _inherit_unset_streaming_fields
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


class _StreamingMockModel(BaseLlm):
  """Streams queued responses as a single model turn."""

  model: str = 'streaming-mock'
  stream_responses: list[LlmResponse] = []

  @classmethod
  def supported_models(cls) -> list[str]:
    return ['streaming-mock']

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    for response in self.stream_responses:
      yield response


def _streamed_turn() -> list[LlmResponse]:
  """Three partial deltas followed by the aggregated final response."""
  deltas = ['Hello ', 'brave ', 'world.']
  responses = [
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text=delta)]
          ),
          partial=True,
      )
      for delta in deltas
  ]
  responses.append(
      LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text=''.join(deltas))],
          )
      )
  )
  return responses


def _rebuilding_callback(callback_context, llm_response: LlmResponse):
  """Returns a fresh LlmResponse, per the after_model_callback contract."""
  if not (llm_response.content and llm_response.content.parts):
    return None
  text = llm_response.content.parts[0].text or ''
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text=text.replace('brave', 'kind'))],
      )
  )


@pytest.mark.asyncio
async def test_after_model_callback_replacement_preserves_partial_in_sse():
  """A rebuilt replacement inherits `partial` from the streamed response."""
  agent = Agent(
      name='test_agent',
      model=_StreamingMockModel(stream_responses=_streamed_turn()),
      after_model_callback=_rebuilding_callback,
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

  assert [event.partial for event in events] == [True, True, True, None]
  assert events[-1].content.parts[0].text == 'Hello kind world.'


@pytest.mark.asyncio
async def test_after_model_callback_explicit_partial_false_is_respected():
  """A replacement that explicitly finalizes a delta is not overridden."""

  def finalize_callback(callback_context, llm_response: LlmResponse):
    if not llm_response.partial:
      return None
    return LlmResponse(content=llm_response.content, partial=False)

  agent = Agent(
      name='test_agent',
      model=_StreamingMockModel(stream_responses=_streamed_turn()),
      after_model_callback=finalize_callback,
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

  assert events[0].partial is False


def test_inherit_unset_streaming_fields_inherits_when_unset():
  original = LlmResponse(partial=True, turn_complete=True)
  replacement = LlmResponse()

  result = _inherit_unset_streaming_fields(original, replacement)

  assert result.partial is True
  assert result.turn_complete is True


def test_inherit_unset_streaming_fields_respects_explicit_values():
  original = LlmResponse(partial=True, turn_complete=True)
  replacement = LlmResponse(partial=False, turn_complete=False)

  result = _inherit_unset_streaming_fields(original, replacement)

  assert result.partial is False
  assert result.turn_complete is False
