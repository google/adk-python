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

"""Tests for the stop_event cancellation mechanism in SSE streaming."""

import asyncio
from typing import AsyncGenerator
from typing import override

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


class StreamingMockModel(testing_utils.MockModel):
  """MockModel that yields multiple chunks per generate_content_async call."""

  chunks_per_call: list[list[LlmResponse]] = Field(default_factory=list)
  call_index: int = -1

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.call_index += 1
    self.requests.append(llm_request)
    for chunk in self.chunks_per_call[self.call_index]:
      yield chunk


@pytest.mark.asyncio
async def test_stop_event_stops_streaming_mid_chunks():
  """Setting stop_event mid-stream should prevent further chunks from being yielded."""
  chunks = [
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text='Hello')]
          ),
          partial=True,
      ),
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text=' world')]
          ),
          partial=True,
      ),
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text=' foo')]
          ),
          partial=True,
      ),
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text=' bar')]
          ),
          partial=True,
      ),
  ]

  mock_model = StreamingMockModel(responses=[], chunks_per_call=[chunks])

  agent = Agent(name='test_agent', model=mock_model)
  stop_event = asyncio.Event()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.stop_event = stop_event

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)
    if len(events) == 2:
      # Signal stop after receiving 2 chunks
      stop_event.set()

  # Should have received exactly 2 chunks (stop was signalled after the 2nd)
  assert len(events) == 2


@pytest.mark.asyncio
async def test_stop_event_not_set_yields_all_chunks():
  """When stop_event is provided but never set, all chunks should be yielded."""
  chunks = [
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text='Hello')]
          ),
          partial=True,
      ),
      LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text=' world')]
          ),
          partial=True,
      ),
  ]

  mock_model = StreamingMockModel(responses=[], chunks_per_call=[chunks])

  agent = Agent(name='test_agent', model=mock_model)
  stop_event = asyncio.Event()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.stop_event = stop_event

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert len(events) == 2


@pytest.mark.asyncio
async def test_stop_event_prevents_next_llm_call():
  """Setting stop_event between LLM calls should prevent the next call."""
  # First LLM call: returns a function call
  fc_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part.from_function_call(
                  name='my_tool', args={'x': '1'}
              )
          ],
      ),
      partial=False,
  )
  # Second LLM call: should NOT be reached
  text_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Result')]
      ),
      partial=False,
      error_code=types.FinishReason.STOP,
  )

  mock_model = testing_utils.MockModel.create(
      responses=[fc_response, text_response]
  )

  def my_tool(x: str) -> str:
    return f'result_{x}'

  agent = Agent(name='test_agent', model=mock_model, tools=[my_tool])
  stop_event = asyncio.Event()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.stop_event = stop_event

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)
    # Stop after first LLM call yields its events
    if event.get_function_calls():
      stop_event.set()

  # Should have events from the first LLM call only
  # The second LLM call (text_response) should NOT have happened
  all_texts = [
      part.text
      for e in events
      if e.content and e.content.parts
      for part in e.content.parts
      if part.text
  ]
  assert 'Result' not in all_texts


@pytest.mark.asyncio
async def test_no_stop_event_works_normally():
  """When no stop_event is provided, everything works as before."""
  response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Done')]
      ),
      partial=False,
      error_code=types.FinishReason.STOP,
  )

  mock_model = testing_utils.MockModel.create(responses=[response])

  agent = Agent(name='test_agent', model=mock_model)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  # No stop_event set (default None)

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Done'
