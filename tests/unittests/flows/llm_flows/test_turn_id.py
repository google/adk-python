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

"""Tests for the turn_id field on streaming events."""

from typing import AsyncGenerator
from typing import override

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


class StreamingMockModel(testing_utils.MockModel):
  """MockModel that yields all responses for a given call at once (simulates streaming chunks)."""

  chunks_per_call: list[list[LlmResponse]] = []
  """Each inner list is yielded during one generate_content_async call."""

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
async def test_partial_chunks_share_same_turn_id():
  """All partial chunks from one LLM call must share the same turn_id."""
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
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert len(events) == 2
  # All chunks must have an integer turn_id
  assert events[0].turn_id == 1
  assert events[1].turn_id == 1
  # But different event ids
  assert events[0].id != events[1].id


@pytest.mark.asyncio
async def test_turn_id_present_on_final_response():
  """A single final response event must carry a turn_id."""
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

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  assert len(events) == 1
  assert events[0].turn_id == 1


@pytest.mark.asyncio
async def test_different_llm_calls_get_different_turn_ids():
  """Events from separate LLM calls (separated by a tool call) must have different turn_ids."""
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
  # Second LLM call: returns text after tool result
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
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()
  events = []
  async for event in flow.run_async(invocation_context):
    events.append(event)

  # We expect: function_call event (turn 1), function_response event, final text event (turn 2)
  events_with_turn_id = [e for e in events if e.turn_id is not None]

  assert len(events_with_turn_id) >= 2
  turn_ids = sorted({e.turn_id for e in events_with_turn_id})
  # Must have consecutive integer turn_ids from separate LLM calls
  assert turn_ids == [1, 2], (
      f'Expected turn_ids [1, 2] for two LLM calls, got: {turn_ids}'
  )
