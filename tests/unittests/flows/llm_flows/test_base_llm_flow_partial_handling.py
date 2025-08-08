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

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
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
