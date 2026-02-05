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

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.mark.asyncio
async def test_postprocess_live_yields_session_resumption_event():
  """Test _postprocess_live yields event for session resumption update."""
  # Create a resumption update as received from Gemini Live API
  resumption_update = types.LiveServerSessionResumptionUpdate(
      new_handle='test-handle-abc123',
      resumable=True,
  )

  # Create LlmResponse with only a resumption update (no content)
  llm_response = LlmResponse(live_session_resumption_update=resumption_update)

  # Set up invocation context
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  # Create the mutable event that _postprocess_live populates
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )

  flow = BaseLlmFlowForTesting()
  llm_request = LlmRequest()
  events = []

  # Collect events from _postprocess_live
  async for event in flow._postprocess_live(
      invocation_context, llm_request, llm_response, model_response_event
  ):
    events.append(event)

  # Verify event is yielded with resumption update
  assert len(events) == 1
  assert events[0].live_session_resumption_update == resumption_update
  assert (
      events[0].live_session_resumption_update.new_handle
      == 'test-handle-abc123'
  )
  assert events[0].live_session_resumption_update.resumable is True

  # Verify invocation context handle is updated for auto-resumption
  assert (
      invocation_context.live_session_resumption_handle == 'test-handle-abc123'
  )


@pytest.mark.asyncio
async def test_postprocess_live_skips_empty_response():
  """Test _postprocess_live skips response with no content or resumption."""
  # Create LlmResponse with no content and no resumption update
  llm_response = LlmResponse()

  # Set up invocation context
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )

  flow = BaseLlmFlowForTesting()
  llm_request = LlmRequest()
  events = []

  # Collect events from _postprocess_live
  async for event in flow._postprocess_live(
      invocation_context, llm_request, llm_response, model_response_event
  ):
    events.append(event)

  # Verify no event is yielded for empty response
  assert len(events) == 0

  # Verify handle remains unset
  assert invocation_context.live_session_resumption_handle is None
