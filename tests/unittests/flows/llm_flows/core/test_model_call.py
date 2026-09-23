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

"""Unit tests for flows.llm_flows.core._model_call."""

from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.core import _model_call
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.utils.context_utils import Aclosing
from google.genai import types
import pytest

from .... import testing_utils


class _FlowForTesting(BaseLlmFlow):
  """Minimal BaseLlmFlow implementation for model call tests."""


class _CfcFlowForTesting(BaseLlmFlow):
  """BaseLlmFlow subclass that stubs run_live so the CFC branch can be driven."""

  async def run_live(self, invocation_context):
    del invocation_context
    yield Event(
        author='root_agent',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='live_hello')]
        ),
        turn_complete=True,
    )


class _SyncOnlyAgent(BaseAgent):
  """An agent supplying the LlmAgent model surface without subclassing it."""

  @property
  def canonical_model(self) -> BaseLlm:
    return LLMRegistry.new_llm('gemini-2.5-flash')

  @property
  def canonical_live_model(self) -> BaseLlm:
    return LLMRegistry.new_llm('gemini-2.0-flash')


async def _drive_one_llm_call(flow: BaseLlmFlow, invocation_context) -> None:
  """Runs `call_llm_async` once, draining whatever it yields."""
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )
  async with Aclosing(
      _model_call.call_llm_async(
          flow,
          invocation_context,
          LlmRequest(model='mock'),
          model_response_event,
      )
  ) as agen:
    async for _ in agen:
      pass


# --- Tests for apply_empty_response_policy ---


@pytest.mark.asyncio
async def test_apply_empty_response_policy_marks_empty_stop_in_non_streaming():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  response = LlmResponse(
      content=types.Content(role='model', parts=[]),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code == _model_call.NO_CONTENT_ERROR_CODE
  assert response.error_message == _model_call.NO_CONTENT_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_apply_empty_response_policy_skips_sse_streaming():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )
  response = LlmResponse(
      content=types.Content(role='model', parts=[]),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code is None


@pytest.mark.asyncio
async def test_apply_empty_response_policy_leaves_non_empty_response_untouched():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='ok')]
      ),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code is None


# --- Tests for resolve_llm ---


@pytest.mark.asyncio
async def test_resolve_llm_reads_an_agent_that_has_only_the_sync_properties():
  agent = _SyncOnlyAgent(name='sync_only')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  llm = await _model_call.resolve_llm(invocation_context)

  assert llm.model == 'gemini-2.5-flash'


@pytest.mark.asyncio
async def test_resolve_llm_reads_live_model_when_live_queue_present():
  agent = _SyncOnlyAgent(name='sync_only')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  llm = await _model_call.resolve_llm(invocation_context)

  assert llm.model == 'gemini-2.0-flash'


@pytest.mark.asyncio
async def test_resolve_llm_rejects_an_agent_with_no_model_at_all():
  agent = BaseAgent(name='no_model')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  with pytest.raises(TypeError, match='canonical_model'):
    await _model_call.resolve_llm(invocation_context)


# --- Tests for call_llm_async ---


@pytest.mark.asyncio
async def test_call_llm_async_stamps_agent_name_label_and_yields_response():
  mock_model = testing_utils.MockModel.create(responses=['hello'])
  agent = Agent(name='billing_agent', model=mock_model)
  flow = _FlowForTesting()
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='hi'
  )
  llm_request = LlmRequest(model='mock')
  event = Event(id=Event.new_id(), invocation_id=ctx.invocation_id, author='a')

  responses = [
      r async for r in _model_call.call_llm_async(flow, ctx, llm_request, event)
  ]

  assert len(responses) == 1
  assert (
      llm_request.config.labels[_model_call.ADK_AGENT_NAME_LABEL_KEY]
      == 'billing_agent'
  )


@pytest.mark.asyncio
async def test_llm_calls_are_counted_against_max_llm_calls():
  """The cap applies on the ordinary (non-CFC) path."""
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=['a', 'b', 'c']),
  )
  flow = _FlowForTesting()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(max_llm_calls=2),
  )

  await _drive_one_llm_call(flow, invocation_context)
  await _drive_one_llm_call(flow, invocation_context)
  assert invocation_context._invocation_cost_manager._number_of_llm_calls == 2

  with pytest.raises(LlmCallsLimitExceededError):
    await _drive_one_llm_call(flow, invocation_context)


@pytest.mark.asyncio
async def test_cfc_llm_calls_are_counted_against_max_llm_calls():
  """support_cfc must not exempt a run from the max_llm_calls spend cap."""
  agent = Agent(
      name='root_agent', model=testing_utils.MockModel.create(responses=[])
  )
  flow = _CfcFlowForTesting()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(
          support_cfc=True,
          streaming_mode=StreamingMode.SSE,
          max_llm_calls=2,
      ),
  )

  await _drive_one_llm_call(flow, invocation_context)
  await _drive_one_llm_call(flow, invocation_context)
  assert invocation_context._invocation_cost_manager._number_of_llm_calls == 2

  with pytest.raises(LlmCallsLimitExceededError):
    await _drive_one_llm_call(flow, invocation_context)
