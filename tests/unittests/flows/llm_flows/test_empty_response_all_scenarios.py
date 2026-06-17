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

"""Comprehensive tests for empty model response retry across all scenarios.

Retry only fires after at least one tool call in the invocation, since
the bug manifests when the model returns empty after processing tool
results.  Each test therefore includes a tool-call response first.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _has_meaningful_content
from google.adk.flows.llm_flows.base_llm_flow import _MAX_EMPTY_RESPONSE_RETRIES
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  pass


def _test_tool_func(key: str = 'value') -> dict:
  """A dummy tool for testing."""
  return {'result': 'ok'}


_TEST_TOOL = FunctionTool(func=_test_tool_func)


def _tool_call_response():
  """A function-call response that triggers has_prior_tool_call."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name='_test_tool_func', args={'key': 'value'}
                  )
              )
          ],
      ),
      partial=False,
  )


def _empty_response():
  return LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=False,
  )


def _good_response(text='Here is the answer.'):
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text=text)],
      ),
      partial=False,
  )


# ---------------------------------------------------------------------------
# Scenario 1: Non-streaming empty response after tool call, then recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario1_non_streaming_empty_then_recovery():
  """Non-streaming: model returns empty parts after tool call, retry recovers."""
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), _empty_response(), _good_response()]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # All 3 responses consumed: tool call + empty (retried) + good
  assert mock_model.response_index == 2
  good_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert any('answer' in t for t in good_texts)


# ---------------------------------------------------------------------------
# Scenario 2: Thought-only final response after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario2_thought_only_final_response_retried():
  """Thought-only non-partial response triggers retry after tool call."""
  thought_only = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Let me think...', thought=True)],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), thought_only, _good_response()]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 2, 'Expected 3 LLM calls (tool + retry)'


# ---------------------------------------------------------------------------
# Scenario 3a: No events at all (breaks, no retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario3a_no_events_at_all_breaks():
  """When _run_one_step yields nothing, loop breaks (no retry)."""
  empty_response = LlmResponse(content=None, partial=False)
  mock_model = testing_utils.MockModel.create(responses=[empty_response])
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 0
  assert len(events) == 0


# ---------------------------------------------------------------------------
# Scenario 3b: Partial event with no meaningful content after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario3b_partial_empty_content_retried():
  """Partial event with empty parts triggers retry after tool call."""
  partial_empty = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), partial_empty, _good_response()]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 2, 'Expected retry on partial empty'


@pytest.mark.asyncio
async def test_scenario3b_partial_thought_only_retried():
  """Partial event with thought-only content triggers retry after tool call."""
  partial_thought = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='Thinking...', thought=True)],
      ),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), partial_thought, _good_response()]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 2, 'Expected retry on partial thought'


# ---------------------------------------------------------------------------
# Scenario 4: Partial event WITH meaningful content (should NOT retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario4_partial_with_meaningful_content_not_retried():
  """Partial event with real text content should NOT trigger retry."""
  partial_with_text = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Partial but real content')],
      ),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(responses=[partial_with_text])
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 0
  partial_events = [e for e in events if e.partial]
  assert len(partial_events) == 1


# ---------------------------------------------------------------------------
# Scenario 5: Empty response exhausts max retries after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario5_empty_exhausts_max_retries():
  """Empty responses stop after max retries."""
  responses = [_tool_call_response()] + [
      _empty_response() for _ in range(_MAX_EMPTY_RESPONSE_RETRIES + 1)
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # tool_call + initial empty + MAX retries
  assert mock_model.response_index == _MAX_EMPTY_RESPONSE_RETRIES + 1


# ---------------------------------------------------------------------------
# Scenario 6: Empty -> Empty -> Good after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario6_multiple_empty_then_recovery():
  """Multiple empty responses followed by good response."""
  responses = [
      _tool_call_response(),
      _empty_response(),
      _empty_response(),
      _good_response('Finally recovered!'),
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 3
  final_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text and not getattr(p, 'thought', False)
  ]
  assert any('recovered' in t.lower() for t in final_texts)


# ---------------------------------------------------------------------------
# Scenario 7: lite_llm streaming fallback
# ---------------------------------------------------------------------------


def test_scenario7_litellm_fallback_response_is_not_partial():
  """Verify the fallback LlmResponse from lite_llm has partial=False."""
  fallback = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=False,
      finish_reason=types.FinishReason.STOP,
      model_version='test-model',
  )
  event = Event(
      invocation_id='test',
      author='test_agent',
      content=fallback.content,
  )
  assert event.is_final_response() is True
  assert _has_meaningful_content(event) is False


# ---------------------------------------------------------------------------
# Scenario 8: Whitespace-only text response after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario8_whitespace_only_response_retried():
  """Response with only whitespace text triggers retry after tool call."""
  whitespace = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='   \n  \t  ')],
      ),
      partial=False,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), whitespace, _good_response()]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 2, 'Expected retry on whitespace'


# ---------------------------------------------------------------------------
# Scenario 9: Function call is meaningful (not retried)
# ---------------------------------------------------------------------------


def test_scenario9_function_call_is_meaningful():
  """A function call event is meaningful and would not trigger retry."""
  event = Event(
      invocation_id='test',
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name='some_tool', args={'key': 'value'}
                  )
              )
          ],
      ),
  )
  assert _has_meaningful_content(event) is True
  assert event.is_final_response() is False


# ---------------------------------------------------------------------------
# Scenario 10: Partial empty then partial with content after tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario10_partial_empty_then_partial_with_content():
  """Partial empty retries, then partial with content breaks normally."""
  partial_empty = LlmResponse(
      content=types.Content(role='model', parts=[]),
      partial=True,
  )
  partial_real = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Streaming chunk')],
      ),
      partial=True,
  )
  mock_model = testing_utils.MockModel.create(
      responses=[_tool_call_response(), partial_empty, partial_real]
  )
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  assert mock_model.response_index == 2


# ---------------------------------------------------------------------------
# Scenario 11: Empty response WITHOUT prior tool call (should NOT retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario11_empty_without_tool_call_not_retried():
  """Empty response on first LLM call (no tool call) should NOT retry."""
  mock_model = testing_utils.MockModel.create(responses=[_empty_response()])
  agent = Agent(name='test_agent', model=mock_model, tools=[_TEST_TOOL])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  events = []
  async for event in BaseLlmFlowForTesting().run_async(ctx):
    events.append(event)

  # Only 1 model call -- no retry without prior tool call
  assert mock_model.response_index == 0
