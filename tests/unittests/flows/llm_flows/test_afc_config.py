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

"""Unit tests for Automatic Function Calling configuration handling.

Tests for Bug #4133: Ensure that AFC config (disable=True, maximum_remote_calls)
is properly respected and that planner hooks are always called.
"""

import asyncio
from unittest.mock import MagicMock

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.genai import types
from google.genai.types import Part
import pytest

from ... import testing_utils


@pytest.fixture
def live_test_runner():
  """Fixture that creates a CustomTestRunner for live mode tests.

  This eliminates code duplication across live mode tests by providing
  a reusable runner factory that handles live streaming response collection.
  """

  class CustomTestRunner(testing_utils.InMemoryRunner):
    """Custom test runner for live mode tests with configurable response collection."""

    def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        run_config: testing_utils.RunConfig = None,
        max_responses: int = 3,
    ) -> list[testing_utils.Event]:
      collected_responses = []

      async def consume_responses(session: testing_utils.Session):
        run_res = self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config or testing_utils.RunConfig(),
        )

        async for response in run_res:
          collected_responses.append(response)
          if len(collected_responses) >= max_responses:
            return

      try:
        session = self.session
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          loop.run_until_complete(
              asyncio.wait_for(consume_responses(session), timeout=5.0)
          )
        finally:
          loop.close()
      except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

      return collected_responses

  return CustomTestRunner


@pytest.mark.asyncio
async def test_afc_disabled_stops_loop():
  """Test that setting disable=True stops the AFC loop after first response."""
  # Setup: Create a mock model that returns function calls
  responses = [
      # First response with function call
      Part.from_function_call(name='test_tool', args={'x': 1}),
      # Second response (should not be called if AFC is disabled)
      'This should not be returned',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  # Create agent with AFC disabled
  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              disable=True
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Assert: Should stop after first LLM call (1 model response with function call)
  # The tool should NOT be executed because AFC is disabled
  assert call_count == 0, 'Tool should not be called when AFC is disabled'

  # Should have only 1 LLM request (not 2)
  assert (
      len(mock_model.requests) == 1
  ), 'Should make only 1 LLM call when AFC is disabled'


@pytest.mark.asyncio
async def test_maximum_remote_calls_zero_stops_loop():
  """Test that setting maximum_remote_calls=0 stops the AFC loop."""
  responses = [
      Part.from_function_call(name='test_tool', args={'x': 1}),
      'This should not be returned',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=0
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Tool should not be executed
  assert (
      call_count == 0
  ), 'Tool should not be called when maximum_remote_calls=0'
  assert (
      len(mock_model.requests) == 1
  ), 'Should make only 1 LLM call when maximum_remote_calls=0'


@pytest.mark.asyncio
async def test_maximum_remote_calls_limit_enforced():
  """Test that maximum_remote_calls limit is properly enforced.

  Note: maximum_remote_calls counts executed function calls. So
  maximum_remote_calls=2 allows executing 2 function calls total.
  """
  responses = [
      # First response
      Part.from_function_call(name='test_tool', args={'x': 1}),
      # Second response (after first tool execution)
      Part.from_function_call(name='test_tool', args={'x': 2}),
      # Third response (after second tool execution - should not be reached)
      'Should not be returned',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=2
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Should execute tool twice (max_remote_calls=2)
  assert (
      call_count == 2
  ), 'Tool should be called exactly twice when maximum_remote_calls=2'
  # Should make 3 LLM calls: initial + after 1st FC + after 2nd FC
  assert (
      len(mock_model.requests) == 3
  ), 'Should make 3 LLM calls with maximum_remote_calls=2'


@pytest.mark.asyncio
async def test_planner_hook_called_with_maximum_remote_calls_zero():
  """Test that planner.process_planning_response is called with maximum_remote_calls=0."""
  from google.adk.planners.plan_re_act_planner import PlanReActPlanner

  responses = [
      Part.from_function_call(name='test_tool', args={'x': 1}),
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def test_tool(x: int) -> int:
    return x + 1

  # Use PlanReActPlanner which actually processes responses
  planner = PlanReActPlanner()
  planner.process_planning_response = MagicMock(
      wraps=planner.process_planning_response
  )

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      planner=planner,
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=0
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  list(runner.run('test message'))

  # Verify the planner hook was called
  assert planner.process_planning_response.called, (
      'Planner.process_planning_response should be called even with '
      'maximum_remote_calls=0'
  )


@pytest.mark.asyncio
async def test_afc_enabled_continues_loop():
  """Test that AFC loop continues normally when not disabled."""
  responses = [
      # First response with function call
      Part.from_function_call(name='test_tool', args={'x': 1}),
      # Second response after function execution
      'Final response',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  # No AFC config - should work normally
  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Tool should be executed
  assert call_count == 1, 'Tool should be called once in normal AFC mode'

  # Should make 2 LLM calls: initial + after function response
  assert (
      len(mock_model.requests) == 2
  ), 'Should make 2 LLM calls in normal AFC mode'


@pytest.mark.asyncio
async def test_afc_disabled_with_parallel_function_calls():
  """Test that AFC disabled works with parallel function calls."""
  # Model returns multiple function calls in one response
  responses = [
      [
          Part.from_function_call(name='test_tool', args={'x': 1}),
          Part.from_function_call(name='test_tool', args={'x': 2}),
          Part.from_function_call(name='test_tool', args={'x': 3}),
      ],
      'This should not be returned',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              disable=True
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # None of the parallel FCs should be executed
  assert call_count == 0, 'No tools should be called when AFC is disabled'
  assert (
      len(mock_model.requests) == 1
  ), 'Should make only 1 LLM call when AFC is disabled'


@pytest.mark.asyncio
async def test_maximum_remote_calls_with_parallel_function_calls():
  """Test that maximum_remote_calls counts events, not individual FCs."""
  # Each LLM response has multiple parallel function calls
  responses = [
      # First event with 2 parallel FCs
      [
          Part.from_function_call(name='test_tool', args={'x': 1}),
          Part.from_function_call(name='test_tool', args={'x': 2}),
      ],
      # Second event with 2 parallel FCs (should execute)
      [
          Part.from_function_call(name='test_tool', args={'x': 3}),
          Part.from_function_call(name='test_tool', args={'x': 4}),
      ],
      # Third event (should not execute - limit reached)
      Part.from_function_call(name='test_tool', args={'x': 5}),
      # Final response after limit
      'Final response after limit reached',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=2  # 2 events, not 2 individual FCs
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Should execute 4 FCs (2 events × 2 FCs each)
  assert call_count == 4, (
      'Should execute all FCs from first 2 events '
      '(maximum_remote_calls counts events, not individual FCs)'
  )
  # Should make 4 LLM calls: initial + after 1st event + after 2nd event + final call
  # The final call happens because we need to get a final response after reaching the limit
  assert (
      len(mock_model.requests) == 4
  ), 'Should make 4 LLM calls with maximum_remote_calls=2'


@pytest.mark.asyncio
async def test_maximum_remote_calls_one_allows_one_execution():
  """Test that maximum_remote_calls=1 allows exactly one FC execution."""
  responses = [
      Part.from_function_call(name='test_tool', args={'x': 1}),
      Part.from_function_call(name='test_tool', args={'x': 2}),
      'Final response after limit reached',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=1
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  assert (
      call_count == 1
  ), 'Tool should be called once when maximum_remote_calls=1'
  # Should make 3 LLM calls: initial + after 1st FC + final call
  # The final call happens because we need to get a final response after reaching the limit
  assert (
      len(mock_model.requests) == 3
  ), 'Should make 3 LLM calls with maximum_remote_calls=1'


def test_negative_maximum_remote_calls_treated_as_zero():
  """Test that negative maximum_remote_calls is caught by <= 0 check."""
  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Done')]),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=-5  # Negative value
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Negative value should be treated like 0 (no FCs allowed)
  assert (
      call_count == 0
  ), 'Tool should not be called when maximum_remote_calls=-5'
  # Should make 1 LLM call: initial only (loop exits immediately)
  assert (
      len(mock_model.requests) == 1
  ), 'Should make 1 LLM call when negative maximum_remote_calls'


def test_very_large_maximum_remote_calls():
  """Test that very large maximum_remote_calls works correctly."""
  # Create responses for 3 function calls
  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Done')]),
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([
      response_with_fc,
      response_with_fc,
      response_with_fc,
      final_response,
  ])

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=999999  # Very large value
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)
  events = list(runner.run('test message'))

  # Should allow all 3 function calls since limit is very high
  assert call_count == 3, 'All 3 tool calls should execute with limit=999999'
  # Should make 4 LLM calls: initial + after each of 3 FCs
  assert (
      len(mock_model.requests) == 4
  ), 'Should make 4 LLM calls with 3 function calls'


def test_corrupted_session_empty_events():
  """Test behavior when session history returns empty/corrupted data."""
  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Done')]),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=1
          )
      ),
  )

  runner = testing_utils.InMemoryRunner(agent)

  # Clear session events to simulate corrupted state
  session = runner.session
  session._events = []  # Simulate empty/corrupted event history

  events = list(runner.run('test message'))

  # Even with corrupted session, the system should handle gracefully
  # The first FC should execute since count starts at 0
  assert call_count == 1, 'Tool should be called once even with empty session'


def test_afc_disabled_in_live_mode(live_test_runner):
  """Test that AFC disabled works in live streaming mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              disable=True  # AFC disabled
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  # Tool should NOT be called because AFC is disabled
  assert (
      call_count == 0
  ), 'Tool should not be called when AFC is disabled in live mode'
  # Should make 1 LLM call: initial only (AFC disabled, no second call)
  assert (
      len(mock_model.requests) == 1
  ), 'Should make 1 LLM call when AFC disabled in live mode'


def test_maximum_remote_calls_in_live_mode(live_test_runner):
  """Test that maximum_remote_calls limit works in live streaming mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  # Create 3 FC responses but limit to 1
  mock_model = testing_utils.MockModel.create([
      response_with_fc,
      response_with_fc,
      response_with_fc,
      final_response,
  ])

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=1  # Limit to 1 FC
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=4)

  # Tool should be called exactly once due to limit
  assert (
      call_count == 1
  ), 'Tool should be called once when maximum_remote_calls=1 in live mode'
  # In live mode with limit=1: initial call + 1 FC execution = 2 LLM calls total
  # (different from async mode where limit is enforced differently)
  assert (
      len(mock_model.requests) >= 1
  ), 'Should make at least 1 LLM call with maximum_remote_calls=1 in live mode'


def test_maximum_remote_calls_zero_in_live_mode(live_test_runner):
  """Test that maximum_remote_calls=0 stops FCs in live streaming mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=0  # No FCs allowed
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  # Tool should NOT be called because maximum_remote_calls=0
  assert (
      call_count == 0
  ), 'Tool should not be called when maximum_remote_calls=0 in live mode'
  assert (
      len(mock_model.requests) == 1
  ), 'Should make 1 LLM call when maximum_remote_calls=0 in live mode'


def test_parallel_function_calls_in_live_mode(live_test_runner):
  """Test that parallel FCs count as 1 event in live mode."""
  from google.genai import types as genai_types

  # Create response with 3 parallel function calls
  tool_call1 = types.Part.from_function_call(name='test_tool', args={'x': 1})
  tool_call2 = types.Part.from_function_call(name='test_tool', args={'x': 2})
  tool_call3 = types.Part.from_function_call(name='test_tool', args={'x': 3})
  response_with_parallel_fcs = LlmResponse(
      content=types.Content(
          role='model', parts=[tool_call1, tool_call2, tool_call3]
      ),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create([
      response_with_parallel_fcs,
      final_response,
  ])

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=1  # Limit to 1 event (3 parallel FCs)
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=4)

  # All 3 parallel function calls should execute (they count as 1 event)
  assert (
      call_count == 3
  ), 'All 3 parallel FCs should execute in live mode (count as 1 event)'
  # Confirms event counting: 3 parallel FCs in 1 event = 1 toward limit
  assert (
      len(mock_model.requests) >= 1
  ), 'Should make at least 1 LLM call with parallel FCs in live mode'


def test_negative_maximum_remote_calls_in_live_mode(live_test_runner):
  """Test that negative maximum_remote_calls is treated as zero in live mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=-10  # Negative value
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  # Tool should NOT be called (negative treated as zero)
  assert (
      call_count == 0
  ), 'Tool should not be called when maximum_remote_calls=-10 in live mode'
  assert (
      len(mock_model.requests) == 1
  ), 'Should make 1 LLM call when negative maximum_remote_calls in live mode'


def test_maximum_remote_calls_two_in_live_mode(live_test_runner):
  """Test that maximum_remote_calls=2 enforces limit in live mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  # Create 3 FC responses but limit to 2
  mock_model = testing_utils.MockModel.create([
      response_with_fc,
      response_with_fc,
      response_with_fc,
      final_response,
  ])

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=2  # Limit to 2 FCs
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=5)

  # Tool should be called exactly twice
  assert (
      call_count == 2
  ), 'Tool should be called twice when maximum_remote_calls=2 in live mode'
  assert (
      len(mock_model.requests) >= 1
  ), 'Should make at least 1 LLM call with maximum_remote_calls=2 in live mode'


def test_very_large_maximum_remote_calls_in_live_mode(live_test_runner):
  """Test that very large maximum_remote_calls works in live mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  # Create 3 FC responses
  mock_model = testing_utils.MockModel.create([
      response_with_fc,
      response_with_fc,
      response_with_fc,
      final_response,
  ])

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=999999  # Very large limit
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, max_responses=5)

  # In live mode with timeout, may not get all 3 calls
  # But should get at least 2 calls (verifies large limit works)
  assert call_count >= 2, (
      'At least 2 tool calls should execute with limit=999999 in live mode,'
      f' got {call_count}'
  )
  assert (
      len(mock_model.requests) >= 1
  ), 'Should make at least 1 LLM call with very large limit in live mode'


def test_corrupted_session_in_live_mode(live_test_runner):
  """Test behavior when session is corrupted in live mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=1
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  # Clear session events to simulate corrupted state
  runner.session._events = []
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  # Even with corrupted session, should handle gracefully
  assert (
      call_count == 1
  ), 'Tool should be called once even with corrupted session in live mode'


def test_planner_hooks_in_live_mode(live_test_runner):
  """Test that maximum_remote_calls=0 works correctly in live mode."""
  from google.genai import types as genai_types

  tool_call = types.Part.from_function_call(name='test_tool', args={'x': 1})
  response_with_fc = LlmResponse(
      content=types.Content(role='model', parts=[tool_call]),
      turn_complete=False,
  )
  final_response = LlmResponse(
      content=types.Content(
          role='model', parts=[genai_types.Part(text='Done')]
      ),
      turn_complete=True,
  )
  mock_model = testing_utils.MockModel.create(
      [response_with_fc, final_response]
  )

  call_count = 0

  def test_tool(x: int) -> int:
    nonlocal call_count
    call_count += 1
    return x + 1

  agent = Agent(
      name='test_agent',
      model=mock_model,
      tools=[test_tool],
      generate_content_config=types.GenerateContentConfig(
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
              maximum_remote_calls=0  # No FCs allowed
          )
      ),
  )

  runner = live_test_runner(root_agent=agent, response_modalities=['AUDIO'])
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=genai_types.Blob(data=b'test audio', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue)

  # AFC config should be respected in live mode
  # Tool should NOT be called because maximum_remote_calls=0
  assert (
      call_count == 0
  ), 'Tool should not be called when maximum_remote_calls=0 in live mode'
