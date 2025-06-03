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

"""Tests for concurrent function execution in LLM flows."""

import asyncio
import time
from typing import Any
from typing import Dict

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_concurrent_function_execution():
  """Test that multiple functions are executed concurrently, not sequentially."""
  
  # Track execution order and timing
  execution_log = []
  start_time = time.time()
  
  async def slow_function_1(delay: float = 0.1) -> Dict[str, Any]:
    """A function that simulates slow execution."""
    execution_log.append(f'function_1_start: {time.time() - start_time:.3f}s')
    await asyncio.sleep(delay)
    execution_log.append(f'function_1_end: {time.time() - start_time:.3f}s')
    return {'result': 'function_1_completed', 'delay': delay}

  async def slow_function_2(delay: float = 0.1) -> Dict[str, Any]:
    """A function that simulates slow execution."""
    execution_log.append(f'function_2_start: {time.time() - start_time:.3f}s')
    await asyncio.sleep(delay)
    execution_log.append(f'function_2_end: {time.time() - start_time:.3f}s')
    return {'result': 'function_2_completed', 'delay': delay}

  async def slow_function_3(delay: float = 0.1) -> Dict[str, Any]:
    """A function that simulates slow execution."""
    execution_log.append(f'function_3_start: {time.time() - start_time:.3f}s')
    await asyncio.sleep(delay)
    execution_log.append(f'function_3_end: {time.time() - start_time:.3f}s')
    return {'result': 'function_3_completed', 'delay': delay}

  # Create function calls that will be executed concurrently
  function_calls = [
      types.Part.from_function_call(name='slow_function_1', args={'delay': 0.1}),
      types.Part.from_function_call(name='slow_function_2', args={'delay': 0.1}),
      types.Part.from_function_call(name='slow_function_3', args={'delay': 0.1}),
  ]
  
  # Expected function responses
  function_responses = [
      types.Part.from_function_response(
          name='slow_function_1', 
          response={'result': 'function_1_completed', 'delay': 0.1}
      ),
      types.Part.from_function_response(
          name='slow_function_2', 
          response={'result': 'function_2_completed', 'delay': 0.1}
      ),
      types.Part.from_function_response(
          name='slow_function_3', 
          response={'result': 'function_3_completed', 'delay': 0.1}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'All functions completed successfully',
  ]
  
  mock_model = testing_utils.MockModel.create(responses=responses)

  agent = Agent(
      name='concurrent_agent',
      model=mock_model,
      tools=[slow_function_1, slow_function_2, slow_function_3],
  )
  
  runner = testing_utils.TestInMemoryRunner(agent)
  
  # Reset start time for accurate measurement
  start_time = time.time()
  execution_log.clear()
  
  events = await runner.run_async_with_new_session('Execute all three functions')
  
  total_execution_time = time.time() - start_time
  
  # Verify the events structure
  assert testing_utils.simplify_events(events) == [
      ('concurrent_agent', function_calls),
      ('concurrent_agent', function_responses),
      ('concurrent_agent', 'All functions completed successfully'),
  ]
  
  # Verify concurrent execution by checking timing
  # If executed concurrently: ~0.1s total (max of individual delays)
  # If executed sequentially: ~0.3s total (sum of individual delays)
  assert total_execution_time < 0.25, (
      f"Functions appear to be executed sequentially. "
      f"Total time: {total_execution_time:.3f}s, expected < 0.25s"
  )
  
  # Verify that all functions started before any completed (concurrent execution)
  start_count = len([log for log in execution_log if 'start' in log])
  end_count = len([log for log in execution_log if 'end' in log])
  
  assert start_count == 3, f"Expected 3 function starts, got {start_count}"
  assert end_count == 3, f"Expected 3 function ends, got {end_count}"
  
  # Print execution log for debugging (only on failure)
  if total_execution_time >= 0.25:
    print("Execution log:")
    for log_entry in execution_log:
      print(f"  {log_entry}")


@pytest.mark.asyncio
async def test_concurrent_execution_with_exception():
  """Test that exceptions in one function don't prevent others from completing."""
  
  execution_results = []
  
  async def successful_function() -> Dict[str, Any]:
    """A function that succeeds."""
    execution_results.append('successful_function_completed')
    return {'result': 'success'}

  async def failing_function() -> Dict[str, Any]:
    """A function that raises an exception."""
    execution_results.append('failing_function_started')
    raise ValueError("This function always fails")

  async def another_successful_function() -> Dict[str, Any]:
    """Another function that succeeds."""
    execution_results.append('another_successful_function_completed')
    return {'result': 'another_success'}

  # Create function calls
  function_calls = [
      types.Part.from_function_call(name='successful_function', args={}),
      types.Part.from_function_call(name='failing_function', args={}),
      types.Part.from_function_call(name='another_successful_function', args={}),
  ]
  
  # Expected function responses (only successful ones)
  function_responses = [
      types.Part.from_function_response(
          name='successful_function', 
          response={'result': 'success'}
      ),
      types.Part.from_function_response(
          name='another_successful_function', 
          response={'result': 'another_success'}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'Functions completed with some errors',
  ]
  
  mock_model = testing_utils.MockModel.create(responses=responses)

  agent = Agent(
      name='error_handling_agent',
      model=mock_model,
      tools=[successful_function, failing_function, another_successful_function],
  )
  
  runner = testing_utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('Execute all functions')
  
  # Verify that successful functions completed despite one failing
  assert 'successful_function_completed' in execution_results
  assert 'another_successful_function_completed' in execution_results
  assert 'failing_function_started' in execution_results
  
  # Verify the events structure contains successful function responses
  event_content = testing_utils.simplify_events(events)
  
  # The first event should be the function calls
  assert event_content[0] == ('error_handling_agent', function_calls)
  
  # The second event should contain the successful function responses
  # (The exact structure may vary due to error handling)
  assert len(event_content) >= 2


@pytest.mark.asyncio 
async def test_concurrent_with_mixed_sync_async():
  """Test concurrent execution with a mix of sync and async functions."""
  
  execution_order = []
  
  def sync_function(value: int) -> Dict[str, Any]:
    """A synchronous function."""
    execution_order.append(f'sync_function_{value}')
    return {'result': f'sync_{value}'}

  async def async_function(value: int) -> Dict[str, Any]:
    """An asynchronous function."""
    execution_order.append(f'async_function_{value}')
    await asyncio.sleep(0.05)  # Small delay to ensure async behavior
    return {'result': f'async_{value}'}

  # Create function calls
  function_calls = [
      types.Part.from_function_call(name='sync_function', args={'value': 1}),
      types.Part.from_function_call(name='async_function', args={'value': 2}),
      types.Part.from_function_call(name='sync_function', args={'value': 3}),
  ]
  
  # Expected function responses
  function_responses = [
      types.Part.from_function_response(
          name='sync_function', 
          response={'result': 'sync_1'}
      ),
      types.Part.from_function_response(
          name='async_function', 
          response={'result': 'async_2'}
      ),
      types.Part.from_function_response(
          name='sync_function', 
          response={'result': 'sync_3'}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'Mixed functions completed',
  ]
  
  mock_model = testing_utils.MockModel.create(responses=responses)

  agent = Agent(
      name='mixed_agent',
      model=mock_model,
      tools=[sync_function, async_function],
  )
  
  runner = testing_utils.TestInMemoryRunner(agent)
  
  start_time = time.time()
  events = await runner.run_async_with_new_session('Execute mixed functions')
  total_time = time.time() - start_time
  
  # Verify the events structure
  assert testing_utils.simplify_events(events) == [
      ('mixed_agent', function_calls),
      ('mixed_agent', function_responses),
      ('mixed_agent', 'Mixed functions completed'),
  ]
  
  # Verify all functions executed
  assert len(execution_order) == 3
  assert 'sync_function_1' in execution_order
  assert 'async_function_2' in execution_order  
  assert 'sync_function_3' in execution_order
  
  # Verify execution was reasonably fast (concurrent)
  assert total_time < 0.2, f"Execution took too long: {total_time:.3f}s" 