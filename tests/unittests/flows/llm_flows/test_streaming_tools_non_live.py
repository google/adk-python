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

"""Tests for streaming tools in non-live mode."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import _execute_streaming_tool_async
from google.adk.flows.llm_flows.functions import _is_streaming_tool
from google.adk.flows.llm_flows.functions import handle_function_calls_async_with_streaming
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


def test_is_streaming_tool_detects_async_generator():
  """Test that _is_streaming_tool correctly identifies async generators."""

  async def regular_function(x: int) -> int:
    return x + 1

  async def streaming_function(x: int) -> AsyncGenerator[dict, None]:
    yield {'result': x + 1}
    yield {'result': x + 2}

  regular_tool = FunctionTool(func=regular_function)
  streaming_tool = FunctionTool(func=streaming_function)

  assert not _is_streaming_tool(regular_tool)
  assert _is_streaming_tool(streaming_tool)


@pytest.mark.asyncio
async def test_streaming_tool_yields_multiple_events():
  """Test that a streaming tool yields multiple Events."""

  async def monitor_stock(symbol: str) -> AsyncGenerator[dict, None]:
    prices = [100, 105, 110]
    for price in prices:
      await asyncio.sleep(0.01)  # Small delay for testing
      yield {'symbol': symbol, 'price': price}

  tool = FunctionTool(func=monitor_stock)
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id='test-call-1',
  )

  events = []
  async for event in _execute_streaming_tool_async(
      tool,
      {'symbol': 'AAPL'},
      tool_context,
      invocation_context,
  ):
    events.append(event)

  # Should have 3 events (one per price)
  assert len(events) == 3
  assert all(isinstance(e, Event) for e in events)
  assert all(
      e.content and e.content.parts and e.content.parts[0].function_response
      for e in events
  )


@pytest.mark.asyncio
async def test_streaming_tool_with_string_results():
  """Test streaming tool that yields string results."""

  async def process_data(data: str) -> AsyncGenerator[str, None]:
    for i, char in enumerate(data):
      await asyncio.sleep(0.01)
      yield f'Processed {i}: {char}'

  tool = FunctionTool(func=process_data)
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id='test-call-2',
  )

  events = []
  async for event in _execute_streaming_tool_async(
      tool,
      {'data': 'ABC'},
      tool_context,
      invocation_context,
  ):
    events.append(event)

  assert len(events) == 3


@pytest.mark.asyncio
async def test_streaming_tool_tracks_task_for_cancellation():
  """Test that streaming tool tasks are tracked for cancellation and cleaned up."""

  async def long_running_task() -> AsyncGenerator[dict, None]:
    for i in range(10):
      await asyncio.sleep(0.1)
      yield {'progress': i}

  tool = FunctionTool(func=long_running_task)
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id='test-call-3',
  )

  # Start the streaming tool and consume it
  async def consume_generator():
    async for event in _execute_streaming_tool_async(
        tool, {}, tool_context, invocation_context
    ):
      # Consume events until cancelled
      pass

  consume_task = asyncio.create_task(consume_generator())

  # Wait a bit for the tool to start
  await asyncio.sleep(0.05)

  # Check that task is tracked
  assert invocation_context.active_streaming_tools is not None
  assert tool.name in invocation_context.active_streaming_tools
  assert invocation_context.active_streaming_tools[tool.name].task is not None

  # Cancel the generator consumption task
  consume_task.cancel()
  try:
    await consume_task
  except asyncio.CancelledError:
    pass

  # Wait a bit for cleanup to complete
  await asyncio.sleep(0.1)

  # Verify cleanup: task should be removed from active_streaming_tools
  # The finally block in _execute_streaming_tool_async should have cleaned it up
  assert tool.name not in invocation_context.active_streaming_tools


@pytest.mark.asyncio
async def test_streaming_tool_handles_errors():
  """Test error handling in streaming tools."""

  async def failing_tool() -> AsyncGenerator[dict, None]:
    yield {'status': 'started'}
    raise ValueError('Test error')
    yield {'status': 'never reached'}  # type: ignore[unreachable]

  tool = FunctionTool(func=failing_tool)
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id='test-call-4',
  )

  events = []
  with pytest.raises(ValueError, match='Test error'):
    async for event in _execute_streaming_tool_async(
        tool, {}, tool_context, invocation_context
    ):
      events.append(event)

  # Should have yielded at least one event before error
  assert len(events) >= 1


@pytest.mark.asyncio
async def test_handle_function_calls_async_with_streaming_separates_tools():
  """Test that handler correctly separates streaming and non-streaming tools."""

  async def regular_tool(x: int) -> int:
    return x * 2

  async def streaming_tool(x: int) -> AsyncGenerator[dict, None]:
    yield {'value': x}
    yield {'value': x * 2}

  regular = FunctionTool(func=regular_tool)
  streaming = FunctionTool(func=streaming_tool)

  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  function_calls = [
      types.FunctionCall(name='regular_tool', args={'x': 5}, id='call-1'),
      types.FunctionCall(name='streaming_tool', args={'x': 3}, id='call-2'),
  ]
  tools_dict = {
      'regular_tool': regular,
      'streaming_tool': streaming,
  }

  events = []
  async for event in handle_function_calls_async_with_streaming(
      invocation_context, function_calls, tools_dict
  ):
    events.append(event)

  # Should have 1 event from regular tool + 2 events from streaming tool
  assert len(events) == 3


@pytest.mark.asyncio
async def test_streaming_tool_with_tool_context():
  """Test that tool_context is correctly passed to streaming tools."""

  async def context_aware_tool(x: int) -> AsyncGenerator[dict, None]:
    # Note: tool_context is injected by FunctionTool, not passed as arg
    yield {
        'value': x,
        'status': 'processed',
    }

  tool = FunctionTool(func=context_aware_tool)
  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(agent)
  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id='test-call-5',
  )

  events = []
  async for event in _execute_streaming_tool_async(
      tool,
      {'x': 42},
      tool_context,
      invocation_context,
  ):
    events.append(event)

  assert len(events) == 1
  response = events[0].content.parts[0].function_response.response
  assert response['value'] == 42
  assert response['status'] == 'processed'
  # Verify the function_call_id is set correctly in the event
  assert events[0].content.parts[0].function_response.id == 'test-call-5'
