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

"""Unit tests for flows.llm_flows.tools._live_caller."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.flows.llm_flows.tools import _live_caller
from google.adk.flows.llm_flows.tools._caller import _PreparedFunctionCall
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


def test_is_live_request_queue_annotation_matches_class_and_string() -> None:
  """Both LiveRequestQueue class and string annotations are recognized."""

  def fn_with_queue(input_stream: LiveRequestQueue, other: int) -> None:
    del input_stream, other

  params = inspect.signature(fn_with_queue).parameters
  assert (
      _live_caller._is_live_request_queue_annotation(params['input_stream'])
      is True
  )
  assert (
      _live_caller._is_live_request_queue_annotation(params['other']) is False
  )


@pytest.mark.asyncio
async def test_emit_streaming_tool_event_enqueues_user_message() -> None:
  """Events yielded by a streaming tool are enqueued as user-role messages on its tool@call_id branch."""
  tool = BaseTool(name='stream_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.function_call_id = 'fc-99'
  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'stream_agent'
  invocation_context._enqueue_event = mock.AsyncMock()

  raw_event = Event(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='progress update')]
      )
  )
  await _live_caller._emit_streaming_tool_event(
      raw_event,
      tool=tool,
      tool_context=tool_context,
      invocation_context=invocation_context,
  )

  invocation_context._enqueue_event.assert_awaited_once()
  enqueued = invocation_context._enqueue_event.call_args.args[0]
  assert isinstance(enqueued, Event)
  assert enqueued.branch == 'stream_tool@fc-99'
  assert enqueued.content is not None
  assert enqueued.content.role == 'user'


def test_is_non_blocking_tool() -> None:
  assert not _live_caller._is_non_blocking_tool(None)

  tool_without_scheduling = BaseTool(name='t1', description='desc')
  assert not _live_caller._is_non_blocking_tool(tool_without_scheduling)

  tool_with_scheduling = BaseTool(
      name='t2',
      description='desc',
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert _live_caller._is_non_blocking_tool(tool_with_scheduling)

  tool_with_behavior_non_blocking = BaseTool(
      name='t3',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
  )
  assert _live_caller._is_non_blocking_tool(tool_with_behavior_non_blocking)

  tool_with_behavior_blocking = BaseTool(
      name='t4',
      description='desc',
      behavior=types.Behavior.BLOCKING,
  )
  assert not _live_caller._is_non_blocking_tool(tool_with_behavior_blocking)

  # Explicit behavior overrides response_scheduling
  tool_blocking_with_scheduling = BaseTool(
      name='t5',
      description='desc',
      behavior=types.Behavior.BLOCKING,
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert not _live_caller._is_non_blocking_tool(tool_blocking_with_scheduling)

  tool_non_blocking_with_scheduling = BaseTool(
      name='t6',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert _live_caller._is_non_blocking_tool(tool_non_blocking_with_scheduling)


def test_is_streaming_tool() -> None:
  assert not _live_caller._is_streaming_tool(None)

  def sync_fn() -> str:
    return 'ok'

  async def async_gen():
    yield 'chunk'

  assert not _live_caller._is_streaming_tool(FunctionTool(sync_fn))
  assert _live_caller._is_streaming_tool(FunctionTool(async_gen))


@pytest.mark.parametrize(
    'has_event_queue,expect_enqueue,expect_session_append',
    [
        (True, True, False),
        (False, False, True),
    ],
)
@pytest.mark.asyncio
async def test_launch_non_blocking_call_live(
    has_event_queue: bool,
    expect_enqueue: bool,
    expect_session_append: bool,
) -> None:
  function_call = types.FunctionCall(name='my_tool', id='call_123')
  tool = BaseTool(name='my_tool', description='')
  event = Event(invocation_id='inv-1', content=types.Content())

  mock_session = mock.MagicMock()
  mock_session_service = mock.MagicMock(append_event=mock.AsyncMock())
  mock_ic = mock.MagicMock(
      _event_queue=asyncio.Queue() if has_event_queue else None,
      session=mock_session,
      session_service=mock_session_service,
      active_non_blocking_tool_tasks={},
      _enqueue_event=mock.AsyncMock(),
  )

  with (
      mock.patch.object(
          _live_caller, '_prepare_single', new_callable=mock.AsyncMock
      ),
      mock.patch.object(
          _live_caller,
          '_execute_single_prepared_call_live',
          new_callable=mock.AsyncMock,
          return_value=event,
      ),
  ):
    await _live_caller._launch_non_blocking_call_live(
        invocation_context=mock_ic,
        function_call=function_call,
        tool=tool,
        tools_dict={'my_tool': tool},
        agent=mock.MagicMock(),
        active_tools_lock=asyncio.Lock(),
        live_session_id='live_session_123',
    )

    task_key = 'my_tool_call_123'
    assert task_key in mock_ic.active_non_blocking_tool_tasks
    await mock_ic.active_non_blocking_tool_tasks[task_key]

  assert event.live_session_id == 'live_session_123'
  assert mock_ic._enqueue_event.await_count == (1 if expect_enqueue else 0)
  assert mock_session_service.append_event.await_count == (
      1 if expect_session_append else 0
  )
  mock_ic.live_request_queue.send_content.assert_called_once_with(event.content)
  assert task_key not in mock_ic.active_non_blocking_tool_tasks


@pytest.mark.asyncio
async def test_execute_prepared_function_calls_live_merges_and_sets_session_id() -> (
    None
):
  mock_ic = mock.MagicMock()
  mock_ic.session.state = {}
  fc_event = Event(invocation_id='inv-1', live_session_id='live-sess-1')

  prepared_1 = _PreparedFunctionCall(
      function_call=types.FunctionCall(name='t1', id='c1', args={}),
      tool=BaseTool(name='t1', description='desc'),
      tool_context=mock.create_autospec(ToolContext, instance=True),
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
  )
  prepared_2 = _PreparedFunctionCall(
      function_call=types.FunctionCall(name='t2', id='c2', args={}),
      tool=BaseTool(name='t2', description='desc'),
      tool_context=mock.create_autospec(ToolContext, instance=True),
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
  )

  ev1 = Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='user', parts=[types.Part.from_text(text='r1')]
      ),
  )
  ev2 = Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='user', parts=[types.Part.from_text(text='r2')]
      ),
  )

  with mock.patch.object(
      _live_caller,
      '_execute_single_prepared_call_live',
      new_callable=mock.AsyncMock,
      side_effect=[ev1, ev2],
  ):
    merged = await _live_caller._execute_prepared_function_calls_live(
        mock_ic, fc_event, [prepared_1, prepared_2], mock.MagicMock()
    )

  assert merged is not None
  assert merged.live_session_id == 'live-sess-1'
  assert merged.content is not None
  assert [p.text for p in merged.content.parts] == ['r1', 'r2']


@pytest.mark.asyncio
async def test_handle_function_calls_live_splits_blocking_and_non_blocking() -> (
    None
):
  blocking_tool = BaseTool(
      name='blocking_tool',
      description='desc',
      behavior=types.Behavior.BLOCKING,
  )
  non_blocking_tool = BaseTool(
      name='non_blocking_tool',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
  )
  tools_dict = {
      'blocking_tool': blocking_tool,
      'non_blocking_tool': non_blocking_tool,
  }

  fc_blocking = types.FunctionCall(name='blocking_tool', id='c_block', args={})
  fc_non_blocking = types.FunctionCall(
      name='non_blocking_tool', id='c_nonblock', args={}
  )
  fc_event = Event(
      invocation_id='inv-1',
      live_session_id='live-sess-99',
      content=types.Content(
          role='model',
          parts=[
              types.Part(function_call=fc_blocking),
              types.Part(function_call=fc_non_blocking),
          ],
      ),
  )

  mock_ic = mock.MagicMock()
  expected_response = Event(invocation_id='inv-1', author='agent')

  with (
      mock.patch.object(
          _live_caller, '_as_llm_agent', return_value=mock.MagicMock()
      ),
      mock.patch.object(
          _live_caller,
          '_launch_non_blocking_call_live',
          new_callable=mock.AsyncMock,
      ) as mock_launch,
      mock.patch.object(
          _live_caller,
          '_prepare_function_calls',
          new_callable=mock.AsyncMock,
          return_value=[mock.MagicMock()],
      ) as mock_prepare,
      mock.patch.object(
          _live_caller,
          '_execute_prepared_function_calls_live',
          new_callable=mock.AsyncMock,
          return_value=expected_response,
      ) as mock_execute,
  ):
    res = await _live_caller.handle_function_calls_live(
        mock_ic, fc_event, tools_dict
    )

  assert res is expected_response
  mock_launch.assert_awaited_once()
  assert mock_launch.call_args.kwargs['function_call'] == fc_non_blocking
  mock_prepare.assert_awaited_once()
  assert mock_prepare.call_args.args[1] == [fc_blocking]
  mock_execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_function_calls_live_only_non_blocking_returns_none() -> (
    None
):
  non_blocking_tool = BaseTool(
      name='non_blocking_tool',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
  )
  tools_dict = {'non_blocking_tool': non_blocking_tool}
  fc_non_blocking = types.FunctionCall(
      name='non_blocking_tool', id='c_nonblock', args={}
  )
  fc_event = Event(
      invocation_id='inv-1',
      content=types.Content(
          role='model',
          parts=[types.Part(function_call=fc_non_blocking)],
      ),
  )

  with (
      mock.patch.object(
          _live_caller, '_as_llm_agent', return_value=mock.MagicMock()
      ),
      mock.patch.object(
          _live_caller,
          '_launch_non_blocking_call_live',
          new_callable=mock.AsyncMock,
      ) as mock_launch,
  ):
    res = await _live_caller.handle_function_calls_live(
        mock.MagicMock(), fc_event, tools_dict
    )

  assert res is None
  mock_launch.assert_awaited_once()
