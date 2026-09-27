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

"""Unit tests for _batch_tool_executor."""

from __future__ import annotations

import asyncio
import contextvars
from unittest import mock

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.tools import _batch_executor as _batch_tool_executor
from google.adk.flows.llm_flows.tools._caller import _PreparedFunctionCall
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


def test_deep_merge_dicts_basic() -> None:
  d1 = {'a': 1, 'nested': {'x': 10}}
  d2 = {'b': 2, 'nested': {'y': 20}}
  merged = _batch_tool_executor.deep_merge_dicts(d1, d2)
  assert merged == {'a': 1, 'b': 2, 'nested': {'x': 10, 'y': 20}}


def test_deep_merge_dicts_overwrite() -> None:
  d1 = {'a': 1, 'nested': {'x': 10}}
  d2 = {'a': 99, 'nested': {'x': 20}}
  merged = _batch_tool_executor.deep_merge_dicts(d1, d2)
  assert merged == {'a': 99, 'nested': {'x': 20}}


def test_merge_parallel_function_response_events_empty() -> None:
  with pytest.raises(ValueError, match='No function response events provided.'):
    _batch_tool_executor.merge_parallel_function_response_events([])


def test_merge_parallel_function_response_events_single() -> None:
  event = Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='result')],
      ),
  )
  merged = _batch_tool_executor.merge_parallel_function_response_events([event])
  assert merged is event


def test_merge_parallel_function_response_events_multiple() -> None:
  event1 = Event(
      invocation_id='inv-1',
      author='agent',
      branch='main',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='part1')],
      ),
      actions=EventActions(state_delta={'key1': 'val1'}),
  )
  event2 = Event(
      invocation_id='inv-1',
      author='agent',
      branch='main',
      content=types.Content(
          role='user',
          parts=[types.Part.from_text(text='part2')],
      ),
      actions=EventActions(state_delta={'key2': 'val2'}),
  )
  merged = _batch_tool_executor.merge_parallel_function_response_events(
      [event1, event2]
  )
  assert merged.invocation_id == 'inv-1'
  assert merged.author == 'agent'
  assert merged.content is not None
  assert merged.content.parts is not None
  assert len(merged.content.parts) == 2
  assert merged.content.parts[0].text == 'part1'
  assert merged.content.parts[1].text == 'part2'
  assert merged.actions.state_delta == {'key1': 'val1', 'key2': 'val2'}


def test_is_non_blocking_tool() -> None:
  assert not _batch_tool_executor._is_non_blocking_tool(None)

  tool_without_scheduling = BaseTool(name='t1', description='desc')
  assert not _batch_tool_executor._is_non_blocking_tool(tool_without_scheduling)

  tool_with_scheduling = BaseTool(
      name='t2',
      description='desc',
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert _batch_tool_executor._is_non_blocking_tool(tool_with_scheduling)

  tool_with_behavior_non_blocking = BaseTool(
      name='t3',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
  )
  assert _batch_tool_executor._is_non_blocking_tool(
      tool_with_behavior_non_blocking
  )

  tool_with_behavior_blocking = BaseTool(
      name='t4',
      description='desc',
      behavior=types.Behavior.BLOCKING,
  )
  assert not _batch_tool_executor._is_non_blocking_tool(
      tool_with_behavior_blocking
  )

  # Explicit behavior overrides response_scheduling
  tool_blocking_with_scheduling = BaseTool(
      name='t5',
      description='desc',
      behavior=types.Behavior.BLOCKING,
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert not _batch_tool_executor._is_non_blocking_tool(
      tool_blocking_with_scheduling
  )

  tool_non_blocking_with_scheduling = BaseTool(
      name='t6',
      description='desc',
      behavior=types.Behavior.NON_BLOCKING,
      response_scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
  )
  assert _batch_tool_executor._is_non_blocking_tool(
      tool_non_blocking_with_scheduling
  )


@pytest.mark.asyncio
async def test_gather_or_cancel_success() -> None:
  async def worker(n: int) -> int:
    return n * 2

  tasks = [asyncio.create_task(worker(i)) for i in range(3)]
  results = await _batch_tool_executor._gather_or_cancel(tasks)
  assert results == [0, 2, 4]


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_siblings_on_failure() -> None:
  cancelled = False

  async def failing_worker() -> None:
    await asyncio.sleep(0.01)
    raise RuntimeError('boom')

  async def slow_worker() -> None:
    nonlocal cancelled
    try:
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      cancelled = True
      raise

  tasks = [
      asyncio.create_task(failing_worker()),
      asyncio.create_task(slow_worker()),
  ]
  with pytest.raises(RuntimeError, match='boom'):
    await _batch_tool_executor._gather_or_cancel(tasks)

  assert cancelled


_probe: contextvars.ContextVar[str] = contextvars.ContextVar(
    'probe', default='unset'
)


def _prepared_call_with_snapshot(value: str) -> _PreparedFunctionCall:
  """Builds a prepared call whose snapshot has `_probe` set to `value`."""

  def take_snapshot() -> contextvars.Context:
    _probe.set(value)
    return contextvars.copy_context()

  return _PreparedFunctionCall(
      function_call=types.FunctionCall(name='t', id='call-1', args={}),
      tool=BaseTool(name='t', description='desc'),
      tool_context=mock.create_autospec(ToolContext, instance=True),
      function_args={},
      # Run in a throwaway context so the setter does not touch this test's own.
      contextvars_snapshot=contextvars.copy_context().run(take_snapshot),
  )


@pytest.mark.asyncio
async def test_start_execute_task_runs_in_the_prepare_snapshot() -> None:
  async def read_then_overwrite_probe() -> str:
    seen = _probe.get()
    _probe.set('set-by-the-tool')
    return seen

  prepared = _prepared_call_with_snapshot('set-by-prepare')

  # The task starts from the snapshot, so it sees what prepare left there.
  task = _batch_tool_executor._start_execute_task(
      prepared, read_then_overwrite_probe()
  )
  assert await task == 'set-by-prepare'

  # It got a copy, so its own write reached neither the snapshot nor the caller.
  assert prepared.contextvars_snapshot[_probe] == 'set-by-prepare'
  assert _probe.get() == 'unset'


@pytest.mark.asyncio
async def test_start_execute_task_keeps_parallel_calls_isolated() -> None:
  async def read_probe() -> str:
    await asyncio.sleep(0)
    return _probe.get()

  prepared = [_prepared_call_with_snapshot(f'call-{i}') for i in range(3)]
  results = await asyncio.gather(*[
      _batch_tool_executor._start_execute_task(call, read_probe())
      for call in prepared
  ])

  assert results == ['call-0', 'call-1', 'call-2']


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
          _batch_tool_executor, '_prepare_single', new_callable=mock.AsyncMock
      ),
      mock.patch.object(
          _batch_tool_executor,
          '_execute_single_prepared_call_live',
          new_callable=mock.AsyncMock,
          return_value=event,
      ),
  ):
    await _batch_tool_executor._launch_non_blocking_call_live(
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
