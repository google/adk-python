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
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.tools import _live_caller
from google.adk.flows.llm_flows.tools._caller import _PreparedFunctionCall
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .... import testing_utils


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
  registry = {}
  mock_ic = mock.MagicMock(
      _event_queue=asyncio.Queue() if has_event_queue else None,
      session=mock_session,
      session_service=mock_session_service,
      _enqueue_event=mock.AsyncMock(),
      _active_background_tools=registry,
  )
  mock_ic._add_background_tool.side_effect = lambda entry: registry.update(
      {entry.function_call_id: entry}
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
        live_session_id='live_session_123',
    )

    entry = registry['call_123']
    assert entry.tool_name == 'my_tool'
    assert not entry.is_generator
    await entry.task

  assert event.live_session_id == 'live_session_123'
  assert mock_ic._enqueue_event.await_count == (1 if expect_enqueue else 0)
  assert mock_session_service.append_event.await_count == (
      1 if expect_session_append else 0
  )
  mock_ic.live_request_queue.send_content.assert_called_once_with(event.content)
  assert 'call_123' not in registry


@pytest.mark.asyncio
async def test_streaming_call_requires_a_call_id() -> None:
  """A streaming call without an id is refused, not tracked under a made-up one."""

  async def watch():
    yield 'watching'

  tool = FunctionTool(watch)

  with pytest.raises(ValueError, match='watch'):
    await _live_caller._process_function_live_helper(
        tool,
        mock.MagicMock(function_call_id=None),
        types.FunctionCall(name='watch'),
        {},
        mock.MagicMock(),
    )


@pytest.mark.asyncio
async def test_launch_non_blocking_call_live_requires_a_call_id() -> None:
  """A call without an id is refused rather than tracked under a made-up one."""
  tool = BaseTool(name='my_tool', description='')

  with pytest.raises(ValueError, match='my_tool'):
    await _live_caller._launch_non_blocking_call_live(
        invocation_context=mock.MagicMock(),
        function_call=types.FunctionCall(name='my_tool'),
        tool=tool,
        tools_dict={'my_tool': tool},
        agent=mock.MagicMock(),
    )


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


# --- Background function calls ------------------------------------------------


async def _context_with_tools(
    *tools: BaseTool, **agent_kwargs: Any
) -> InvocationContext:
  """Returns an invocation context whose agent has `tools`."""
  agent = Agent(
      name='test_agent',
      model=testing_utils.MockModel.create(responses=[]),
      tools=list(tools),
      **agent_kwargs,
  )
  return await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )


def _function_call_event(*function_calls: types.FunctionCall) -> Event:
  """Returns a model event carrying `function_calls`."""
  return Event(
      invocation_id='inv-1',
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[types.Part(function_call=fc) for fc in function_calls],
      ),
  )


def _gated_lookup_tool(
    name: str = 'gated_lookup_tool',
) -> tuple[FunctionTool, asyncio.Event]:
  """Returns a tool that answers only once the returned event is set."""
  release = asyncio.Event()

  async def gated_lookup_tool(label: str) -> dict[str, str]:
    await release.wait()
    return {'label': label}

  gated_lookup_tool.__name__ = name
  return FunctionTool(gated_lookup_tool), release


async def lookup_tool(label: str) -> dict[str, str]:
  """Looks up `label` right away."""
  return {'label': label}


async def broken_tool(label: str) -> dict[str, str]:
  """Always fails."""
  raise RuntimeError(f'broken_tool failed on purpose (label={label!r})')


async def cancelled_tool(label: str) -> dict[str, str]:
  """Always cancels."""
  raise asyncio.CancelledError()


async def broken_streaming_tool(label: str) -> AsyncGenerator[str, None]:
  """Always fails."""
  yield f'started {label}'
  raise RuntimeError(
      f'broken_streaming_tool failed on purpose (label={label!r})'
  )


async def streaming_tool(label: str) -> AsyncGenerator[str, None]:
  """Streams updates for `label` until cancelled."""
  yield f'started {label}'
  await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_launch_live_calls_runs_each_call_in_its_own_task() -> None:
  """Each call runs in its own background task."""
  tool_a, release_a = _gated_lookup_tool('gated_lookup_tool_a')
  tool_b, release_b = _gated_lookup_tool('gated_lookup_tool_b')
  invocation_context = await _context_with_tools(tool_a, tool_b)
  event = _function_call_event(
      types.FunctionCall(
          name='gated_lookup_tool_a', id='fc-a', args={'label': 'A'}
      ),
      types.FunctionCall(
          name='gated_lookup_tool_b', id='fc-b', args={'label': 'B'}
      ),
  )

  _live_caller.launch_function_calls_live(
      invocation_context,
      event,
      {tool_a.name: tool_a, tool_b.name: tool_b},
      asyncio.Queue(),
  )

  active = invocation_context._active_background_tools or {}
  task_a, task_b = active['fc-a'].task, active['fc-b'].task
  assert task_a is not task_b
  assert not task_a.done() and not task_b.done()

  release_a.set()
  release_b.set()
  await asyncio.gather(task_a, task_b)


@pytest.mark.asyncio
async def test_launch_live_calls_queues_responses_in_finish_order() -> None:
  """Each call answers in an event of its own, in the order the calls finish."""
  tool_a, release_a = _gated_lookup_tool('gated_lookup_tool_a')
  tool_b, release_b = _gated_lookup_tool('gated_lookup_tool_b')
  invocation_context = await _context_with_tools(tool_a, tool_b)
  event = _function_call_event(
      types.FunctionCall(
          name='gated_lookup_tool_a', id='fc-a', args={'label': 'A'}
      ),
      types.FunctionCall(
          name='gated_lookup_tool_b', id='fc-b', args={'label': 'B'}
      ),
  )
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller.launch_function_calls_live(
      invocation_context,
      event,
      {tool_a.name: tool_a, tool_b.name: tool_b},
      function_response_queue,
  )
  active = invocation_context._active_background_tools or {}
  task_a, task_b = active['fc-a'].task, active['fc-b'].task

  release_b.set()
  await task_b

  assert not task_a.done()
  first_event = function_response_queue.get_nowait()
  assert [fr.id for fr in first_event.get_function_responses()] == ['fc-b']

  release_a.set()
  await task_a

  second_event = function_response_queue.get_nowait()
  assert [fr.id for fr in second_event.get_function_responses()] == ['fc-a']


@pytest.mark.asyncio
async def test_launch_live_calls_keeps_sibling_running_on_failure() -> None:
  """A call that raises gets an error response; its sibling keeps running."""
  broken = FunctionTool(broken_tool)
  gated, release = _gated_lookup_tool()
  invocation_context = await _context_with_tools(broken, gated)
  event = _function_call_event(
      types.FunctionCall(
          name='broken_tool', id='fc-broken', args={'label': 'A'}
      ),
      types.FunctionCall(
          name='gated_lookup_tool', id='fc-gated', args={'label': 'B'}
      ),
  )
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller.launch_function_calls_live(
      invocation_context,
      event,
      {broken.name: broken, gated.name: gated},
      function_response_queue,
  )
  gated_task = invocation_context._active_background_tools['fc-gated'].task
  broken_event = await asyncio.wait_for(function_response_queue.get(), 1)

  (broken_response,) = broken_event.get_function_responses()
  assert broken_response.id == 'fc-broken'
  assert 'error' in broken_response.response
  assert not gated_task.done()

  release.set()
  await gated_task

  gated_event = function_response_queue.get_nowait()
  assert [fr.id for fr in gated_event.get_function_responses()] == ['fc-gated']


@pytest.mark.asyncio
async def test_launch_live_calls_keeps_sibling_running_on_cancel() -> None:
  """A call that is cancelled leaves its sibling running."""
  cancelled = FunctionTool(cancelled_tool)
  gated, release = _gated_lookup_tool()
  invocation_context = await _context_with_tools(cancelled, gated)
  event = _function_call_event(
      types.FunctionCall(
          name='cancelled_tool', id='fc-cancelled', args={'label': 'A'}
      ),
      types.FunctionCall(
          name='gated_lookup_tool', id='fc-gated', args={'label': 'B'}
      ),
  )
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller.launch_function_calls_live(
      invocation_context,
      event,
      {cancelled.name: cancelled, gated.name: gated},
      function_response_queue,
  )
  active_tools = invocation_context._active_background_tools
  cancelled_task = active_tools['fc-cancelled'].task
  gated_task = active_tools['fc-gated'].task
  await asyncio.gather(cancelled_task, return_exceptions=True)

  assert cancelled_task.cancelled()
  assert not gated_task.done()

  release.set()
  await gated_task

  gated_event = function_response_queue.get_nowait()
  assert [fr.id for fr in gated_event.get_function_responses()] == ['fc-gated']


@pytest.mark.asyncio
async def test_launch_live_call_stores_call_in_registry_while_running() -> None:
  """The running call is tracked in the registry under its call id."""
  tool, release = _gated_lookup_tool()
  invocation_context = await _context_with_tools(tool)
  function_call = types.FunctionCall(
      name='gated_lookup_tool', id='fc-1', args={'label': 'A'}
  )

  _live_caller._launch_function_call_live(
      invocation_context,
      function_call,
      {tool.name: tool},
      invocation_context.agent,
      asyncio.Queue(),
      None,
  )

  entry = (invocation_context._active_background_tools or {})['fc-1']
  assert entry.tool_name == 'gated_lookup_tool'
  assert entry.function_call_id == 'fc-1'
  assert not entry.task.done()

  release.set()
  await entry.task


@pytest.mark.asyncio
async def test_launch_live_call_returns_correct_response_when_done() -> None:
  """The call's response names the call and carries the tool's result."""
  tool = FunctionTool(lookup_tool)
  invocation_context = await _context_with_tools(tool)
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='lookup_tool', id='fc-1', args={'label': 'A'}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  (function_response,) = (
      function_response_queue.get_nowait().get_function_responses()
  )
  assert function_response.name == 'lookup_tool'
  assert function_response.id == 'fc-1'
  assert function_response.response == {'label': 'A'}


@pytest.mark.asyncio
async def test_launch_live_call_stores_response_in_queue_when_done() -> None:
  """The response is put on the queue once the call is done, not before."""
  tool, release = _gated_lookup_tool()
  invocation_context = await _context_with_tools(tool)
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(
          name='gated_lookup_tool', id='fc-1', args={'label': 'A'}
      ),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  task = invocation_context._active_background_tools['fc-1'].task
  await asyncio.sleep(0)
  assert function_response_queue.empty()

  release.set()
  await task

  assert function_response_queue.qsize() == 1
  response_event = function_response_queue.get_nowait()
  assert [fr.id for fr in response_event.get_function_responses()] == ['fc-1']


@pytest.mark.asyncio
async def test_launch_live_call_sets_matching_live_session_id() -> None:
  """The response carries the live session id of the call's event."""
  tool = FunctionTool(lookup_tool)
  invocation_context = await _context_with_tools(tool)
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='lookup_tool', id='fc-1', args={'label': 'A'}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      'live-1',
  )
  await invocation_context._active_background_tools['fc-1'].task

  assert function_response_queue.get_nowait().live_session_id == 'live-1'


@pytest.mark.asyncio
async def test_launch_live_call_queues_nothing_for_long_running_tool() -> None:
  """A long-running call that returns nothing puts nothing on the queue."""

  def start_job(label: str) -> None:
    """Starts a job whose result arrives later."""

  tool = LongRunningFunctionTool(start_job)
  invocation_context = await _context_with_tools(tool)
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='start_job', id='fc-1', args={'label': 'A'}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  assert function_response_queue.empty()


@pytest.mark.asyncio
async def test_launch_live_call_queues_error_on_exception() -> None:
  """A call that raises puts an error response for the call on the queue."""
  tool = FunctionTool(broken_tool)
  invocation_context = await _context_with_tools(tool)
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='broken_tool', id='fc-1', args={'label': 'A'}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  (function_response,) = (
      function_response_queue.get_nowait().get_function_responses()
  )
  assert function_response.id == 'fc-1'
  assert function_response.response == {
      'error': 'Invoking `broken_tool()` failed with an internal error.'
  }


@pytest.mark.asyncio
async def test_launch_live_call_queues_error_on_streaming_exception() -> None:
  """A stream that raises puts an error response for the call on the queue."""
  tool = FunctionTool(broken_streaming_tool)
  invocation_context = await _context_with_tools(tool)
  invocation_context.live_request_queue = LiveRequestQueue()
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(
          name='broken_streaming_tool', id='fc-1', args={'label': 'A'}
      ),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  function_response_queue.get_nowait()  # The "pending" response.
  (function_response,) = (
      function_response_queue.get_nowait().get_function_responses()
  )
  assert function_response.id == 'fc-1'
  assert function_response.response == {
      'error': (
          'Invoking `broken_streaming_tool()` failed with an internal error.'
      )
  }


@pytest.mark.asyncio
async def test_launch_live_call_removes_call_from_registry_when_done() -> None:
  """The call's registry entry is removed once the call is done."""
  tool, release = _gated_lookup_tool()
  invocation_context = await _context_with_tools(tool)

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(
          name='gated_lookup_tool', id='fc-1', args={'label': 'A'}
      ),
      {tool.name: tool},
      invocation_context.agent,
      asyncio.Queue(),
      None,
  )
  task = invocation_context._active_background_tools['fc-1'].task
  assert 'fc-1' in invocation_context._active_background_tools

  release.set()
  await task

  assert 'fc-1' not in (invocation_context._active_background_tools or {})


@pytest.mark.asyncio
async def test_launch_live_call_streams_in_the_call_task() -> None:
  """A streaming call answers "pending", then streams in the same task."""
  tool = FunctionTool(streaming_tool)
  invocation_context = await _context_with_tools(tool)
  invocation_context.live_request_queue = LiveRequestQueue()
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='streaming_tool', id='fc-1', args={'label': 'A'}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  entry = invocation_context._active_background_tools['fc-1']
  pending_event = await asyncio.wait_for(function_response_queue.get(), 1)
  chunk = await asyncio.wait_for(invocation_context.live_request_queue.get(), 1)

  (pending_response,) = pending_event.get_function_responses()
  assert 'pending' in pending_response.response['status']
  assert chunk.content.parts[0].function_response.id == 'fc-1'
  assert invocation_context._active_background_tools['fc-1'] is entry
  assert entry.is_generator
  assert not entry.task.done()

  entry.task.cancel()
  await asyncio.gather(entry.task, return_exceptions=True)
  assert 'fc-1' not in (invocation_context._active_background_tools or {})


@pytest.mark.asyncio
async def test_launch_live_call_no_stream_if_before_callback_answers() -> None:
  """A streaming call answered by a before-tool callback never streams."""
  stream_started = False

  async def watch() -> AsyncGenerator[str, None]:
    nonlocal stream_started
    stream_started = True
    yield 'chunk'

  tool = FunctionTool(watch)
  invocation_context = await _context_with_tools(
      tool, before_tool_callback=mock.Mock(return_value={'answer': 'callback'})
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='watch', id='fc-1', args={}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  (function_response,) = (
      function_response_queue.get_nowait().get_function_responses()
  )
  assert function_response.response == {'answer': 'callback'}
  assert not stream_started


@pytest.mark.asyncio
async def test_launch_live_call_no_stream_if_after_callback_fails() -> None:
  """A streaming call whose after-tool callback raises never streams."""
  stream_started = False

  async def watch() -> AsyncGenerator[str, None]:
    nonlocal stream_started
    stream_started = True
    yield 'chunk'

  tool = FunctionTool(watch)
  invocation_context = await _context_with_tools(
      tool, after_tool_callback=mock.Mock(side_effect=RuntimeError('boom'))
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  function_response_queue: asyncio.Queue[Event] = asyncio.Queue()

  _live_caller._launch_function_call_live(
      invocation_context,
      types.FunctionCall(name='watch', id='fc-1', args={}),
      {tool.name: tool},
      invocation_context.agent,
      function_response_queue,
      None,
  )
  await invocation_context._active_background_tools['fc-1'].task

  (function_response,) = (
      function_response_queue.get_nowait().get_function_responses()
  )
  assert 'error' in function_response.response
  assert not stream_started
