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

from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.core import _function_call_postprocessor
from google.adk.flows.llm_flows.core._resume import decide_step_resume
from google.adk.flows.llm_flows.core._resume import ResumeAction
from google.adk.flows.llm_flows.tools import _batch_executor as _batch_tool_executor
from google.adk.flows.llm_flows.tools._caller import _PreparedFunctionCall
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow._errors import NodeInterruptedError
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


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_siblings_on_cancelled_error() -> None:
  """A child-raised CancelledError cancels unfinished sibling tasks."""
  started = asyncio.Event()
  sibling_cancelled = False
  sibling_finished = False

  async def failing_worker() -> None:
    await started.wait()
    raise asyncio.CancelledError()

  async def slow_worker() -> None:
    nonlocal sibling_cancelled, sibling_finished
    started.set()
    try:
      await asyncio.sleep(10)
      sibling_finished = True
    except asyncio.CancelledError:
      sibling_cancelled = True
      raise

  tasks = [
      asyncio.create_task(failing_worker()),
      asyncio.create_task(slow_worker()),
  ]
  with pytest.raises(asyncio.CancelledError):
    await _batch_tool_executor._gather_or_cancel(tasks)

  assert sibling_cancelled
  assert not sibling_finished
  assert all(t.done() for t in tasks)


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_siblings_on_self_cancel() -> None:
  """A child that cancels its own task still tears down unfinished siblings."""
  started = asyncio.Event()
  sibling_cancelled = False
  sibling_finished = False

  async def self_cancelling_worker() -> None:
    await started.wait()
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    await asyncio.sleep(0)

  async def slow_worker() -> None:
    nonlocal sibling_cancelled, sibling_finished
    started.set()
    try:
      await asyncio.sleep(10)
      sibling_finished = True
    except asyncio.CancelledError:
      sibling_cancelled = True
      raise

  tasks = [
      asyncio.create_task(self_cancelling_worker()),
      asyncio.create_task(slow_worker()),
  ]
  with pytest.raises(asyncio.CancelledError):
    await _batch_tool_executor._gather_or_cancel(tasks)

  assert sibling_cancelled
  assert not sibling_finished
  assert all(t.done() for t in tasks)


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_and_awaits_tasks_on_cancelled_error() -> (
    None
):
  """Cancelling the gather task cancels and awaits unfinished child tasks."""
  started = asyncio.Event()
  cleanup_finished = False

  async def fast_cancel_worker() -> None:
    await asyncio.sleep(10)

  async def slow_cleanup_worker() -> None:
    nonlocal cleanup_finished
    started.set()
    try:
      await asyncio.sleep(10)
    finally:
      await asyncio.sleep(0.02)
      cleanup_finished = True

  fast_task = asyncio.create_task(fast_cancel_worker())
  slow_task = asyncio.create_task(slow_cleanup_worker())
  gather_task = asyncio.create_task(
      _batch_tool_executor._gather_or_cancel([fast_task, slow_task])
  )
  await started.wait()

  gather_task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await gather_task

  assert fast_task.cancelled()
  assert slow_task.cancelled()
  assert cleanup_finished


@pytest.mark.asyncio
async def test_gather_or_cancel_cancels_tasks_on_generator_exit() -> None:
  """Closing the coroutine mid-await (GeneratorExit) still cancels the child tasks."""

  async def slow_worker() -> None:
    await asyncio.sleep(10)

  task = asyncio.create_task(slow_worker())
  coro = _batch_tool_executor._gather_or_cancel([task])
  coro.send(None)
  coro.close()
  await asyncio.sleep(0)

  assert task.cancelled()


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


@pytest.mark.asyncio
async def test_execute_prepared_function_calls_sets_live_session_id_and_merges() -> (
    None
):
  mock_ic = mock.MagicMock()
  mock_ic.session.state = {}
  prepared_1 = _prepared_call_with_snapshot('call-1')
  prepared_2 = _prepared_call_with_snapshot('call-2')

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
  call_executor = mock.AsyncMock(side_effect=[ev1, ev2])

  merged = await _batch_tool_executor._execute_prepared_function_calls(
      mock_ic,
      [prepared_1, prepared_2],
      call_executor=call_executor,
      live_session_id='live-sess-1',
  )

  assert merged is not None
  assert merged.live_session_id == 'live-sess-1'
  assert merged.content is not None
  assert [p.text for p in merged.content.parts] == ['r1', 'r2']


def _response_event(name: str, **actions) -> Event:
  return Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='user',
          parts=[types.Part.from_function_response(name=name, response={})],
      ),
      actions=EventActions(**actions),
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'name, actions, kept',
    [
        ('plain', {}, True),
        ('state_delta', {'state_delta': {'k': 'v'}}, True),
        ('transfer', {'transfer_to_agent': 'other'}, False),
        ('escalate', {'escalate': True}, False),
        (
            'confirmation',
            {'requested_tool_confirmations': {'id': ToolConfirmation()}},
            False,
        ),
        ('set_model_response', {'set_model_response': {'a': 1}}, False),
        (FINISH_TASK_TOOL_NAME, {}, False),
    ],
)
async def test_keep_completed_responses_keeps_only_finished_plain_results(
    name: str, actions: dict, kept: bool
) -> None:
  """Finished plain results are merged and kept; control actions are not."""
  mock_ic = mock.MagicMock()
  mock_ic.session.state = {}
  mock_ic.live_request_queue = None
  mock_ic.is_aborted = False

  async def returns(event):
    return event

  async def raises():
    raise RuntimeError('boom')

  tasks = [
      asyncio.create_task(returns(_response_event('fast'))),
      asyncio.create_task(asyncio.Event().wait()),
      asyncio.create_task(raises()),
      asyncio.create_task(returns(None)),
      asyncio.create_task(returns(_response_event(name, **actions))),
  ]
  await asyncio.sleep(0)
  tasks[1].cancel()
  await asyncio.gather(*tasks, return_exceptions=True)

  # The cancelled call, the call that raised and the None result get nothing.
  merged = _batch_tool_executor._keep_completed_responses(mock_ic, tasks)
  assert merged is not None
  assert [r.name for r in merged.get_function_responses()] == (
      ['fast', name] if kept else ['fast']
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'live_request_queue, is_aborted',
    [
        (mock.MagicMock(), False),
        (None, True),
    ],
)
async def test_keep_completed_responses_skips_live_or_aborted_invocation(
    live_request_queue: object, is_aborted: bool
) -> None:
  """Live mode and aborted invocations do not keep completed responses."""
  mock_ic = mock.MagicMock()
  mock_ic.session.state = {}
  mock_ic.live_request_queue = live_request_queue
  mock_ic.is_aborted = is_aborted

  async def returns(event):
    return event

  task = asyncio.create_task(returns(_response_event('fast')))
  await task

  assert _batch_tool_executor._keep_completed_responses(mock_ic, [task]) is None


@pytest.mark.asyncio
async def test_execute_and_postprocess_preserve_completed_responses_on_node_interrupted() -> (
    None
):
  """Completed sibling tool responses are yielded on Turn 1 and skipped on Turn 2 resume."""
  # Arrange
  mock_ic = mock.MagicMock()
  mock_ic.session.state = {}
  mock_ic.live_request_queue = None
  mock_ic.is_aborted = False
  mock_ic.is_resumable = True
  mock_ic.branch = None
  mock_ic.agent.name = 'agent'
  mock_ic.should_pause_invocation.return_value = False

  prepared_1 = _prepared_call_with_snapshot('call-1')
  prepared_1.function_call = types.FunctionCall(
      name='fast', id='call-1', args={}
  )
  prepared_1.tool = BaseTool(name='fast', description='desc')
  prepared_2 = _prepared_call_with_snapshot('call-2')
  prepared_2.function_call = types.FunctionCall(
      name='slow', id='call-2', args={}
  )
  prepared_2.tool = BaseTool(name='slow', description='desc')

  fast_event = _response_event('fast', state_delta={'saved': True})
  fast_event.content.parts[0].function_response.id = 'call-1'
  slow_event = _response_event('slow', state_delta={'finished': True})
  slow_event.content.parts[0].function_response.id = 'call-2'

  fast_runs = 0
  slow_runs = 0

  async def run_one(prepared: _PreparedFunctionCall) -> Event | None:
    nonlocal fast_runs, slow_runs
    if prepared is prepared_1:
      fast_runs += 1
      return fast_event
    slow_runs += 1
    await asyncio.sleep(0)
    if slow_runs == 1:
      raise NodeInterruptedError()
    return slow_event

  async def fake_handle_function_calls(ic, fc_event, tools_dict):
    call_ids = {fc.id for fc in fc_event.get_function_calls()}
    active_prepared = [
        p
        for p in (prepared_1, prepared_2)
        if not call_ids or p.function_call.id in call_ids
    ]
    return await _batch_tool_executor._execute_prepared_function_calls(
        ic, active_prepared, call_executor=run_one
    )

  llm_request = mock.MagicMock()
  llm_request.tools_dict = {'fast': prepared_1.tool, 'slow': prepared_2.tool}
  fc_event = Event(
      invocation_id='inv-1',
      author='agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(function_call=prepared_1.function_call),
              types.Part(function_call=prepared_2.function_call),
          ],
      ),
  )

  # Act / Assert - Turn 1: fast completes and is yielded before NodeInterruptedError.
  emitted: list[Event] = []
  with mock.patch(
      'google.adk.flows.llm_flows.functions.handle_function_calls_async',
      side_effect=fake_handle_function_calls,
  ):
    with pytest.raises(NodeInterruptedError):
      async for (
          ev
      ) in _function_call_postprocessor.postprocess_handle_function_calls_async(
          mock_ic, fc_event, llm_request
      ):
        emitted.append(ev)

    assert [ev.id for ev in emitted] == [fast_event.id]
    assert emitted[0].actions.state_delta == {'saved': True}
    assert fast_runs == 1
    assert slow_runs == 1

    # Turn 2: resume replays only slow (call-2) without re-running fast (call-1).
    sub_reply = Event(
        invocation_id='inv-1',
        author='user',
        branch='agent.slow@call-2',
        content=types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id='sub-1',
                        name='adk_request_input',
                        response={'answer': 'ok'},
                    )
                )
            ],
        ),
    )
    mock_ic._get_events.return_value = [fc_event, *emitted, sub_reply]
    decision = decide_step_resume(mock_ic, llm_request.tools_dict)
    assert decision.action is ResumeAction.REPLAY_CALLS
    assert [fc.id for fc in decision.replay_event().get_function_calls()] == [
        'call-2'
    ]

    turn2_emitted: list[Event] = []
    async for (
        ev
    ) in _function_call_postprocessor.postprocess_handle_function_calls_async(
        mock_ic, decision.replay_event(), llm_request
    ):
      turn2_emitted.append(ev)

  assert [ev.id for ev in turn2_emitted] == [slow_event.id]
  assert fast_runs == 1
  assert slow_runs == 2


@pytest.mark.asyncio
async def test_handle_function_calls_live_compat_wrapper() -> None:
  mock_context = mock.MagicMock()
  mock_event = mock.MagicMock()
  mock_tools = {'tool': mock.MagicMock()}
  mock_result = mock.MagicMock()

  with mock.patch(
      'google.adk.flows.llm_flows.tools._live_caller.handle_function_calls_live',
      new_callable=mock.AsyncMock,
      return_value=mock_result,
  ) as mock_handle:
    res = await _batch_tool_executor.handle_function_calls_live(
        mock_context, mock_event, mock_tools
    )
    assert res is mock_result
    mock_handle.assert_awaited_once_with(
        invocation_context=mock_context,
        function_call_event=mock_event,
        tools_dict=mock_tools,
    )
