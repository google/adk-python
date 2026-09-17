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

"""Tests for the executor interceptor pipeline and its context object."""

from __future__ import annotations

from unittest.mock import Mock

from google.adk.a2a import _compat
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.a2a.executor.executor_context import ExecutorContext
from google.adk.a2a.executor.utils import execute_after_agent_interceptors
from google.adk.a2a.executor.utils import execute_after_event_interceptors
from google.adk.a2a.executor.utils import execute_before_agent_interceptors
from google.adk.events.event import Event
from google.adk.runners import Runner
import pytest


def _executor_context() -> ExecutorContext:
  return ExecutorContext(
      app_name='test-app',
      user_id='test-user',
      session_id='test-session',
      runner=Mock(spec=Runner),
  )


def _adk_event() -> Event:
  return Event(author='test-agent', invocation_id='inv-1')


def _a2a_event(task_id: str):
  return _compat.make_task_status_update_event(
      task_id=task_id,
      context_id='ctx-1',
      status=_compat.make_task_status(_compat.TS_WORKING),
      final=False,
  )


# -----------------------------------------------------------------------------
# execute_before_agent_interceptors
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize('interceptors', [None, []])
async def test_execute_before_agent_interceptors_no_hooks_returns_context(
    interceptors,
):
  context = Mock(name='request-context')
  assert await execute_before_agent_interceptors(context, interceptors) is (
      context
  )


@pytest.mark.asyncio
async def test_execute_before_agent_interceptors_threads_context_in_order():
  original, first_out, second_out = (
      Mock(name='original'),
      Mock(name='first-out'),
      Mock(name='second-out'),
  )
  seen = []

  async def first(context):
    seen.append(context)
    return first_out

  async def second(context):
    seen.append(context)
    return second_out

  result = await execute_before_agent_interceptors(
      original,
      [
          ExecuteInterceptor(before_agent=first),
          ExecuteInterceptor(before_agent=second),
      ],
  )

  # Each hook must see the previous hook's return value, not the original.
  assert seen == [original, first_out]
  assert result is second_out


@pytest.mark.asyncio
async def test_execute_before_agent_interceptors_skips_interceptor_without_hook():
  original, replacement = Mock(name='original'), Mock(name='replacement')

  async def replace(context):
    del context
    return replacement

  result = await execute_before_agent_interceptors(
      original,
      [
          ExecuteInterceptor(after_event=_unused_after_event),
          ExecuteInterceptor(before_agent=replace),
      ],
  )

  assert result is replacement


async def _unused_after_event(executor_context, a2a_event, adk_event):
  raise AssertionError('after_event must not run in the before_agent phase')


# -----------------------------------------------------------------------------
# execute_after_event_interceptors
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize('interceptors', [None, []])
async def test_execute_after_event_interceptors_no_hooks_returns_single_event(
    interceptors,
):
  event = _a2a_event('task-1')

  result = await execute_after_event_interceptors(
      event, _executor_context(), _adk_event(), interceptors
  )

  assert result == [event]


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_single_return_replaces_event():
  replacement = _a2a_event('replacement')

  async def replace(executor_context, a2a_event, adk_event):
    del executor_context, a2a_event, adk_event
    return replacement

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      _executor_context(),
      _adk_event(),
      [ExecuteInterceptor(after_event=replace)],
  )

  assert result == [replacement]


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_list_return_fans_out_in_order():
  first, second = _a2a_event('first'), _a2a_event('second')

  async def fan_out(executor_context, a2a_event, adk_event):
    del executor_context, a2a_event, adk_event
    return [first, second]

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      _executor_context(),
      _adk_event(),
      [ExecuteInterceptor(after_event=fan_out)],
  )

  assert result == [first, second]


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_none_return_drops_the_event():
  async def drop(executor_context, a2a_event, adk_event):
    del executor_context, a2a_event, adk_event
    return None

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      _executor_context(),
      _adk_event(),
      [ExecuteInterceptor(after_event=drop)],
  )

  assert result == []


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_drop_halts_later_hooks():
  later_calls = []

  async def drop(executor_context, a2a_event, adk_event):
    del executor_context, a2a_event, adk_event
    return None

  async def later(executor_context, a2a_event, adk_event):
    del executor_context, adk_event
    later_calls.append(a2a_event)
    return a2a_event

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      _executor_context(),
      _adk_event(),
      [
          ExecuteInterceptor(after_event=drop),
          ExecuteInterceptor(after_event=later),
      ],
  )

  assert result == []
  # Dropping the event ends the chain; downstream hooks never see it.
  assert later_calls == []


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_later_hook_sees_each_fanned_event():
  first, second = _a2a_event('first'), _a2a_event('second')
  executor_context, adk_event = _executor_context(), _adk_event()
  seen = []

  async def fan_out(ctx, a2a_event, event):
    del ctx, a2a_event, event
    return [first, second]

  async def observe(ctx, a2a_event, event):
    seen.append((ctx, a2a_event, event))
    return a2a_event

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      executor_context,
      adk_event,
      [
          ExecuteInterceptor(after_event=fan_out),
          ExecuteInterceptor(after_event=observe),
      ],
  )

  # The second hook runs once per event the first produced, not once for the
  # event that entered the chain.
  assert [event for _, event, _ in seen] == [first, second]
  assert all(ctx is executor_context for ctx, _, _ in seen)
  assert all(event is adk_event for _, _, event in seen)
  assert result == [first, second]


@pytest.mark.asyncio
async def test_execute_after_event_interceptors_skips_interceptor_without_hook():
  replacement = _a2a_event('replacement')

  async def replace(executor_context, a2a_event, adk_event):
    del executor_context, a2a_event, adk_event
    return replacement

  result = await execute_after_event_interceptors(
      _a2a_event('task-1'),
      _executor_context(),
      _adk_event(),
      [
          ExecuteInterceptor(before_agent=_unused_before_agent),
          ExecuteInterceptor(after_event=replace),
      ],
  )

  assert result == [replacement]


async def _unused_before_agent(context):
  raise AssertionError('before_agent must not run in the after_event phase')


# -----------------------------------------------------------------------------
# execute_after_agent_interceptors
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize('interceptors', [None, []])
async def test_execute_after_agent_interceptors_no_hooks_returns_final_event(
    interceptors,
):
  final_event = _a2a_event('task-1')

  result = await execute_after_agent_interceptors(
      _executor_context(), final_event, interceptors
  )

  assert result is final_event


@pytest.mark.asyncio
async def test_execute_after_agent_interceptors_runs_in_reverse_order():
  entered, outer_out, inner_out = (
      _a2a_event('entered'),
      _a2a_event('outer'),
      _a2a_event('inner'),
  )
  seen = []

  async def outer(executor_context, final_event):
    del executor_context
    seen.append(final_event)
    return outer_out

  async def inner(executor_context, final_event):
    del executor_context
    seen.append(final_event)
    return inner_out

  result = await execute_after_agent_interceptors(
      _executor_context(),
      entered,
      [
          ExecuteInterceptor(after_agent=outer),
          ExecuteInterceptor(after_agent=inner),
      ],
  )

  # after_agent unwinds the interceptor stack: the last-registered hook runs
  # first, and each hook sees the previous one's return value.
  assert seen == [entered, inner_out]
  assert result is outer_out


@pytest.mark.asyncio
async def test_execute_after_agent_interceptors_skips_interceptor_without_hook():
  replacement = _a2a_event('replacement')

  async def replace(executor_context, final_event):
    del executor_context, final_event
    return replacement

  result = await execute_after_agent_interceptors(
      _executor_context(),
      _a2a_event('task-1'),
      [
          ExecuteInterceptor(after_agent=replace),
          ExecuteInterceptor(before_agent=_unused_before_agent),
      ],
  )

  assert result is replacement


# -----------------------------------------------------------------------------
# ExecutorContext
# -----------------------------------------------------------------------------
def test_executor_context_exposes_each_constructor_argument():
  runner = Mock(spec=Runner)
  context = ExecutorContext(
      app_name='app-value',
      user_id='user-value',
      session_id='session-value',
      runner=runner,
  )

  assert context.app_name == 'app-value'
  assert context.user_id == 'user-value'
  assert context.session_id == 'session-value'
  assert context.runner is runner
