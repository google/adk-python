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

"""Tests for ParallelAgent.branch_idle_timeout_secs (issue #5455).

The bug: when an upstream model stream silently stalls (connection stays open,
no chunks arrive, no exception), process_an_agent in _merge_agent_run blocks
forever on events_for_one_agent.__anext__(), queue.get() in the merge loop
waits forever, and the SSE stream never closes.

The fix: _iter_with_idle_timeout wraps each __anext__ call with
asyncio.wait_for so that a stalled branch raises asyncio.TimeoutError instead
of hanging indefinitely.
"""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.parallel_agent import _iter_with_idle_timeout
from google.adk.agents.parallel_agent import _merge_agent_run
from google.adk.agents.parallel_agent import _merge_agent_run_pre_3_11
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
from typing_extensions import override

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_event(author: str, text: str = 'hello') -> Event:
  return Event(
      author=author,
      invocation_id='test-inv',
      content=types.Content(parts=[types.Part(text=text)]),
  )


async def _make_ctx(agent: BaseAgent) -> InvocationContext:
  svc = InMemorySessionService()
  session = await svc.create_session(app_name='app', user_id='user')
  return InvocationContext(
      invocation_id='test-inv',
      agent=agent,
      session=session,
      session_service=svc,
      resumability_config=ResumabilityConfig(is_resumable=False),
  )


class _NormalAgent(BaseAgent):
  """Yields one event then exits."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield _make_event(self.name)


class _StallingAgent(BaseAgent):
  """Yields one event then stalls forever (simulates silent model freeze).

  This is the core reproduction of issue #5455: the branch produces some
  events (tool call responses arrive) then the model stream stops emitting
  chunks without closing the connection.
  """

  stall_after: int = 1
  """Number of events to yield before stalling."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for i in range(self.stall_after):
      yield _make_event(self.name, f'event-{i}')
    # Simulate a connection that is open but silent.
    await asyncio.sleep(3600)  # stall for 1 hour
    yield _make_event(self.name, 'never-reached')  # pragma: no cover


class _SlowAgent(BaseAgent):
  """Yields one event after a short but noticeable delay."""

  delay: float = 0.05

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    await asyncio.sleep(self.delay)
    yield _make_event(self.name)


# ---------------------------------------------------------------------------
# Unit tests for _iter_with_idle_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_with_idle_timeout_passes_events_through():
  """Normal generator: all events pass through unmodified."""

  async def _gen():
    for i in range(3):
      yield _make_event('agent', f'msg-{i}')

  events = [
      e
      async for e in _iter_with_idle_timeout(
          _gen(), timeout_secs=1.0, branch_name='test'
      )
  ]
  assert len(events) == 3
  assert [e.content.parts[0].text for e in events] == [
      'msg-0',
      'msg-1',
      'msg-2',
  ]


@pytest.mark.asyncio
async def test_iter_with_idle_timeout_raises_on_stall():
  """Generator that stalls: TimeoutError is raised with a descriptive message."""

  async def _stalling_gen():
    yield _make_event('agent', 'first')
    await asyncio.sleep(3600)  # simulate silent stall
    yield _make_event('agent', 'never')  # pragma: no cover

  gen = _stalling_gen()
  it = _iter_with_idle_timeout(gen, timeout_secs=0.05, branch_name='my-branch')

  first = await it.__anext__()
  assert first.content.parts[0].text == 'first'

  with pytest.raises(asyncio.TimeoutError) as exc_info:
    await it.__anext__()

  assert 'my-branch' in str(exc_info.value)
  assert (
      '0.05' in str(exc_info.value)
      or '0.1' in str(exc_info.value)
      or 'idle' in str(exc_info.value).lower()
  )
  await gen.aclose()


@pytest.mark.asyncio
async def test_iter_with_idle_timeout_stops_on_empty_generator():
  """Empty generator terminates cleanly."""

  async def _empty():
    return
    yield  # make it an async generator  # noqa: unreachable

  events = [
      e
      async for e in _iter_with_idle_timeout(
          _empty(), timeout_secs=1.0, branch_name='empty'
      )
  ]
  assert events == []


# ---------------------------------------------------------------------------
# Integration tests for _merge_agent_run (3.11+) and _merge_agent_run_pre_3_11
# ---------------------------------------------------------------------------


async def _gen_from_events(events: list[Event]) -> AsyncGenerator[Event, None]:
  for e in events:
    yield e


async def _stalling_gen(
    yield_first: bool = True,
) -> AsyncGenerator[Event, None]:
  if yield_first:
    yield _make_event('stalling')
  await asyncio.sleep(3600)
  yield _make_event('never')  # pragma: no cover


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'merge_fn',
    [
        _merge_agent_run,
        _merge_agent_run_pre_3_11,
    ],
)
async def test_merge_no_timeout_normal(merge_fn):
  """Without timeout, two normal generators merge correctly."""
  runs = [
      _gen_from_events([_make_event('a', '1'), _make_event('a', '2')]),
      _gen_from_events([_make_event('b', '3')]),
  ]
  events = [e async for e in merge_fn(runs)]
  authors = {e.author for e in events}
  assert authors == {'a', 'b'}
  assert len(events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'merge_fn',
    [
        _merge_agent_run,
        _merge_agent_run_pre_3_11,
    ],
)
async def test_merge_timeout_stalled_branch_raises(merge_fn):
  """A stalled branch raises TimeoutError, unblocking the merge loop."""
  runs = [
      _gen_from_events([_make_event('normal', 'ok')]),
      _stalling_gen(yield_first=True),
  ]
  with pytest.raises(
      (asyncio.TimeoutError, ExceptionGroup, BaseExceptionGroup)
  ):
    events = []
    async for e in merge_fn(
        runs,
        branch_names=['normal', 'stalling'],
        branch_idle_timeout_secs=0.05,
    ):
      events.append(e)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'merge_fn',
    [
        _merge_agent_run,
        _merge_agent_run_pre_3_11,
    ],
)
async def test_merge_timeout_does_not_fire_for_fast_branches(merge_fn):
  """Timeout is generous enough that normal (fast) branches complete cleanly."""
  runs = [
      _gen_from_events([_make_event('a', '1'), _make_event('a', '2')]),
      _gen_from_events([_make_event('b', '3')]),
  ]
  events = [
      e
      async for e in merge_fn(
          runs,
          branch_idle_timeout_secs=5.0,  # generous
      )
  ]
  assert len(events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'merge_fn',
    [
        _merge_agent_run,
        _merge_agent_run_pre_3_11,
    ],
)
async def test_merge_timeout_none_disables_guard(merge_fn):
  """branch_idle_timeout_secs=None preserves the original unbounded behaviour."""
  runs = [
      _gen_from_events([_make_event('a')]),
      _gen_from_events([_make_event('b')]),
  ]
  # Should complete without any timeout machinery
  events = [e async for e in merge_fn(runs, branch_idle_timeout_secs=None)]
  assert len(events) == 2


# ---------------------------------------------------------------------------
# End-to-end ParallelAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_agent_no_timeout_field_by_default():
  """branch_idle_timeout_secs defaults to None (no timeout)."""
  agent = ParallelAgent(name='p', sub_agents=[])
  assert agent.branch_idle_timeout_secs is None


@pytest.mark.asyncio
async def test_parallel_agent_completes_normally_with_timeout_set():
  """With timeout set, normal branches complete and all events arrive."""
  a1 = _NormalAgent(name='a1')
  a2 = _NormalAgent(name='a2')
  parent = ParallelAgent(
      name='parent',
      sub_agents=[a1, a2],
      branch_idle_timeout_secs=5.0,
  )
  ctx = await _make_ctx(parent)
  events = [e async for e in parent.run_async(ctx)]
  authors = {e.author for e in events}
  assert 'a1' in authors
  assert 'a2' in authors


@pytest.mark.asyncio
async def test_parallel_agent_stalled_branch_raises_timeout():
  """A branch that stalls causes the ParallelAgent to raise TimeoutError.

  This is the direct reproduction of issue #5455: one sub-agent produces some
  events (like tool responses) then the model stream silently stops.  Without
  branch_idle_timeout_secs the SSE stream hangs forever.  With it, a
  TimeoutError is raised promptly so the caller can close the stream.
  """
  normal = _NormalAgent(name='normal')
  stalling = _StallingAgent(name='stalling', stall_after=1)
  parent = ParallelAgent(
      name='parent',
      sub_agents=[normal, stalling],
      branch_idle_timeout_secs=0.1,
  )
  ctx = await _make_ctx(parent)

  with pytest.raises(
      (asyncio.TimeoutError, ExceptionGroup, BaseExceptionGroup)
  ):
    async for _ in parent.run_async(ctx):
      pass


@pytest.mark.asyncio
async def test_parallel_agent_stalled_from_start_raises_timeout():
  """Branch that stalls immediately (no events at all) also detected."""
  normal = _NormalAgent(name='normal')
  stalling = _StallingAgent(name='stalling', stall_after=0)
  parent = ParallelAgent(
      name='parent',
      sub_agents=[normal, stalling],
      branch_idle_timeout_secs=0.1,
  )
  ctx = await _make_ctx(parent)

  with pytest.raises(
      (asyncio.TimeoutError, ExceptionGroup, BaseExceptionGroup)
  ):
    async for _ in parent.run_async(ctx):
      pass


@pytest.mark.asyncio
async def test_parallel_agent_slow_but_not_stalled_completes():
  """A slow branch (delay < timeout) is not incorrectly cancelled."""
  slow = _SlowAgent(name='slow', delay=0.05)
  normal = _NormalAgent(name='normal')
  parent = ParallelAgent(
      name='parent',
      sub_agents=[normal, slow],
      branch_idle_timeout_secs=5.0,  # generous — should not fire
  )
  ctx = await _make_ctx(parent)
  events = [e async for e in parent.run_async(ctx)]
  authors = {e.author for e in events}
  assert 'normal' in authors
  assert 'slow' in authors


@pytest.mark.asyncio
async def test_parallel_agent_timeout_logged_as_warning(caplog):
  """Warning is logged with branch name and timeout duration when stall fires."""
  import logging

  stalling = _StallingAgent(name='stalling_branch', stall_after=0)
  parent = ParallelAgent(
      name='parent',
      sub_agents=[stalling],
      branch_idle_timeout_secs=0.05,
  )
  ctx = await _make_ctx(parent)

  with pytest.raises(
      (asyncio.TimeoutError, ExceptionGroup, BaseExceptionGroup)
  ):
    with caplog.at_level(logging.WARNING):
      async for _ in parent.run_async(ctx):
        pass

  assert any('stalling_branch' in r.message for r in caplog.records)
  assert any(
      'idle' in r.message.lower() or 'stall' in r.message.lower()
      for r in caplog.records
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason='ExceptionGroup only available on Python 3.11+',
)
async def test_parallel_agent_timeout_error_is_timeout_in_exception_group():
  """On 3.11+ the raised ExceptionGroup wraps a TimeoutError."""
  stalling = _StallingAgent(name='stalling', stall_after=0)
  parent = ParallelAgent(
      name='parent',
      sub_agents=[stalling],
      branch_idle_timeout_secs=0.05,
  )
  ctx = await _make_ctx(parent)

  with pytest.raises(BaseExceptionGroup) as exc_info:
    async for _ in parent.run_async(ctx):
      pass

  flat = exc_info.value.exceptions
  assert any(isinstance(e, asyncio.TimeoutError) for e in flat)
