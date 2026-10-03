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

"""Tests for FirstMatchNode (race fan-out; cancel losers on first match)."""

import asyncio
import time
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.workflow import FirstMatchNode
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._workflow import Workflow
from pydantic import ConfigDict
import pytest
from typing_extensions import override

from .workflow_testing_utils import get_outputs
from .workflow_testing_utils import run_workflow

# Branch lifecycle events, recorded as (state, name). Module-global so it
# survives the scheduler's ``model_copy`` of branch nodes.
_EVENTS: list[tuple[str, str]] = []


class _SleepBranch(BaseNode):
  """A branch that sleeps, then yields ``result`` -- unless cancelled first."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  delay: float = 0.0
  result: Any = None

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    try:
      await asyncio.sleep(self.delay)
    except asyncio.CancelledError:
      _EVENTS.append(('cancelled', self.name))
      raise
    _EVENTS.append(('done', self.name))
    yield self.result


def _found(result: Any) -> bool:
  return isinstance(result, dict) and bool(result.get('found'))


@pytest.mark.asyncio
async def test_first_match_cancels_slower_branches():
  _EVENTS.clear()
  fast = _SleepBranch(
      name='fast', delay=0.05, result={'found': True, 'src': 'fast'}
  )
  slow1 = _SleepBranch(name='slow1', delay=5.0, result={'found': True})
  slow2 = _SleepBranch(name='slow2', delay=5.0, result={'found': True})
  race = FirstMatchNode(
      name='race', nodes=[fast, slow1, slow2], match=_found
  )
  wf = Workflow(name='w_first', edges=[('START', race)])

  start = time.perf_counter()
  events, _, _ = await run_workflow(wf)
  elapsed = time.perf_counter() - start

  # The fast branch's result is the node output.
  assert {'found': True, 'src': 'fast'} in get_outputs(events)
  # We returned in ~fast time, not waiting on the 5s losers.
  assert elapsed < 2.0
  # The winner finished; both losers were cancelled mid-flight (never "done").
  assert ('done', 'fast') in _EVENTS
  assert ('cancelled', 'slow1') in _EVENTS
  assert ('cancelled', 'slow2') in _EVENTS
  assert ('done', 'slow1') not in _EVENTS
  assert ('done', 'slow2') not in _EVENTS


@pytest.mark.asyncio
async def test_no_branch_matches_yields_default_and_runs_all():
  _EVENTS.clear()
  a = _SleepBranch(name='a', delay=0.02, result={'found': False})
  b = _SleepBranch(name='b', delay=0.04, result={'found': False})
  race = FirstMatchNode(
      name='race',
      nodes=[a, b],
      match=_found,
      no_match_output={'found': False, 'reason': 'nobody had it'},
  )
  wf = Workflow(name='w_none', edges=[('START', race)])

  events, _, _ = await run_workflow(wf)

  assert {'found': False, 'reason': 'nobody had it'} in get_outputs(events)
  # With no early winner, every branch is allowed to finish.
  assert ('done', 'a') in _EVENTS
  assert ('done', 'b') in _EVENTS


@pytest.mark.asyncio
async def test_max_parallel_one_never_starts_later_branches_after_win():
  """Rank-ordered gate: a win short-circuits before lower-ranked reads start."""
  _EVENTS.clear()
  first = _SleepBranch(name='first', delay=0.02, result={'found': True})
  second = _SleepBranch(name='second', delay=0.02, result={'found': True})
  race = FirstMatchNode(
      name='race', nodes=[first, second], match=_found, max_parallel=1
  )
  wf = Workflow(name='w_serial', edges=[('START', race)])

  events, _, _ = await run_workflow(wf)

  assert {'found': True} in get_outputs(events)
  assert ('done', 'first') in _EVENTS
  # The second branch was never launched -- not even started, so not cancelled.
  assert ('done', 'second') not in _EVENTS
  assert ('cancelled', 'second') not in _EVENTS


@pytest.mark.asyncio
async def test_failing_branch_does_not_sink_the_race():
  _EVENTS.clear()

  class _Boom(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @override
    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      raise RuntimeError('branch blew up')
      yield  # pragma: no cover

  boom = _Boom(name='boom')
  good = _SleepBranch(name='good', delay=0.05, result={'found': True})
  race = FirstMatchNode(name='race', nodes=[boom, good], match=_found)
  wf = Workflow(name='w_flaky', edges=[('START', race)])

  events, _, _ = await run_workflow(wf)

  # A single failing source must not deny the answer another source can give.
  assert {'found': True} in get_outputs(events)
  assert ('done', 'good') in _EVENTS
