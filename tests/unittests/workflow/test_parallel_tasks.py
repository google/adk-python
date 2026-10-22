# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner-backed contracts for owned parallel workflow tasks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.state import StateSchemaError
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from . import testing_utils

_MAX_WAIT = 5


async def _run(coordinator, **workflow_options):
  runner = testing_utils.InMemoryRunner(
      node=Workflow(
          name='team', edges=[(START, coordinator)], **workflow_options
      )
  )
  try:
    return await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
  finally:
    await runner.runner.close()


def _response(text: str) -> LlmResponse:
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text=text)])
  )


async def test_four_agents_overlap_while_coordinator_does_independent_work():
  """Spawning is nonblocking and all four real Agent model calls overlap."""
  entered: asyncio.Queue[str] = asyncio.Queue()
  release = asyncio.Event()
  active = 0
  peak = 0
  observations = {}

  class GatedModel(testing_utils.MockModel):

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      nonlocal active, peak
      item = llm_request.contents[-1].parts[0].text
      active += 1
      peak = max(peak, active)
      entered.put_nowait(item)
      try:
        await release.wait()
        yield _response(item)
      finally:
        active -= 1

  agents = [
      LlmAgent(name=f'worker_{i}', model=GatedModel(responses=[]))
      for i in range(4)
  ]

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> list[str]:
    async with ctx.parallel_tasks() as tasks:
      handles = [tasks.spawn(agent, str(i)) for i, agent in enumerate(agents)]
      observations['started'] = [await entered.get() for _ in handles]
      observations['coordinator_work'] = sum(range(10))
      observations['statuses'] = [handle.status for handle in handles]
      release.set()
      done, pending = await tasks.wait(mode='all')
      assert done == handles
      assert pending == []
      observations['outputs'] = [await handle.result() for handle in handles]
      return observations['outputs']

  await _run(coordinate)

  assert set(observations['started']) == {'0', '1', '2', '3'}
  assert observations['coordinator_work'] == 45
  assert observations['statuses'] == ['running'] * 4
  assert observations['outputs'] == ['0', '1', '2', '3']
  assert peak == 4
  assert active == 0


async def test_first_drains_model_and_tool_losers_before_returning():
  """A selected result closes losing model/tool work before parent proceeds."""
  model_entered = asyncio.Event()
  tool_entered = asyncio.Event()
  model_closed = asyncio.Event()
  tool_closed = asyncio.Event()
  never = asyncio.Event()
  calls = []
  observations = {}

  class BlockedModel(testing_utils.MockModel):

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      calls.append('blocked model')
      model_entered.set()
      try:
        await never.wait()
        yield _response('must not complete')
      finally:
        model_closed.set()

  async def blocked_tool() -> str:
    calls.append('blocked tool')
    tool_entered.set()
    try:
      await never.wait()
      return 'must not complete'
    finally:
      tool_closed.set()

  tool_model = testing_utils.MockModel.create([
      types.Part.from_function_call(name='blocked_tool', args={}),
      'must not make another model call',
  ])
  model_worker = LlmAgent(name='model_worker', model=BlockedModel(responses=[]))
  tool_worker = LlmAgent(
      name='tool_worker', model=tool_model, tools=[blocked_tool]
  )

  async def accepted() -> str:
    await model_entered.wait()
    await tool_entered.wait()
    return 'accepted'

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      losers = [
          tasks.spawn(model_worker, 'work'),
          tasks.spawn(tool_worker, 'work'),
      ]
      expected = tasks.spawn(accepted)
      winner = await tasks.first()
      assert winner is expected
      assert model_closed.is_set()
      assert tool_closed.is_set()
      observations['statuses'] = [handle.status for handle in losers]
      observations['calls_at_selection'] = list(calls)
      return await winner.result()

  await _run(coordinate)

  assert observations['statuses'] == ['cancelled', 'cancelled']
  assert calls == observations['calls_at_selection']
  assert len(tool_model.requests) == 1


@pytest.mark.parametrize('async_predicate', [False, True])
async def test_first_rejects_fastest_and_selects_next_acceptable(
    async_predicate,
):
  """Rejected outputs leave other candidates running; both predicate forms work."""
  release_acceptable = asyncio.Event()
  blocked_entered = asyncio.Event()
  blocked_closed = asyncio.Event()
  seen = []

  async def candidate(node_input: str) -> str:
    if node_input == 'rejected':
      await blocked_entered.wait()
    elif node_input == 'accepted':
      await release_acceptable.wait()
    else:
      blocked_entered.set()
      try:
        await asyncio.Event().wait()
      finally:
        blocked_closed.set()
    return node_input

  def accept(output):
    seen.append(output)
    if output == 'rejected':
      release_acceptable.set()
    return output == 'accepted'

  async def accept_async(output):
    return accept(output)

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      handles = [
          tasks.spawn(candidate, value)
          for value in ['rejected', 'accepted', 'blocked']
      ]
      winner = await tasks.first(
          accept=accept_async if async_predicate else accept
      )
      assert winner is handles[1]
      assert blocked_closed.is_set()
      return winner.output

  await _run(coordinate)
  assert seen == ['rejected', 'accepted']


async def test_first_with_cap_one_never_starts_queued_losers():
  """The winner decision pauses admission before a freed slot starts more work."""
  started = []

  async def candidate(node_input: str) -> str:
    started.append(node_input)
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      handles = [
          tasks.spawn(candidate, value)
          for value in ['winner', 'queued1', 'queued2']
      ]
      winner = await tasks.first()
      assert winner is handles[0]
      assert [handle.status for handle in handles] == [
          'completed',
          'cancelled',
          'cancelled',
      ]
      return winner.output

  await _run(coordinate)
  assert started == ['winner']


async def test_cancelled_result_observer_does_not_cancel_worker():
  """Cancellation of an observer is separate from cancellation of owned work."""
  entered = asyncio.Event()
  release = asyncio.Event()
  closed = asyncio.Event()

  async def candidate() -> str:
    entered.set()
    try:
      await release.wait()
      return 'complete'
    finally:
      closed.set()

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      handle = tasks.spawn(candidate)
      await entered.wait()
      observer = asyncio.create_task(handle.result())
      observer.cancel()
      with pytest.raises(asyncio.CancelledError):
        await observer
      assert handle.status == 'running'
      assert not closed.is_set()
      release.set()
      return await handle.result()

  await _run(coordinate)
  assert closed.is_set()


async def test_completion_ties_select_spawn_order():
  """When every candidate is already complete, first picks the first spawned."""

  async def candidate(node_input: str) -> str:
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      handles = [tasks.spawn(candidate, value) for value in ['first', 'second']]
      await tasks.wait(mode='all')
      winner = await tasks.first()
      assert winner is handles[0]
      assert await tasks.first() is winner
      return winner.output

  await _run(coordinate)


@pytest.mark.parametrize('all_fail', [False, True])
async def test_first_tolerates_partial_failure_and_surfaces_all_failure(
    all_fail,
):
  """A failed candidate cannot prevent a viable candidate from winning."""

  async def candidate(node_input: str) -> str:
    if node_input == 'failure' or all_fail:
      raise ValueError(f'{node_input} failed')
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      tasks.spawn(candidate, 'failure')
      success = tasks.spawn(candidate, 'success')
      winner = await tasks.first()
      assert winner is success
      return winner.output

  if all_fail:
    with pytest.raises(ValueError, match='failure failed'):
      await _run(coordinate)
  else:
    await _run(coordinate)


async def test_predicate_error_closes_in_flight_workers():
  """An acceptance callback error propagates only after sibling cleanup."""
  entered = asyncio.Event()
  closed = asyncio.Event()

  async def candidate(node_input: str) -> str:
    if node_input == 'blocked':
      entered.set()
      try:
        await asyncio.Event().wait()
      finally:
        closed.set()
    await entered.wait()
    return node_input

  def accept(output):
    raise LookupError('predicate failed')

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      tasks.spawn(candidate, 'blocked')
      tasks.spawn(candidate, 'complete')
      await tasks.first(accept=accept)
    return 'unreachable'

  with pytest.raises(LookupError, match='predicate failed'):
    await _run(coordinate)
  assert closed.is_set()


async def test_parent_cancellation_drains_workers_without_durable_cancel_record():
  """Aborting the invocation cleans up work without recording a user decision."""
  entered: asyncio.Queue[str] = asyncio.Queue()
  closed = set()

  async def candidate(node_input: str) -> str:
    entered.put_nowait(node_input)
    try:
      await asyncio.Event().wait()
    finally:
      closed.add(node_input)
    return 'unreachable'

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> list[str]:
    async with ctx.parallel_tasks() as tasks:
      tasks.spawn(candidate, 'one')
      tasks.spawn(candidate, 'two')
      return await tasks.gather()

  runner = testing_utils.InMemoryRunner(
      node=Workflow(name='team', edges=[(START, coordinate)])
  )
  running = asyncio.create_task(runner.run_async('start'))
  try:
    assert {
        await asyncio.wait_for(entered.get(), _MAX_WAIT) for _ in range(2)
    } == {'one', 'two'}
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(running, _MAX_WAIT)
    assert closed == {'one', 'two'}
    assert not any(
        key.endswith('parallel_tasks')
        for event in runner.session.events
        for key in event.custom_metadata or {}
    )
  finally:
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    await runner.runner.close()


async def test_first_retry_waits_for_cleanup_after_timeout():
  """A cached winner cannot bypass incomplete cleanup or renew its deadline."""
  entered = asyncio.Event()
  release = asyncio.Event()
  closed = asyncio.Event()
  cancellations = []

  async def stubborn() -> str:
    entered.set()
    try:
      while not release.is_set():
        try:
          await release.wait()
        except asyncio.CancelledError:
          cancellations.append('cancelled')
      return 'late'
    finally:
      closed.set()

  async def accepted() -> str:
    await entered.wait()
    return 'winner'

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(cancellation_timeout=0.01) as tasks:
      loser = tasks.spawn(stubborn)
      winner = tasks.spawn(accepted)
      try:
        with pytest.raises(TimeoutError, match='did not stop within'):
          await tasks.first()
        assert loser.status == 'running'
        assert not closed.is_set()
        deadline = loser._cancel_deadline
        assert deadline is not None
        with pytest.raises(TimeoutError, match='did not stop within'):
          await tasks.first()
        assert loser.status == 'running'
        assert not closed.is_set()
        assert loser._cancel_deadline == deadline
        assert cancellations == ['cancelled']
      finally:
        release.set()
        await closed.wait()
        with pytest.raises(asyncio.CancelledError):
          await loser.result()
      assert await tasks.first() is winner
      assert loser.status == 'cancelled'
      assert cancellations == ['cancelled']
      return winner.output

  await _run(coordinate)
  assert closed.is_set()


async def test_all_rejected_results_raise_clear_error():
  """Rejecting every successful output is an explicit no-winner outcome."""

  async def candidate(node_input: str) -> str:
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      tasks.spawn(candidate, 'one')
      tasks.spawn(candidate, 'two')
      await tasks.first(accept=lambda output: False)
    return 'unreachable'

  with pytest.raises(
      ValueError, match='No parallel task produced an acceptable'
  ):
    await _run(coordinate)


async def test_explicit_cancel_stops_running_worker_and_releases_queued_slot():
  """Awaited cancellation drains a worker and permits the next queued task."""
  entered = asyncio.Event()
  closed = asyncio.Event()
  started = []

  async def candidate(node_input: str) -> str:
    started.append(node_input)
    if node_input == 'blocked':
      entered.set()
      try:
        await asyncio.Event().wait()
      finally:
        closed.set()
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      blocked = tasks.spawn(candidate, 'blocked')
      queued = tasks.spawn(candidate, 'queued')
      await entered.wait()
      await tasks.cancel(blocked)
      assert blocked.status == 'cancelled'
      assert closed.is_set()
      assert blocked.cancel() is False
      with pytest.raises(asyncio.CancelledError):
        await blocked.result()
      return await queued.result()

  await _run(coordinate)
  assert started == ['blocked', 'queued']


def _resumable_runner(coordinate, is_resumable):
  return testing_utils.InMemoryRunner(
      app=App(
          name='team',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )


async def _resume(runner, events, interrupt_id):
  return await asyncio.wait_for(
      runner.run_async(
          new_message=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=interrupt_id,
                          name='user_input',
                          response={'text': 'approved'},
                      )
                  )
              ],
          ),
          invocation_id=events[0].invocation_id,
      ),
      _MAX_WAIT,
  )


@pytest.mark.parametrize('is_resumable', [False, True])
async def test_winner_and_cancelled_losers_survive_downstream_hitl(
    is_resumable,
):
  """Resume reuses the chosen result without rerunning predicate or losers."""
  calls = []
  predicate_calls = []
  winners = []

  async def candidate(node_input: str) -> str:
    calls.append(node_input)
    return node_input

  def accept(output):
    predicate_calls.append(output)
    return True

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      first = tasks.spawn(candidate, 'winner')
      loser = tasks.spawn(candidate, 'queued')
      winner = await tasks.first(accept=accept)
      assert winner is first
      assert loser.status == 'cancelled'
      winners.append(winner.id)
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Approve winner?')
      return
    yield Event(output=winner.output)

  runner = _resumable_runner(coordinate, is_resumable)
  try:
    first_events = await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
    resumed = await _resume(runner, first_events, 'approval')
    assert any(event.output == 'winner' for event in resumed)
  finally:
    await runner.runner.close()

  assert len(winners) == 2
  assert winners[0] == winners[1]
  assert calls == ['winner']
  assert predicate_calls == ['winner']


@pytest.mark.parametrize('is_resumable', [False, True])
async def test_merge_disposition_survives_resume_and_preserves_external_edit(
    is_resumable,
):
  """A committed patch is not reapplied over edits made while awaiting input."""
  calls = []

  async def proposal() -> dict[str, str]:
    calls.append('proposal')
    return {'answer': 'selected', 'unselected': 'must not merge'}

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks() as tasks:
      task = tasks.spawn(proposal)
      await task.result()
      await tasks.merge_result(task, keys=['answer'])
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(
          interrupt_id='approval', message='Approve merged answer?'
      )
      return
    yield Event(output=ctx.state['answer'])

  runner = _resumable_runner(coordinate, is_resumable)
  try:
    first_events = await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
    assert runner.session.state['answer'] == 'selected'
    assert 'unselected' not in runner.session.state
    await runner.runner.session_service.append_event(
        runner.session,
        Event(
            author='user',
            actions=EventActions(state_delta={'answer': 'external edit'}),
        ),
    )
    resumed = await _resume(runner, first_events, 'approval')
    assert any(event.output == 'external edit' for event in resumed)
    assert runner.session.state['answer'] == 'external edit'
  finally:
    await runner.runner.close()
  assert calls == ['proposal']


@pytest.mark.parametrize(
    'initial,changed',
    [(False, 0), ({'enabled': False}, {'enabled': 0}), ('before', 'after')],
)
async def test_merge_conflicts_are_type_sensitive_and_atomic(initial, changed):
  """A conflict rejects every proposed key, including bool/number changes."""

  async def proposal() -> dict[str, Any]:
    return {'new_key': 'must not leak', 'conflict': 'proposal'}

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    ctx.state['conflict'] = initial
    async with ctx.parallel_tasks() as tasks:
      task = tasks.spawn(proposal)
      await task.result()
      ctx.state['conflict'] = changed
      with pytest.raises(ValueError, match='State changed before merging'):
        await tasks.merge_result(task)
      assert 'new_key' not in ctx.state
      assert ctx.state['conflict'] == changed
      await tasks.merge_result(
          task, keys=['conflict'], expected={'conflict': changed}
      )
      assert ctx.state['conflict'] == 'proposal'
    return 'checked'

  await _run(coordinate)


@pytest.mark.parametrize('is_resumable', [False, True])
async def test_gather_pauses_queue_for_hitl_and_replays_completed_workers(
    is_resumable,
):
  """An approval request pauses bounded admission and resumes only pending work."""
  calls = []
  outputs = []

  @node(rerun_on_resume=True)
  async def candidate(
      ctx: Context, node_input: str
  ) -> AsyncGenerator[Event, None]:
    calls.append(node_input)
    if node_input == 'approval' and 'worker_approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='worker_approval', message='Continue?')
      return
    yield Event(output=node_input)

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> list[str]:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      for value in ['complete', 'approval', 'queued']:
        tasks.spawn(candidate, value)
      result = await tasks.gather()
      outputs.append(result)
      return result

  runner = _resumable_runner(coordinate, is_resumable)
  try:
    first_events = await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
    assert calls == ['complete', 'approval']
    assert outputs == []
    await _resume(runner, first_events, 'worker_approval')
  finally:
    await runner.runner.close()

  assert calls == ['complete', 'approval', 'approval', 'queued']
  assert outputs == [['complete', 'approval', 'queued']]


@pytest.mark.parametrize('is_resumable', [False, True])
async def test_cancelled_waiting_worker_cannot_block_unrelated_resume(
    is_resumable,
):
  """Cancelling a waiting child clears its interruption across later resumes."""
  calls = []
  results = []

  @node(rerun_on_resume=True)
  async def candidate(
      ctx: Context, node_input: str
  ) -> AsyncGenerator[Event, None]:
    calls.append(node_input)
    if node_input == 'cancelled_approval':
      yield RequestInput(interrupt_id='unused_approval', message='Not required')
      return
    yield Event(output=node_input)

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      cancelled = tasks.spawn(candidate, 'cancelled_approval')
      survivor = tasks.spawn(candidate, 'survivor')
      if cancelled.status != 'cancelled':
        done, pending = await tasks.wait(mode='all')
        assert done == [cancelled]
        assert pending == [survivor]
        assert cancelled.status == 'waiting'
        await tasks.cancel(cancelled)
      result = await survivor.result()
      results.append(result)
    if 'parent_approval' not in ctx.resume_inputs:
      yield RequestInput(
          interrupt_id='parent_approval', message='Approve survivor?'
      )
      return
    yield Event(output=result)

  runner = _resumable_runner(coordinate, is_resumable)
  try:
    first_events = await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
    assert calls == ['cancelled_approval', 'survivor']
    resumed = await _resume(runner, first_events, 'parent_approval')
    assert any(event.output == 'survivor' for event in resumed)
  finally:
    await runner.runner.close()

  assert calls == ['cancelled_approval', 'survivor']
  assert results == ['survivor', 'survivor']


@pytest.mark.parametrize('is_resumable', [False, True])
async def test_new_invocation_does_not_reuse_previous_winner_or_patch(
    is_resumable,
):
  """Replay dispositions are scoped to a continuation, not future user turns."""
  calls = []
  predicates = []
  answers = []

  async def proposal() -> dict[str, int]:
    calls.append('run')
    return {'answer': len(calls)}

  def accept(output):
    predicates.append(output['answer'])
    return True

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> int:
    async with ctx.parallel_tasks() as tasks:
      tasks.spawn(proposal)
      winner = await tasks.first(accept=accept)
      await tasks.merge_result(winner)
      answers.append(ctx.state['answer'])
      return ctx.state['answer']

  runner = _resumable_runner(coordinate, is_resumable)
  try:
    await asyncio.wait_for(runner.run_async('first task'), _MAX_WAIT)
    await asyncio.wait_for(runner.run_async('new task'), _MAX_WAIT)
    assert runner.session.state['answer'] == 2
  finally:
    await runner.runner.close()

  assert calls == ['run', 'run']
  assert predicates == [1, 2]
  assert answers == [1, 2]


async def test_merge_validates_all_selected_schema_fields_before_writing():
  """An invalid later field cannot leave a valid earlier field partially merged."""

  class Schema(BaseModel):
    good: str
    count: int

  async def proposal() -> dict[str, Any]:
    return {'good': 'must not leak', 'count': 'not an integer'}

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      handle = tasks.spawn(proposal)
      await handle.result()
      with pytest.raises(StateSchemaError, match='count'):
        await tasks.merge_result(handle)
      assert 'good' not in ctx.state
      assert 'count' not in ctx.state
      await tasks.merge_result(handle, keys=['good'])
      assert ctx.state['good'] == 'must not leak'
    return 'checked'

  await _run(coordinate, state_schema=Schema)


async def test_merge_rejects_bad_selection_and_foreign_handle():
  """Selection and ownership errors cannot accidentally modify session state."""

  async def proposal() -> dict[str, int]:
    return {'answer': 42}

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(name='outer') as outer:
      handle = outer.spawn(proposal)
      await handle.result()
      with pytest.raises(ValueError, match='not a string'):
        await outer.merge_result(handle, keys='answer')
      with pytest.raises(ValueError, match='Every selected key'):
        await outer.merge_result(handle, keys=['missing'])
      async with ctx.parallel_tasks(name='inner') as inner:
        with pytest.raises(ValueError, match='different parallel scope'):
          await inner.merge_result(handle)
        with pytest.raises(ValueError, match='different parallel scope'):
          await inner.cancel(handle)
      assert 'answer' not in ctx.state
      await outer.merge_result(handle)
      ctx.state['answer'] = 99
      await outer.merge_result(handle)
      assert ctx.state['answer'] == 99
    return 'checked'

  await _run(coordinate)


async def test_first_closes_losing_node_generator_after_progress_event():
  """A losing generator is finalized even after it emitted native progress."""
  progressed = asyncio.Event()
  closed = asyncio.Event()
  outputs = []

  @node
  async def streaming() -> AsyncGenerator[Event, None]:
    try:
      yield Event(custom_metadata={'progress': 'started'})
      progressed.set()
      await asyncio.Event().wait()
      outputs.append('late output')
      yield Event(output='late output')
    finally:
      closed.set()

  async def accepted() -> str:
    await progressed.wait()
    return 'winner'

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      loser = tasks.spawn(streaming)
      winner = tasks.spawn(accepted)
      assert await tasks.first() is winner
      assert closed.is_set()
      assert loser.status == 'cancelled'
      return winner.output

  events = await _run(coordinate)
  assert any(
      (event.custom_metadata or {}).get('progress') == 'started'
      for event in events
  )
  assert outputs == []


async def test_handle_cancel_request_prevents_queued_worker_from_starting():
  """Synchronous cancellation requests are persisted before queued admission."""
  entered = asyncio.Event()
  release = asyncio.Event()
  started = []

  async def candidate(node_input: str) -> str:
    started.append(node_input)
    entered.set()
    await release.wait()
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      running = tasks.spawn(candidate, 'running')
      queued = tasks.spawn(candidate, 'cancelled')
      await entered.wait()
      assert queued.cancel() is True
      release.set()
      done, pending = await tasks.wait(mode='all')
      assert done == [running, queued]
      assert pending == []
      assert queued.status == 'cancelled'
      return await running.result()

  await _run(coordinate)
  assert started == ['running']


@pytest.mark.parametrize('concurrent', [False, True])
@pytest.mark.parametrize('explicit_run_ids', [False, True])
async def test_tool_calls_own_distinct_groups_and_current_selection(
    concurrent, explicit_run_ids
):
  """Each tool invocation owns its task IDs, selection, and merge decisions.

  The same real Agent calls one coordinating FunctionTool twice in a turn.
  Each tool uses the default group name and the same candidate names/run IDs,
  but must select the output requested by that call's current arguments.
  """
  both_entered = asyncio.Event()
  entered = []
  ids = []
  selected = {}
  predicate_calls = {'a': [], 'b': []}

  async def candidate(node_input: str) -> dict[str, str]:
    return {'answer': node_input}

  async def choose(desired: str, ctx: Context) -> str:
    """Choose the requested candidate and merge its answer."""
    async with ctx.parallel_tasks(max_parallel_workers=1) as tasks:
      handles = [
          tasks.spawn(
              candidate,
              value,
              **({'run_id': f'candidate-{value}'} if explicit_run_ids else {}),
          )
          for value in ['a', 'b']
      ]
      ids.extend(handle.id for handle in handles)
      entered.append(desired)
      if len(entered) == 2:
        both_entered.set()
      if concurrent:
        await both_entered.wait()

      def accept(output):
        predicate_calls[desired].append(output['answer'])
        return output['answer'] == desired

      winner = await tasks.first(accept=accept)
      expected = (
          {'answer': ctx.state['answer']} if 'answer' in ctx.state else None
      )
      await tasks.merge_result(winner, expected=expected)
      selected[desired] = winner.output['answer']
      return winner.output['answer']

  calls = [
      types.Part.from_function_call(name='choose', args={'desired': desired})
      for desired in ['a', 'b']
  ]
  model = testing_utils.MockModel.create(
      [calls, 'complete'] if concurrent else [*calls, 'complete']
  )
  agent = LlmAgent(name='coordinator', model=model, tools=[choose])
  runner = testing_utils.InMemoryRunner(root_agent=agent)
  try:
    events = await asyncio.wait_for(runner.run_async('start'), _MAX_WAIT)
    assert selected == {'a': 'a', 'b': 'b'}
    assert predicate_calls == {'a': ['a'], 'b': ['a', 'b']}
    assert len(ids) == len(set(ids)) == 4
    merged_answers = [
        event.actions.state_delta['answer']
        for event in events
        if 'answer' in event.actions.state_delta
    ]
    assert sorted(merged_answers) == ['a', 'b']
    if not concurrent:
      assert runner.session.state['answer'] == 'b'
  finally:
    both_entered.set()
    await runner.runner.close()
