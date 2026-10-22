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

"""Replay of owned tasks whose agent transfers cross workflow schedulers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._errors import DynamicNodeFailError
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._replay_manager import ReplayManager
from google.genai import types
import pytest

from . import testing_utils


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('cancel_from_callback', [False, True])
@pytest.mark.parametrize('final_interruption', [False, True])
async def test_cancel_at_final_output_agrees_with_replayed_status(
    is_resumable: bool,
    cancel_from_callback: bool,
    final_interruption: bool,
):
  """A cancellation racing final output has one live and replay disposition.

  Setup: Hold the worker's final event until cancellation can be requested.
  Act: Cancel from the progress callback or await group cancellation, then
    pause for human input and resume.
  Assert: Both outcomes agree with the durable decision; the worker runs once.
  """
  observations = {'calls': [], 'results': [], 'statuses': [], 'requests': []}
  final_output_seen = asyncio.Event()
  release_final_output = asyncio.Event()

  class CancelAtOutput(BasePlugin):

    async def on_event_callback(self, *, invocation_context, event):
      at_boundary = event.output == 'done' or 'unused_approval' in (
          event.long_running_tool_ids or set()
      )
      if at_boundary and 'parallel_task' in (event.custom_metadata or {}):
        if cancel_from_callback:
          observations['requests'].append(observations['task'].cancel())
        else:
          final_output_seen.set()
          await release_final_output.wait()
      return None

  @node(rerun_on_resume=True)
  async def worker() -> AsyncGenerator[Event, None]:
    observations['calls'].append('worker')
    if final_interruption:
      yield RequestInput(interrupt_id='unused_approval', message='Not needed')
      return
    yield Event(output='done')

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks() as group:
      task = group.spawn(worker)
      observations['task'] = task
      if not cancel_from_callback and 'approval' not in ctx.resume_inputs:
        await final_output_seen.wait()
        release_final_output.set()
        await group.cancel(task)
      try:
        outcome = await task.result()
      except asyncio.CancelledError:
        outcome = 'cancelled'
      observations['results'].append(outcome)
      observations['statuses'].append(task.status)
    assert not ctx.interrupt_ids
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Continue?')
      return
    yield Event(output=outcome)

  runner = testing_utils.InMemoryRunner(
      app=App(
          name='cancellation_race',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          plugins=[CancelAtOutput(name='cancel_at_output')],
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 3)
    assert observations['requests'] == ([True] if cancel_from_callback else [])
    cancelled = any(
        observations['task'].id
        in (event.custom_metadata or {})
        .get('__adk_internal_parallel_tasks', {})
        .get('cancelled', [])
        for event in runner.session.events
    )
    expected_result = 'cancelled' if cancelled else 'done'
    expected_status = 'cancelled' if cancelled else 'completed'
    if final_interruption:
      assert cancelled
    assert observations['results'] == [expected_result]
    assert observations['statuses'] == [expected_status]
    response = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id='approval',
                    name='adk_request_input',
                    response={'result': 'yes'},
                )
            )
        ],
    )
    resumed = await asyncio.wait_for(
        runner.run_async(response, invocation_id=events[0].invocation_id), 3
    )
    assert any(event.output == expected_result for event in resumed)
  finally:
    await runner.runner.close()

  assert observations['calls'] == ['worker']
  assert observations['results'] == [expected_result, expected_result]
  assert observations['statuses'] == [expected_status, expected_status]


@pytest.mark.parametrize('explicit_group', [False, True])
async def test_legacy_worker_preserves_event_metadata_and_outer_ownership(
    explicit_group: bool,
):
  """Legacy workers add no task identity and inherit any public task owner."""
  owner_ids = []
  captured = []

  @node(parallel_worker=True, max_parallel_workers=1)
  async def worker(node_input: str) -> AsyncGenerator[Event, None]:
    yield Event(
        output=f'{node_input}_processed',
        custom_metadata={'source': node_input},
    )

  def produce() -> list[str]:
    return ['first', 'second']

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> list[str]:
    async with ctx.parallel_tasks() as tasks:
      task = tasks.spawn(worker, produce())
      owner_ids.append(task.id)
      return await task.result()

  def capture(node_input: list[str]) -> list[str]:
    captured.append(node_input)
    return node_input

  workflow = Workflow(
      name='legacy_metadata',
      edges=[
          (START, coordinate, capture)
          if explicit_group
          else (START, produce, worker, capture)
      ],
  )
  runner = testing_utils.InMemoryRunner(node=workflow)
  try:
    events = await runner.run_async('start')
    worker_events = [
        event for event in events if 'source' in (event.custom_metadata or {})
    ]
    assert len(worker_events) == 2
    assert captured == [['first_processed', 'second_processed']]
    for event in worker_events:
      expected = {'source': event.output.removesuffix('_processed')}
      if explicit_group:
        expected.update({
            'parallel_task': {'task_id': owner_ids[0], 'status': 'running'},
            '__adk_internal_parallel_task_owners': owner_ids,
        })
      assert event.custom_metadata == expected
  finally:
    await runner.runner.close()


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('other_fails', [False, True])
@pytest.mark.parametrize('observe_after_first', [False, True])
async def test_first_preserves_previously_observed_nonwinner_on_resume(
    is_resumable: bool,
    other_fails: bool,
    observe_after_first: bool,
):
  """Selection preserves observed terminal results and failures across replay.

  Setup: Inspect a terminal candidate before or after selecting another winner.
  Act: Select, pause for human input, and resume the same coordinator.
  Assert: Terminal status stays consistent; completed work is reused and
    failed work retains the native scheduler's retry behavior.
  """
  calls = []
  observed = []

  async def worker(node_input: str) -> str:
    calls.append(node_input)
    if other_fails and node_input == 'other':
      raise ValueError('Rejected candidate')
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(max_parallel_workers=1) as group:
      group.spawn(worker, 'winner')
      other = group.spawn(worker, 'other')
      await group.wait(mode='all')
      if observe_after_first:
        winner = await group.first(lambda output: output == 'winner')
      try:
        result = await other.result()
      except DynamicNodeFailError:
        result = 'failed'
      observed.append((result, other.status))
      if not observe_after_first:
        winner = await group.first(lambda output: output == 'winner')
      assert other.status == ('failed' if other_fails else 'completed')
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Continue?')
      return
    yield Event(output=winner.output)

  runner = testing_utils.InMemoryRunner(
      app=App(
          name='completed_candidate',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  expected = ('failed', 'failed') if other_fails else ('other', 'completed')
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 3)
    assert observed == [expected]
    response = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id='approval',
                    name='adk_request_input',
                    response={'result': 'yes'},
                )
            )
        ],
    )
    resumed = await asyncio.wait_for(
        runner.run_async(response, invocation_id=events[0].invocation_id), 3
    )
    assert any(event.output == 'winner' for event in resumed)
  finally:
    await runner.runner.close()

  # Native replay retries failed nodes; completed nodes reuse their output.
  assert calls == ['winner', 'other'] + (['other'] if other_fails else [])
  assert observed == [expected, expected]


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('retry_before_winner', [False, True])
async def test_restored_winner_drains_a_failed_siblings_unfinished_retry(
    is_resumable: bool,
    retry_before_winner: bool,
):
  """Restoring a winner cancels an unfinished retry before returning it."""
  calls = []
  retry_started = asyncio.Event()
  retry_closed = asyncio.Event()

  async def worker(node_input: str) -> str:
    calls.append(node_input)
    if node_input == 'other':
      if calls.count('other') == 1:
        raise ValueError('Candidate failed on its first attempt')
      retry_started.set()
      try:
        await asyncio.Event().wait()
      finally:
        retry_closed.set()
    return node_input

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(
        max_parallel_workers=1 if retry_before_winner else 2
    ) as group:
      if retry_before_winner:
        other = group.spawn(worker, 'other')
        group.spawn(worker, 'winner')
      else:
        group.spawn(worker, 'winner')
        other = group.spawn(worker, 'other')
      if 'approval' in ctx.resume_inputs:
        await retry_started.wait()
      else:
        await group.wait(mode='all')
        assert other.status == 'failed'
      winner = await group.first(lambda output: output == 'winner')
      if 'approval' in ctx.resume_inputs:
        assert retry_closed.is_set()
        assert other.status == 'cancelled'
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Continue?')
      return
    yield Event(output=winner.output)

  runner = testing_utils.InMemoryRunner(
      app=App(
          name='retried_candidate',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 3)
    response = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id='approval',
                    name='adk_request_input',
                    response={'result': 'yes'},
                )
            )
        ],
    )
    resumed = await asyncio.wait_for(
        runner.run_async(response, invocation_id=events[0].invocation_id), 3
    )
    assert any(event.output == 'winner' for event in resumed)
  finally:
    await runner.runner.close()

  assert calls == (
      ['other', 'winner', 'other']
      if retry_before_winner
      else ['winner', 'other', 'other']
  )
  assert retry_closed.is_set()


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('workflow_sibling', [False, True])
async def test_restored_winner_retains_completed_sibling_with_bounded_admission(
    is_resumable: bool,
    workflow_sibling: bool,
):
  """A retained sibling queued behind the restored winner hydrates normally."""
  calls = []
  observed = []

  async def worker(node_input: str) -> str:
    calls.append(node_input)
    return node_input

  other_node = (
      Workflow(name='other_workflow', edges=[(START, worker)])
      if workflow_sibling
      else worker
  )

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(max_parallel_workers=1) as group:
      group.spawn(worker, 'winner')
      other = group.spawn(other_node, 'other')
      if 'approval' not in ctx.resume_inputs:
        await group.wait(mode='all')
      winner = await group.first(lambda output: output == 'winner')
      # Resume enters first() before the second handle can hydrate. Selection
      # must preserve this known completed sibling and admit it through the cap.
      assert other.status == 'completed'
      observed.append(await other.result())
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Continue?')
      return
    yield Event(output=winner.output)

  runner = testing_utils.InMemoryRunner(
      app=App(
          name='retained_candidate',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 3)
    response = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id='approval',
                    name='adk_request_input',
                    response={'result': 'yes'},
                )
            )
        ],
    )
    resumed = await asyncio.wait_for(
        runner.run_async(response, invocation_id=events[0].invocation_id), 3
    )
    assert any(event.output == 'winner' for event in resumed)
  finally:
    await runner.runner.close()

  assert calls == ['winner', 'other']
  assert observed == ['other', 'other']


async def test_selection_timeout_preserves_retained_workflow_until_retry():
  """Timing out a losing retry must not cancel retained native hydration."""
  calls = []
  resuming = asyncio.Event()
  retry_started = asyncio.Event()
  retry_closed = asyncio.Event()
  release_retry = asyncio.Event()
  hydration_started = asyncio.Event()
  hydration_cancelled = asyncio.Event()
  release_hydration = asyncio.Event()

  async def worker(node_input: str) -> str:
    calls.append(node_input)
    if node_input == 'failed':
      if calls.count('failed') == 1:
        raise ValueError('Candidate failed on its first attempt')
      retry_started.set()
      try:
        while not release_retry.is_set():
          try:
            await release_retry.wait()
          except asyncio.CancelledError:
            pass
      finally:
        retry_closed.set()
    return node_input

  class GatedWorkflow(Workflow):

    async def _run_impl(self, *, ctx, node_input):
      if resuming.is_set():
        hydration_started.set()
        try:
          await release_hydration.wait()
        except asyncio.CancelledError:
          hydration_cancelled.set()
          raise
      async for event in super()._run_impl(ctx=ctx, node_input=node_input):
        yield event

  retained_node = GatedWorkflow(name='retained', edges=[(START, worker)])

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> AsyncGenerator[Event, None]:
    async with ctx.parallel_tasks(cancellation_timeout=0.01) as group:
      group.spawn(worker, 'winner')
      failed = group.spawn(worker, 'failed')
      retained = group.spawn(retained_node, 'retained')
      if not resuming.is_set():
        await group.wait(mode='all')
        winner = await group.first(lambda output: output == 'winner')
      else:
        selection = None
        try:
          await retry_started.wait()
          await hydration_started.wait()
          with pytest.raises(TimeoutError, match='did not stop within'):
            await group.first()
          assert retained.status == 'running'
          assert not hydration_cancelled.is_set()
          release_retry.set()
          await retry_closed.wait()
          with pytest.raises(asyncio.CancelledError):
            await failed.result()
          selection = asyncio.create_task(group.first())
          await asyncio.sleep(0)
          assert not selection.done()
          release_hydration.set()
          winner = await selection
          assert await retained.result() == 'retained'
          assert retained.status == 'completed'
          assert not hydration_cancelled.is_set()
        finally:
          release_retry.set()
          release_hydration.set()
          if selection is not None:
            await asyncio.gather(selection, return_exceptions=True)
    if 'approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='approval', message='Continue?')
      return
    yield Event(output=winner.output)

  runner = testing_utils.InMemoryRunner(
      app=App(
          name='retry_retained',
          root_agent=Workflow(name='team', edges=[(START, coordinate)]),
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 3)
    resuming.set()
    response = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id='approval',
                    name='adk_request_input',
                    response={'result': 'yes'},
                )
            )
        ],
    )
    resumed = await asyncio.wait_for(
        runner.run_async(response, invocation_id=events[0].invocation_id), 3
    )
    assert any(event.output == 'winner' for event in resumed)
  finally:
    release_retry.set()
    release_hydration.set()
    await runner.runner.close()

  assert calls == ['winner', 'failed', 'retained', 'failed']


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('nested_group', [False, True])
async def test_cancelled_transfer_to_foreign_scheduler_does_not_block_resume(
    is_resumable: bool,
    nested_group: bool,
):
  """A cancelled parent transfer cannot retain an unrelated replay barrier.

  The outer agent calls an inner workflow through a tool. Its child worker
  transfers back to that outer agent, crossing from the inner scheduler to the
  outer scheduler, and pauses for input. Another worker wins. Later resuming
  an unrelated input request must skip the cancelled hop and completed work.
  """
  chosen = []
  cancelled_ids = []

  def abandoned_approval() -> None:
    """An approval requested by the losing transferred worker."""

  child_model = testing_utils.MockModel.create([
      types.Part.from_function_call(
          name='transfer_to_agent', args={'agent_name': 'outer_agent'}
      ),
  ])
  child_agent = LlmAgent(name='child_agent', model=child_model)

  @node(rerun_on_resume=True)
  async def nested(ctx: Context, node_input: str) -> str:
    async with ctx.parallel_tasks() as tasks:
      return await tasks.spawn(child_agent, node_input).result()

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      losing = tasks.spawn(nested if nested_group else child_agent, 'work')

      async def accepted() -> str:
        try:
          await losing.result()
        except NodeInterruptedError:
          return 'winner'
        raise AssertionError('The transferred worker should request approval.')

      tasks.spawn(accepted)
      winner = await tasks.first()
      chosen.append(winner.output)
      cancelled_ids.append(losing.id)
      assert losing.status == 'cancelled'
      return winner.output

  inner = Workflow(name='inner', edges=[(START, coordinate)])

  async def run_inner(tool_context: ToolContext) -> str:
    """Run the inner collaboration workflow."""
    return await tool_context.run_node(inner, node_input='work')

  outer_model = testing_utils.MockModel.create([
      types.Part.from_function_call(name='run_inner', args={}),
      types.Part.from_function_call(name='abandoned_approval', args={}),
      'outer complete',
  ])
  outer_agent = LlmAgent(
      name='outer_agent',
      model=outer_model,
      sub_agents=[child_agent],
      tools=[run_inner, LongRunningFunctionTool(abandoned_approval)],
  )

  @node(rerun_on_resume=True)
  async def approval(ctx: Context) -> AsyncGenerator[Event, None]:
    if 'keep_approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='keep_approval', message='Continue?')
      return
    yield Event(output='finished')

  workflow = Workflow(name='outer', edges=[(START, outer_agent, approval)])
  runner = testing_utils.InMemoryRunner(
      app=App(
          name='transfer_tasks',
          root_agent=workflow,
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 5)
    assert chosen == ['winner']
    assert len(outer_model.requests) == 3
    cancelled_hops = [
        event
        for event in runner.session.events
        if cancelled_ids[0]
        in (event.custom_metadata or {}).get(
            '__adk_internal_parallel_task_owners', []
        )
        and event.long_running_tool_ids
    ]
    assert cancelled_hops
    assert all(
        not event.node_info.path.startswith(cancelled_ids[0] + '/')
        and event.node_info.path != cancelled_ids[0]
        for event in cancelled_hops
    )
    response = types.Content(
        role='user',
        parts=[
            types.Part.from_function_response(
                name='adk_request_input',
                response={'result': 'approved'},
            )
        ],
    )
    response.parts[0].function_response.id = 'keep_approval'
    resumed = await asyncio.wait_for(
        runner.run_async(
            new_message=response,
            invocation_id=events[0].invocation_id,
        ),
        5,
    )
    assert any(event.output == 'finished' for event in resumed)
    assert chosen == ['winner']
    assert len(outer_model.requests) == 3
    assert len(child_model.requests) == 1
  finally:
    await runner.runner.close()


async def test_cancellation_index_rebuilds_after_rewind():
  """Rewinding the decision permits a new execution of the same task path."""
  agent = LlmAgent(name='wf', model=testing_utils.MockModel.create([]))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.invocation_id = 'run'
  ctx = Context(invocation, node=agent, run_id='1')
  worker_path = 'wf@1/worker@one'
  invocation.session.events.extend([
      Event(
          author='worker',
          invocation_id='run',
          output='discarded',
          node_info=NodeInfo(path=worker_path),
      ),
      Event(
          author='wf',
          invocation_id='run',
          node_info=NodeInfo(path='wf@1/__parallel_tasks_control@parallel-1'),
          custom_metadata={
              '__adk_internal_parallel_tasks': {
                  'group': 'wf@1/parallel@1',
                  'cancelled': [worker_path],
              }
          },
      ),
  ])
  replay = ReplayManager()
  recovered, sequence = replay.scan_workflow_events(ctx)
  assert 'worker@one' not in recovered
  assert 'worker@one' not in sequence

  invocation.session.events.extend([
      Event(
          author='user',
          invocation_id='rewind',
          actions=EventActions(rewind_before_invocation_id='run'),
      ),
      Event(
          author='worker',
          invocation_id='run',
          output='new execution',
          node_info=NodeInfo(path=worker_path),
          custom_metadata={'parallel_task': {'task_id': []}},
      ),
  ])
  recovered, sequence = replay.scan_workflow_events(ctx)
  assert recovered['worker@one'].output == 'new execution'
  assert sequence == ['worker@one']


@pytest.mark.parametrize('task_id', [{'arbitrary': True}, [], 'wf@1/loser@one'])
async def test_cancellation_filter_ignores_unrelated_public_metadata(task_id):
  """Arbitrary user metadata cannot break replay after a cancellation."""
  agent = LlmAgent(name='wf', model=testing_utils.MockModel.create([]))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.invocation_id = 'run'
  ctx = Context(invocation, node=agent, run_id='1')
  invocation.session.events.extend([
      Event(
          author='wf',
          invocation_id='run',
          node_info=NodeInfo(path='wf@1/__parallel_tasks_control@parallel-1'),
          custom_metadata={
              '__adk_internal_parallel_tasks': {
                  'group': 'wf@1/parallel@1',
                  'cancelled': ['wf@1/loser@one'],
              }
          },
      ),
      Event(
          author='worker',
          invocation_id='run',
          output='kept',
          node_info=NodeInfo(path='wf@1/worker@one'),
          custom_metadata={'parallel_task': {'task_id': task_id}},
      ),
  ])
  recovered, sequence = ReplayManager().scan_workflow_events(ctx)
  assert recovered['worker@one'].output == 'kept'
  assert sequence == ['worker@one']
