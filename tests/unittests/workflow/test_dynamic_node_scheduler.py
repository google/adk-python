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

"""Tests for DynamicNodeScheduler.

Verifies the three scheduling cases (fresh, dedup, resume) and the
lazy event scan that reconstructs dynamic node state.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeRun
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeState
from google.adk.workflow._errors import WorkflowInvariantError
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_state import NodeStatus
from google.adk.workflow._workflow import _LoopState
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

# --- Fixtures ---


def _make_parent_ctx(events=None):
  """Create a minimal parent Context with mock IC."""
  ic = MagicMock()
  ic.invocation_id = 'inv-1'
  ic.session = MagicMock()
  ic.session.state = {}
  ic.session.events = events or []
  ic.run_config = None

  collected = []

  async def _enqueue(event):
    collected.append(event)

  ic._enqueue_event = AsyncMock(side_effect=_enqueue)

  ctx = MagicMock(spec=Context)
  ctx._invocation_context = ic
  ctx.node_path = 'wf/parent'
  ctx.run_id = 'run-parent'
  ctx.event_author = 'wf'
  ctx._workflow_scheduler = None
  ctx._output_for_ancestors = []
  ctx._output_delegated = False

  return ctx, collected


def _make_event(
    path='',
    output=None,
    interrupt_ids=None,
    run_id=None,
    author='node',
    invocation_id='inv-1',
    output_for=None,
):
  """Create a minimal Event for session event lists."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = author
  event.output = output
  event.error_code = None
  event.partial = False
  event.node_info = MagicMock(spec=NodeInfo)
  event.node_info.path = path
  event.node_info.output_for = output_for
  event.node_info.message_as_output = None
  event.branch = None
  event.isolation_scope = None
  event.long_running_tool_ids = set(interrupt_ids) if interrupt_ids else None
  event.content = None
  event.actions = None
  return event


def _make_fr_event(fc_id, response, invocation_id='inv-1'):
  """Create a user FR event."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = 'user'
  event.output = None
  event.error_code = None
  event.node_info = MagicMock(spec=NodeInfo)
  event.node_info.path = ''
  event.node_info.message_as_output = None
  event.branch = None
  event.isolation_scope = None
  event.long_running_tool_ids = None

  fr = MagicMock()
  fr.id = fc_id
  fr.response = response

  part = MagicMock()
  part.function_response = fr

  content = MagicMock()
  content.parts = [part]
  event.content = content
  return event


# =========================================================================
# _rehydrate_from_events — lazy scan
# =========================================================================


@pytest.mark.asyncio
async def test_rehydrate_finds_completed_node():
  """Scan finds output event → node marked COMPLETED."""
  events = [
      _make_event(
          path='wf/parent/child@r-1',
          output='result',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  assert 'wf/parent/child@r-1' in ls.runs
  run = ls.runs['wf/parent/child@r-1']
  assert run.recovered_state is not None
  assert run.recovered_state.output == 'result'


@pytest.mark.asyncio
async def test_rehydrate_ignores_events_from_different_invocation():
  """Scan ignores events with a different invocation_id."""
  events = [
      _make_event(
          path='wf/parent/child@r-1',
          output='result',
          invocation_id='inv-different',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ctx._invocation_context.invocation_id = 'inv-current'
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  assert 'wf/parent/child@r-1' not in ls.runs


@pytest.mark.asyncio
async def test_rehydrate_finds_interrupted_node():
  """Scan finds interrupt event → node marked WAITING."""
  events = [
      _make_event(
          path='wf/parent/child@r-1',
          interrupt_ids=['fc-1'],
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  assert 'wf/parent/child@r-1' in ls.runs
  run = ls.runs['wf/parent/child@r-1']
  assert run.recovered_state is not None
  assert 'fc-1' in run.recovered_state.interrupt_ids


@pytest.mark.asyncio
async def test_rehydrate_with_target_run_id_skips_others():
  """Scan with unique path only rehydrates that specific run."""
  events = [
      _make_event(
          path='wf/parent/child@r-1',
          output='result-1',
      ),
      _make_event(
          path='wf/parent/child@r-2',
          output='result-2',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  # When targeting r-2
  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-2')

  # Then only r-2 is in state
  assert 'wf/parent/child@r-2' in ls.runs
  assert 'wf/parent/child@r-1' not in ls.runs
  run = ls.runs['wf/parent/child@r-2']
  assert run.recovered_state is not None
  assert run.recovered_state.output == 'result-2'


@pytest.mark.asyncio
async def test_rehydrate_includes_delegated():
  """Scan includes events delegated to that run."""
  events = [
      _make_event(
          path='wf/parent/child@r-target/inner@r-inner',
          output='delegated-val',
          output_for=['wf/parent/child@r-target'],
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-target')

  assert 'wf/parent/child@r-target' in ls.runs
  run = ls.runs['wf/parent/child@r-target']
  assert run.recovered_state is not None
  assert run.recovered_state.output == 'delegated-val'


@pytest.mark.asyncio
async def test_rehydrate_resolves_interrupt_with_fr():
  """Scan finds interrupt + FR → all resolved, ready to re-run."""
  events = [
      _make_event(
          path='wf/parent/child@r-1',
          interrupt_ids=['fc-1'],
      ),
      _make_fr_event('fc-1', {'approved': True}),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  run = ls.runs['wf/parent/child@r-1']
  assert run.recovered_state is not None
  assert 'fc-1' in run.recovered_state.resolved_ids


@pytest.mark.asyncio
async def test_rehydrate_no_events_does_nothing():
  """Scan with no matching events does not populate dynamic_nodes."""
  events = [
      _make_event(path='wf/other/node', output='x'),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  assert 'wf/parent/child@r-1' not in ls.runs


@pytest.mark.asyncio
async def test_rehydrate_subtree_interrupt():
  """Interrupts from nested descendants are collected."""
  events = [
      _make_event(
          path='wf/parent/child@r-1/inner@r-inner',
          interrupt_ids=['fc-deep'],
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  assert 'wf/parent/child@r-1' in ls.runs
  run = ls.runs['wf/parent/child@r-1']
  assert run.recovered_state is not None
  assert 'fc-deep' in run.recovered_state.interrupt_ids


@pytest.mark.asyncio
async def test_rehydrate_parallel_worker_interrupts():
  """Interrupts from parallel child nodes sharing the parent's path."""
  events = [
      _make_event(
          # Child has exact same path as parent
          path='wf/parent/parallel',
          interrupt_ids=['fc-1'],
          run_id='r-child-1',
      ),
      _make_event(
          path='wf/parent/parallel',
          interrupt_ids=['fc-2'],
          run_id='r-child-2',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  # Rehydrate the parent which has run_id 'r-parent'
  scheduler._rehydrate_from_events(ctx, 'wf/parent/parallel')

  assert 'wf/parent/parallel' in ls.runs
  run = ls.runs['wf/parent/parallel']
  assert run.recovered_state is not None
  assert 'fc-1' in run.recovered_state.interrupt_ids
  assert 'fc-2' in run.recovered_state.interrupt_ids


@pytest.mark.asyncio
async def test_rehydrate_output_for_delegation():
  """Output via output_for delegation is recognized."""
  events = [
      _make_event(
          path='wf/parent/child@r-1/inner@r-inner',
          output='delegated',
          output_for=['wf/parent/child@r-1'],
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child@r-1')

  run = ls.runs['wf/parent/child@r-1']
  assert run.recovered_state is not None
  assert run.recovered_state.output == 'delegated'


# =========================================================================
# __call__ — dispatch logic
# =========================================================================


# =========================================================================
# DefaultNodeScheduler — standalone scheduler
# =========================================================================


@pytest.mark.asyncio
async def test_fresh_execution_runs_node():
  """DefaultNodeScheduler runs a fresh node just like DynamicNodeScheduler."""

  class _Child(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'ct: {node_input}'

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'ct: data'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  child_ctx = await tracker(
      ctx,
      _Child(name='child'),
      'data',
      node_name='child',
      run_id='1',
  )

  assert child_ctx.output == 'ct: data'


@pytest.mark.asyncio
async def test_completed_dedup_returns_cached():
  """DefaultNodeScheduler returns cached output for completed nodes."""
  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  # Pre-populate state as if node already completed.
  from google.adk.workflow.utils._rehydration_utils import _ChildScanState

  tracker._state.runs['wf/parent/child@r-1'] = DynamicNodeRun(
      state=NodeState(run_id='r-1'),
      recovered_state=_ChildScanState(
          run_id='r-1',
          output='cached',
      ),
  )

  child_ctx = await tracker(
      ctx,
      BaseNode(name='child'),
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'cached'


@pytest.mark.asyncio
async def test_concurrent_dedup_returns_running_task():
  """Scheduler deduplicates concurrent executions of the same running task."""
  import asyncio

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  # Mock an active running task (not done yet!)
  running_task = asyncio.Future()

  tracker._state.runs['wf/parent/child@r-1'] = DynamicNodeRun(
      state=NodeState(run_id='r-1'),
      task=running_task,
  )

  # Dispatch the scheduler in the background
  scheduler_task = asyncio.create_task(
      tracker(
          ctx,
          BaseNode(name='child'),
          'input',
          node_name='child',
          run_id='r-1',
      )
  )

  # Let the event loop run one tick to execute the scheduler interception
  await asyncio.sleep(0)

  # Resolve the running task dynamically
  mock_context = MagicMock(spec=Context)
  running_task.set_result(mock_context)

  res_ctx = await scheduler_task
  assert res_ctx is mock_context


@pytest.mark.asyncio
async def test_waiting_resolved_resumes_node():
  """DefaultNodeScheduler re-runs nodes with resolved interrupts."""

  class _Resumable(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-1"]}'
        return
      yield 'should not reach here'

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  # Pre-populate state as if node interrupted and was resolved.
  from google.adk.workflow.utils._rehydration_utils import _ChildScanState

  tracker._state.runs['wf/parent/child@r-1'] = DynamicNodeRun(
      state=NodeState(run_id='r-1'),
      recovered_state=_ChildScanState(
          run_id='r-1',
          interrupt_ids={'fc-1'},
          resolved_ids={'fc-1'},
          resolved_responses={'fc-1': 'approved'},
      ),
  )

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'resumed: approved'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  child_ctx = await tracker(
      ctx,
      _Resumable(name='child'),
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'resumed: approved'


@pytest.mark.asyncio
async def test_waiting_unresolved_propagates_interrupts():
  """DefaultNodeScheduler propagates unresolved interrupts."""
  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  from google.adk.workflow.utils._rehydration_utils import _ChildScanState

  tracker._state.runs['wf/parent/child@r-1'] = DynamicNodeRun(
      state=NodeState(run_id='r-1'),
      recovered_state=_ChildScanState(
          run_id='r-1',
          interrupt_ids={'fc-1'},
      ),
  )

  child_ctx = await tracker(
      ctx,
      BaseNode(name='child'),
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.interrupt_ids == {'fc-1'}
  assert 'fc-1' in tracker._state.interrupt_ids


@pytest.mark.asyncio
async def test_calling_waiting_node_without_rerun_raises_value_error():
  """Calling a dynamic node that is waiting for output with rerun_on_resume=False raises ValueError."""

  # Given a dynamic node waiting for output with rerun_on_resume=False
  class _WaitingNode(BaseNode):
    wait_for_output: bool = True

    async def _run_impl(self, *, ctx, node_input):
      yield 'should not reach here'

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  from google.adk.workflow.utils._rehydration_utils import _ChildScanState

  ls.runs['wf/parent/child@r-1'] = DynamicNodeRun(
      state=NodeState(run_id='r-1'),
      recovered_state=_ChildScanState(
          run_id='r-1',
          interrupt_ids={'pause_req'},
          resolved_ids={'pause_req'},
      ),
  )
  scheduler = DynamicNodeScheduler(state=ls)

  # When it is called again
  # Then it raises ValueError
  with pytest.raises(
      ValueError, match='is waiting for output but was called again'
  ):
    await scheduler(
        ctx,
        _WaitingNode(name='child'),
        'input',
        node_name='child',
        run_id='r-1',
    )


def test_get_dynamic_tasks_excludes_done_tasks():
  """get_dynamic_tasks should not return completed tasks."""
  import asyncio

  loop = asyncio.new_event_loop()
  running_task = None
  try:

    async def _done():
      return None

    # run_until_complete returns the coroutine's result, so the run entry has
    # to hold the task itself for the done-task filter to be exercised at all.
    done_task = loop.create_task(_done())
    loop.run_until_complete(done_task)
    running_task = loop.create_task(asyncio.sleep(9999))

    state = DynamicNodeState()
    state.runs['path/done@r-1'] = DynamicNodeRun(
        state=NodeState(run_id='r-1'),
        task=done_task,
    )
    state.runs['path/running@r-2'] = DynamicNodeRun(
        state=NodeState(run_id='r-2'),
        task=running_task,
    )
    state.runs['path/no-task@r-3'] = DynamicNodeRun(
        state=NodeState(run_id='r-3'),
        task=None,
    )

    tasks = state.get_dynamic_tasks()

    assert tasks == [running_task]
  finally:
    # Cancelling without draining leaves the task pending at close() and the
    # sleep coroutine unawaited, which surfaces as a warning in later tests.
    if running_task is not None:
      running_task.cancel()
      loop.run_until_complete(
          asyncio.gather(running_task, return_exceptions=True)
      )
    loop.close()


class _ModelA(BaseModel):
  x: int


@pytest.mark.asyncio
async def test_runtime_schema_validation_passes():
  """Tests that runtime schema validation passes when input matches schema."""
  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  node = BaseNode(name='child', input_schema=_ModelA)

  # We mock _run_node_internal to avoid full execution, we only care about validation in __call__
  scheduler._run_node_internal = AsyncMock(return_value=MagicMock(spec=Context))

  await scheduler(
      ctx,
      node,
      {'x': 1},
      node_name='child',
      run_id='1',
  )
  # Should not raise


@pytest.mark.asyncio
async def test_runtime_schema_validation_raises():
  """Tests that runtime schema validation raises when input mismatches schema."""
  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  node = BaseNode(name='child', input_schema=_ModelA)

  with pytest.raises(ValidationError):
    await scheduler(
        ctx,
        node,
        {'x': 'string'},  # Invalid type for x
        node_name='child',
        run_id='1',
    )


@pytest.mark.asyncio
async def test_runtime_schema_validation_missing_schema_passes():
  """Tests that runtime schema validation passes when no schema is defined."""
  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  node = BaseNode(name='child')  # No input schema

  scheduler._run_node_internal = AsyncMock(return_value=MagicMock(spec=Context))

  await scheduler(
      ctx,
      node,
      {'x': 1},
      node_name='child',
      run_id='1',
  )
  # Should not raise


@pytest.mark.asyncio
async def test_runtime_schema_validation_content_fallback():
  """Tests that runtime schema validation handles Content objects by extraction."""
  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  node = BaseNode(name='child', input_schema=_ModelA)

  scheduler._run_node_internal = AsyncMock(return_value=MagicMock(spec=Context))

  from google.genai import types

  msg = types.Content(parts=[types.Part(text='{"x": 1}')], role='user')

  await scheduler(
      ctx,
      node,
      msg,
      node_name='child',
      run_id='1',
  )
  # Should not raise


# =========================================================================
# Replay Sequence Ordering preservation for Dynamic Nodes
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_replay_ordering_preserved(
    request: pytest.FixtureRequest,
):
  """Test that parallel dynamic nodes maintain their chronological completion order during replay."""
  import asyncio

  from google.adk.events.request_input import RequestInput
  from google.adk.workflow import node
  from google.adk.workflow import START
  from google.adk.workflow._workflow import Workflow
  from google.genai import types

  from .. import testing_utils

  execution_order = []
  recorded_winner_vals = []

  @node
  async def source_a(*, ctx, node_input):
    await asyncio.sleep(0.1)
    execution_order.append('source_a_executed')
    yield 'result_a'

  @node
  async def source_b(*, ctx, node_input):
    # No sleep, completes immediately in Run 1
    execution_order.append('source_b_executed')
    yield 'result_b'

  @node(rerun_on_resume=True)
  async def hitl_node(*, ctx, node_input):
    if 'req_h' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='req_h', message='input h')
      return
    execution_order.append(f'hitl_resumed_with_{node_input}')
    yield f'h_{node_input}'

  @node(rerun_on_resume=True)
  async def parent(*, ctx, node_input):
    completed_order = []

    async def run_and_record(node_func, run_id):
      res = await ctx.run_node(node_func, run_id=run_id)
      completed_order.append(res)
      return res

    task_a = asyncio.create_task(run_and_record(source_a, 'a'))
    task_b = asyncio.create_task(run_and_record(source_b, 'b'))

    await asyncio.wait([task_a, task_b], return_when=asyncio.ALL_COMPLETED)

    winner_val = completed_order[0]
    recorded_winner_vals.append(winner_val)

    await ctx.run_node(hitl_node, node_input=winner_val, run_id='h')

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, parent)])
  runner = testing_utils.InMemoryRunner(node=agent)

  # Run 1: source_b finishes first, source_a finishes second. hitl_node interrupts.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  req_events = [e for e in events1 if e.long_running_tool_ids]
  assert len(req_events) == 1
  assert execution_order == ['source_b_executed', 'source_a_executed']

  invocation_id = events1[0].invocation_id

  # Clear execution order to track replay/resume behavior accurately
  execution_order.clear()
  recorded_winner_vals.clear()

  # Run 2: Resume with response
  resume_payload = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(  # type: ignore[call-arg]  # Third-party SDK signature
                  id='req_h',
                  name='user_input',
                  response={'text': 'response_h'},
              )
          ),
      ],
  )

  await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Assert source_a and source_b were replayed from cache in exact historical order,
  # ensuring winner_val correctly resolves to 'result_b' without re-execution.
  assert recorded_winner_vals == ['result_b']


@pytest.mark.asyncio
async def test_node_with_clone_uses_clone():
  """DynamicNodeScheduler uses node.clone() if available."""

  class MockNodeWithClone(BaseNode):
    clone_called: bool = False

    async def _run_impl(self, *, ctx, node_input):
      yield 'output'

    def clone(self, update=None):
      copied = self.model_copy(update=update)
      copied.clone_called = True
      return copied

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'output'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  node = MockNodeWithClone(name='child')

  await tracker(
      ctx,
      node,
      'data',
      node_name='child',
      run_id='1',
  )

  ctx._run_node_standalone.assert_called_once()
  called_node = ctx._run_node_standalone.call_args[0][0]
  assert called_node.clone_called is True
  assert called_node.name == 'child'


@pytest.mark.asyncio
async def test_node_without_clone_uses_model_copy():
  """DynamicNodeScheduler falls back to model_copy() if clone is not available."""

  class MockNodeWithoutClone(BaseNode):
    model_copy_called: bool = False

    async def _run_impl(self, *, ctx, node_input):
      yield 'output'

    def model_copy(self, *, update=None, deep=False):
      copied = super().model_copy(update=update, deep=deep)
      copied.model_copy_called = True
      return copied

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'output'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  node = MockNodeWithoutClone(name='child')

  await tracker(
      ctx,
      node,
      'data',
      node_name='child',
      run_id='1',
  )

  ctx._run_node_standalone.assert_called_once()
  called_node = ctx._run_node_standalone.call_args[0][0]
  assert called_node.model_copy_called is True
  assert called_node.name == 'child'


@pytest.mark.asyncio
async def test_node_with_clone_preserves_parent_agent():
  """DynamicNodeScheduler preserves parent_agent if it was cleared by clone()."""

  class MockParentAgent(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'parent'

  class MockNodeWithCloneAndParent(BaseNode):
    parent_agent: MockParentAgent | None = None

    async def _run_impl(self, *, ctx, node_input):
      yield 'output'

    def clone(self, update=None):
      copied = self.model_copy(update=update)
      # Simulate BaseAgent.clone behavior of clearing parent_agent
      copied.parent_agent = None
      return copied

  ctx, _ = _make_parent_ctx()
  tracker = DynamicNodeScheduler(state=DynamicNodeState())

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'output'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  parent_node = MockParentAgent(name='parent')
  node = MockNodeWithCloneAndParent(name='child', parent_agent=parent_node)

  await tracker(
      ctx,
      node,
      'data',
      node_name='child',
      run_id='1',
  )

  ctx._run_node_standalone.assert_called_once()
  called_node = ctx._run_node_standalone.call_args[0][0]
  # Verify parent_agent was restored
  assert called_node.parent_agent is parent_node
  assert called_node.name == 'child'


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_auto_generates_sequential_run_id():
  """DynamicNodeScheduler assigns sequential run IDs when run_id is None."""

  class SimpleNode(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'out: {node_input}'

  ctx, _ = _make_parent_ctx()
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  mock_child_ctx1 = MagicMock(spec=Context)
  mock_child_ctx1.error = None
  mock_child_ctx1.interrupt_ids = set()
  mock_child_ctx1.output = 'out: 1'
  mock_child_ctx1.actions = MagicMock()
  mock_child_ctx1.actions.transfer_to_agent = None

  mock_child_ctx2 = MagicMock(spec=Context)
  mock_child_ctx2.error = None
  mock_child_ctx2.interrupt_ids = set()
  mock_child_ctx2.output = 'out: 2'
  mock_child_ctx2.actions = MagicMock()
  mock_child_ctx2.actions.transfer_to_agent = None

  ctx._run_node_standalone = AsyncMock(
      side_effect=[mock_child_ctx1, mock_child_ctx2]
  )

  node = SimpleNode(name='worker')

  # First execution without run_id -> assigns '1'
  await scheduler(ctx, node, 'task1', node_name='worker')
  assert ctx._run_node_standalone.call_count == 1
  assert ctx._run_node_standalone.call_args_list[0].kwargs.get('run_id') == '1'
  assert state.run_counters[ctx.node_path]['worker'] == 1

  # Second execution without run_id -> assigns '2'
  await scheduler(ctx, node, 'task2', node_name='worker')
  assert ctx._run_node_standalone.call_count == 2
  assert ctx._run_node_standalone.call_args_list[1].kwargs.get('run_id') == '2'
  assert state.run_counters[ctx.node_path]['worker'] == 2


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_standalone_defaults_run_id_to_1():
  """When enable_replay=False, DynamicNodeScheduler defaults run_id to '1' without incrementing state."""

  class SimpleNode(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'out: {node_input}'

  ctx, _ = _make_parent_ctx()
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state, enable_replay=False)

  mock_child_ctx1 = MagicMock(spec=Context)
  mock_child_ctx1.error = None
  mock_child_ctx1.interrupt_ids = set()
  mock_child_ctx1.output = 'out: 1'
  mock_child_ctx1.actions = MagicMock()
  mock_child_ctx1.actions.transfer_to_agent = None

  mock_child_ctx2 = MagicMock(spec=Context)
  mock_child_ctx2.error = None
  mock_child_ctx2.interrupt_ids = set()
  mock_child_ctx2.output = 'out: 2'
  mock_child_ctx2.actions = MagicMock()
  mock_child_ctx2.actions.transfer_to_agent = None

  ctx._run_node_standalone = AsyncMock(
      side_effect=[mock_child_ctx1, mock_child_ctx2]
  )

  node = SimpleNode(name='worker')

  # First execution without run_id -> defaults to '1'
  await scheduler(ctx, node, 'task1', node_name='worker')
  assert ctx._run_node_standalone.call_count == 1
  assert ctx._run_node_standalone.call_args_list[0].kwargs.get('run_id') == '1'
  # Node is passed through directly without cloning
  assert ctx._run_node_standalone.call_args_list[0].args[0] is node

  # Second execution of same node without run_id -> still '1'
  await scheduler(ctx, node, 'task2', node_name='worker')
  assert ctx._run_node_standalone.call_count == 2
  assert ctx._run_node_standalone.call_args_list[1].kwargs.get('run_id') == '1'
  assert ctx._run_node_standalone.call_args_list[1].args[0] is node

  # State counters and runs remain empty with replay off
  assert not state.run_counters
  assert not state.runs


@pytest.mark.asyncio
async def test_dynamic_node_state_maintains_independent_run_counters():
  """DynamicNodeState maintains independent counters for different nodes and parent paths."""
  state = DynamicNodeState()
  # Unscoped / root parent
  assert state.next_run_id('agent_a') == '1'
  assert state.next_run_id('agent_a') == '2'
  assert state.next_run_id('agent_b') == '1'
  assert state.next_run_id('agent_a') == '3'
  assert state.next_run_id('agent_b') == '2'
  assert state.run_counters[''] == {'agent_a': 3, 'agent_b': 2}

  # Scoped parents (e.g. parallel branches)
  assert state.next_run_id('child', parent_path='branch_1') == '1'
  assert state.next_run_id('child', parent_path='branch_1') == '2'
  assert state.next_run_id('child', parent_path='branch_2') == '1'
  assert state.run_counters['branch_1'] == {'child': 2}
  assert state.run_counters['branch_2'] == {'child': 1}


@pytest.mark.asyncio
async def test_static_and_dynamic_node_sharing_a_name_do_not_collide():
  """A static graph node and a dynamic node of the same name get distinct run IDs.

  Both allocators -- `Workflow._next_run_id` for static graph nodes and
  `DynamicNodeScheduler` for `ctx.run_node()` -- draw from the same
  `_LoopState` counter, so two runs of the same name under one parent can no
  longer be assigned the same run_id (and therefore the same node_path).
  """

  class SimpleNode(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'out: {node_input}'

  ctx, _ = _make_parent_ctx()
  loop_state = _LoopState()
  scheduler = DynamicNodeScheduler(state=loop_state)

  mock_child_ctx = MagicMock(spec=Context)
  mock_child_ctx.error = None
  mock_child_ctx.interrupt_ids = set()
  mock_child_ctx.output = 'out: task'
  mock_child_ctx.actions = MagicMock()
  mock_child_ctx.actions.transfer_to_agent = None
  ctx._run_node_standalone = AsyncMock(return_value=mock_child_ctx)

  # The static graph node 'worker' runs first and takes run_id '1'.
  static_run_id = Workflow._next_run_id(
      loop_state, 'worker', parent_path=ctx.node_path
  )

  # A dynamic node of the same name under the same parent continues the same
  # sequence instead of restarting at '1'.
  await scheduler(ctx, SimpleNode(name='worker'), 'task', node_name='worker')
  dynamic_run_id = ctx._run_node_standalone.call_args.kwargs.get('run_id')

  assert static_run_id == '1'
  assert dynamic_run_id == '2'

  # A later static run of the same name keeps advancing the shared counter.
  assert (
      Workflow._next_run_id(loop_state, 'worker', parent_path=ctx.node_path)
      == '3'
  )
  assert loop_state.run_counters[ctx.node_path] == {'worker': 3}


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_handles_agent_transfer_loop():
  """DynamicNodeScheduler loops through sequential agent transfers natively."""
  agent_b = LlmAgent(name='agent_b', rerun_on_resume=True)
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  root = LlmAgent(
      name='root', sub_agents=[agent_a, agent_b], rerun_on_resume=True
  )
  agent_a.parent_agent = root
  agent_b.parent_agent = root

  ctx, _ = _make_parent_ctx()
  ctx.node = root
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  child_ctx_a = MagicMock(spec=Context)
  child_ctx_a.error = None
  child_ctx_a.interrupt_ids = set()
  child_ctx_a.actions = EventActions(transfer_to_agent='agent_b')
  child_ctx_a.output = None

  child_ctx_b = MagicMock(spec=Context)
  child_ctx_b.error = None
  child_ctx_b.interrupt_ids = set()
  child_ctx_b.actions = EventActions()
  child_ctx_b.output = 'transferred_result'

  ctx._run_node_standalone = AsyncMock(side_effect=[child_ctx_a, child_ctx_b])

  final_ctx = await scheduler(
      ctx, agent_a, 'initial_input', node_name='agent_a'
  )

  assert final_ctx is child_ctx_b
  assert final_ctx.output == 'transferred_result'
  assert ctx._run_node_standalone.call_count == 2
  # First hop
  assert ctx._run_node_standalone.call_args_list[0].kwargs.get('run_id') == '1'
  # Second hop
  assert ctx._run_node_standalone.call_args_list[1].kwargs.get('run_id') == '1'
  assert state.run_counters[ctx.node_path] == {'agent_a': 1, 'agent_b': 1}


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_transfer_stops_on_interrupt():
  """DynamicNodeScheduler stops transferring and returns context if target interrupts."""
  agent_b = LlmAgent(name='agent_b', rerun_on_resume=True)
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  root = LlmAgent(
      name='root', sub_agents=[agent_a, agent_b], rerun_on_resume=True
  )
  agent_a.parent_agent = root
  agent_b.parent_agent = root

  ctx, _ = _make_parent_ctx()
  ctx.node = root
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  child_ctx_a = MagicMock(spec=Context)
  child_ctx_a.error = None
  child_ctx_a.interrupt_ids = set()
  child_ctx_a.actions = EventActions(transfer_to_agent='agent_b')
  child_ctx_a.output = None

  child_ctx_b = MagicMock(spec=Context)
  child_ctx_b.error = None
  child_ctx_b.interrupt_ids = {'hitl_1'}
  child_ctx_b.actions = EventActions()
  child_ctx_b.output = None

  ctx._run_node_standalone = AsyncMock(side_effect=[child_ctx_a, child_ctx_b])

  final_ctx = await scheduler(
      ctx, agent_a, 'initial_input', node_name='agent_a'
  )

  assert final_ctx is child_ctx_b
  assert 'hitl_1' in final_ctx.interrupt_ids
  assert 'hitl_1' in state.interrupt_ids


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_transfer_raises_on_invalid_target():
  """DynamicNodeScheduler raises ValueError when transferring to an unknown agent."""
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  root = LlmAgent(name='root', sub_agents=[agent_a], rerun_on_resume=True)
  agent_a.parent_agent = root

  ctx, _ = _make_parent_ctx()
  ctx.node = root
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  child_ctx_a = MagicMock(spec=Context)
  child_ctx_a.error = None
  child_ctx_a.interrupt_ids = set()
  child_ctx_a.actions = EventActions(transfer_to_agent='nonexistent_agent')
  child_ctx_a.output = None

  ctx._run_node_standalone = AsyncMock(return_value=child_ctx_a)

  with pytest.raises(
      ValueError, match="Transfer target agent 'nonexistent_agent' not found."
  ):
    await scheduler(ctx, agent_a, 'input', node_name='agent_a')


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_foreign_scheduler_raises_workflow_invariant_error():
  """Verifies that an unsupported foreign scheduler raises WorkflowInvariantError naming the offending type."""
  ctx, _ = _make_parent_ctx()

  class CustomForeignScheduler:
    pass

  ctx._workflow_scheduler = CustomForeignScheduler()
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  with pytest.raises(WorkflowInvariantError) as exc_info:
    await scheduler(
        ctx,
        BaseNode(name='test_node'),
        'input_data',
        node_name='test_node',
    )

  assert 'CustomForeignScheduler' in str(exc_info.value)
  assert 'cannot own a transfer hop' in str(exc_info.value)


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_transfer_to_foreign_scheduler_raises_workflow_invariant_error():
  """Verifies that a transfer hop to a parent context with a foreign scheduler raises WorkflowInvariantError."""
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  agent_b = LlmAgent(name='agent_b', rerun_on_resume=True)
  parent = LlmAgent(name='parent', sub_agents=[agent_a], rerun_on_resume=True)
  root = LlmAgent(
      name='root', sub_agents=[parent, agent_b], rerun_on_resume=True
  )
  agent_a.parent_agent = parent
  parent.parent_agent = root
  agent_b.parent_agent = root

  class OtherForeignScheduler:
    pass

  root_ctx, _ = _make_parent_ctx()
  root_ctx.node = root
  parent_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=parent,
      run_id='1',
  )
  # Install foreign scheduler on root_ctx AFTER building parent_ctx so Hop 1
  # executes on parent_ctx using the current scheduler and transfers to agent_b
  # (whose parent is root_ctx) on Hop 2.
  root_ctx._workflow_scheduler = OtherForeignScheduler()

  child_ctx_a = Context(
      root_ctx._invocation_context,
      parent_ctx=parent_ctx,
      node=agent_a,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='parent'),
  )
  child_ctx_a.output = 'agent_a_out'
  parent_ctx._run_node_standalone = AsyncMock(return_value=child_ctx_a)

  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  with pytest.raises(WorkflowInvariantError) as exc_info:
    await scheduler(
        parent_ctx,
        agent_a,
        'init_input',
        node_name='agent_a',
    )

  # Verify Hop 1 actually ran before Hop 2 failed on the foreign scheduler.
  parent_ctx._run_node_standalone.assert_awaited_once()
  assert 'OtherForeignScheduler' in str(exc_info.value)
  assert 'cannot own a transfer hop' in str(exc_info.value)


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_resumed_run_prefers_recovered_isolation_scope():
  """Resuming an interrupted node prefers its recovered isolation_scope over a newly computed override_isolation_scope."""
  ctx, _ = _make_parent_ctx()
  state = DynamicNodeState()
  scheduler = DynamicNodeScheduler(state=state)

  node_path = f'{ctx.node_path}/task_node@1'
  recovered = _ChildScanState(
      run_id='1',
      interrupt_ids={'req-1'},
      isolation_scope='wf@1/task_node@1',
  )
  run = DynamicNodeRun(
      state=NodeState(
          status=NodeStatus.WAITING, run_id='1', interrupts=['req-1']
      ),
      recovered_state=recovered,
  )
  state.runs[node_path] = run

  node = LlmAgent(name='task_node', rerun_on_resume=True)
  expected_child_ctx = MagicMock(spec=Context)
  expected_child_ctx.error = False
  expected_child_ctx.interrupt_ids = set()
  expected_child_ctx.actions = EventActions()
  expected_child_ctx.output = 'resumed_output'
  ctx._run_node_standalone = AsyncMock(return_value=expected_child_ctx)

  with patch(
      'google.adk.workflow._dynamic_node_scheduler.check_interception'
  ) as mock_check:
    mock_result = MagicMock()
    mock_result.should_run = True
    mock_result.resume_inputs = {'req-1': 'user_reply'}
    mock_check.return_value = mock_result

    result_ctx = await scheduler(
        ctx,
        node,
        'input_data',
        node_name='task_node',
        run_id='1',
        override_isolation_scope='recomputed_task_scope',
    )

  assert result_ctx is expected_child_ctx
  ctx._run_node_standalone.assert_awaited_once()
  assert (
      ctx._run_node_standalone.call_args.kwargs['override_isolation_scope']
      == 'wf@1/task_node@1'
  )


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_transfer_defers_to_target_parent_scheduler(
    mocker,
):
  """A transfer hands the chain to the scheduler owning the target's parent."""
  # Arrange
  child = LlmAgent(name='child', rerun_on_resume=True)
  parent = LlmAgent(name='parent', sub_agents=[child], rerun_on_resume=True)
  root = LlmAgent(name='root', sub_agents=[parent], rerun_on_resume=True)
  child.parent_agent = parent
  parent.parent_agent = root

  root_ctx, _ = _make_parent_ctx()
  root_ctx.node = root
  parent_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=parent,
      run_id='1',
  )

  child_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=parent_ctx,
      node=child,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='parent'),
  )
  parent_ctx2 = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=parent,
      run_id='2',
      event_actions=EventActions(),
  )
  parent_ctx2.output = 'parent_output'

  # root_ctx owns a different scheduler; installed after parent_ctx is built
  # so parent_ctx does not inherit it and gets the transfer-only default.
  root_scheduler = DynamicNodeScheduler(state=DynamicNodeState())
  root_scheduler._execute_step = AsyncMock(return_value=parent_ctx2)
  root_ctx._workflow_scheduler = root_scheduler

  mock_standalone = mocker.patch(
      'google.adk.workflow._dynamic_node_scheduler.run_node_standalone',
      return_value=child_ctx,
  )

  scheduler = DynamicNodeScheduler(
      state=DynamicNodeState(), enable_replay=False
  )

  # Act
  result_ctx = await scheduler(
      parent_ctx, child, node_input='child_input', node_name='child'
  )

  # Assert
  assert result_ctx.output == 'parent_output'
  # The hop after the transfer went through root_ctx's scheduler, not the
  # one installed on parent_ctx.
  assert mock_standalone.call_count == 1
  root_scheduler._execute_step.assert_awaited_once()
  assert root_scheduler._execute_step.await_args.args[0] is root_ctx
  assert root_scheduler._execute_step.await_args.args[1] is parent


@pytest.mark.asyncio
async def test_dynamic_node_scheduler_transfer_restores_use_as_output_on_hop_back_to_calling_ctx(
    mocker,
):
  """A transfer chain that leaves the calling context and hops back restores use_as_output."""
  # Arrange: root -> parent -> [child1, child2]
  child1 = LlmAgent(name='child1', rerun_on_resume=True)
  child2 = LlmAgent(name='child2', rerun_on_resume=True)
  parent = LlmAgent(
      name='parent', sub_agents=[child1, child2], rerun_on_resume=True
  )
  root = LlmAgent(name='root', sub_agents=[parent], rerun_on_resume=True)
  child1.parent_agent = parent
  child2.parent_agent = parent
  parent.parent_agent = root

  root_ctx, _ = _make_parent_ctx()
  root_ctx.node = root
  parent_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=parent,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='child2'),
  )

  child1_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=parent_ctx,
      node=child1,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='parent'),
  )
  child2_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=parent_ctx,
      node=child2,
      run_id='1',
      event_actions=EventActions(),
  )
  child2_ctx.output = 'final_output'

  # root_ctx has a different scheduler
  root_scheduler = DynamicNodeScheduler(state=DynamicNodeState())
  root_scheduler._execute_step = AsyncMock(return_value=parent_ctx)
  root_ctx._workflow_scheduler = root_scheduler

  # parent_ctx's standalone runs: child1 then child2
  mock_standalone = mocker.patch(
      'google.adk.workflow._dynamic_node_scheduler.run_node_standalone',
      side_effect=[child1_ctx, child2_ctx],
  )

  scheduler = DynamicNodeScheduler(state=DynamicNodeState())

  # Act
  result_ctx = await scheduler(
      parent_ctx,
      child1,
      node_input='init',
      node_name='child1',
      use_as_output=True,
  )

  # Assert
  assert result_ctx.output == 'final_output'
  # Hop 1 (child1 on parent_ctx): use_as_output=True
  assert mock_standalone.call_args_list[0].kwargs['use_as_output'] is True
  # Hop 2 (parent on root_ctx via root_scheduler): use_as_output=False
  root_scheduler._execute_step.assert_awaited_once()
  assert (
      root_scheduler._execute_step.await_args.kwargs['use_as_output'] is False
  )
  # Hop 3 (child2 hopped back to parent_ctx): use_as_output restored to True!
  assert mock_standalone.call_args_list[1].kwargs['use_as_output'] is True
