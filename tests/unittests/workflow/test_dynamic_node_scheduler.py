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

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeRun
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeState
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import _LoopState
from pydantic import BaseModel
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
  ctx._child_run_counters = {}

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

  with pytest.raises(
      ValueError,
      match=r"Runtime schema validation failed for dynamic node 'child'",
  ):
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
# __call__ — Agent Transfer logic
# =========================================================================


@pytest.mark.asyncio
async def test_scheduler_handles_child_transfer():
  """Scheduler processes CHILD relationship by nesting next context."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  target = LlmAgent(name='target')
  current = LlmAgent(name='current', sub_agents=[target])
  root = LlmAgent(name='root', sub_agents=[current])

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  child_ctx1 = MagicMock()
  child_ctx1.node_path = 'wf/parent/current@1'
  child_ctx1.parent_ctx = ctx
  child_ctx1.actions = EventActions(transfer_to_agent='target')
  child_ctx1.error = None
  child_ctx1.interrupt_ids = set()
  child_ctx1._invocation_context = ctx._invocation_context
  child_ctx1._child_run_counters = {}

  child_ctx2 = MagicMock()
  child_ctx2.node_path = 'wf/parent/current@1/target@1'
  child_ctx2.parent_ctx = child_ctx1
  child_ctx2.actions = EventActions()
  child_ctx2.error = None
  child_ctx2.interrupt_ids = set()
  child_ctx2._invocation_context = ctx._invocation_context
  child_ctx2._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(side_effect=[child_ctx1, child_ctx2])

  # Act
  final_ctx = await scheduler(
      ctx,
      current,
      'input',
      node_name='current',
      run_id='1',
  )

  # Assert
  assert final_ctx is child_ctx2
  calls = scheduler._run_node_internal.call_args_list
  assert len(calls) == 2

  # First call: current node
  args1, kwargs1 = calls[0]
  assert args1[0] is ctx
  assert args1[1] is current
  assert args1[2] == 'current'
  assert args1[4] == '1'

  # Second call: target node (transferred CHILD)
  args2, kwargs2 = calls[1]
  assert args2[0] is child_ctx1
  assert args2[1] is target
  assert args2[2] == 'target'
  assert args2[4] == '1'


@pytest.mark.asyncio
async def test_scheduler_handles_sibling_transfer():
  """Scheduler processes SIBLING relationship by sharing parent context."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  current = LlmAgent(name='current')
  target = LlmAgent(name='target')
  root = LlmAgent(name='root', sub_agents=[current, target])

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  child_ctx1 = MagicMock()
  child_ctx1.node_path = 'wf/parent/current@1'
  child_ctx1.parent_ctx = ctx
  child_ctx1.actions = EventActions(transfer_to_agent='target')
  child_ctx1.error = None
  child_ctx1.interrupt_ids = set()
  child_ctx1._invocation_context = ctx._invocation_context
  child_ctx1._child_run_counters = {}

  child_ctx2 = MagicMock()
  child_ctx2.node_path = 'wf/parent/target@1'
  child_ctx2.parent_ctx = ctx
  child_ctx2.actions = EventActions()
  child_ctx2.error = None
  child_ctx2.interrupt_ids = set()
  child_ctx2._invocation_context = ctx._invocation_context
  child_ctx2._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(side_effect=[child_ctx1, child_ctx2])

  # Act
  final_ctx = await scheduler(
      ctx,
      current,
      'input',
      node_name='current',
      run_id='1',
  )

  # Assert
  assert final_ctx is child_ctx2
  calls = scheduler._run_node_internal.call_args_list
  assert len(calls) == 2

  # Second call: target node (transferred SIBLING)
  args2, kwargs2 = calls[1]
  assert args2[0] is ctx
  assert args2[1] is target
  assert args2[2] == 'target'
  assert args2[4] == '1'


@pytest.mark.asyncio
async def test_scheduler_handles_parent_transfer():
  """Scheduler processes PARENT relationship by truncating parent context."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  current = LlmAgent(name='current')
  target = LlmAgent(name='target', sub_agents=[current])
  root = LlmAgent(name='root', sub_agents=[target])

  ctx, _ = _make_parent_ctx()
  ctx._child_run_counters = {'target': 1}
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  target_ctx = MagicMock()
  target_ctx.node_path = 'wf/parent/target@1'
  target_ctx.node = target
  target_ctx.parent_ctx = ctx
  target_ctx._invocation_context = ctx._invocation_context
  target_ctx._child_run_counters = {}

  child_ctx1 = MagicMock()
  child_ctx1.node_path = 'wf/parent/target@1/current@1'
  child_ctx1.node = current
  child_ctx1.parent_ctx = target_ctx
  child_ctx1.actions = EventActions(transfer_to_agent='target')
  child_ctx1.error = None
  child_ctx1.interrupt_ids = set()
  child_ctx1._invocation_context = ctx._invocation_context
  child_ctx1._child_run_counters = {}

  child_ctx2 = MagicMock()
  child_ctx2.node_path = 'wf/parent/target@2'
  child_ctx2.parent_ctx = ctx
  child_ctx2.actions = EventActions()
  child_ctx2.error = None
  child_ctx2.interrupt_ids = set()
  child_ctx2._invocation_context = ctx._invocation_context
  child_ctx2._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(side_effect=[child_ctx1, child_ctx2])

  # Act
  final_ctx = await scheduler(
      target_ctx,
      current,
      'input',
      node_name='current',
      run_id='1',
  )

  # Assert
  assert final_ctx is child_ctx2
  calls = scheduler._run_node_internal.call_args_list
  assert len(calls) == 2

  # Second call: target node (transferred ANCESTOR)
  args2, kwargs2 = calls[1]
  assert args2[0] is ctx
  assert args2[1] is target
  assert args2[2] == 'target'
  assert args2[4] == '2'


@pytest.mark.asyncio
async def test_scheduler_raises_value_error_on_self_transfer():
  """Scheduler raises ValueError when agent attempts to transfer to itself."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  current = LlmAgent(name='current')
  root = LlmAgent(name='root', sub_agents=[current])

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  child_ctx = MagicMock()
  child_ctx.node_path = 'wf/parent/current@1'
  child_ctx.parent_ctx = ctx
  child_ctx.actions = EventActions(transfer_to_agent='current')
  child_ctx.error = None
  child_ctx.interrupt_ids = set()
  child_ctx._invocation_context = ctx._invocation_context
  child_ctx._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(return_value=child_ctx)

  # Act & Assert
  with pytest.raises(ValueError, match='cannot transfer to itself'):
    await scheduler(
        ctx,
        current,
        'input',
        node_name='current',
        run_id='1',
    )


@pytest.mark.asyncio
async def test_scheduler_handles_parent_transfer_bypassed_on_resume():
  """Scheduler processes PARENT relationship when parent was bypassed on resume."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  current = LlmAgent(name='current')
  target = LlmAgent(name='target', sub_agents=[current])
  root = LlmAgent(name='root', sub_agents=[target])

  current.parent_agent = target
  target.parent_agent = root

  ctx, _ = _make_parent_ctx()
  ctx.node = None
  ctx._child_run_counters = {'target': 1}
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  target_ctx = MagicMock()
  target_ctx.node_path = 'current@1'
  target_ctx.node = current
  target_ctx.parent_ctx = ctx
  target_ctx._invocation_context = ctx._invocation_context
  target_ctx._child_run_counters = {}

  child_ctx1 = MagicMock()
  child_ctx1.node_path = 'current@1'
  child_ctx1.node = current
  child_ctx1.parent_ctx = ctx
  child_ctx1.actions = EventActions(transfer_to_agent='target')
  child_ctx1.error = None
  child_ctx1.interrupt_ids = set()
  child_ctx1._invocation_context = ctx._invocation_context
  child_ctx1._child_run_counters = {}

  child_ctx2 = MagicMock()
  child_ctx2.node_path = 'target@2'
  child_ctx2.parent_ctx = ctx
  child_ctx2.actions = EventActions()
  child_ctx2.error = None
  child_ctx2.interrupt_ids = set()
  child_ctx2._invocation_context = ctx._invocation_context
  child_ctx2._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(side_effect=[child_ctx1, child_ctx2])

  # Act
  final_ctx = await scheduler(
      ctx,
      current,
      'input',
      node_name='current',
      run_id='1',
  )

  # Assert
  assert final_ctx is child_ctx2
  calls = scheduler._run_node_internal.call_args_list
  assert len(calls) == 2

  # First call: current node
  args1, kwargs1 = calls[0]
  assert args1[0] is ctx
  assert args1[1] is current
  assert args1[2] == 'current'
  assert args1[4] == '1'

  # Second call: target node (transferred PARENT)
  args2, kwargs2 = calls[1]
  assert args2[0] is ctx
  assert args2[1] is target
  assert args2[2] == 'target'
  assert args2[4] == '2'


@pytest.mark.asyncio
async def test_scheduler_handles_three_layer_agent_transfers_round_trip():
  """Verify 3-layer agent transfers (Root -> Child -> Grandchild -> Child -> Root)."""
  # Arrange
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.events.event_actions import EventActions
  from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
  from google.adk.workflow._workflow import _LoopState

  grandchild = LlmAgent(name='grandchild')
  child = LlmAgent(name='child', sub_agents=[grandchild])
  root = LlmAgent(name='root', sub_agents=[child])

  grandchild.parent_agent = child
  child.parent_agent = root

  # ctx is the root context (node = None, parent_ctx = None)
  ctx = MagicMock()
  ctx.node = None
  ctx.parent_ctx = None
  ctx.node_path = ''
  ctx._child_run_counters = {'child': 1}
  ctx._output_for_ancestors = []

  ic = MagicMock()
  ic.invocation_id = 'inv-1'
  ic.session = MagicMock()
  ic.session.state = {}
  ic.session.events = []
  ic.run_config = None
  ctx._invocation_context = ic

  # root_ctx is the parent context of child (node = root, parent_ctx = ctx)
  root_ctx = MagicMock()
  root_ctx.node = root
  root_ctx.parent_ctx = ctx
  root_ctx.node_path = 'root@1'
  root_ctx._child_run_counters = {'child': 1}
  root_ctx._invocation_context = ic

  ls = _LoopState()
  scheduler = DynamicNodeScheduler(state=ls)

  # Step 1: root delegates to child
  child_ctx1 = MagicMock()
  child_ctx1.node_path = 'root@1/child@1'
  child_ctx1.node = child
  child_ctx1.parent_ctx = root_ctx
  child_ctx1.actions = EventActions(transfer_to_agent='grandchild')
  child_ctx1.error = None
  child_ctx1.interrupt_ids = set()
  child_ctx1._invocation_context = ic
  child_ctx1._child_run_counters = {}

  # Step 2: child delegates to grandchild
  grandchild_ctx1 = MagicMock()
  grandchild_ctx1.node_path = 'root@1/child@1/grandchild@1'
  grandchild_ctx1.node = grandchild
  grandchild_ctx1.parent_ctx = child_ctx1
  grandchild_ctx1.actions = EventActions(transfer_to_agent='child')
  grandchild_ctx1.error = None
  grandchild_ctx1.interrupt_ids = set()
  grandchild_ctx1._invocation_context = ic
  grandchild_ctx1._child_run_counters = {}

  # Step 3: grandchild transfers back to child
  child_ctx2 = MagicMock()
  child_ctx2.node_path = 'root@1/child@2'
  child_ctx2.node = child
  child_ctx2.parent_ctx = root_ctx
  child_ctx2.actions = EventActions(transfer_to_agent='root')
  child_ctx2.error = None
  child_ctx2.interrupt_ids = set()
  child_ctx2._invocation_context = ic
  child_ctx2._child_run_counters = {}

  # Step 4: child transfers back to root
  ctx._child_run_counters = {'root': 1}
  root_ctx2 = MagicMock()
  root_ctx2.node_path = 'root@2'
  root_ctx2.node = root
  root_ctx2.parent_ctx = ctx
  root_ctx2.actions = EventActions()
  root_ctx2.error = None
  root_ctx2.interrupt_ids = set()
  root_ctx2._invocation_context = ic
  root_ctx2._child_run_counters = {}

  scheduler._run_node_internal = AsyncMock(
      side_effect=[child_ctx1, grandchild_ctx1, child_ctx2, root_ctx2]
  )

  # Act
  final_ctx = await scheduler(
      root_ctx,
      child,
      'input',
      node_name='child',
      run_id='1',
  )

  # Assert
  assert final_ctx is root_ctx2
  calls = scheduler._run_node_internal.call_args_list
  assert len(calls) == 4

  # 1st call: child (scheduled by root)
  args1, kwargs1 = calls[0]
  assert args1[0] is root_ctx
  assert args1[1] is child
  assert args1[2] == 'child'
  assert args1[4] == '1'

  # 2nd call: grandchild (transferred CHILD from child)
  args2, kwargs2 = calls[1]
  assert args2[0] is child_ctx1
  assert args2[1] is grandchild
  assert args2[2] == 'grandchild'
  assert args2[4] == '1'

  # 3rd call: child (transferred PARENT from grandchild)
  args3, kwargs3 = calls[2]
  assert args3[0] is root_ctx
  assert args3[1] is child
  assert args3[2] == 'child'
  assert args3[4] == '2'

  # 4th call: root (transferred PARENT from child)
  args4, kwargs4 = calls[3]
  assert args4[0] is ctx
  assert args4[1] is root
  assert args4[2] == 'root'
  assert args4[4] == '2'


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
