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

"""Tests for dynamic node executor module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import Workflow
from google.adk.workflow._dynamic_node_executor import run_node_internal
from google.adk.workflow._dynamic_node_executor import run_node_standalone
from google.adk.workflow._errors import DynamicNodeFailError
from google.adk.workflow._errors import NodeInterruptedError
import pytest


def _make_context(
    node: BaseNode | None = None,
    *,
    rerun_on_resume: bool = True,
    workflow_scheduler: Any = None,
) -> Context:
  """Creates a minimal Context with real InvocationContext for testing."""
  mock_agent = MagicMock(spec=BaseAgent)
  real_session = Session(
      id='test_session', app_name='test_app', user_id='test_user'
  )
  real_session_service = InMemorySessionService()

  ic = InvocationContext(
      invocation_id='test-inv-id',
      agent=mock_agent,
      session=real_session,
      session_service=real_session_service,
  )
  object.__setattr__(ic, '_enqueue_event', AsyncMock())

  if node is None:
    node = FunctionNode(
        func=lambda: None,
        name='test_parent_node',
        rerun_on_resume=rerun_on_resume,
    )
  ctx = Context(
      invocation_context=ic,
      node=node,
      run_id='0',
  )
  ctx._node_rerun_on_resume = rerun_on_resume
  ctx._workflow_scheduler = workflow_scheduler
  ctx._child_run_counters = {}
  return ctx


async def test_run_node_internal_standalone_returns_child_output():
  """Standalone dynamic node execution returns the child node output."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda node_input: f'hello {node_input}',
      name='greeter',
      rerun_on_resume=True,
  )

  # Act
  result = await run_node_internal(parent_ctx, child_node, node_input='world')

  # Assert
  assert result == 'hello world'


async def test_run_node_internal_returns_child_context_and_handles_resume_inputs(
    mocker,
):
  """run_node_internal returns child Context and forwards resume_inputs to run_node_standalone."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda node_input: f'echo {node_input}',
      name='echoer',
      rerun_on_resume=True,
  )

  child_ctx = _make_context(child_node)
  child_ctx.output = 'echo data'

  mock_standalone = mocker.patch(
      'google.adk.workflow._dynamic_node_executor.run_node_standalone',
      return_value=child_ctx,
  )

  # Act
  result_ctx = await run_node_internal(
      parent_ctx,
      child_node,
      node_input='data',
      return_ctx=True,
      resume_inputs={'some_key': 'some_val'},
  )

  # Assert 1: Returns child context object itself
  assert result_ctx is child_ctx
  assert result_ctx.output == 'echo data'

  # Assert 2: resume_inputs was correctly passed to run_node_standalone
  mock_standalone.assert_called_once()
  _, kwargs = mock_standalone.call_args
  assert kwargs.get('resume_inputs') == {'some_key': 'some_val'}


async def test_run_node_internal_resume_inputs_propagates_end_to_end():
  """End-to-end verification that resume_inputs reaches child context in standalone execution."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda node_input: f'echo {node_input}',
      name='echoer',
      rerun_on_resume=True,
  )

  # Act
  child_ctx = await run_node_internal(
      parent_ctx,
      child_node,
      node_input='data',
      return_ctx=True,
      resume_inputs={'intr_1': 'val_1'},
  )

  # Assert
  assert isinstance(child_ctx, Context)
  assert child_ctx.output == 'echo data'
  assert child_ctx.resume_inputs == {'intr_1': 'val_1'}


async def test_run_node_internal_raises_when_rerun_on_resume_is_false():
  """Calling run_node_internal from a non-resumable parent raises ValueError."""
  # Arrange
  parent_ctx = _make_context(rerun_on_resume=False)
  child_node = FunctionNode(
      func=lambda: 'ok',
      name='child',
      rerun_on_resume=True,
  )

  # Act & Assert
  with pytest.raises(ValueError, match='A node must have rerun_on_resume=True'):
    await run_node_internal(parent_ctx, child_node)


async def test_run_node_internal_use_as_output_marks_delegated_and_rejects_duplicate():
  """Output delegation marks parent delegated and rejects duplicate delegates."""
  # Arrange
  parent_ctx = _make_context()
  child1 = FunctionNode(func=lambda: 1, name='c1', rerun_on_resume=True)
  child2 = FunctionNode(func=lambda: 2, name='c2', rerun_on_resume=True)

  # Act 1: First delegate succeeds
  res1 = await run_node_internal(parent_ctx, child1, use_as_output=True)
  assert res1 == 1
  assert parent_ctx._output_delegated is True

  # Act 2 & Assert: Second delegate on non-Workflow node fails
  with pytest.raises(ValueError, match='already has a use_as_output delegate'):
    await run_node_internal(parent_ctx, child2, use_as_output=True)


async def test_run_node_internal_workflow_scheduler_validates_numeric_run_id():
  """Workflow scheduler rejects explicit numeric run_ids unless validation is skipped."""
  # Arrange
  mock_scheduler = AsyncMock()
  parent_ctx = _make_context(workflow_scheduler=mock_scheduler)
  child_node = FunctionNode(
      func=lambda: 'ok', name='child', rerun_on_resume=True
  )

  # Act & Assert 1: Reject purely numeric run_id in workflow mode
  with pytest.raises(ValueError, match='must contain non-numeric characters'):
    await run_node_internal(
        parent_ctx,
        child_node,
        run_id='123',
        skip_run_id_validation=False,
    )

  # Act 2: Allow purely numeric run_id when skip_run_id_validation is True
  mock_child_ctx = _make_context()
  mock_child_ctx.output = 'bypassed'
  mock_scheduler.return_value = mock_child_ctx

  res = await run_node_internal(
      parent_ctx,
      child_node,
      run_id='123',
      skip_run_id_validation=True,
  )
  assert res == 'bypassed'


async def test_run_node_internal_delegates_run_id_allocation_to_scheduler():
  """Executor forwards run_id=None so the scheduler allocates sequential IDs.

  The sequential allocation itself is covered by
  test_dynamic_node_scheduler.py::test_dynamic_node_scheduler_auto_generates_sequential_run_id.
  """
  # Arrange
  mock_scheduler = AsyncMock()
  parent_ctx = _make_context(workflow_scheduler=mock_scheduler)
  child_node = FunctionNode(
      func=lambda: 'ok', name='child', rerun_on_resume=True
  )

  mock_child_ctx1 = _make_context()
  mock_child_ctx1.output = 'out1'
  mock_child_ctx2 = _make_context()
  mock_child_ctx2.output = 'out2'
  mock_scheduler.side_effect = [mock_child_ctx1, mock_child_ctx2]

  # Act
  res1 = await run_node_internal(parent_ctx, child_node)
  res2 = await run_node_internal(parent_ctx, child_node)

  # Assert
  assert res1 == 'out1'
  assert res2 == 'out2'
  assert mock_scheduler.call_count == 2
  assert mock_scheduler.call_args_list[0].kwargs['run_id'] is None
  assert mock_scheduler.call_args_list[1].kwargs['run_id'] is None


async def test_run_node_internal_propagates_child_error():
  """Child node failures are surfaced as DynamicNodeFailError."""
  # Arrange
  parent_ctx = _make_context()

  def _failing_fn(node_input):
    del node_input
    raise RuntimeError('custom error')

  child_node = FunctionNode(
      func=_failing_fn,
      name='failing_node',
      rerun_on_resume=True,
  )

  # Act & Assert
  with pytest.raises(
      DynamicNodeFailError, match='Dynamic node failing_node failed'
  ):
    await run_node_internal(parent_ctx, child_node)


async def test_run_node_internal_merges_interrupt_ids_and_raises(mocker):
  """Interrupted child merges interrupt_ids into parent and raises NodeInterruptedError."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda node_input: 'noop', name='child', rerun_on_resume=True
  )

  child_ctx = _make_context(child_node)
  child_ctx._interrupt_ids = {'test-intr-1', 'test-intr-2'}

  mocker.patch(
      'google.adk.workflow._dynamic_node_executor.run_node_standalone',
      return_value=child_ctx,
  )

  # Act & Assert
  with pytest.raises(NodeInterruptedError):
    await run_node_internal(parent_ctx, child_node)

  assert 'test-intr-1' in parent_ctx._interrupt_ids
  assert 'test-intr-2' in parent_ctx._interrupt_ids


async def test_run_node_internal_return_ctx_preserves_interrupted_without_raising(
    mocker,
):
  """Interrupted child with return_ctx=True returns the child Context without raising."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda node_input: 'noop', name='child', rerun_on_resume=True
  )

  child_ctx = _make_context(child_node)
  child_ctx._interrupt_ids = {'test-intr-1'}

  mocker.patch(
      'google.adk.workflow._dynamic_node_executor.run_node_standalone',
      return_value=child_ctx,
  )

  # Act
  res_ctx = await run_node_internal(parent_ctx, child_node, return_ctx=True)

  # Assert
  assert res_ctx is child_ctx
  assert 'test-intr-1' in res_ctx.interrupt_ids


async def test_run_node_internal_raise_on_wait_raises_when_child_waiting(
    mocker,
):
  """raise_on_wait=True surfaces waiting nodes with no output as NodeInterruptedError."""
  # Arrange
  parent_ctx = _make_context()
  child_node = FunctionNode(
      func=lambda: None,
      name='waiting_node',
      rerun_on_resume=True,
  )
  child_node.wait_for_output = True

  child_ctx = _make_context(child_node)
  child_ctx.output = None

  mocker.patch(
      'google.adk.workflow._dynamic_node_executor.run_node_standalone',
      return_value=child_ctx,
  )

  # Act & Assert
  with pytest.raises(NodeInterruptedError):
    await run_node_internal(
        parent_ctx,
        child_node,
        raise_on_wait=True,
    )


async def test_run_node_internal_executes_sequential_agent_transfers():
  """run_node_internal loops through sequential agent transfers until terminal output."""
  # Arrange
  agent_b = LlmAgent(name='agent_b', rerun_on_resume=True)
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  root = LlmAgent(
      name='root', sub_agents=[agent_a, agent_b], rerun_on_resume=True
  )
  agent_a.parent_agent = root
  agent_b.parent_agent = root

  root_ctx = _make_context(node=root)

  child_ctx_a = Context(
      invocation_context=root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=agent_a,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='agent_b'),
  )

  child_ctx_b = Context(
      invocation_context=root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=agent_b,
      run_id='1',
      event_actions=EventActions(),
  )
  child_ctx_b.output = 'terminal_output'

  mock_scheduler = AsyncMock(side_effect=[child_ctx_a, child_ctx_b])
  root_ctx._workflow_scheduler = mock_scheduler

  # Act
  result = await run_node_internal(root_ctx, agent_a, node_input='init')

  # Assert
  assert result == 'terminal_output'
  assert mock_scheduler.call_count == 2


async def test_run_node_standalone_runs_node_directly():
  """run_node_standalone invokes NodeRunner and returns the child Context."""
  # Arrange
  parent_ctx = _make_context()
  node = FunctionNode(
      func=lambda node_input: f'standalone_{node_input}',
      name='runner_test',
      rerun_on_resume=True,
  )

  # Act
  result_ctx = await run_node_standalone(parent_ctx, node, node_input='payload')

  # Assert
  assert isinstance(result_ctx, Context)
  assert result_ctx.output == 'standalone_payload'


async def test_run_node_standalone_passes_resume_inputs_to_node_runner():
  """run_node_standalone forwards resume_inputs to NodeRunner populating child Context."""
  # Arrange
  parent_ctx = _make_context()
  node = FunctionNode(
      func=lambda node_input: f'resumed_{node_input}',
      name='resume_runner_test',
      rerun_on_resume=True,
  )

  # Act
  result_ctx = await run_node_standalone(
      parent_ctx,
      node,
      node_input='payload',
      resume_inputs={'intr_key': 'intr_val'},
  )

  # Assert
  assert isinstance(result_ctx, Context)
  assert result_ctx.output == 'resumed_payload'
  assert result_ctx.resume_inputs == {'intr_key': 'intr_val'}


async def test_run_node_internal_parent_transfer_routes_to_parent_agent():
  """Parent transfer routes execution to parent agent climbing up the context tree."""
  # Arrange
  child = LlmAgent(name='child', rerun_on_resume=True)
  parent = LlmAgent(name='parent', sub_agents=[child], rerun_on_resume=True)
  root = LlmAgent(name='root', sub_agents=[parent], rerun_on_resume=True)
  child.parent_agent = parent
  parent.parent_agent = root

  root_ctx = _make_context(node=root)
  root_ctx._child_run_counters = {'parent': 1}

  parent_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=parent,
      run_id='1',
  )
  parent_ctx._child_run_counters = {'child': 1}

  mock_scheduler = AsyncMock()
  root_ctx._workflow_scheduler = mock_scheduler
  parent_ctx._workflow_scheduler = mock_scheduler

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

  mock_scheduler.side_effect = [child_ctx, parent_ctx2]

  # Act
  result = await run_node_internal(
      parent_ctx, child, node_input='child_input', use_as_output=True
  )

  # Assert
  assert result == 'parent_output'
  assert mock_scheduler.call_count == 2
  calls = mock_scheduler.call_args_list
  assert calls[0][0][0] is parent_ctx
  assert calls[0][0][1].name == child.name
  assert calls[0][1].get('use_as_output') is True

  assert calls[1][0][0] is root_ctx
  assert calls[1][0][1].name == parent.name
  assert calls[1][1].get('use_as_output') is False


async def test_run_node_internal_standalone_sibling_transfer(mocker):
  """Standalone sibling transfer executes target agent without a workflow scheduler."""
  # Arrange
  agent_b = LlmAgent(name='agent_b', rerun_on_resume=True)
  agent_a = LlmAgent(name='agent_a', rerun_on_resume=True)
  root = LlmAgent(
      name='root', sub_agents=[agent_a, agent_b], rerun_on_resume=True
  )
  agent_a.parent_agent = root
  agent_b.parent_agent = root

  root_ctx = _make_context(node=root)
  root_ctx._child_run_counters = {}
  root_ctx._workflow_scheduler = None

  child_ctx_a = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=agent_a,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='agent_b'),
  )

  child_ctx_b = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=agent_b,
      run_id='1',
      event_actions=EventActions(),
  )
  child_ctx_b.output = 'standalone_b_output'

  mock_run_standalone = mocker.patch(
      'google.adk.workflow._dynamic_node_executor.run_node_standalone',
      side_effect=[child_ctx_a, child_ctx_b],
  )

  # Act
  result = await run_node_internal(root_ctx, agent_a, node_input='a_input')

  # Assert
  assert result == 'standalone_b_output'
  assert mock_run_standalone.call_count == 2


async def test_run_node_internal_child_transfer_routes_downward():
  """Child transfer routes execution to a sub-agent downward in the agent hierarchy."""
  # Arrange
  child = LlmAgent(name='child', rerun_on_resume=True)
  parent = LlmAgent(name='parent', sub_agents=[child], rerun_on_resume=True)
  child.parent_agent = parent

  parent_ctx = _make_context(node=None)
  mock_scheduler = AsyncMock()
  parent_ctx._workflow_scheduler = mock_scheduler

  parent_run_ctx = Context(
      parent_ctx._invocation_context,
      parent_ctx=parent_ctx,
      node=parent,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='child'),
  )

  child_run_ctx = Context(
      parent_ctx._invocation_context,
      parent_ctx=parent_run_ctx,
      node=child,
      run_id='1',
      event_actions=EventActions(),
  )
  child_run_ctx.output = 'child_output'

  mock_scheduler.side_effect = [parent_run_ctx, child_run_ctx]

  # Act
  result = await run_node_internal(
      parent_ctx, parent, node_input='parent_input'
  )

  # Assert
  assert result == 'child_output'
  assert mock_scheduler.call_count == 2
  calls = mock_scheduler.call_args_list
  assert calls[0][0][0] is parent_ctx
  assert calls[0][0][1].name == parent.name
  assert calls[1][0][0] is parent_run_ctx
  assert calls[1][0][1].name == child.name


async def test_run_node_internal_three_layer_transfer_round_trip():
  """Verifies 3-layer round trip transfer (Root -> Child -> Grandchild -> Child -> Root)."""
  # Arrange
  grandchild = LlmAgent(name='grandchild', rerun_on_resume=True)
  child = LlmAgent(name='child', sub_agents=[grandchild], rerun_on_resume=True)
  root = LlmAgent(name='root', sub_agents=[child], rerun_on_resume=True)
  grandchild.parent_agent = child
  child.parent_agent = root

  root_ctx = _make_context(node=None)
  mock_scheduler = AsyncMock()
  root_ctx._workflow_scheduler = mock_scheduler

  root_run_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=root,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='child'),
  )
  child_run_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_run_ctx,
      node=child,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='grandchild'),
  )
  grandchild_run_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=child_run_ctx,
      node=grandchild,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='child'),
  )
  child_run_ctx2 = Context(
      root_ctx._invocation_context,
      parent_ctx=root_run_ctx,
      node=child,
      run_id='2',
      event_actions=EventActions(transfer_to_agent='root'),
  )
  root_run_ctx2 = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=root,
      run_id='2',
      event_actions=EventActions(),
  )
  root_run_ctx2.output = 'final_root_output'

  mock_scheduler.side_effect = [
      root_run_ctx,
      child_run_ctx,
      grandchild_run_ctx,
      child_run_ctx2,
      root_run_ctx2,
  ]

  # Act
  result = await run_node_internal(root_ctx, root, node_input='start')

  # Assert
  assert result == 'final_root_output'
  assert mock_scheduler.call_count == 5
  calls = mock_scheduler.call_args_list
  assert calls[0][0][0] is root_ctx
  assert calls[1][0][0] is root_run_ctx
  assert calls[2][0][0] is child_run_ctx
  assert calls[3][0][0] is root_run_ctx
  assert calls[4][0][0] is root_ctx


async def test_run_node_internal_transfer_preserves_use_as_output_for_original_context():
  """Verifies use_as_output is preserved when transferring back to the initial context."""
  # Arrange
  child = LlmAgent(name='child', rerun_on_resume=True)
  root = LlmAgent(name='root', sub_agents=[child], rerun_on_resume=True)
  child.parent_agent = root

  root_ctx = _make_context(node=None)
  mock_scheduler = AsyncMock()
  root_ctx._workflow_scheduler = mock_scheduler

  root_run_ctx1 = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=root,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='child'),
  )
  child_run_ctx = Context(
      root_ctx._invocation_context,
      parent_ctx=root_run_ctx1,
      node=child,
      run_id='1',
      event_actions=EventActions(transfer_to_agent='root'),
  )
  root_run_ctx2 = Context(
      root_ctx._invocation_context,
      parent_ctx=root_ctx,
      node=root,
      run_id='2',
      event_actions=EventActions(),
  )
  root_run_ctx2.output = 'final_output'

  mock_scheduler.side_effect = [
      root_run_ctx1,
      child_run_ctx,
      root_run_ctx2,
  ]

  # Act
  result = await run_node_internal(
      root_ctx, root, node_input='start', use_as_output=True
  )

  # Assert
  assert result == 'final_output'
  assert mock_scheduler.call_count == 3
  calls = mock_scheduler.call_args_list
  assert calls[0][1].get('use_as_output') is True
  assert calls[1][1].get('use_as_output') is False
  assert calls[2][1].get('use_as_output') is True
