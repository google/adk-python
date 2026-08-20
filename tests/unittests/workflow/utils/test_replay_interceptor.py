# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ReplayInterceptor.

Verifies that ReplayInterceptor correctly checks and manages workflow resumption
replay interception.
"""

from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeRun
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from google.adk.workflow.utils._replay_interceptor import check_interception
from google.adk.workflow.utils._replay_interceptor import create_mock_context
from google.adk.workflow.utils._replay_interceptor import InterceptionResult
import pytest


def test_same_turn_completed():
  """Same-turn completed run intercepts and returns cached output."""
  # Given a same-turn completed run
  run = DynamicNodeRun(
      state=NodeState(status=NodeStatus.COMPLETED),
      output='cached-out',
      transfer_to_agent='target-agent',
  )

  # When checked
  result = check_interception(
      node=BaseNode(name='node'),
      current_run=run,
  )

  # Then it intercepts with cached results
  assert not result.should_run
  assert result.output == 'cached-out'
  assert result.transfer_to_agent == 'target-agent'


def test_same_turn_waiting():
  """Same-turn waiting run intercepts and returns unresolved interrupts."""
  # Given a same-turn waiting run
  run = DynamicNodeRun(
      state=NodeState(status=NodeStatus.WAITING, interrupts=['fc-1']),
  )

  # When checked
  result = check_interception(
      node=BaseNode(name='node'),
      current_run=run,
  )

  # Then it intercepts and keeps waiting
  assert not result.should_run
  assert result.interrupts == {'fc-1'}


def test_cross_turn_unresolved_interrupts_no_rerun():
  """Cross-turn unresolved interrupts keep waiting without rerun."""
  # Given unresolved interrupts and node without rerun_on_resume
  recovered = _ChildScanState(
      run_id='1',
      interrupt_ids={'fc-1', 'fc-2'},
      resolved_ids={'fc-1'},
  )
  node = BaseNode(name='node', rerun_on_resume=False)

  # When checked
  result = check_interception(
      node=node,
      recovered=recovered,
  )

  # Then it stays waiting on unresolved interrupts
  assert not result.should_run
  assert result.interrupts == {'fc-2'}


def test_cross_turn_unresolved_interrupts_rerun():
  """Cross-turn unresolved interrupts with rerun resolves progress and reruns."""
  # Given unresolved interrupts and node with rerun_on_resume
  recovered = _ChildScanState(
      run_id='1',
      interrupt_ids={'fc-1', 'fc-2'},
      resolved_ids={'fc-1'},
      resolved_responses={'fc-1': 'ans'},
  )
  node = BaseNode(name='node', rerun_on_resume=True)

  # When checked
  result = check_interception(
      node=node,
      recovered=recovered,
  )

  # Then it reruns with partial resolved inputs
  assert result.should_run
  assert result.resume_inputs == {'fc-1': 'ans'}


def test_cross_turn_completed():
  """Cross-turn completed run fast-forwards output and route."""
  # Given a completed run from history
  recovered = _ChildScanState(
      run_id='1',
      output='past-out',
      route='route-a',
  )
  node = BaseNode(name='node')

  # When checked
  result = check_interception(
      node=node,
      recovered=recovered,
  )

  # Then it fast-forwards with cached output and route
  assert not result.should_run
  assert result.output == 'past-out'
  assert result.route == 'route-a'


def test_cross_turn_all_resolved_no_rerun():
  """Cross-turn all resolved run without rerun auto-completes with responses."""
  # Given all resolved interrupts and node without rerun_on_resume
  recovered = _ChildScanState(
      run_id='1',
      interrupt_ids={'fc-1'},
      resolved_ids={'fc-1'},
      resolved_responses={'fc-1': 'ans'},
  )
  node = BaseNode(name='node', rerun_on_resume=False)

  # When checked
  result = check_interception(
      node=node,
      recovered=recovered,
  )

  # Then it auto-completes
  assert not result.should_run
  assert result.output == 'ans'


def test_cross_turn_all_resolved_rerun():
  """Cross-turn all resolved run with rerun triggers rerun with responses."""
  # Given all resolved interrupts and node with rerun_on_resume
  recovered = _ChildScanState(
      run_id='1',
      interrupt_ids={'fc-1'},
      resolved_ids={'fc-1'},
      resolved_responses={'fc-1': 'ans'},
  )
  node = BaseNode(name='node', rerun_on_resume=True)

  # When checked
  result = check_interception(
      node=node,
      recovered=recovered,
  )

  # Then it reruns
  assert result.should_run
  assert result.resume_inputs == {'fc-1': 'ans'}


# --- create_mock_context ---


def _parent_ctx(branch=None):
  """A root Context standing in for the parent of an intercepted node."""
  ic = InvocationContext(
      invocation_id='inv-1',
      agent=MagicMock(spec=BaseAgent),
      session=Session(id='s', app_name='app', user_id='u'),
      session_service=InMemorySessionService(),
      branch=branch,
  )
  return Context(ic, node_path='wf@1')


def test_create_mock_context_fast_forward_carries_cached_results():
  """A fast-forwarded node exposes its cached results without executing."""
  parent = _parent_ctx()
  result = InterceptionResult(
      should_run=False,
      output='past-out',
      route='route-a',
      transfer_to_agent='target-agent',
  )

  ctx = create_mock_context(
      parent_ctx=parent,
      node=BaseNode(name='node'),
      run_id='1',
      result=result,
      ancestors=['wf@1'],
      node_path='wf@1/node@1',
  )

  assert ctx.output == 'past-out'
  # Marked emitted so the orchestrator does not re-emit the cached output.
  assert ctx._output_emitted is True
  assert ctx.route == 'route-a'
  assert ctx.actions.transfer_to_agent == 'target-agent'
  assert ctx._output_for_ancestors == ['wf@1']
  assert ctx.node_path == 'wf@1/node@1'


def test_create_mock_context_waiting_result_captures_interrupts_only():
  """A node paused on interrupts must not look like it produced an output."""
  parent = _parent_ctx()
  result = InterceptionResult(should_run=False, interrupts={'fc-1', 'fc-2'})

  ctx = create_mock_context(
      parent_ctx=parent,
      node=BaseNode(name='node'),
      run_id='1',
      result=result,
      ancestors=[],
      node_path='wf@1/node@1',
  )

  assert ctx.interrupt_ids == {'fc-1', 'fc-2'}
  assert ctx.output is None
  assert ctx._output_emitted is False
  assert ctx.route is None
  assert ctx.actions.transfer_to_agent is None


def test_create_mock_context_branch_override_does_not_touch_parent():
  """Overriding the branch is scoped to the replayed child's context."""
  parent = _parent_ctx(branch='root')
  result = InterceptionResult(should_run=False, output='out')

  ctx = create_mock_context(
      parent_ctx=parent,
      node=BaseNode(name='node'),
      run_id='1',
      result=result,
      ancestors=[],
      node_path='wf@1/node@1',
      branch='root.sub',
  )

  assert ctx.branch == 'root.sub'
  assert parent.branch == 'root'
