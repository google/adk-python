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

from google.adk.agents.context import Context
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeRun
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from google.adk.workflow.utils._replay_interceptor import check_interception
import pytest


def _make_parent_ctx():
  ctx = MagicMock(spec=Context)
  ctx._invocation_context = MagicMock()
  ctx.resume_inputs = {}
  return ctx


def test_same_turn_completed():
  """Same-turn completed run intercepts and returns cached output."""
  # Given a same-turn completed run
  run = DynamicNodeRun(
      state=NodeState(status=NodeStatus.COMPLETED),
      output='cached-out',
      transfer_to_agent='target-agent',
  )
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=BaseNode(name='node'),
      current_run=run,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=BaseNode(name='node'),
      current_run=run,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=node,
      recovered=recovered,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=node,
      recovered=recovered,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=node,
      recovered=recovered,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()
  ctx.resume_inputs = {
      'fc-1': {'result': 'ans'}
  }  # Simulate FunctionResponse dict

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=node,
      recovered=recovered,
      curr_parent_ctx=ctx,
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
  ctx = _make_parent_ctx()

  # When checked
  result = check_interception(
      node_path='wf/node@1',
      node=node,
      recovered=recovered,
      curr_parent_ctx=ctx,
  )

  # Then it reruns
  assert result.should_run
  assert result.resume_inputs == {'fc-1': 'ans'}
