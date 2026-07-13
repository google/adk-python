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

"""Tests for ReplayManager utility."""

from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.workflow.utils._replay_manager import ReplayManager
import pytest


def test_replay_manager_init() -> None:
  """Tests that ReplayManager initializes with empty state."""
  mgr = ReplayManager()
  assert mgr.recovered_executions == {}
  assert mgr.sequence_barrier is None


def _make_event(
    path='', output=None, interrupt_ids=None, invocation_id='inv-1'
):
  """Create a minimal Event for session event lists."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = 'node'
  event.output = output
  event.partial = False
  event.node_info = MagicMock(spec=NodeInfo)
  event.node_info.path = path
  event.node_info.output_for = None
  event.node_info.message_as_output = None
  event.branch = None
  event.isolation_scope = None
  event.long_running_tool_ids = set(interrupt_ids) if interrupt_ids else None
  event.content = None
  event.actions = None
  return event


@pytest.mark.asyncio
async def test_scan_workflow_events():
  """Scan workflow events populates recovered_executions and sequence_barrier."""
  mgr = ReplayManager()
  events = [
      _make_event(path='wf/child1@1', output='out1'),
      _make_event(path='wf/child2@1', output='out2'),
  ]
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = 'inv-1'
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = events
  ctx.node_path = 'wf'

  recovered, sequence = mgr.scan_workflow_events(ctx)

  assert 'child1@1' in recovered
  assert 'child2@1' in recovered
  assert sequence == ['child1@1', 'child2@1']
  assert mgr.sequence_barrier is not None


@pytest.mark.asyncio
async def test_scan_child_events_ignores_descendant_run_id_resets():
  """scan_workflow_events only resets run_id from direct child events."""
  mgr = ReplayManager()

  event1 = Event(
      author='node',
      node_info=NodeInfo(path='wf@1/child@1', run_id='1'),
      invocation_id='test_inv',
  )
  event2 = Event(
      author='node',
      node_info=NodeInfo(path='wf@1/child@1/grandchild@2', run_id='2'),
      invocation_id='test_inv',
  )

  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = 'test_inv'
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [event1, event2]
  ctx.node_path = 'wf@1'

  children, _ = mgr.scan_workflow_events(ctx)

  # Assert child 'child' run_id remains '1' (not '2' from the descendant).
  assert children['child@1'].run_id == '1'
