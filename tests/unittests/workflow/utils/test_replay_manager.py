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


def test_new_replay_manager_has_empty_state() -> None:
  """A freshly created ReplayManager initializes with empty state maps."""
  mgr = ReplayManager()

  assert mgr.recovered_executions == {}
  assert mgr.sequence_barrier is None


def _make_event(
    path="", output=None, interrupt_ids=None, invocation_id="inv-1"
):
  """Create a minimal Event for session event lists."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = "node"
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
async def test_scan_workflow_events_populates_recovered_executions_and_sequence_barrier():
  """Scanning workflow events populates recovered child states and execution barrier."""
  mgr = ReplayManager()
  events = [
      _make_event(path="wf/child1@1", output="out1"),
      _make_event(path="wf/child2@1", output="out2"),
  ]
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-1"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = events
  ctx.node_path = "wf"

  recovered, sequence = mgr.scan_workflow_events(ctx)

  assert "child1@1" in recovered
  assert "child2@1" in recovered
  assert sequence == ["child1@1", "child2@1"]
  assert mgr.sequence_barrier is not None


@pytest.mark.asyncio
async def test_scan_workflow_events_preserves_direct_child_run_id():
  """Scanning workflow events derives run_id from direct child events rather than descendants."""
  mgr = ReplayManager()
  event1 = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child@1", run_id="1"),
      invocation_id="test_inv",
  )
  event2 = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child@1/grandchild@2", run_id="2"),
      invocation_id="test_inv",
  )
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "test_inv"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [event1, event2]
  ctx.node_path = "wf@1"

  children, _ = mgr.scan_workflow_events(ctx)

  assert children["child@1"].run_id == "1"


def test_build_event_index_groups_events_by_parent_and_transitive_ancestors():
  """Building event index categorizes events under direct parent and ancestor paths."""
  from google.genai import types

  mgr = ReplayManager()
  e_a = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child_a@1"),
      invocation_id="inv-1",
  )
  e_b = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child_a@1/grandchild_b@1"),
      invocation_id="inv-1",
  )
  e_c = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child_c@1"),
      invocation_id="inv-1",
      long_running_tool_ids=["fc-1"],
  )
  e_user = Event(
      author="user",
      invocation_id="inv-1",
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name="RequestInput", id="fc-1", response={"result": "ok"}
                  )
              )
          ]
      ),
  )
  events = [e_a, e_b, e_c, e_user]

  mgr._build_event_index(events, invocation_id="inv-1")

  assert mgr._events_by_parent["wf@1"] == [e_a, e_c, e_user]
  assert mgr._events_by_parent["wf@1/child_a@1"] == [e_b]
  assert e_b in mgr._transitive_events_by_parent["wf@1/child_a@1"]
  assert e_b in mgr._transitive_events_by_parent["wf@1"]
  assert e_a in mgr._transitive_events_by_parent["wf@1"]
  assert e_a not in mgr._transitive_events_by_parent.get("wf@1/child_a@1", [])
  assert e_user in mgr._transitive_events_by_parent["wf@1"]


def test_get_events_for_rehydration_lazily_builds_event_index():
  """Requesting rehydration events initializes event index when unbuilt."""
  mgr = ReplayManager()
  e_a = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/child_a@1"),
      invocation_id="inv-1",
  )
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-1"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [e_a]

  assert not mgr._events_by_parent

  events = mgr.get_events_for_rehydration(ctx, "wf@1/child_a@1")

  assert mgr._events_by_parent
  assert events == [e_a]


def test_scan_workflow_events_recovers_children_from_transitive_descendant_events():
  """Scanning workflow events recovers child nodes when events are emitted deep in child subtrees."""
  mgr = ReplayManager()
  e_descendant = _make_event(
      path="wf@1/child_a@1/grandchild_b@1", output="deep_out"
  )
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-1"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [e_descendant]
  ctx.node_path = "wf@1"

  recovered, _ = mgr.scan_workflow_events(ctx)

  assert "child_a@1" in recovered
