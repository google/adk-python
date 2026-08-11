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

import asyncio
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


def test_scan_workflow_events_sequence_excludes_prior_invocation_events():
  """Replay sequence covers only the current invocation.

  A session may hold a completed earlier invocation followed by a second
  invocation that pauses for human input. Terminal events from the earlier
  invocation must not enter the replay sequence, otherwise the sequence
  barrier blocks on a node that never runs during the resume.
  """
  mgr = ReplayManager()
  # Completed earlier invocation in the same session.
  prior = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/finish@1", run_id="1"),
      invocation_id="inv-1",
      output="prior_out",
  )
  # Current invocation, ending on an unresolved RequestInput interrupt.
  current_first = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/alpha@1", run_id="1"),
      invocation_id="inv-2",
      output="alpha_out",
  )
  current_pending = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/beta@1", run_id="1"),
      invocation_id="inv-2",
      long_running_tool_ids=["clarify:1"],
  )

  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-2"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [
      prior,
      current_first,
      current_pending,
  ]
  ctx.node_path = "wf@1"

  recovered, sequence = mgr.scan_workflow_events(ctx)

  assert sequence == ["alpha@1", "beta@1"]
  # Sequence and recovered state must agree; disagreement was the defect.
  assert "finish@1" not in recovered
  # The fix belongs in _scan_sequence, NOT in the event index: the index
  # deliberately spans the whole session so multi-turn context stays visible
  # during rehydration. Filtering there instead would pass the assertions
  # above while silently breaking cross-turn context.
  assert prior in mgr._transitive_events_by_parent["wf@1"]


def test_prepare_parent_sequence_barrier_excludes_prior_invocation_events():
  """Dynamic-node sequence barriers are also scoped to the current invocation."""
  mgr = ReplayManager()
  prior = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/finish@1", run_id="1"),
      invocation_id="inv-1",
      output="prior_out",
  )
  current = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/alpha@1", run_id="1"),
      invocation_id="inv-2",
      output="alpha_out",
  )

  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-2"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [prior, current]
  ctx.node_path = "wf@1"

  barrier = mgr.prepare_parent_sequence_barrier(ctx, "wf@1")

  assert barrier.sequence == ["alpha@1"]
  assert prior in mgr._events_by_parent["wf@1"]


@pytest.mark.asyncio
async def test_scan_workflow_events_sequence_empty_when_all_events_are_prior():
  """A session holding only prior-invocation events yields a non-blocking barrier."""
  mgr = ReplayManager()
  prior = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/finish@1", run_id="1"),
      invocation_id="inv-1",
      output="prior_out",
  )

  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-2"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [prior]
  ctx.node_path = "wf@1"

  _, sequence = mgr.scan_workflow_events(ctx)

  assert sequence == []
  # An empty sequence must fast-forward rather than deadlock.
  await asyncio.wait_for(mgr.sequence_barrier.wait("anything"), timeout=1)


def _recorded_two_step_ctx():
  """A ctx whose session records alpha completing before beta."""
  alpha = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/alpha@1", run_id="1"),
      invocation_id="inv-1",
      output="alpha_out",
  )
  beta = Event(
      author="node",
      node_info=NodeInfo(path="wf@1/beta@1", run_id="1"),
      invocation_id="inv-1",
      output="beta_out",
  )
  ctx = MagicMock()
  ctx._invocation_context = MagicMock()
  ctx._invocation_context.invocation_id = "inv-1"
  ctx._invocation_context.session = MagicMock()
  ctx._invocation_context.session.events = [alpha, beta]
  ctx.node_path = "wf@1"
  return ctx


@pytest.mark.asyncio
async def test_wait_sequence_holds_second_key_until_first_advances():
  """Replay follows the recorded order: beta cannot start before alpha ends."""
  mgr = ReplayManager()
  ctx = _recorded_two_step_ctx()
  barrier = mgr.prepare_parent_sequence_barrier(ctx, "wf@1")
  assert barrier.sequence == ["alpha@1", "beta@1"]

  # The first recorded key is already open.
  await asyncio.wait_for(mgr.wait_sequence("wf@1", "alpha@1"), timeout=1)

  beta_started = False

  async def _wait_beta():
    nonlocal beta_started
    await mgr.wait_sequence("wf@1", "beta@1")
    beta_started = True

  task = asyncio.create_task(_wait_beta())
  await asyncio.sleep(0.05)
  assert not beta_started

  await mgr.advance_sequence("wf@1", "alpha@1")

  await asyncio.wait_for(task, timeout=1)
  assert beta_started


@pytest.mark.asyncio
async def test_advance_sequence_with_diverging_key_keeps_barrier_closed():
  """An out-of-order completion must not open the barrier for the next key.

  Replay diverged from the recording (beta finished before alpha), so the
  barrier stays shut and the waiter fails loudly instead of proceeding in an
  order the recording never contained.
  """
  mgr = ReplayManager()
  ctx = _recorded_two_step_ctx()
  barrier = mgr.prepare_parent_sequence_barrier(ctx, "wf@1")
  barrier.timeout_sec = 0.05

  # beta reports completion first — not what was recorded.
  await mgr.advance_sequence("wf@1", "beta@1")

  assert barrier.current_index == 0
  with pytest.raises(RuntimeError, match="Replay divergence detected"):
    await mgr.wait_sequence("wf@1", "beta@1")


@pytest.mark.asyncio
async def test_wait_sequence_without_barrier_for_path_does_not_block():
  """A parent path with no recorded sequence fast-forwards instead of raising."""
  mgr = ReplayManager()
  ctx = _recorded_two_step_ctx()
  mgr.prepare_parent_sequence_barrier(ctx, "wf@1")

  # "other@1" was never prepared, so nothing constrains it.
  await asyncio.wait_for(mgr.wait_sequence("other@1", "beta@1"), timeout=1)


@pytest.mark.asyncio
async def test_advance_sequence_for_unprepared_path_leaves_other_barriers_alone():
  """Advancing an unprepared parent path is a no-op, not a cross-path advance."""
  mgr = ReplayManager()
  ctx = _recorded_two_step_ctx()
  barrier = mgr.prepare_parent_sequence_barrier(ctx, "wf@1")

  await mgr.advance_sequence("other@1", "alpha@1")

  assert barrier.current_index == 0
  assert not barrier.events["beta@1"].is_set()
