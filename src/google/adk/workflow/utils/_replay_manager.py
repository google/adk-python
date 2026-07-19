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

"""ReplayManager — unified orchestrator for event rehydration, interception, and sequence barriers."""

from __future__ import annotations

import logging

from ...agents.context import Context
from ...events._node_path_builder import _NodePathBuilder
from ...events.event import Event
from ._rehydration_utils import _ChildScanState
from ._rehydration_utils import _reconstruct_node_states
from ._rehydration_utils import is_terminal_event
from ._replay_sequence_barrier import ReplaySequenceBarrier

logger = logging.getLogger("google_adk." + __name__)


class ReplayManager:
  """Unifies rehydration, replay interception, and sequence barrier synchronization across static and dynamic nodes."""

  def __init__(self) -> None:
    self._recovered_executions: dict[str, _ChildScanState] = {}
    self._sequence_barrier: ReplaySequenceBarrier | None = None
    self._parent_sequence_barriers: dict[str, ReplaySequenceBarrier] = {}
    self._events_by_parent: dict[str, list[Event]] = {}
    self._transitive_events_by_parent: dict[str, list[Event]] = {}

  @property
  def recovered_executions(self) -> dict[str, _ChildScanState]:
    """Recovered child states from event scan."""
    return self._recovered_executions

  @property
  def sequence_barrier(self) -> ReplaySequenceBarrier | None:
    """Sequence barrier for deterministic replay ordering."""
    return self._sequence_barrier

  def _ensure_index(self, ctx: Context) -> None:
    """Ensures event indexes are initialized and up-to-date with current session.

    In multi-turn sessions, new events are added to session history on each turn.
    Rebuilding the index whenever event count changes ensures rehydration
    always operates on the complete event stream across turns.
    """
    ic = ctx._invocation_context
    events = ic.session.events
    if getattr(self, "_indexed_event_count", -1) != len(events):
      self._build_event_index(events, ic.invocation_id)
      self._indexed_event_count = len(events)

  def _build_event_index(self, events: list[Event], invocation_id: str) -> None:
    """Builds index of events grouped by parent path (both direct and transitive)."""
    self._events_by_parent = {}
    self._transitive_events_by_parent = {}
    fc_to_parent: dict[str, str] = {}

    from ._workflow_hitl_utils import get_request_input_interrupt_ids

    for event in events:
      if event.author == "user":
        self._index_user_event(event, fc_to_parent)
        continue

      path = event.node_info.path or ""
      if not path:
        continue

      path_builder = _NodePathBuilder.from_string(path)
      parent_path = str(path_builder.parent) if path_builder.parent else ""

      self._add_event_to_index(parent_path, event)

      # Track interrupts to route future user responses
      interrupt_ids = set(event.long_running_tool_ids or [])
      interrupt_ids.update(get_request_input_interrupt_ids(event))
      for fid in interrupt_ids:
        fc_to_parent[fid] = parent_path

  def _index_user_event(
      self, event: Event, fc_to_parent: dict[str, str]
  ) -> None:
    """Routes user response events to parent path based on function call IDs."""
    if not event.content or not event.content.parts:
      return
    matched = False
    added_parents: set[str] = set()
    for part in event.content.parts:
      fr = part.function_response
      if fr and fr.id and fr.id in fc_to_parent:
        parent = fc_to_parent[fr.id]
        if parent not in added_parents:
          self._add_event_to_index(parent, event)
          added_parents.add(parent)
          matched = True

    if not matched:
      # General user prompt event: add to root ("")
      self._events_by_parent.setdefault("", []).append(event)
      self._transitive_events_by_parent.setdefault("", []).append(event)

  def get_events_for_rehydration(
      self, ctx: Context, node_path: str
  ) -> list[Event]:
    """Retrieves pre-filtered session events relevant to rehydrating a node path.

    Instead of performing an O(N) linear scan over all session events, this
    queries pre-indexed transitive events under the node's parent path. Top-level
    user prompts are merged in to ensure multi-turn conversation context is visible
    while preserving strict session chronological ordering.
    """
    if not node_path:
      return []

    self._ensure_index(ctx)
    path_builder = _NodePathBuilder.from_string(node_path)
    parent_builder = path_builder.parent
    if not parent_builder or not str(parent_builder):
      return ctx._invocation_context.session.events
    parent_path = str(parent_builder)

    node_events = self._transitive_events_by_parent.get(parent_path, [])
    if not node_events:
      return ctx._invocation_context.session.events

    # Top-level user text prompts live under root key ("").
    # Merge them so multi-turn turn inputs remain visible during state reconstruction.
    root_events = self._events_by_parent.get("", [])
    user_prompts = [
        e for e in root_events if e.author == "user" and e not in node_events
    ]

    if not user_prompts:
      return node_events

    # Retain exact chronological ordering of session events.
    session_events = ctx._invocation_context.session.events
    event_ids = {id(e) for e in node_events}.union(id(e) for e in user_prompts)
    return [e for e in session_events if id(e) in event_ids]

  def _add_event_to_index(self, parent_path: str, event: Event) -> None:
    """Indexes an event under its direct parent and all ancestor paths up to root."""
    self._events_by_parent.setdefault(parent_path, []).append(event)

    # Propagate event up through all ancestor paths so parent and grandparent nodes
    # can query all sub-tree events in O(1) via _transitive_events_by_parent.
    curr: _NodePathBuilder | None = (
        _NodePathBuilder.from_string(parent_path) if parent_path else None
    )
    while curr is not None and str(curr):
      self._transitive_events_by_parent.setdefault(str(curr), []).append(event)
      curr = curr.parent

    self._transitive_events_by_parent.setdefault("", []).append(event)

  def _scan_sequence(
      self,
      events: list[Event],
      ctx: Context,
      base_path: str,
      strict_direct_child: bool = False,
  ) -> list[str]:
    """Extract chronological child completion sequence under base_path."""
    base_path_builder = _NodePathBuilder.from_string(base_path)
    sequence: list[str] = []

    for event in events:
      event_node_path = event.node_info.path or ""
      event_path_builder = _NodePathBuilder.from_string(event_node_path)

      if not event_path_builder.is_descendant_of(base_path_builder):
        continue

      child_path = base_path_builder.get_direct_child(event_path_builder)
      if strict_direct_child and event_path_builder != child_path:
        continue

      segment: str = child_path.leaf_segment

      if is_terminal_event(event):
        if segment in sequence:
          sequence.remove(segment)
        sequence.append(segment)

    return sequence

  def scan_workflow_events(
      self, ctx: Context
  ) -> tuple[dict[str, _ChildScanState], list[str]]:
    """Scan session events for direct child workflow nodes and initialize sequence barrier."""
    ic = ctx._invocation_context

    # Build the index
    self._build_event_index(ic.session.events, ic.invocation_id)

    # Use transitive parent events for static child nodes so deeper descendant events (e.g. delegated outputs/interrupts) are recovered
    filtered_events = self._transitive_events_by_parent.get(ctx.node_path, [])
    raw_results = _reconstruct_node_states(
        events=filtered_events,
        base_path=ctx.node_path,
        group_by_direct_child=True,
        invocation_id=ic.invocation_id,
    )

    # Use transitive events for sequence (strict_direct_child = False)
    transitive_events = self._transitive_events_by_parent.get(ctx.node_path, [])
    sequence = self._scan_sequence(
        transitive_events, ctx, ctx.node_path, strict_direct_child=False
    )

    self._recovered_executions = raw_results
    self._sequence_barrier = ReplaySequenceBarrier(sequence)
    return raw_results, sequence

  def prepare_parent_sequence_barrier(
      self, ctx: Context, parent_path: str
  ) -> ReplaySequenceBarrier:
    """Ensure a sequence barrier is set up for dynamic nodes under parent_path."""
    if parent_path not in self._parent_sequence_barriers:
      self._ensure_index(ctx)

      # Dynamic parent path uses strict_direct_child=True.
      # So we use the direct parent index.
      events = self._events_by_parent.get(parent_path, [])
      seq = self._scan_sequence(
          events, ctx, parent_path, strict_direct_child=True
      )
      self._parent_sequence_barriers[parent_path] = ReplaySequenceBarrier(seq)
    return self._parent_sequence_barriers[parent_path]

  async def advance_sequence(self, parent_path: str, key: str) -> None:
    """Advance sequence barrier if initialized for parent_path."""
    if parent_path in self._parent_sequence_barriers:
      self._parent_sequence_barriers[parent_path].check_and_advance(key)

  async def wait_sequence(self, parent_path: str, key: str) -> None:
    """Wait for sequence barrier if initialized for parent_path."""
    if parent_path in self._parent_sequence_barriers:
      await self._parent_sequence_barriers[parent_path].wait(key)
