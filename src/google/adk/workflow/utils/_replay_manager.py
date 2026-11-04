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
from ...events._branch_path import _BranchPath
from ...events._interrupts import is_interrupt_event
from ...events._node_path_builder import _NodePathBuilder
from ...events._node_path_builder import parent_node_path
from ...events._rewind_events import _apply_rewinds
from ...events.event import Event
from ._rehydration_utils import _ChildScanState
from ._rehydration_utils import _event_has_result
from ._rehydration_utils import _reconstruct_node_states
from ._rehydration_utils import is_terminal_event
from ._replay_sequence_barrier import ReplaySequenceBarrier

logger = logging.getLogger("google_adk." + __name__)


class ReplayManager:
  """Unifies rehydration, replay interception, and sequence barrier synchronization across static and dynamic nodes."""

  def __init__(self) -> None:
    self._sequence_barrier: ReplaySequenceBarrier | None = None
    self._parent_sequence_barriers: dict[str, ReplaySequenceBarrier] = {}
    self._events_by_parent: dict[str, list[Event]] = {}
    self._transitive_events_by_parent: dict[str, list[Event]] = {}
    self._fc_to_parent: dict[str, str] = {}
    self._live_events: list[Event] = []
    self._indexed_invocation_id: str | None = None
    self._indexed_event_count: int = 0
    self._indexed_last_event: Event | None = None

  @property
  def sequence_barrier(self) -> ReplaySequenceBarrier | None:
    """Sequence barrier for deterministic replay ordering."""
    return self._sequence_barrier

  def _ensure_index(self, ctx: Context) -> list[Event]:
    """Ensures event indexes are initialized and up-to-date with current session.

    Returns the session events the index now describes. A caller that needs
    those events should use the returned list rather than reading them off
    `ctx` a second time, so the events it works with are the same ones the
    index covers.

    Events are appended to the session as the run proceeds and across turns, so
    the index is extended with whatever arrived since it was last updated.
    Rebuilding it from scratch on every append costs O(N) each time, which is
    quadratic over a run.

    The index is rebuilt only when the events it describes are no longer a
    prefix of the session -- after a compaction or a rewind -- because the
    existing buckets then hold events the session no longer has.

    Only `ctx`'s invocation is indexed. A ReplayManager belongs to one workflow
    run, so every query it answers is for that run's invocation; indexing the
    rest of the session would cost time and memory nothing reads back. An empty
    invocation id indexes everything.
    """
    ic = ctx._invocation_context
    raw_events = ic.session.events
    if (
        self._indexed_event_count > 0
        and self._indexed_invocation_id == ic.invocation_id
        and self._indexed_prefix_is_intact(raw_events)
    ):
      delta = raw_events[self._indexed_event_count :]
      has_new_rewind = any(
          ev.actions and ev.actions.rewind_before_invocation_id for ev in delta
      )
      if not has_new_rewind:
        if delta:
          self._live_events.extend(e for e in delta if self._in_scope(e))
          self._index_events(delta)
          self._record_indexed_through(raw_events)
        return self._live_events

    self._indexed_invocation_id = ic.invocation_id
    self._build_event_index(_apply_rewinds(raw_events))
    self._record_indexed_through(raw_events)
    return self._live_events

  def _in_scope(self, event: Event) -> bool:
    """Whether `event` belongs to the invocation the index describes."""
    return (
        not self._indexed_invocation_id
        or event.invocation_id == self._indexed_invocation_id
    )

  def _indexed_prefix_is_intact(self, events: list[Event]) -> bool:
    """Whether the already-indexed events are still a prefix of `events`.

    Comparing the count alone misses a compaction or a rewind that replaces
    events without changing how many there are, which would leave the index
    describing events that are gone. The last indexed event is held by
    reference and compared by identity, so the check neither reads an event's
    fields nor can be fooled by a freed object's address being reused.
    """
    if self._indexed_event_count > len(events):
      return False
    if self._indexed_event_count == 0:
      return True
    return events[self._indexed_event_count - 1] is self._indexed_last_event

  def _record_indexed_through(self, events: list[Event]) -> None:
    """Remembers how far `events` has been indexed."""
    self._indexed_event_count = len(events)
    self._indexed_last_event = events[-1] if events else None

  def _build_event_index(self, events: list[Event]) -> None:
    """Builds index of events grouped by parent path (both direct and transitive).

    Only the indexed invocation's events are grouped, so the queries that read
    these buckets need no filtering of their own.
    """
    self._events_by_parent = {}
    self._transitive_events_by_parent = {}
    self._fc_to_parent = {}
    self._live_events = [e for e in events if self._in_scope(e)]
    self._index_events(events)
    self._record_indexed_through(events)

  def _index_events(self, events: list[Event]) -> None:
    """Adds `events` to the index, continuing from what is already there.

    Interrupt ownership (`_fc_to_parent`) carries across calls: a user response
    indexed now may answer a function call indexed in an earlier batch. It is
    tracked for every event rather than only the indexed invocation's, so a
    reply still routes to an interrupt raised by an earlier invocation.
    """
    from ...events._interrupts import extract_event_interrupt_ids

    for event in events:
      if event.author == "user":
        if self._in_scope(event):
          self._index_user_event(event, self._fc_to_parent)
        continue

      path = event.node_info.path or ""
      if not path:
        continue

      parent_path = parent_node_path(path)

      if self._in_scope(event):
        self._add_event_to_index(parent_path, event)

      # Track interrupts to route future user responses
      for fid in extract_event_interrupt_ids(event):
        self._fc_to_parent[fid] = parent_path

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

    if event.branch:
      # Match the branch's run ids exactly. A substring test on the raw branch
      # string indexes the event under a parent whose function-call id merely
      # happens to be contained in another id.
      branch_run_ids = _BranchPath.from_string(event.branch).run_ids
      for fid, parent in fc_to_parent.items():
        if fid in branch_run_ids and parent not in added_parents:
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
    while preserving strict session chronological ordering. The index holds only
    the current invocation's events, so the result is limited to them.
    """
    if not node_path:
      return []

    session_events = self._ensure_index(ctx)
    parent_path = parent_node_path(node_path)
    if not parent_path:
      return session_events

    node_events = self._transitive_events_by_parent.get(parent_path, [])
    if not node_events:
      return session_events

    # Top-level user text prompts live under root key ("").
    # Merge them so multi-turn turn inputs remain visible during state reconstruction.
    root_events = self._events_by_parent.get("", [])
    # Membership is by identity, matching the `id()` set used below: both lists
    # are views onto `session.events`, so an event is already indexed under this
    # node only if it is the same object. `e not in node_events` would instead
    # invoke `Event.__eq__` per probe, deep-comparing whole events and making
    # this loop quadratic in the session size.
    node_event_ids = {id(e) for e in node_events}
    user_prompts = [
        e
        for e in root_events
        if e.author == "user" and id(e) not in node_event_ids
    ]

    if not user_prompts:
      return node_events

    # Retain exact chronological ordering of session events.
    event_ids = node_event_ids.union(id(e) for e in user_prompts)
    return [e for e in session_events if id(e) in event_ids]

  def indexed_invocation_events(self, ctx: Context) -> list[Event] | None:
    """Returns ctx's invocation's live events, or None before any indexing."""
    if not self._indexed_event_count:
      return None
    return self._ensure_index(ctx)

  def _add_event_to_index(self, parent_path: str, event: Event) -> None:
    """Indexes an event under its direct parent and all ancestor paths up to root."""
    self._events_by_parent.setdefault(parent_path, []).append(event)

    # Propagate event up through all ancestor paths so parent and grandparent nodes
    # can query all sub-tree events in O(1) via _transitive_events_by_parent.
    curr = parent_path
    while curr:
      self._transitive_events_by_parent.setdefault(curr, []).append(event)
      curr = parent_node_path(curr)

    self._transitive_events_by_parent.setdefault("", []).append(event)

  def _scan_sequence(
      self,
      events: list[Event],
      ctx: Context,
      base_path: str,
      strict_direct_child: bool = False,
  ) -> tuple[list[str], set[str]]:
    """Extract chronological child completion sequence under base_path.

    Returns the sequence and the keys in it placed by a recorded result rather
    than an interrupt.
    """
    base_path_builder = _NodePathBuilder.from_string(base_path)
    sequence: list[str] = []
    completed: set[str] = set()
    invocation_id = ctx._invocation_context.invocation_id

    for event in events:
      # An earlier invocation's terminal event would block the barrier on a
      # node that never runs during this resume.
      if invocation_id and event.invocation_id != invocation_id:
        continue

      event_node_path = event.node_info.path or ""
      event_path_builder = _NodePathBuilder.from_string(event_node_path)

      if not event_path_builder.is_descendant_of(base_path_builder):
        continue

      child_path = base_path_builder.get_direct_child(event_path_builder)
      if strict_direct_child and event_path_builder != child_path:
        continue

      # Only direct child events, events with delegated output for the child,
      # or descendant interrupt events represent terminal outcomes for the child itself.
      is_child_event = (
          event_path_builder == child_path
          or bool(
              event.node_info
              and event.node_info.output_for
              and str(child_path) in event.node_info.output_for
          )
          or is_interrupt_event(event)
      )
      if not is_child_event:
        continue

      segment: str = child_path.leaf_segment

      if is_terminal_event(event):
        has_result = _event_has_result(event)
        # An error with no result means the child did not complete: on resume
        # it runs again rather than being replayed, so it holds no place in
        # the replay order. A later attempt that succeeds places it there.
        if (
            event.error_code is not None
            and not has_result
            and not is_interrupt_event(event)
        ):
          if segment in sequence and segment not in completed:
            sequence.remove(segment)
          continue
        # Ignore re-emitted echoes for already completed children.
        if segment in completed:
          continue
        if segment in sequence:
          sequence.remove(segment)
        sequence.append(segment)
        # Only non-interrupt terminal outcomes mark the child completed.
        if not is_interrupt_event(event) and has_result:
          completed.add(segment)

    return sequence, completed

  def scan_workflow_events(
      self, ctx: Context
  ) -> tuple[dict[str, _ChildScanState], list[str]]:
    """Scan session events for direct child workflow nodes and initialize sequence barrier."""
    ic = ctx._invocation_context

    self._ensure_index(ctx)

    # Use transitive parent events for static child nodes so deeper descendant events (e.g. delegated outputs/interrupts) are recovered
    transitive_events = self._transitive_events_by_parent.get(ctx.node_path, [])
    raw_results = _reconstruct_node_states(
        events=transitive_events,
        base_path=ctx.node_path,
        group_by_direct_child=True,
        invocation_id=ic.invocation_id,
    )

    # Use transitive events for sequence (strict_direct_child = False)
    sequence, completed = self._scan_sequence(
        transitive_events, ctx, ctx.node_path, strict_direct_child=False
    )

    self._sequence_barrier = ReplaySequenceBarrier(
        sequence, pending=frozenset(sequence) - completed
    )
    self._parent_sequence_barriers[ctx.node_path] = self._sequence_barrier
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
      seq, completed = self._scan_sequence(
          events, ctx, parent_path, strict_direct_child=True
      )
      self._parent_sequence_barriers[parent_path] = ReplaySequenceBarrier(
          seq, pending=frozenset(seq) - completed
      )
    return self._parent_sequence_barriers[parent_path]

  async def advance_sequence(self, parent_path: str, key: str) -> None:
    """Advance sequence barrier if initialized for parent_path."""
    if parent_path in self._parent_sequence_barriers:
      self._parent_sequence_barriers[parent_path].check_and_advance(key)

  async def release_pending_sequence(self, parent_path: str, key: str) -> None:
    """Advance past `key` if its place came from an interrupt, not a result.

    A rerun produces its result live, so a place its interrupt took must not
    hold back replayed siblings recorded after that interrupt.
    """
    barrier = self._parent_sequence_barriers.get(parent_path)
    if barrier is not None and key in barrier.pending:
      barrier.check_and_advance(key)

  async def wait_sequence(self, parent_path: str, key: str) -> None:
    """Wait for sequence barrier if initialized for parent_path."""
    if parent_path in self._parent_sequence_barriers:
      await self._parent_sequence_barriers[parent_path].wait(key)
