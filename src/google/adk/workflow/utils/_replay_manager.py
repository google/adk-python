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

  @property
  def recovered_executions(self) -> dict[str, _ChildScanState]:
    """Recovered child states from event scan."""
    return self._recovered_executions

  @property
  def sequence_barrier(self) -> ReplaySequenceBarrier | None:
    """Sequence barrier for deterministic replay ordering."""
    return self._sequence_barrier

  def _scan_sequence(
      self, ctx: Context, base_path: str, strict_direct_child: bool = False
  ) -> list[str]:
    """Extract chronological child completion sequence under base_path."""
    ic = ctx._invocation_context
    base_path_builder = _NodePathBuilder.from_string(base_path)
    sequence: list[str] = []

    for event in ic.session.events:
      if event.invocation_id != ic.invocation_id:
        continue

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
    raw_results = _reconstruct_node_states(
        events=ic.session.events,
        base_path=ctx.node_path,
        group_by_direct_child=True,
        invocation_id=ic.invocation_id,
    )

    sequence = self._scan_sequence(
        ctx, ctx.node_path, strict_direct_child=False
    )

    self._recovered_executions = raw_results
    self._sequence_barrier = ReplaySequenceBarrier(sequence)
    return raw_results, sequence

  def prepare_parent_sequence_barrier(
      self, ctx: Context, parent_path: str
  ) -> ReplaySequenceBarrier:
    """Ensure a sequence barrier is set up for dynamic nodes under parent_path."""
    if parent_path not in self._parent_sequence_barriers:
      seq = self._scan_sequence(ctx, parent_path, strict_direct_child=True)
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
