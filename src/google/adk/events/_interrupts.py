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

"""Unified interrupt index derived from session events.

All pause mechanisms (LRO tools, tool confirmations, credential requests, and
workflow `RequestInput`) share the same wire representation: a `FunctionCall`
whose interrupt ID is recorded in `Event.long_running_tool_ids` (with a legacy
fallback for `adk_request_input` / `adk_request_credential` calls exported
without `long_running_tool_ids`).

This module builds a pure, session-event-derived index mapping each interrupt ID
to its `OpenInterrupt` metadata so the router and workflow rehydration/replay
consumers share one definition of interrupt ownership.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from ..utils._function_call_names import REQUEST_EUC_FUNCTION_CALL_NAME
from ..utils._function_call_names import REQUEST_INPUT_FUNCTION_CALL_NAME
from ._branch_path import _BranchPath
from .event import Event


@dataclasses.dataclass(frozen=True)
class OpenInterrupt:
  """Metadata for an open interrupt emitted in an event stream."""

  author: str | None
  ancestor_authors: tuple[str, ...] = ()

  @property
  def candidate_authors(self) -> tuple[str, ...]:
    """Authors from innermost node to outermost tool-branch caller."""
    authors: list[str] = []
    if self.author:
      authors.append(self.author)
    for ancestor_author in self.ancestor_authors:
      if ancestor_author and ancestor_author not in authors:
        authors.append(ancestor_author)
    return tuple(authors)


def extract_event_interrupt_ids(event: Event) -> set[str]:
  """Returns all interrupt IDs carried by `event`.

  Includes `event.long_running_tool_ids` plus the fallback extraction from
  `adk_request_input` and `adk_request_credential` function calls for older
  serialized sessions where `long_running_tool_ids` was omitted.
  """
  interrupt_ids = set(event.long_running_tool_ids or [])
  for fc in event.get_function_calls():
    if fc.id is not None and fc.name in (
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        REQUEST_EUC_FUNCTION_CALL_NAME,
    ):
      interrupt_ids.add(fc.id)
  return interrupt_ids


def index_open_interrupts(events: Sequence[Event]) -> dict[str, OpenInterrupt]:
  """Indexes interrupts in `events` that have not yet been answered.

  An interrupt is considered answered when a subsequent user `FunctionResponse`
  in `events` carries `fr.id == interrupt_id`, or arrives on a sub-branch whose
  run IDs contain `interrupt_id`.
  """
  index: dict[str, OpenInterrupt] = {}
  call_author_by_id: dict[str, str] = {}

  for event in events:
    branch_run_ids = (
        _BranchPath.from_string(event.branch).ordered_run_ids
        if event.branch
        else ()
    )
    if event.author and event.author != 'user':
      for fc in event.get_function_calls():
        if fc.id:
          call_author_by_id[fc.id] = event.author

    if event.author == 'user':
      for fr in event.get_function_responses():
        if fr.id and fr.id in index:
          del index[fr.id]

      if branch_run_ids and event.get_function_responses() and index:
        for int_id in list(index):
          if int_id in branch_run_ids:
            del index[int_id]

    interrupt_ids = extract_event_interrupt_ids(event)
    if not interrupt_ids:
      continue

    ancestor_authors: list[str] = []
    for run_id in reversed(branch_run_ids):
      anc_author = call_author_by_id.get(run_id)
      if anc_author and anc_author not in ancestor_authors:
        ancestor_authors.append(anc_author)

    for interrupt_id in interrupt_ids:
      index[interrupt_id] = OpenInterrupt(
          author=event.author,
          ancestor_authors=tuple(ancestor_authors),
      )

  return index
