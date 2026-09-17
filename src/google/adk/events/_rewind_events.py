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

"""Helpers for resolving which events are live after rewinds."""

from __future__ import annotations

from .event import Event


def _apply_rewinds(events: list[Event]) -> list[Event]:
  """Returns ``events`` with rewound invocations removed.

  Iterates backward. When an event carries
  ``actions.rewind_before_invocation_id == X``, drops that event together with
  every event between it and the earliest event of invocation ``X`` (inclusive),
  then resumes the backward walk from there.

  This is the single source of truth for "which events are live" after rewinds.
  Both LLM prompt building (``google.adk.flows.llm_flows.contents``) and context
  compaction (``google.adk.apps.compaction``) must agree on it, otherwise
  rewound content can leak back into prompts through a compaction summary.

  Args:
    events: The full event history, in chronological order.

  Returns:
    The chronological subset of ``events`` that survives all rewinds.
  """
  kept: list[Event] = []
  i = len(events) - 1
  while i >= 0:
    event = events[i]
    if event.actions and event.actions.rewind_before_invocation_id:
      rewind_invocation_id = event.actions.rewind_before_invocation_id
      for j in range(0, i, 1):
        if events[j].invocation_id == rewind_invocation_id:
          i = j
          break
    else:
      kept.append(event)
    i -= 1
  kept.reverse()
  return kept
