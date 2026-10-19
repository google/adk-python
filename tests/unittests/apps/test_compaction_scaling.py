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

"""Compacted histories must not require all-pairs interval comparisons."""

import math

from google.adk.apps.compaction import _latest_compaction_event
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context import _compaction
from google.genai import types
import pytest


class _CountedTimestamp(float):
  comparisons = 0

  def __lt__(self, other):
    type(self).comparisons += 1
    return super().__lt__(other)

  def __le__(self, other):
    type(self).comparisons += 1
    return super().__le__(other)

  def __gt__(self, other):
    type(self).comparisons += 1
    return super().__gt__(other)

  def __ge__(self, other):
    type(self).comparisons += 1
    return super().__ge__(other)


@pytest.mark.parametrize('operation', ['contents', 'latest_summary'])
def test_long_compacted_history_has_bounded_interval_comparisons(operation):
  """Bound work without depending on machine speed or wall-clock timing."""
  count = 256
  events = []
  for i in range(count):
    start = float(i * 10)
    for offset in range(8):
      events.append(Event(timestamp=start + offset, author='user'))
    compaction = EventCompaction(
        start_timestamp=start,
        end_timestamp=start + 7,
        compacted_content=types.Content(
            role='model', parts=[types.Part(text=f'summary {i}')]
        ),
    )
    # Assign float subclasses after validation so the comparisons in the real
    # production path can be counted without replacing any ADK helper.
    compaction.start_timestamp = _CountedTimestamp(start)
    compaction.end_timestamp = _CountedTimestamp(start + 7)
    events.append(
        Event(
            timestamp=start + 8,
            author='user',
            invocation_id=f'summary-{i}',
            actions=EventActions(compaction=compaction),
        )
    )

  _CountedTimestamp.comparisons = 0
  if operation == 'contents':
    result = _compaction._process_compaction_events(events)
    assert len(result) == count
    workload = len(events)
  else:
    assert _latest_compaction_event(events) is events[-1]
    workload = count
  # Generous O(N log C) comparison budget, far below an all-pairs scan.
  assert _CountedTimestamp.comparisons < 8 * workload * math.ceil(
      math.log2(count)
  )
