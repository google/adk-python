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

"""Check interval-query behavior against the direct compaction definition."""

import json
import random

from google.adk.apps.compaction import _latest_compaction_event
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context import _compaction
from google.genai import types
import pytest


def _summary(start, end, index):
  return Event(
      author='summarizer',
      timestamp=100.0 + index,
      invocation_id=f'summary-{index}',
      branch='root',
      actions=EventActions(
          compaction=EventCompaction(
              start_timestamp=start,
              end_timestamp=end,
              compacted_content=types.Content(
                  role='model', parts=[types.Part(text=f'summary {index}')]
              ),
          )
      ),
  )


def _reference(events):
  """Use the direct containment and inclusive coverage rules as an oracle."""
  ranges = [
      (
          i,
          e.actions.compaction.start_timestamp,
          e.actions.compaction.end_timestamp,
      )
      for i, e in enumerate(events)
      if e.actions.compaction
      and e.actions.compaction.start_timestamp is not None
      and e.actions.compaction.end_timestamp is not None
      and e.actions.compaction.compacted_content is not None
  ]
  kept = [
      (i, start, end)
      for i, start, end in ranges
      if not any(
          j != i
          and other_start <= start
          and other_end >= end
          and (other_start < start or other_end > end or j > i)
          for j, other_start, other_end in ranges
      )
  ]
  output = []
  for i, _, end in kept:
    source = events[i]
    output.append((
        end,
        i,
        Event(
            timestamp=end,
            author='agent',
            branch=source.branch,
            invocation_id=source.invocation_id,
            content=source.actions.compaction.compacted_content,
            actions=source.actions,
        ),
    ))
  for i, event in enumerate(events):
    if event.actions.compaction:
      continue
    if not any(start <= event.timestamp <= end for _, start, end in kept):
      output.append((event.timestamp, i, event))
  output.sort(key=lambda item: (item[0], item[1]))
  latest = events[max(i for i, _, _ in kept)] if kept else None
  return [event for _, _, event in output], latest


def _signature(events):
  # Materialized summaries receive fresh IDs. Every other field must match;
  # JSON also gives NaN a stable representation for these compatibility cases.
  return json.dumps(
      [event.model_dump(mode='json', exclude={'id'}) for event in events],
      sort_keys=True,
  )


@pytest.mark.parametrize(
    'ranges',
    [
        [],
        [(1, 2), (4, 5)],
        [(1, 4), (3, 6)],
        [(1, 4), (4, 6)],
        [(1, 6), (2, 3), (2, 5)],
        [(1, 6), (1, 6), (1, 6)],
        [(1, 2), (1, 5), (3, 5)],
        [(2, 2), (4, 4)],
        [(5, 1), (6, 2)],
        [(5, 1), (0, 2)],
        [(float('-inf'), 2), (1, float('inf'))],
        [(float('-inf'), float('-inf'))],
        [(float('nan'), 2), (1, 4)],
        [(1, float('nan')), (1, 4)],
    ],
)
def test_compaction_preserves_range_and_boundary_semantics(ranges):
  """Keep summaries, coverage, attribution and event order unchanged."""
  events = [
      Event(timestamp=float(i), author='user', invocation_id=f'raw-{i}')
      for i in range(7)
  ] + [_summary(start, end, i) for i, (start, end) in enumerate(ranges)]
  before = _signature(events)
  expected, latest = _reference(events)
  actual = _compaction._process_compaction_events(events, 'agent')
  assert _signature(actual) == _signature(expected)
  assert _latest_compaction_event(events) is latest
  assert _signature(events) == before


def test_compaction_preserves_shuffled_history_views():
  """Compare small randomized histories, including duplicate ranges and ties."""
  randomizer = random.Random(59173)
  for case in range(300):
    events = [
        Event(
            author='user',
            timestamp=float(randomizer.randrange(-5, 20)),
            invocation_id=f'raw-{i}',
        )
        for i in range(15)
    ]
    for i in range(20):
      first, last = sorted(randomizer.choices(range(-5, 20), k=2))
      events.append(_summary(float(first), float(last), i))
    randomizer.shuffle(events)
    expected, latest = _reference(events)
    assert _signature(
        _compaction._process_compaction_events(events, 'agent')
    ) == _signature(expected), case
    assert _latest_compaction_event(events) is latest, case
