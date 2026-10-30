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

from __future__ import annotations

from google.adk.events import get_active_events
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions


def _make_event(
    invocation_id: str,
    author: str = "user",
    rewind_before_invocation_id: str | None = None,
) -> Event:
  actions = EventActions()
  if rewind_before_invocation_id:
    actions.rewind_before_invocation_id = rewind_before_invocation_id

  return Event(
      invocation_id=invocation_id,
      author=author,
      actions=actions,
  )


def test_get_active_events_empty():
  assert get_active_events([]) == []


def test_get_active_events_returns_all_when_no_rewinds():
  event_a = _make_event("inv-1", "user")
  event_b = _make_event("inv-1", "agent")
  event_c = _make_event("inv-2", "user")

  result = get_active_events([event_a, event_b, event_c])

  assert len(result) == 3
  assert result[0] is event_a
  assert result[1] is event_b
  assert result[2] is event_c


def test_get_active_events_single_rewind():
  event_a = _make_event("inv-1", "user")
  event_b = _make_event("inv-1", "agent")
  event_c = _make_event("inv-2", "user")
  event_d = _make_event("inv-2", "agent")
  rewind_event = _make_event(
      "inv-rewind", "user", rewind_before_invocation_id="inv-2"
  )

  events = [event_a, event_b, event_c, event_d, rewind_event]
  result = get_active_events(events)

  assert len(result) == 2
  assert result[0] is event_a
  assert result[1] is event_b


def test_get_active_events_multiple_sequential_rewinds():
  event_a = _make_event("inv-1", "user")
  event_b = _make_event("inv-2", "user")
  rewind_1 = _make_event(
      "inv-rewind-1", "user", rewind_before_invocation_id="inv-2"
  )
  event_c = _make_event("inv-3", "user")
  event_d = _make_event("inv-4", "user")
  rewind_2 = _make_event(
      "inv-rewind-2", "user", rewind_before_invocation_id="inv-4"
  )

  events = [event_a, event_b, rewind_1, event_c, event_d, rewind_2]
  result = get_active_events(events)

  assert len(result) == 2
  assert result[0] is event_a
  assert result[1] is event_c


def test_get_active_events_rewind_to_first_invocation():
  event_a = _make_event("inv-1", "user")
  event_b = _make_event("inv-2", "user")
  event_c = _make_event("inv-3", "user")
  rewind_event = _make_event(
      "inv-rewind", "user", rewind_before_invocation_id="inv-1"
  )

  events = [event_a, event_b, event_c, rewind_event]
  assert get_active_events(events) == []


def test_get_active_events_rewind_target_not_found():
  event_a = _make_event("inv-1", "user")
  rewind_event = _make_event(
      "inv-rewind", "user", rewind_before_invocation_id="inv-nonexistent"
  )

  events = [event_a, rewind_event]
  result = get_active_events(events)

  assert len(result) == 1
  assert result[0] is event_a


def test_get_active_events_preserves_chronological_order():
  event_a = _make_event("inv-1", "user")
  event_b = _make_event("inv-2", "user")
  event_c = _make_event("inv-3", "user")
  event_d = _make_event("inv-4", "user")
  rewind_event = _make_event(
      "inv-rewind", "user", rewind_before_invocation_id="inv-3"
  )
  event_e = _make_event("inv-5", "user")

  events = [event_a, event_b, event_c, event_d, rewind_event, event_e]
  result = get_active_events(events)

  assert [e.invocation_id for e in result] == ["inv-1", "inv-2", "inv-5"]
