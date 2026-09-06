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

"""Unit tests for Session.get_events() and Session.filter_events() methods."""

from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
import pytest


def _make_event(
    invocation_id: str,
    author: str = "user",
    rewind_before_invocation_id: str | None = None,
) -> Event:
  """Helper to create a mock Event with specified properties."""
  actions = EventActions()
  if rewind_before_invocation_id:
    actions.rewind_before_invocation_id = rewind_before_invocation_id

  return Event(
      invocation_id=invocation_id,
      author=author,
      actions=actions,
  )


def _make_session(events: list[Event]) -> Session:
  """Helper to create a Session with specified events."""
  return Session(
      id="test-session-id",
      app_name="test-app",
      user_id="test-user",
      events=events,
  )


class TestGetEvents:
  """Tests for Session.get_events() method."""

  def test_get_events_returns_all_events(self):
    """Test that get_events returns all events in the session."""
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-1", "agent")
    event_c = _make_event("inv-2", "user")

    session = _make_session([event_a, event_b, event_c])

    result = session.get_events()

    assert len(result) == 3
    assert result[0] is event_a
    assert result[1] is event_b
    assert result[2] is event_c

  def test_get_events_returns_empty_list_when_no_events(self):
    """Test that get_events returns empty list for session with no events."""
    session = _make_session([])

    result = session.get_events()

    assert result == []

  def test_get_events_returns_same_reference_as_events_property(self):
    """Test that get_events returns the same list as session.events."""
    event_a = _make_event("inv-1")
    session = _make_session([event_a])

    result = session.get_events()

    assert result is session.events


class TestFilterEventsNoRewind:
  """Tests for Session.filter_events() when no rewind events exist."""

  def test_filter_events_returns_all_when_no_rewinds(self):
    """Test that all events are returned when there are no rewind events."""
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-1", "agent")
    event_c = _make_event("inv-2", "user")

    session = _make_session([event_a, event_b, event_c])

    result = session.filter_events()

    assert len(result) == 3
    assert result[0].invocation_id == "inv-1"
    assert result[0].author == "user"
    assert result[1].invocation_id == "inv-1"
    assert result[1].author == "agent"
    assert result[2].invocation_id == "inv-2"

  def test_filter_events_returns_empty_list_when_no_events(self):
    """Test that filter_events returns empty list for empty session."""
    session = _make_session([])

    result = session.filter_events()

    assert result == []

  def test_filter_events_exclude_rewound_false_returns_all(self):
    """Test that exclude_rewound=False returns all events."""
    event_a = _make_event("inv-1")
    event_b = _make_event("inv-2")

    session = _make_session([event_a, event_b])

    result = session.filter_events(exclude_rewound=False)

    assert len(result) == 2
    assert result is session.events  # Should return same reference


class TestFilterEventsWithRewind:
  """Tests for Session.filter_events() with rewind events."""

  def test_filter_events_single_rewind(self):
    """Test that events from rewound invocation are filtered out."""
    # inv-1: user message + agent response (should remain)
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-1", "agent")

    # inv-2: user message + agent response (should be filtered)
    event_c = _make_event("inv-2", "user")
    event_d = _make_event("inv-2", "agent")

    # Rewind event targeting inv-2
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-2"
    )

    session = _make_session([event_a, event_b, event_c, event_d, rewind_event])

    result = session.filter_events()

    # Should only contain events from inv-1
    assert len(result) == 2
    assert result[0].invocation_id == "inv-1"
    assert result[0].author == "user"
    assert result[1].invocation_id == "inv-1"
    assert result[1].author == "agent"

  def test_filter_events_multiple_sequential_rewinds(self):
    """Test multiple sequential rewinds filter correctly."""
    # inv-1: should remain
    event_a = _make_event("inv-1", "user")

    # inv-2: should be filtered by first rewind
    event_b = _make_event("inv-2", "user")

    # First rewind: removes inv-2
    rewind_1 = _make_event(
        "inv-rewind-1", "user", rewind_before_invocation_id="inv-2"
    )

    # inv-3: should remain
    event_c = _make_event("inv-3", "user")

    # inv-4: should be filtered by second rewind
    event_d = _make_event("inv-4", "user")

    # Second rewind: removes inv-4
    rewind_2 = _make_event(
        "inv-rewind-2", "user", rewind_before_invocation_id="inv-4"
    )

    session = _make_session(
        [event_a, event_b, rewind_1, event_c, event_d, rewind_2]
    )

    result = session.filter_events()

    # Should only contain inv-1 and inv-3
    assert len(result) == 2
    assert result[0].invocation_id == "inv-1"
    assert result[1].invocation_id == "inv-3"

  def test_filter_events_rewind_to_first_invocation(self):
    """Test rewind that goes back to the very first invocation."""
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-2", "user")
    event_c = _make_event("inv-3", "user")

    # Rewind all the way back to inv-1
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-1"
    )

    session = _make_session([event_a, event_b, event_c, rewind_event])

    result = session.filter_events()

    # All events should be filtered out
    assert len(result) == 0

  def test_filter_events_with_exclude_rewound_false_includes_all(self):
    """Test that exclude_rewound=False includes rewound events."""
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-2", "user")
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-2"
    )

    session = _make_session([event_a, event_b, rewind_event])

    result = session.filter_events(exclude_rewound=False)

    # Should include all events including rewound ones
    assert len(result) == 3


class TestFilterEventsEdgeCases:
  """Edge case tests for Session.filter_events()."""

  def test_filter_events_rewind_target_not_found(self):
    """Test behavior when rewind target invocation doesn't exist."""
    event_a = _make_event("inv-1", "user")

    # Rewind targeting non-existent invocation
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-nonexistent"
    )

    session = _make_session([event_a, rewind_event])

    result = session.filter_events()

    # Should still filter out the rewind event itself, keep inv-1
    assert len(result) == 1
    assert result[0].invocation_id == "inv-1"

  def test_filter_events_preserves_chronological_order(self):
    """Test that filtered events maintain chronological order."""
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-2", "user")
    event_c = _make_event("inv-3", "user")
    event_d = _make_event("inv-4", "user")

    # Rewind inv-3 only
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-3"
    )

    event_e = _make_event("inv-5", "user")

    session = _make_session(
        [event_a, event_b, event_c, event_d, rewind_event, event_e]
    )

    result = session.filter_events()

    # Should be inv-1, inv-2, inv-5 in order
    assert len(result) == 3
    assert result[0].invocation_id == "inv-1"
    assert result[1].invocation_id == "inv-2"
    assert result[2].invocation_id == "inv-5"

  def test_filter_events_single_event_no_rewind(self):
    """Test with single event and no rewind."""
    event_a = _make_event("inv-1", "user")
    session = _make_session([event_a])

    result = session.filter_events()

    assert len(result) == 1
    assert result[0].invocation_id == "inv-1"

  def test_filter_events_multiple_events_same_invocation(self):
    """Test filtering with multiple events in the same invocation."""
    # Multiple events in inv-1
    event_a = _make_event("inv-1", "user")
    event_b = _make_event("inv-1", "agent")
    event_c = _make_event("inv-1", "agent")  # Multiple agent responses

    # inv-2 events
    event_d = _make_event("inv-2", "user")
    event_e = _make_event("inv-2", "agent")

    # Rewind to inv-2 - should remove all inv-2 events
    rewind_event = _make_event(
        "inv-rewind", "user", rewind_before_invocation_id="inv-2"
    )

    session = _make_session(
        [event_a, event_b, event_c, event_d, event_e, rewind_event]
    )

    result = session.filter_events()

    # Should only have the 3 events from inv-1
    assert len(result) == 3
    assert all(e.invocation_id == "inv-1" for e in result)
