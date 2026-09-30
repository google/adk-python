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

from datetime import datetime
from datetime import timezone

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.sessions import _session_util
from google.adk.sessions.schemas.shared import DEFAULT_MAX_VARCHAR_LENGTH
from google.adk.sessions.schemas.v0 import _truncate_str
from google.adk.sessions.schemas.v0 import StorageEvent
from google.adk.sessions.schemas.v1 import StorageEvent as V1StorageEvent
from google.adk.sessions.session import Session
from google.genai import types


def test_storage_event_v0_to_event_rehydrates_compaction_model():
  compaction = EventCompaction(
      start_timestamp=1.0,
      end_timestamp=2.0,
      compacted_content=types.Content(
          role="user",
          parts=[types.Part(text="compacted")],
      ),
  )
  actions = EventActions(compaction=compaction)
  storage_event = StorageEvent(
      id="event_id",
      invocation_id="invocation_id",
      author="author",
      actions=actions,
      session_id="session_id",
      app_name="app_name",
      user_id="user_id",
      timestamp=datetime.fromtimestamp(3.0, tz=timezone.utc),
  )

  event = storage_event.to_event()

  assert event.actions is not None
  assert isinstance(event.actions.compaction, EventCompaction)
  assert event.actions.compaction.start_timestamp == 1.0
  assert event.actions.compaction.end_timestamp == 2.0


def test_truncate_str_returns_none_for_none():
  assert _truncate_str(None, 256) is None


def test_truncate_str_returns_short_string_unchanged():
  short = "short message"
  assert _truncate_str(short, 256) == short


def test_truncate_str_returns_exact_length_string_unchanged():
  exact = "a" * DEFAULT_MAX_VARCHAR_LENGTH
  assert _truncate_str(exact, DEFAULT_MAX_VARCHAR_LENGTH) == exact


def test_truncate_str_truncates_long_string():
  long_msg = "x" * 1000
  result = _truncate_str(long_msg, DEFAULT_MAX_VARCHAR_LENGTH)
  assert result is not None
  assert len(result) == DEFAULT_MAX_VARCHAR_LENGTH
  assert result.endswith("...[truncated]")


def test_from_event_truncates_long_error_message():
  long_error = "Malformed function call: " + "a" * 1000
  session = Session(
      app_name="app",
      user_id="user",
      id="session_id",
      state={},
      events=[],
      last_update_time=0.0,
  )
  event = Event(
      id="event_id",
      invocation_id="inv_id",
      author="agent",
      timestamp=1.0,
      error_code="MALFORMED_FUNCTION_CALL",
      error_message=long_error,
  )

  storage_event = StorageEvent.from_event(session, event)

  assert storage_event.error_message is not None
  assert len(storage_event.error_message) == DEFAULT_MAX_VARCHAR_LENGTH
  assert storage_event.error_message.endswith("...[truncated]")
  assert storage_event.error_code == "MALFORMED_FUNCTION_CALL"


def test_from_event_preserves_short_error_message():
  short_error = "Something went wrong"
  session = Session(
      app_name="app",
      user_id="user",
      id="session_id",
      state={},
      events=[],
      last_update_time=0.0,
  )
  event = Event(
      id="event_id",
      invocation_id="inv_id",
      author="agent",
      timestamp=1.0,
      error_code="SOME_ERROR",
      error_message=short_error,
  )

  storage_event = StorageEvent.from_event(session, event)

  assert storage_event.error_message == short_error


def test_storage_event_v0_timestamp_round_trip_uses_utc():
  session = Session(app_name="app", user_id="user", id="session")
  event = Event(author="agent", timestamp=1.0)

  storage_event = StorageEvent.from_event(session, event)

  assert storage_event.timestamp == datetime(1970, 1, 1, 0, 0, 1)
  assert storage_event.to_event().timestamp == 1.0


def test_storage_event_v1_timestamp_round_trip_uses_utc():
  session = Session(app_name="app", user_id="user", id="session")
  event = Event(author="agent", timestamp=1.0)

  storage_event = V1StorageEvent.from_event(session, event)

  assert storage_event.timestamp == datetime(1970, 1, 1, 0, 0, 1)
  assert storage_event.to_event().timestamp == 1.0


def _event_with_recent_fields() -> Event:
  return Event(
      id="event_id",
      invocation_id="inv",
      author="agent",
      isolation_scope="fc-1",
      output={"result": "done"},
      content=types.Content(role="model", parts=[types.Part(text="hi")]),
  )


def _session() -> Session:
  return Session(id="s", app_name="app", user_id="u")


def test_storage_event_v0_loses_the_fields_it_has_no_column_for():
  """Pins the loss the legacy-schema warning exists for.

  The legacy schema gives an event one column per field, so a field added
  after it has nowhere to go. On read-back the event looks as though the field
  was never set, which is what lets an agent misbehave in silence.
  """
  restored = StorageEvent.from_event(
      _session(), _event_with_recent_fields()
  ).to_event()

  assert restored.isolation_scope is None
  assert restored.output is None


def test_storage_event_v1_keeps_them():
  """The current schema stores the whole event, so nothing is dropped."""
  restored = V1StorageEvent.from_event(
      _session(), _event_with_recent_fields()
  ).to_event()

  assert restored.isolation_scope == "fc-1"
  assert restored.output == {"result": "done"}


def test_stored_event_fields_tracks_the_columns():
  """The kept set is read off the columns, so it cannot drift from them."""
  stored = StorageEvent.stored_event_fields()

  assert "long_running_tool_ids" in stored
  assert "long_running_tool_ids_json" not in stored
  assert {"content", "author", "branch", "actions"} <= stored


def test_v0_reports_every_event_field_it_cannot_hold():
  """The reported loss is derived from Event, so it covers fields added later.

  A hand-written list would name whichever fields were topical the day it was
  written and silently stop being true afterwards.
  """
  dropped = set(
      _session_util.event_fields_not_stored(StorageEvent.stored_event_fields())
  )

  assert {"isolation_scope", "output", "node_info"} <= dropped
  # Nothing the table has a column for is reported as lost.
  assert not dropped & StorageEvent.stored_event_fields()
  # Every reported field really is one the round trip loses.
  restored = StorageEvent.from_event(
      _session(), _event_with_recent_fields()
  ).to_event()
  for field in dropped:
    assert getattr(restored, field) == type(restored).model_fields[
        field
    ].get_default(call_default_factory=True), field


def test_v1_reports_no_loss():
  """The current schema stores the whole event, so there is nothing to warn."""
  assert not _session_util.event_fields_not_stored(Event.model_fields)
