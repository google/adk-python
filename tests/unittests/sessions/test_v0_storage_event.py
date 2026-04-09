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
from google.adk.sessions.schemas.shared import DEFAULT_MAX_VARCHAR_LENGTH
from google.adk.sessions.schemas.v0 import _truncate_str
from google.adk.sessions.schemas.v0 import StorageEvent
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
