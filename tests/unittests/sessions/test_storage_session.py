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

"""Tests for StorageSession.to_session in both storage schemas."""

import contextlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import os
import time

from google.adk.events.event import Event
from google.adk.sessions.schemas import v0
from google.adk.sessions.schemas import v1
import pytest

# A naive timestamp, as SQLite and PostgreSQL hand it back to SQLAlchemy.
_NAIVE_UPDATE_TIME = datetime(2026, 1, 2, 3, 4, 5, 123456)
# The same instant, expressed in a non-UTC zone.
_AWARE_UPDATE_TIME = datetime(
    2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone(timedelta(hours=5))
)


@pytest.fixture(params=[v0, v1], ids=["v0", "v1"])
def schema(request):
  """Runs each test against both the pickle (v0) and JSON (v1) schemas."""
  return request.param


def _storage_session(schema, update_time):
  return schema.StorageSession(
      app_name="my_app",
      user_id="u1",
      id="s1",
      update_time=update_time,
  )


@contextlib.contextmanager
def _pinned_local_timezone(name: str):
  """Pins the process timezone for the duration of the block.

  ``time.tzset`` is POSIX-only, so on other platforms the block runs in the
  host zone instead. Restoring ``TZ`` without a second ``tzset`` would leave
  the C library pinned for the rest of the session, so both are undone.
  """
  if not hasattr(time, "tzset"):
    yield
    return
  previous = os.environ.get("TZ")
  os.environ["TZ"] = name
  time.tzset()
  try:
    yield
  finally:
    if previous is None:
      os.environ.pop("TZ", None)
    else:
      os.environ["TZ"] = previous
    time.tzset()


def test_to_session_without_arguments_yields_empty_state_and_events(schema):
  """The identity columns are copied and the containers default to empty."""
  session = _storage_session(schema, _NAIVE_UPDATE_TIME).to_session()

  assert session.app_name == "my_app"
  assert session.user_id == "u1"
  assert session.id == "s1"
  assert session.state == {}
  assert session.events == []


def test_to_session_carries_supplied_state_and_events(schema):
  """Caller-supplied state and events are attached unchanged."""
  event = Event(invocation_id="inv1", author="user")

  session = _storage_session(schema, _NAIVE_UPDATE_TIME).to_session(
      state={"k": "v"}, events=[event]
  )

  assert session.state == {"k": "v"}
  assert [e.invocation_id for e in session.events] == ["inv1"]


def test_to_session_reads_naive_update_time_as_utc(schema):
  """A naive stored timestamp means UTC, not the machine's local zone."""
  # Pin a non-UTC zone so reading the naive value as local time would produce
  # a different epoch than reading it as UTC.
  with _pinned_local_timezone("America/Los_Angeles"):
    session = _storage_session(schema, _NAIVE_UPDATE_TIME).to_session()

  assert (
      session.last_update_time
      == _NAIVE_UPDATE_TIME.replace(tzinfo=timezone.utc).timestamp()
  )
  # The marker keeps the stored wall-clock reading verbatim so it can be
  # compared against the value read back from storage.
  assert session._storage_update_marker == "2026-01-02T03:04:05.123456"


def test_to_session_normalizes_aware_update_time_marker_to_utc(schema):
  """An offset-aware timestamp keeps its instant and normalizes its marker."""
  session = _storage_session(schema, _AWARE_UPDATE_TIME).to_session()

  assert session.last_update_time == _AWARE_UPDATE_TIME.timestamp()
  assert session._storage_update_marker == "2026-01-01T22:04:05.123456+00:00"
