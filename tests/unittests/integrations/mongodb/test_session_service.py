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

"""Tests for MongoDbSessionService, backed by mongomock."""

from __future__ import annotations

from google.adk.errors import StaleSessionError
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types
import mongomock
import pytest

APP = "test_app"
USER = "test_user"
USER_2 = "other_user"


@pytest.fixture
def client():
  return mongomock.MongoClient()


@pytest.fixture
def service(client):
  return MongoDbSessionService(mongo_client=client, database_name="test_db")


def _text_event(
    text: str, *, author: str = "user", timestamp: float = 1.0
) -> Event:
  return Event(
      invocation_id="inv",
      author=author,
      content=types.Content(
          role="user" if author == "user" else "model",
          parts=[types.Part(text=text)],
      ),
      timestamp=timestamp,
  )


@pytest.mark.asyncio
async def test_create_and_get_session(service):
  session = await service.create_session(
      app_name=APP,
      user_id=USER,
      state={"session_key": "s1", "app:app_key": "a1", "user:user_key": "u1"},
      session_id="s1",
  )

  assert session.id == "s1"
  assert session.state["session_key"] == "s1"
  assert session.state["app:app_key"] == "a1"
  assert session.state["user:user_key"] == "u1"
  assert session.events == []

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state == session.state
  assert fetched._storage_update_marker == "0"

  # Missing session returns None.
  assert (
      await service.get_session(app_name=APP, user_id=USER, session_id="nope")
      is None
  )


@pytest.mark.asyncio
async def test_create_duplicate_session_raises(service):
  await service.create_session(app_name=APP, user_id=USER, session_id="s1")
  with pytest.raises(AlreadyExistsError):
    await service.create_session(app_name=APP, user_id=USER, session_id="s1")


@pytest.mark.asyncio
async def test_constructor_validates_client_args(client):
  with pytest.raises(ValueError):
    MongoDbSessionService(database_name="db")
  with pytest.raises(ValueError):
    MongoDbSessionService(
        database_name="db",
        mongo_client=client,
        connection_string="mongodb://localhost:27017",
    )


@pytest.mark.asyncio
async def test_append_event_persists_and_merges_state(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = _text_event("hello mongodb", timestamp=10.0)
  event.actions = EventActions(
      state_delta={"turn": 1, "user:theme": "dark", "temp:scratch": "x"}
  )
  await service.append_event(session, event)

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert len(fetched.events) == 1
  assert fetched.events[0].content.parts[0].text == "hello mongodb"
  assert fetched.state["turn"] == 1
  assert fetched.state["user:theme"] == "dark"
  # temp state lives on the in-memory session but is never persisted.
  assert "temp:scratch" in session.state
  assert "temp:scratch" not in fetched.state
  assert fetched._storage_update_marker == "1"
  # The in-memory session advanced too.
  assert session._storage_update_marker == "1"
  assert len(session.events) == 1

  # User state is shared across the user's sessions.
  session2 = await service.create_session(
      app_name=APP, user_id=USER, session_id="s2"
  )
  assert session2.state["user:theme"] == "dark"
  assert await service.get_user_state(app_name=APP, user_id=USER) == {
      "theme": "dark"
  }


@pytest.mark.asyncio
async def test_append_event_missing_session_raises(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  session.id = "deleted"
  with pytest.raises(SessionNotFoundError):
    await service.append_event(session, _text_event("hi"))


@pytest.mark.asyncio
async def test_append_event_stale_session_raises(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  stale_copy = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )

  await service.append_event(session, _text_event("first", timestamp=1.0))
  # stale_copy still holds revision marker "0" while storage is at "1".
  with pytest.raises(StaleSessionError):
    await service.append_event(stale_copy, _text_event("second", timestamp=2.0))


@pytest.mark.asyncio
async def test_append_event_is_idempotent_on_retry(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = _text_event("hello", timestamp=1.0)
  await service.append_event(session, event)

  # Simulate a storage-level retry of the same event write: the document is
  # replaced in place instead of duplicated.
  service._events().replace_one(
      {"_id": f"{APP}/{USER}/s1/{event.id}"},
      {
          "app_name": APP,
          "user_id": USER,
          "session_id": "s1",
          "timestamp": event.timestamp,
          "event_data": event.model_dump(exclude_none=True, mode="json"),
      },
      upsert=True,
  )
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert len(fetched.events) == 1


@pytest.mark.asyncio
async def test_get_session_config_filters_events(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  for i in range(1, 5):
    await service.append_event(
        session, _text_event(f"event {i}", timestamp=float(i))
    )

  # num_recent_events=0 -> no events loaded.
  fetched = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id="s1",
      config=GetSessionConfig(num_recent_events=0),
  )
  assert fetched.events == []

  # num_recent_events=2 -> two most recent, in chronological order.
  fetched = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id="s1",
      config=GetSessionConfig(num_recent_events=2),
  )
  assert [e.content.parts[0].text for e in fetched.events] == [
      "event 3",
      "event 4",
  ]

  # after_timestamp -> only events at or after the cursor.
  fetched = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id="s1",
      config=GetSessionConfig(after_timestamp=2.5),
  )
  assert [e.content.parts[0].text for e in fetched.events] == [
      "event 3",
      "event 4",
  ]


@pytest.mark.asyncio
async def test_list_sessions_sorted_oldest_first(service):
  await service.create_session(app_name=APP, user_id=USER, session_id="s1")
  await service.create_session(app_name=APP, user_id=USER, session_id="s2")
  await service.create_session(app_name=APP, user_id=USER_2, session_id="s3")

  response = await service.list_sessions(app_name=APP, user_id=USER)
  assert [s.id for s in response.sessions] == ["s1", "s2"]

  response = await service.list_sessions(app_name=APP)
  assert {s.id for s in response.sessions} == {"s1", "s2", "s3"}


@pytest.mark.asyncio
async def test_delete_session_removes_session_and_events(service):
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("bye", timestamp=1.0))

  await service.delete_session(app_name=APP, user_id=USER, session_id="s1")

  assert (
      await service.get_session(app_name=APP, user_id=USER, session_id="s1")
      is None
  )
  assert service._events().count_documents({}) == 0


@pytest.mark.asyncio
async def test_state_keys_with_dots_round_trip(service):
  """MongoDB forbids dots in document keys; JSON-encoded state does not care."""
  session = await service.create_session(
      app_name=APP, user_id=USER, state={"nested.key": 1}, session_id="s1"
  )
  await service.append_event(
      session,
      _text_event("x", timestamp=1.0),
  )
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched.state["nested.key"] == 1
