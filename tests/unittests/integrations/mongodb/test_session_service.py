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

"""Tests for MongoDbSessionService, backed by an in-memory fake client."""

from __future__ import annotations

from typing import Any
from unittest import mock

from google.adk.errors import StaleSessionError
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.integrations.mongodb import _client
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types
import pytest

from ._fake_mongo import FakeCursor
from ._fake_mongo import FakeMongoClient
from ._fake_mongo import OperationFailure

APP = "test_app"
USER = "test_user"
USER_2 = "other_user"


@pytest.fixture
def client() -> Any:
  return FakeMongoClient()


@pytest.fixture
def service(client: Any) -> MongoDbSessionService:
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
async def test_create_and_get_session(
    service: MongoDbSessionService,
) -> None:
  session = await service.create_session(
      app_name=APP,
      user_id=USER,
      state={
          "session_key": "s1",
          "coords": (1, 2),
          "app:app_key": "a1",
          "app:tags": ("x", "y"),
          "user:user_key": "u1",
          "user:roles": ("r1",),
      },
      session_id="s1",
  )

  assert session.id == "s1"
  assert session.state["session_key"] == "s1"
  assert session.state["coords"] == [1, 2]
  assert session.state["app:app_key"] == "a1"
  assert session.state["app:tags"] == ["x", "y"]
  assert session.state["user:user_key"] == "u1"
  assert session.state["user:roles"] == ["r1"]
  assert session.events == []

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state == session.state
  assert fetched.last_update_time == session.last_update_time
  assert fetched._storage_update_marker == "0"

  # Missing session returns None.
  assert (
      await service.get_session(app_name=APP, user_id=USER, session_id="nope")
      is None
  )


@pytest.mark.asyncio
async def test_create_duplicate_session_raises(
    service: MongoDbSessionService,
) -> None:
  await service.create_session(
      app_name=APP,
      user_id=USER,
      state={"app:app_key": "initial", "user:user_key": "initial"},
      session_id="s1",
  )
  with pytest.raises(AlreadyExistsError):
    await service.create_session(
        app_name=APP,
        user_id=USER,
        state={"app:app_key": "polluted", "user:user_key": "polluted"},
        session_id="s1",
    )

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state["app:app_key"] == "initial"
  assert fetched.state["user:user_key"] == "initial"


@pytest.mark.asyncio
async def test_constructor_validates_client_args(client: Any) -> None:
  with pytest.raises(ValueError):
    MongoDbSessionService(database_name="db")
  with pytest.raises(ValueError):
    MongoDbSessionService(
        database_name="db",
        mongo_client=client,
        connection_string="mongodb://localhost:27017",
    )


@pytest.mark.asyncio
async def test_append_event_persists_and_merges_state(
    service: MongoDbSessionService,
) -> None:
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  session.state["direct_write"] = "dropped"
  event = _text_event("hello mongodb", timestamp=10.0)
  event.actions = EventActions(
      state_delta={"turn": 1, "user:theme": "dark", "temp:scratch": "x"}
  )
  await service.append_event(session, event)

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert len(fetched.events) == 1
  assert fetched.events[0].content is not None
  assert fetched.events[0].content.parts is not None
  assert fetched.events[0].content.parts[0].text == "hello mongodb"
  assert fetched.state["turn"] == 1
  assert fetched.state["user:theme"] == "dark"
  assert "direct_write" not in fetched.state
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
async def test_append_event_missing_session_raises(
    service: MongoDbSessionService,
) -> None:
  session = await service.create_session(
      app_name=APP,
      user_id=USER,
      state={"app:app_key": "initial", "user:user_key": "initial"},
      session_id="s1",
  )
  session.id = "deleted"
  bad_event = _text_event("hi")
  bad_event.actions = EventActions(
      state_delta={"app:app_key": "polluted", "user:user_key": "polluted"}
  )
  with pytest.raises(SessionNotFoundError):
    await service.append_event(session, bad_event)

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state["app:app_key"] == "initial"
  assert fetched.state["user:user_key"] == "initial"


@pytest.mark.asyncio
async def test_append_event_stale_session_raises(
    service: MongoDbSessionService,
) -> None:
  session = await service.create_session(
      app_name=APP,
      user_id=USER,
      state={"app:app_key": "initial", "user:user_key": "initial"},
      session_id="s1",
  )
  stale_copy = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert stale_copy is not None

  await service.append_event(session, _text_event("first", timestamp=1.0))
  # stale_copy still holds revision marker "0" while storage is at "1".
  stale_event = _text_event("second", timestamp=2.0)
  stale_event.actions = EventActions(
      state_delta={"app:app_key": "polluted", "user:user_key": "polluted"}
  )
  with pytest.raises(StaleSessionError):
    await service.append_event(stale_copy, stale_event)

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state["app:app_key"] == "initial"
  assert fetched.state["user:user_key"] == "initial"


@pytest.mark.asyncio
async def test_append_event_rejects_duplicate_event_id(
    service: MongoDbSessionService,
    client: Any,
) -> None:
  """A reused event id is rejected before any write."""
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  first = _text_event("first", timestamp=1.0)
  first.actions = EventActions(state_delta={"k": "first"})
  await service.append_event(session, first)

  second = _text_event("second", timestamp=2.0)
  second.id = first.id
  second.actions = EventActions(
      state_delta={"k": "second", "app:a": 1, "user:u": 2}
  )
  with pytest.raises(AlreadyExistsError):
    await service.append_event(session, second)

  assert len(session.events) == 1
  assert client["test_db"]["events"].count_documents({}) == 1
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state == {"k": "first"}
  assert [
      e.content.parts[0].text
      for e in fetched.events
      if e.content and e.content.parts
  ] == ["first"]

  third = _text_event("third", timestamp=3.0)
  third.actions = EventActions(state_delta={"k": "third"})
  await service.append_event(session, third)

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state == {"k": "third"}
  assert [
      e.content.parts[0].text
      for e in fetched.events
      if e.content and e.content.parts
  ] == ["first", "third"]


@pytest.mark.asyncio
async def test_slash_in_keys_does_not_collide(
    service: MongoDbSessionService,
) -> None:
  s1 = await service.create_session(
      app_name="a/b",
      user_id="c",
      state={"k": "v1", "user:uk": "u1"},
      session_id="d/e",
  )
  s2 = await service.create_session(
      app_name="a",
      user_id="b/c",
      state={"k": "v2", "user:uk": "u2"},
      session_id="d/e",
  )
  s3 = await service.create_session(
      app_name="a",
      user_id="b/c/d",
      state={"k": "v3", "user:uk": "u3"},
      session_id="e",
  )

  e1 = _text_event("m1", timestamp=1.0)
  e1.id = "x/y"
  await service.append_event(s1, e1)

  e3 = _text_event("m3", timestamp=2.0)
  e3.id = "y"
  await service.append_event(s3, e3)

  fetched_1 = await service.get_session(
      app_name="a/b", user_id="c", session_id="d/e"
  )
  fetched_2 = await service.get_session(
      app_name="a", user_id="b/c", session_id="d/e"
  )
  fetched_3 = await service.get_session(
      app_name="a", user_id="b/c/d", session_id="e"
  )
  assert fetched_1 is not None
  assert fetched_2 is not None
  assert fetched_3 is not None
  assert fetched_1.state == {"k": "v1", "user:uk": "u1"}
  assert fetched_2.state == {"k": "v2", "user:uk": "u2"}
  assert fetched_3.state == {"k": "v3", "user:uk": "u3"}
  assert len(fetched_1.events) == 1
  assert len(fetched_2.events) == 0
  assert len(fetched_3.events) == 1


@pytest.mark.asyncio
async def test_get_session_config_filters_events(
    service: MongoDbSessionService,
) -> None:
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
  assert fetched is not None
  assert fetched.events == []

  # num_recent_events=2 -> two most recent, in chronological order.
  fetched = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id="s1",
      config=GetSessionConfig(num_recent_events=2),
  )
  assert fetched is not None
  assert [
      e.content.parts[0].text
      for e in fetched.events
      if e.content and e.content.parts
  ] == [
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
  assert fetched is not None
  assert [
      e.content.parts[0].text
      for e in fetched.events
      if e.content and e.content.parts
  ] == [
      "event 3",
      "event 4",
  ]


@pytest.mark.asyncio
async def test_get_session_breaks_timestamp_ties_in_append_order(
    service: MongoDbSessionService,
) -> None:
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("first", timestamp=1.0))
  await service.append_event(session, _text_event("second", timestamp=1.0))

  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert [
      e.content.parts[0].text
      for e in fetched.events
      if e.content and e.content.parts
  ] == ["first", "second"]

  recent = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id="s1",
      config=GetSessionConfig(num_recent_events=1),
  )
  assert recent is not None
  assert [
      e.content.parts[0].text
      for e in recent.events
      if e.content and e.content.parts
  ] == ["second"]


@pytest.mark.asyncio
async def test_list_sessions_sorted_oldest_first(
    service: MongoDbSessionService,
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  await service.create_session(
      app_name=APP,
      user_id=USER,
      state={"user:lang": "en"},
      session_id="s1",
  )
  await service.create_session(app_name=APP, user_id=USER, session_id="s2")
  await service.create_session(
      app_name=APP,
      user_id=USER_2,
      state={"user:lang": "fr"},
      session_id="s3",
  )

  response = await service.list_sessions(app_name=APP, user_id=USER)
  assert [s.id for s in response.sessions] == ["s1", "s2"]
  assert [s.state["user:lang"] for s in response.sessions] == ["en", "en"]

  response = await service.list_sessions(app_name=APP, user_id="")
  assert response.sessions == []

  find_queries: list[dict[str, Any]] = []
  orig_find = client["test_db"]["user_states"].find

  def _recording_find(
      query: dict[str, Any], *, session: Any = None
  ) -> FakeCursor:
    find_queries.append(query)
    return orig_find(query, session=session)

  monkeypatch.setattr(client["test_db"]["user_states"], "find", _recording_find)

  response = await service.list_sessions(app_name=APP)
  assert {s.id for s in response.sessions} == {"s1", "s2", "s3"}
  assert {s.id: s.state["user:lang"] for s in response.sessions} == {
      "s1": "en",
      "s2": "en",
      "s3": "fr",
  }
  assert len(find_queries) == 1


@pytest.mark.asyncio
async def test_delete_session_removes_session_and_events(
    service: MongoDbSessionService,
    client: Any,
) -> None:
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("bye", timestamp=1.0))

  await service.delete_session(app_name=APP, user_id=USER, session_id="s1")

  assert (
      await service.get_session(app_name=APP, user_id=USER, session_id="s1")
      is None
  )
  assert client["test_db"]["events"].count_documents({}) == 0


@pytest.mark.asyncio
async def test_state_keys_with_dots_round_trip(
    service: MongoDbSessionService,
) -> None:
  """MongoDB forbids dots in document keys; JSON-encoded state does not care."""
  session = await service.create_session(
      app_name=APP, user_id=USER, state={"nested.key": 1}, session_id="s1"
  )
  event = _text_event("x", timestamp=1.0)
  event.actions = EventActions(
      state_delta={"event.dotted.key": 2, "$dollar.key": 3}
  )
  await service.append_event(session, event)
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.state["nested.key"] == 1
  assert fetched.state["event.dotted.key"] == 2
  assert fetched.state["$dollar.key"] == 3
  assert fetched.events[0].actions.state_delta == {
      "event.dotted.key": 2,
      "$dollar.key": 3,
  }


@pytest.mark.asyncio
async def test_ensure_indexes_creates_recommended_indexes(
    client: Any,
) -> None:
  """ensure_indexes=True creates the events and sessions secondary indexes."""
  service = MongoDbSessionService(
      mongo_client=client, database_name="test_db", ensure_indexes=True
  )
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("hi", timestamp=1.0))

  events_indexes = client["test_db"]["events"].index_information()
  assert "app_user_session_ts" in events_indexes
  sessions_indexes = client["test_db"]["sessions"].index_information()
  assert "app_user" in sessions_indexes


@pytest.mark.asyncio
async def test_indexes_created_by_default(
    service: MongoDbSessionService, client: Any
) -> None:
  """Indexes are on by default: no opt-in is needed for the read indexes."""
  await service.create_session(app_name=APP, user_id=USER, session_id="s1")

  assert "app_user_session_ts" in (
      client["test_db"]["events"].index_information()
  )
  assert "app_user" in (client["test_db"]["sessions"].index_information())


@pytest.mark.asyncio
async def test_ensure_indexes_false_skips_index_creation(client: Any) -> None:
  """Opting out leaves the implicit _id indexes as the only ones present."""
  service = MongoDbSessionService(
      mongo_client=client, database_name="test_db", ensure_indexes=False
  )
  await service.create_session(app_name=APP, user_id=USER, session_id="s1")

  assert "app_user_session_ts" not in (
      client["test_db"]["events"].index_information()
  )
  assert "app_user" not in (client["test_db"]["sessions"].index_information())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("not authorized on test_db", 13),
        ("user is not allowed to do action [createIndex] on [test_db]", 8000),
    ],
)
async def test_index_privilege_error_falls_back_to_unindexed_reads(
    service: MongoDbSessionService,
    client: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    code: int,
) -> None:
  """A denied createIndex warns once and never breaks reads or writes."""
  create_index_calls: list[str] = []

  def _denied(keys: Any, *, name: str) -> str:
    del keys
    create_index_calls.append(name)
    raise OperationFailure(message, code=code)

  monkeypatch.setattr(client["test_db"]["events"], "create_index", _denied)

  with caplog.at_level("WARNING"):
    session = await service.create_session(
        app_name=APP, user_id=USER, session_id="s1"
    )
    await service.append_event(session, _text_event("hi", timestamp=1.0))
    fetched = await service.get_session(
        app_name=APP, user_id=USER, session_id="s1"
    )

  assert fetched is not None
  assert len(fetched.events) == 1
  assert create_index_calls == ["app_user_session_ts"]
  warnings = [
      r for r in caplog.records if "app_user_session_ts" in r.getMessage()
  ]
  assert len(warnings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_name", "existing_keys"),
    [
        # 86: the name exists with keys from an earlier release.
        (
            "app_user_session_ts",
            [("app_name", 1), ("user_id", 1), ("session_id", 1)],
        ),
        # 85: the same keys exist under a DBA-chosen name.
        (
            "dba_events_idx",
            [
                ("app_name", 1),
                ("user_id", 1),
                ("session_id", 1),
                ("timestamp", -1),
                ("revision", -1),
            ],
        ),
    ],
)
async def test_index_conflict_warns_once_and_session_calls_work(
    service: MongoDbSessionService,
    client: Any,
    caplog: pytest.LogCaptureFixture,
    existing_name: str,
    existing_keys: list[tuple[str, int]],
) -> None:
  """A conflicting existing index warns once and is not retried."""
  client["test_db"]["events"].create_index(existing_keys, name=existing_name)

  with caplog.at_level("WARNING"):
    session = await service.create_session(
        app_name=APP, user_id=USER, session_id="s1"
    )
    await service.append_event(session, _text_event("hi", timestamp=1.0))
    fetched = await service.get_session(
        app_name=APP, user_id=USER, session_id="s1"
    )

  assert fetched is not None
  assert len(fetched.events) == 1
  assert "app_user" in client["test_db"]["sessions"].index_information()
  warnings = [
      r for r in caplog.records if "app_user_session_ts" in r.getMessage()
  ]
  assert len(warnings) == 1


@pytest.mark.asyncio
async def test_transient_index_error_warns_and_retries(
    service: MongoDbSessionService,
    client: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A transient index error does not fail the call; the next call retries."""
  events = client["test_db"]["events"]
  real_create_index = events.create_index
  failures = [OperationFailure("connection reset", code=6)]

  def _flaky(keys: Any, *, name: str) -> str:
    if failures:
      raise failures.pop()
    return real_create_index(keys, name=name)

  monkeypatch.setattr(events, "create_index", _flaky)

  with caplog.at_level("WARNING"):
    await service.create_session(app_name=APP, user_id=USER, session_id="s1")
  assert "app_user_session_ts" not in events.index_information()
  assert any("connection reset" in r.getMessage() for r in caplog.records)

  await service.create_session(app_name=APP, user_id=USER, session_id="s2")
  assert "app_user_session_ts" in events.index_information()


@pytest.mark.asyncio
async def test_client_without_sessions_falls_back_to_direct_writes(
    service: MongoDbSessionService,
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """In-memory fake has no sessions: writes run directly and verdict caches."""
  start_session_calls: list[int] = []

  def _start_session() -> Any:
    start_session_calls.append(1)
    raise NotImplementedError("Sessions are not supported by FakeMongoClient")

  monkeypatch.setattr(client, "start_session", _start_session)

  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("hi", timestamp=1.0))

  assert len(start_session_calls) == 1
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.events[0].content is not None
  assert fetched.events[0].content.parts is not None
  assert fetched.events[0].content.parts[0].text == "hi"


@pytest.mark.asyncio
async def test_writes_run_inside_transaction_when_supported() -> None:
  """A transaction-capable client wraps every write sequence in a session."""
  client = mock.MagicMock()
  mongo_session = client.start_session.return_value
  mongo_session.with_transaction.side_effect = lambda callback: callback(
      mongo_session
  )
  db = client["test_db"]
  db["app_states"].find_one.return_value = None
  db["user_states"].find_one.return_value = None
  db["sessions"].find_one.return_value = {"revision": 0}
  db["sessions"].find_one_and_update.return_value = {"revision": 1}
  # MagicMock hands back one child for every collection name, so the
  # duplicate-event lookup is told apart by its key.
  db["events"].find_one.side_effect = lambda query, **_: (
      None if "event_id" in query["_id"] else {"revision": 0}
  )

  service = MongoDbSessionService(mongo_client=client, database_name="test_db")
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("hi", timestamp=1.0))

  # create_session and append_event each ran their writes in a transaction.
  assert mongo_session.with_transaction.call_count == 2
  assert db["sessions"].insert_one.call_args.kwargs["session"] is mongo_session
  assert (
      db["sessions"].find_one_and_update.call_args.kwargs["session"]
      is mongo_session
  )
  assert db["events"].insert_one.call_args.kwargs["session"] is mongo_session


class _UnsupportedTransactionSession:
  """Fake client session whose transaction fails like a standalone mongod."""

  def __enter__(self) -> _UnsupportedTransactionSession:
    return self

  def __exit__(self, *args: Any) -> None:
    del args

  def with_transaction(self, callback: Any) -> Any:
    del callback
    raise OperationFailure(
        "Transaction numbers are only allowed on a replica set member or"
        " mongos",
        code=20,
    )


@pytest.mark.asyncio
async def test_falls_back_when_transactions_not_supported(
    service: MongoDbSessionService,
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Standalone mongod: first write falls back, later writes skip attempt."""
  start_session_calls: list[int] = []

  def _start_session() -> _UnsupportedTransactionSession:
    start_session_calls.append(1)
    return _UnsupportedTransactionSession()

  monkeypatch.setattr(client, "start_session", _start_session)

  session = await service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await service.append_event(session, _text_event("hi", timestamp=1.0))

  # create_session attempted a transaction, failed, and fell back;
  # append_event then skipped the attempt because the verdict was cached.
  assert len(start_session_calls) == 1
  fetched = await service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert fetched is not None
  assert fetched.events[0].content is not None
  assert fetched.events[0].content.parts is not None
  assert fetched.events[0].content.parts[0].text == "hi"


@pytest.mark.asyncio
async def test_close_does_not_close_injected_client(
    service: MongoDbSessionService,
    client: Any,
) -> None:
  """close() leaves a caller-owned client open."""
  await service.close()

  assert client.closed is False


@pytest.mark.asyncio
async def test_close_closes_client_created_from_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """close() closes the client created from connection_string."""
  created_client = FakeMongoClient()
  received_uris: list[str] = []

  def _get_mongo_client(connection_string: str) -> Any:
    received_uris.append(connection_string)
    return created_client

  monkeypatch.setattr(_client, "get_mongo_client", _get_mongo_client)
  owned_service = MongoDbSessionService(
      database_name="test_db",
      connection_string="mongodb://localhost:27017",
  )

  await owned_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert (
      await owned_service.get_session(
          app_name=APP, user_id=USER, session_id="s1"
      )
      is not None
  )

  await owned_service.close()

  assert received_uris == ["mongodb://localhost:27017"]
  assert created_client.closed is True
