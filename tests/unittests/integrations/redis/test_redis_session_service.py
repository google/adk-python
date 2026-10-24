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

"""Unit tests for RedisSessionService."""

from __future__ import annotations

import json
from unittest import mock

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.integrations.redis._config import RedisSessionServiceConfig
from google.adk.integrations.redis._redis_session_service import RedisSessionService
from google.adk.integrations.redis._redis_session_service import _escape_glob
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from ._fake_redis import FakeRedisAsync


@pytest.fixture
def fake_redis():
  return FakeRedisAsync()


@pytest.fixture
def session_service(fake_redis):
  config = RedisSessionServiceConfig(
      ttl_seconds=3600,
      key_prefix="test:session:",
  )
  return RedisSessionService(config=config, redis_client=fake_redis)


def test_key_components_are_unambiguous_and_type_separated(session_service):
  assert (
      RedisSessionService()
      ._session_key("app1", "user1", "session1")
      .startswith("adk:session:v2:")
  )
  assert (
      session_service._session_key("app1", "user1", "session1")
      == "test:session:v2:s2_YXBwMQ.dXNlcjE.c2Vzc2lvbjE"
  )
  victim_key = session_service._session_key("billing", "tenant:victim", "fixed")
  assert (
      victim_key == "test:session:v2:s2_YmlsbGluZw.dGVuYW50OnZpY3RpbQ.Zml4ZWQ"
  )
  assert victim_key != session_service._session_key(
      "billing", "tenant", "victim:fixed"
  )
  assert session_service._user_state_key(
      "billing:tenant", "victim"
  ) != session_service._user_state_key("billing", "tenant:victim")
  assert session_service._session_key(
      "user_state", "billing", "alice"
  ) != session_service._user_state_key("billing", "alice")
  assert session_service._session_key(
      "app_state", "billing", "alice"
  ) != session_service._app_state_key("billing:alice")
  assert (
      session_service._app_state_key("billing")
      == "test:session:v2:a2_YmlsbGluZw"
  )
  for key in (
      victim_key,
      session_service._user_state_key("billing:tenant", "victim"),
      session_service._app_state_key("billing:tenant"),
  ):
    encoded_suffix = key.removeprefix("test:session:v2:")
    assert ":" not in encoded_suffix
    assert not any(char in encoded_suffix for char in "*?[]\\")

  assert _escape_glob("prefix:*?[x]") == r"prefix:\*\?\[x\]"
  assert _escape_glob("a\\b") == "a\\\\b"


@pytest.mark.asyncio
async def test_versioned_keys_ignore_legacy_session_and_state_keys(
    session_service, fake_redis
):
  legacy_session = Session(
      id="fixed",
      app_name="billing",
      user_id="tenant:victim",
      state={"private": "legacy-session"},
      events=[],
      last_update_time=1.0,
  )
  legacy_key = "test:session:billing:tenant:victim:fixed"
  await fake_redis.set(legacy_key, legacy_session.model_dump_json())
  legacy_session_at_user_state_key = Session(
      id="alice",
      app_name="user_state",
      user_id="billing",
      state={"private": "legacy-session-at-state-key"},
      events=[],
      last_update_time=1.0,
  )
  await fake_redis.set(
      "test:session:user_state:billing:alice",
      legacy_session_at_user_state_key.model_dump_json(),
  )
  await fake_redis.set("test:session:app_state:billing", '{"old": true}')

  assert (
      await session_service.get_session(
          app_name="billing",
          user_id="tenant:victim",
          session_id="fixed",
      )
      is None
  )
  assert (
      await session_service.get_user_state(app_name="billing", user_id="alice")
      == {}
  )
  assert (
      await session_service.list_sessions(
          app_name="billing", user_id="tenant:victim"
      )
  ).sessions == []
  assert (
      await session_service.list_sessions(app_name="billing")
  ).sessions == []
  await session_service.delete_session(
      app_name="billing", user_id="tenant:victim", session_id="fixed"
  )
  assert await fake_redis.get(legacy_key) == legacy_session.model_dump_json()
  new_session = await session_service.create_session(
      app_name="billing", user_id="alice", session_id="new"
  )
  assert new_session.state == {}


@pytest.mark.parametrize("binary_scan_keys", [False, True])
@pytest.mark.asyncio
async def test_list_sessions_ignores_legacy_key_matching_v2_scan(
    session_service, fake_redis, binary_scan_keys
):
  legacy_session = Session(
      id="bar",
      app_name="v2",
      user_id="s2_djI.foo",
      state={"private": "legacy-session"},
      events=[],
      last_update_time=1.0,
  )
  await fake_redis.set(
      "test:session:v2:s2_djI.foo:bar", legacy_session.model_dump_json()
  )
  current = await session_service.create_session(
      app_name="v2", user_id="alice", session_id="live"
  )
  if binary_scan_keys:
    original_scan_iter = fake_redis.scan_iter
    original_get = fake_redis.get

    async def scan_bytes(match):
      async for key in original_scan_iter(match):
        yield key.encode("utf-8")

    async def get_with_bytes(key):
      return await original_get(
          key.decode("utf-8") if isinstance(key, bytes) else key
      )

    fake_redis.scan_iter = scan_bytes
    fake_redis.get = get_with_bytes

  assert (await session_service.list_sessions(app_name="v2")).sessions == [
      current
  ]


@pytest.mark.asyncio
async def test_create_session(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="user1",
      state={"key1": "val1", "user:pref": "dark", "app:version": "1.0"},
  )

  assert session.app_name == "app1"
  assert session.user_id == "user1"
  assert session.state["key1"] == "val1"
  assert session.state["user:pref"] == "dark"
  assert session.state["app:version"] == "1.0"
  assert session.id is not None


@pytest.mark.asyncio
async def test_create_session_already_exists(session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="user1",
      session_id="sess_123",
  )

  with pytest.raises(AlreadyExistsError):
    await session_service.create_session(
        app_name="app1",
        user_id="user1",
        session_id="sess_123",
    )


@pytest.mark.asyncio
async def test_get_session(session_service):
  created = await session_service.create_session(
      app_name="app1",
      user_id="user1",
      session_id="sess_abc",
      state={"foo": "bar"},
  )

  fetched = await session_service.get_session(
      app_name="app1",
      user_id="user1",
      session_id="sess_abc",
  )

  assert fetched is not None
  assert fetched.id == created.id
  assert fetched.state["foo"] == "bar"


@pytest.mark.asyncio
async def test_get_session_not_found(session_service):
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="user1",
      session_id="nonexistent",
  )
  assert fetched is None


@pytest.mark.asyncio
async def test_get_list_and_delete_reject_colon_key_alias(session_service):
  """A colon-delimited identity alias cannot read, list, or delete a session."""
  victim = await session_service.create_session(
      app_name="billing",
      user_id="tenant:victim",
      session_id="fixed",
      state={"secret": "victim-only"},
  )
  assert session_service._session_key(
      "billing", "tenant:victim", "fixed"
  ) != session_service._session_key("billing", "tenant", "victim:fixed")

  assert (
      await session_service.get_session(
          app_name="billing", user_id="tenant", session_id="victim:fixed"
      )
      is None
  )
  assert (
      await session_service.list_sessions(app_name="billing", user_id="tenant")
  ).sessions == []
  assert [
      s.id
      for s in (
          await session_service.list_sessions(app_name="billing")
      ).sessions
  ] == [victim.id]

  await session_service.delete_session(
      app_name="billing", user_id="tenant", session_id="victim:fixed"
  )
  fetched = await session_service.get_session(
      app_name="billing", user_id="tenant:victim", session_id="fixed"
  )
  assert fetched is not None
  assert fetched.state["secret"] == "victim-only"


@pytest.mark.asyncio
async def test_delete_session_removes_corrupt_record(
    session_service, fake_redis
):
  """Deletion works even if the stored session cannot be deserialized."""
  key = session_service._session_key("billing", "tenant:victim", "fixed")
  await fake_redis.set(key, "{corrupt json")
  await session_service.delete_session(
      app_name="billing", user_id="tenant:victim", session_id="fixed"
  )
  assert await fake_redis.get(key) is None


@pytest.mark.asyncio
async def test_append_event_does_not_commit_state_after_identity_race(
    session_service, fake_redis
):
  """A replacement identity aborts queued session and state writes."""
  victim = await session_service.create_session(
      app_name="billing",
      user_id="tenant:victim",
      session_id="fixed",
  )
  alias = Session(
      id="victim:fixed",
      app_name="billing",
      user_id="tenant",
      state={"owner": "replacement"},
      events=[],
      last_update_time=victim.last_update_time,
  )
  key = session_service._session_key("billing", "tenant:victim", "fixed")
  original_get = fake_redis.get
  replaced = False

  async def replace_after_read(read_key):
    nonlocal replaced
    raw = await original_get(read_key)
    if read_key == key and not replaced:
      replaced = True
      await fake_redis.delete(key)
      await fake_redis.set(key, alias.model_dump_json())
    return raw

  fake_redis.get = replace_after_read
  event = Event(
      author="user",
      actions=EventActions(
          state_delta={
              "app:should_not_write": True,
              "user:should_not_write": True,
              "session:should_not_write": True,
          }
      ),
  )
  with pytest.raises(SessionNotFoundError):
    await session_service.append_event(victim, event)

  assert await original_get(session_service._app_state_key("billing")) is None
  assert (
      await original_get(
          session_service._user_state_key("billing", "tenant:victim")
      )
      is None
  )
  remaining = await original_get(key)
  assert remaining is not None
  assert Session.model_validate_json(remaining).user_id == "tenant"
  assert Session.model_validate_json(remaining).state == {
      "owner": "replacement"
  }


@pytest.mark.asyncio
async def test_append_event_rejects_mismatched_stored_identity_without_persistent_mutation(
    session_service, fake_redis
):
  """A mismatched Redis record cannot cause persistent cross-identity writes."""
  victim = await session_service.create_session(
      app_name="billing",
      user_id="tenant:victim",
      session_id="fixed",
      state={"secret": "victim-only"},
  )
  attacker_view = Session(
      id="victim:fixed",
      app_name="billing",
      user_id="tenant",
      state={"attacker": "local-only"},
      events=[],
      last_update_time=victim.last_update_time,
  )
  attacker_key = session_service._session_key(
      "billing", "tenant", "victim:fixed"
  )
  await fake_redis.set(attacker_key, victim.model_dump_json())
  event = Event(
      author="user",
      actions=EventActions(
          state_delta={
              "app:alias_write": True,
              "user:alias_write": True,
              "session_alias_write": True,
          }
      ),
  )

  with pytest.raises(SessionNotFoundError):
    await session_service.append_event(attacker_view, event)

  assert await fake_redis.get(session_service._app_state_key("billing")) is None
  assert (
      await session_service.get_user_state(app_name="billing", user_id="tenant")
      == {}
  )
  assert (
      await session_service.get_user_state(
          app_name="billing", user_id="tenant:victim"
      )
      == {}
  )
  stored = await session_service.get_session(
      app_name="billing", user_id="tenant:victim", session_id="fixed"
  )
  assert stored is not None
  assert stored.events == []
  assert stored.state == {"secret": "victim-only"}
  assert await fake_redis.get(attacker_key) == victim.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_append_event_to_unknown_session_writes_nothing(
    session_service, fake_redis, partial
):
  """A rejected append leaves no session, app state or user state behind."""
  session = Session(app_name="app1", user_id="u1", id="never_created")
  event = Event(
      author="agent",
      partial=partial,
      actions=EventActions(
          state_delta={"count": 1, "user:score": 100, "app:status": "active"}
      ),
  )

  with pytest.raises(SessionNotFoundError):
    await session_service.append_event(session, event)

  assert fake_redis._store == {}
  assert session.events == []


@pytest.mark.asyncio
async def test_get_session_with_event_filter(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="user1",
  )

  for i in range(5):
    event = Event(author=f"user_{i}")
    await session_service.append_event(session, event)

  config = GetSessionConfig(num_recent_events=2)
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="user1",
      session_id=session.id,
      config=config,
  )

  assert fetched is not None
  assert len(fetched.events) == 2
  assert fetched.events[-1].author == "user_4"


@pytest.mark.asyncio
async def test_get_session_with_num_recent_events_zero(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="user1",
  )

  for i in range(5):
    event = Event(author=f"user_{i}")
    await session_service.append_event(session, event)

  config = GetSessionConfig(num_recent_events=0)
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="user1",
      session_id=session.id,
      config=config,
  )

  assert fetched is not None
  assert fetched.events == []


@pytest.mark.asyncio
async def test_get_session_with_after_timestamp(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="user1",
  )

  for i in range(5):
    event = Event(author=f"user_{i}", timestamp=float(100 + i))
    await session_service.append_event(session, event)

  config = GetSessionConfig(after_timestamp=103.0)
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="user1",
      session_id=session.id,
      config=config,
  )

  assert fetched is not None
  assert len(fetched.events) == 2
  assert [e.author for e in fetched.events] == ["user_3", "user_4"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, start",
    [
        (None, 0),
        (GetSessionConfig(num_recent_events=2), 2),
        (GetSessionConfig(num_recent_events=0), 4),
        (GetSessionConfig(after_timestamp=102), 2),
        (GetSessionConfig(num_recent_events=3, after_timestamp=102), 2),
    ],
)
async def test_append_event_preserves_filtered_history(
    session_service, config, start
):
  """A filtered view can grow without removing stored events."""
  session = await session_service.create_session(
      app_name="app1", user_id="u1", state={"original": "value"}
  )
  events = [
      Event(
          author="user",
          invocation_id=f"invocation-{i}",
          timestamp=100.0 + i,
          content=types.Content(parts=[types.Part(text=f"event-{i}")]),
      )
      for i in range(6)
  ]
  for event in events[:4]:
    await session_service.append_event(session, event)
  view = await session_service.get_session(
      app_name="app1", user_id="u1", session_id=session.id, config=config
  )
  assert view.events == events[start:4]
  events[4].actions.state_delta = {
      "added": 1,
      "app:mode": "test",
      "user:theme": "dark",
      "temp:scratch": "local",
  }

  for event in events[4:]:
    assert await session_service.append_event(view, event) is event

  stored = await session_service.get_session(
      app_name="app1", user_id="u1", session_id=session.id
  )
  assert stored.events == events
  assert view.events == events[start:]
  assert stored.state == {
      "original": "value",
      "added": 1,
      "app:mode": "test",
      "user:theme": "dark",
  }
  assert view.state["temp:scratch"] == "local"
  assert "temp:scratch" not in stored.events[4].actions.state_delta


@pytest.mark.asyncio
async def test_partial_event_preserves_filtered_history(session_service):
  session = await session_service.create_session(app_name="app1", user_id="u1")
  old_event = Event(author="user", invocation_id="old")
  await session_service.append_event(session, old_event)
  view = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id=session.id,
      config=GetSessionConfig(num_recent_events=0),
  )
  partial = Event(author="agent", invocation_id="partial", partial=True)

  assert await session_service.append_event(view, partial) is partial

  stored = await session_service.get_session(
      app_name="app1", user_id="u1", session_id=session.id
  )
  assert stored.events == [old_event]
  assert view.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["append", "delete", "expire"])
async def test_append_event_retries_storage_changes(
    session_service, fake_redis, monkeypatch, change
):
  """A concurrent append is retained; missing storage keeps recreation behavior."""
  session = await session_service.create_session(app_name="app1", user_id="u1")
  old_event = Event(author="user", invocation_id="old")
  await session_service.append_event(session, old_event)
  view = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id=session.id,
      config=GetSessionConfig(num_recent_events=0),
  )
  other_event = Event(author="agent", invocation_id="other")
  new_event = Event(author="user", invocation_id="new")
  key = session_service._session_key("app1", "u1", session.id)
  original_get = fake_redis.get
  changed = False

  async def get_with_concurrent_change(read_key):
    nonlocal changed
    raw = await original_get(read_key)
    if read_key == key and not changed:
      changed = True
      if change == "append":
        await session_service.append_event(session, other_event)
      elif change == "delete":
        await session_service.delete_session(
            app_name="app1", user_id="u1", session_id=session.id
        )
      else:
        fake_redis.advance_time(3601)
    return raw

  monkeypatch.setattr(fake_redis, "get", get_with_concurrent_change)
  await session_service.append_event(view, new_event)

  stored = await session_service.get_session(
      app_name="app1", user_id="u1", session_id=session.id
  )
  expected = [old_event, other_event] if change == "append" else []
  assert stored.events == expected + [new_event]
  assert view.events == [new_event]
  fake_redis.advance_time(3599)
  assert await original_get(key) is not None
  fake_redis.advance_time(2)
  assert await original_get(key) is None


@pytest.mark.asyncio
async def test_append_event_preserves_concurrent_state_delta(session_service):
  """State committed between get_session and append_event is preserved."""
  session = await session_service.create_session(
      app_name="app1", user_id="u1", state={"initial": "v1"}
  )
  view = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id=session.id,
      config=GetSessionConfig(num_recent_events=0),
  )
  intervening_event = Event(
      author="writer2",
      invocation_id="inv-2",
      actions=EventActions(state_delta={"writer2_key": "v2"}),
  )
  await session_service.append_event(session, intervening_event)

  new_event = Event(
      author="writer1",
      invocation_id="inv-1",
      actions=EventActions(state_delta={"writer1_key": "v3"}),
  )
  await session_service.append_event(view, new_event)

  stored = await session_service.get_session(
      app_name="app1", user_id="u1", session_id=session.id
  )
  assert stored.events == [intervening_event, new_event]
  assert stored.state == {
      "initial": "v1",
      "writer2_key": "v2",
      "writer1_key": "v3",
  }


@pytest.mark.asyncio
async def test_list_sessions(session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s2",
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u2",
      session_id="s3",
  )

  resp_u1 = await session_service.list_sessions(app_name="app1", user_id="u1")
  session_ids_u1 = {s.id for s in resp_u1.sessions}
  assert session_ids_u1 == {"s1", "s2"}

  resp_all = await session_service.list_sessions(app_name="app1")
  session_ids_all = {s.id for s in resp_all.sessions}
  assert session_ids_all == {"s1", "s2", "s3"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id, expected",
    [
        ("u1", [("u1", "s1"), ("u1", "s2")]),
        (None, [("u2", "s0"), ("u1", "s1"), ("u1", "s2"), ("u2", "s1")]),
    ],
)
async def test_list_sessions_ordered_by_activity_with_stable_ties(
    session_service, user_id, expected
):
  """Sessions are oldest first, with ties ordered by user and session id."""
  with mock.patch(
      "google.adk.integrations.redis._redis_session_service.time"
  ) as clock:
    for owner, session_id, timestamp in (
        ("u2", "s1", 20.0),
        ("u1", "s2", 20.0),
        ("u1", "s1", 20.0),
        ("u2", "s0", 10.0),
    ):
      clock.time.return_value = timestamp
      await session_service.create_session(
          app_name="app1", user_id=owner, session_id=session_id
      )

  response = await session_service.list_sessions(
      app_name="app1", user_id=user_id
  )

  assert [(s.user_id, s.id) for s in response.sessions] == expected


@pytest.mark.asyncio
async def test_list_sessions_glob_metacharacters_match_literally(
    session_service, fake_redis
):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u2",
      session_id="s2",
  )

  for app_name, user_id, expected_pattern in (
      ("app1", "*", "test:session:v2:s2_YXBwMQ.Kg.*"),
      ("app1", "u?", "test:session:v2:s2_YXBwMQ.dT8.*"),
      ("app1", "[u]1", "test:session:v2:s2_YXBwMQ.W3VdMQ.*"),
      ("app1", "u\\1", "test:session:v2:s2_YXBwMQ.dVwx.*"),
      ("*", "u1", "test:session:v2:s2_Kg.dTE.*"),
  ):
    fake_redis.scan_patterns.clear()
    resp = await session_service.list_sessions(
        app_name=app_name, user_id=user_id
    )
    assert fake_redis.scan_patterns == [expected_pattern], (app_name, user_id)
    assert resp.sessions == [], (app_name, user_id)

  fake_redis.scan_patterns.clear()
  resp = await session_service.list_sessions(app_name="*")
  assert fake_redis.scan_patterns == ["test:session:v2:s2_Kg.*"]
  assert resp.sessions == []


@pytest.mark.asyncio
async def test_list_sessions_empty_user_id_is_a_filter_not_a_wildcard(
    session_service,
):
  await session_service.create_session(
      app_name="app1",
      user_id="",
      session_id="s1",
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s2",
  )

  resp = await session_service.list_sessions(app_name="app1", user_id="")
  assert [s.id for s in resp.sessions] == ["s1"]


@pytest.mark.asyncio
async def test_list_sessions_excludes_user_ids_sharing_a_prefix(
    session_service,
):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u1:sub",
      session_id="s2",
  )

  resp = await session_service.list_sessions(app_name="app1", user_id="u1")
  assert [s.id for s in resp.sessions] == ["s1"]


@pytest.mark.asyncio
async def test_delete_session(session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="to_delete",
  )

  await session_service.delete_session(
      app_name="app1",
      user_id="u1",
      session_id="to_delete",
  )

  fetched = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id="to_delete",
  )
  assert fetched is None


@pytest.mark.asyncio
async def test_get_user_state(session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      state={"user:theme": "dark", "user:locale": "en"},
  )

  user_state = await session_service.get_user_state(
      app_name="app1",
      user_id="u1",
  )
  assert user_state == {"theme": "dark", "locale": "en"}


@pytest.mark.asyncio
async def test_append_event_and_state_delta(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="u1",
  )

  event = Event(
      author="agent",
      actions=EventActions(
          state_delta={
              "count": 1,
              "user:score": 100,
              "app:status": "active",
          }
      ),
  )

  await session_service.append_event(session, event)

  fetched = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id=session.id,
  )

  assert fetched is not None
  assert len(fetched.events) == 1
  assert fetched.state["count"] == 1
  assert fetched.state["user:score"] == 100
  assert fetched.state["app:status"] == "active"


@pytest.mark.asyncio
async def test_append_event_stamps_session_with_event_timestamp(
    session_service,
):
  """The session records when the event happened, not when it was appended.

  `last_update_time` is what `list_sessions` orders by, so stamping it with the
  wall clock makes an event that is replayed, re-delivered or imported push a
  session forward to its append time instead of its own. Every other backend
  stores `event.timestamp`; the shared contract test asserts the same.
  """
  session = await session_service.create_session(
      app_name="app1",
      user_id="u1",
  )

  event_timestamp = session.last_update_time + 10
  event = Event(
      author="agent",
      invocation_id="inv1",
      timestamp=event_timestamp,
  )

  # Pin the wall clock far from the event's own timestamp so the current
  # implementation cannot agree with the expected value by coincidence.
  with mock.patch(
      "google.adk.integrations.redis._redis_session_service.time"
  ) as clock:
    clock.time.return_value = event_timestamp + 100
    await session_service.append_event(session, event)

  assert session.last_update_time == pytest.approx(event_timestamp, abs=1e-6)

  fetched = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id=session.id,
  )
  assert fetched is not None
  assert fetched.last_update_time == pytest.approx(event_timestamp, abs=1e-6)


@pytest.mark.asyncio
async def test_app_and_user_state_ttl(fake_redis, session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"user:pref": "dark", "app:name": "demo"},
  )

  session_key = session_service._session_key("app1", "u1", "s1")
  app_key = session_service._app_state_key("app1")
  user_key = session_service._user_state_key("app1", "u1")
  assert fake_redis._ex_store[session_key] == 3600
  assert fake_redis._ex_store[app_key] == 3600
  assert fake_redis._ex_store[user_key] == 3600


@pytest.mark.asyncio
async def test_session_ttl_expired(fake_redis, session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"user:pref": "dark", "app:name": "demo", "key1": "val1"},
  )

  # Verify session exists before expiration
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  assert fetched is not None

  # Advance time past TTL (3600 seconds)
  fake_redis.advance_time(3601)

  # Session should now be expired
  assert (
      await session_service.get_session(
          app_name="app1",
          user_id="u1",
          session_id="s1",
      )
      is None
  )

  # User state should also be expired
  assert (
      await session_service.get_user_state(
          app_name="app1",
          user_id="u1",
      )
      == {}
  )

  # list_sessions should return empty
  resp = await session_service.list_sessions(app_name="app1", user_id="u1")
  assert resp.sessions == []


@pytest.mark.asyncio
async def test_session_storage_only_contains_session_state(
    fake_redis, session_service
):
  session = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={
          "topic": "weather",
          "user:pref": "dark",
          "app:env": "prod",
          "temp:scratch": "temp_value",
      },
  )

  # Check in-memory returned session has all merged and temp keys
  assert session.state["topic"] == "weather"
  assert session.state["user:pref"] == "dark"
  assert session.state["app:env"] == "prod"
  assert session.state["temp:scratch"] == "temp_value"

  # Check what is directly saved in Redis under the session key
  session_key = session_service._session_key("app1", "u1", "s1")
  raw_session = json.loads(fake_redis._store[session_key])
  assert raw_session["state"] == {"topic": "weather"}
  assert "user:pref" not in raw_session["state"]
  assert "app:env" not in raw_session["state"]
  assert "temp:scratch" not in raw_session["state"]


@pytest.mark.asyncio
async def test_dynamic_user_and_app_state_propagation(session_service):
  s1 = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"user:theme": "dark", "s1_key": "val1"},
  )
  s2 = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s2",
      state={"s2_key": "val2"},
  )

  # Both sessions initially see the user state
  assert s1.state["user:theme"] == "dark"
  assert s2.state["user:theme"] == "dark"

  # s2 updates user:theme to "light" via append_event
  event = Event(
      author="agent",
      actions=EventActions(state_delta={"user:theme": "light"}),
  )
  await session_service.append_event(s2, event)

  # Reload s1 via get_session: it should dynamically reflect "light"
  reloaded_s1 = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  assert reloaded_s1 is not None
  assert reloaded_s1.state["user:theme"] == "light"
  assert reloaded_s1.state["s1_key"] == "val1"


@pytest.mark.asyncio
async def test_temp_state_not_persisted(session_service):
  session = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"temp:code": 1234, "persist_me": "yes"},
  )
  assert session.state.get("temp:code") == 1234

  # When re-fetching the session, temp state is gone
  fetched = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  assert fetched is not None
  assert "temp:code" not in fetched.state
  assert fetched.state["persist_me"] == "yes"


@pytest.mark.asyncio
async def test_list_sessions_state_merging(session_service):
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"user:lang": "en", "app:mode": "fast", "s1": 1},
  )
  await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s2",
      state={"s2": 2},
  )

  resp = await session_service.list_sessions(app_name="app1", user_id="u1")
  assert len(resp.sessions) == 2
  for s in resp.sessions:
    assert s.state["user:lang"] == "en"
    assert s.state["app:mode"] == "fast"


@pytest.mark.asyncio
async def test_cumulative_user_state_creation(session_service):
  s1 = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
      state={"user:theme": "dark", "s1_key": "val1"},
  )
  assert s1.state["user:theme"] == "dark"
  assert "user:lang" not in s1.state

  # Create s2 for the same user with an additional user state key
  s2 = await session_service.create_session(
      app_name="app1",
      user_id="u1",
      session_id="s2",
      state={"user:lang": "en", "s2_key": "val2"},
  )

  # s2 should see both user states (cumulative) and only its own session state
  assert s2.state["user:theme"] == "dark"
  assert s2.state["user:lang"] == "en"
  assert s2.state["s2_key"] == "val2"
  assert "s1_key" not in s2.state

  # Re-fetching s1 should now dynamically include both cumulative user states
  reloaded_s1 = await session_service.get_session(
      app_name="app1",
      user_id="u1",
      session_id="s1",
  )
  assert reloaded_s1 is not None
  assert reloaded_s1.state["user:theme"] == "dark"
  assert reloaded_s1.state["user:lang"] == "en"
  assert reloaded_s1.state["s1_key"] == "val1"
  assert "s2_key" not in reloaded_s1.state

  # get_user_state should return the cumulative user state
  user_state = await session_service.get_user_state(
      app_name="app1",
      user_id="u1",
  )
  assert user_state == {"theme": "dark", "lang": "en"}


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_append_event_to_unknown_session_writes_nothing(
    session_service, fake_redis, partial
):
  """A rejected append leaves no session, app state or user state behind."""
  session = Session(app_name="app1", user_id="u1", id="never_created")
  event = Event(
      author="agent",
      partial=partial,
      actions=EventActions(
          state_delta={"count": 1, "user:score": 100, "app:status": "active"}
      ),
  )

  with pytest.raises(SessionNotFoundError):
    await session_service.append_event(session, event)

  assert fake_redis._store == {}
  assert session.events == []
