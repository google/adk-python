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

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.integrations.redis._config import RedisSessionServiceConfig
from google.adk.integrations.redis._redis_session_service import RedisSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
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
