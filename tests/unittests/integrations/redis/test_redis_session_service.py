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

import time

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.integrations.redis.redis_session_service import RedisSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
import pytest


@pytest.fixture
def fake_redis_client():
  fakeredis = pytest.importorskip("fakeredis")
  return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def service(fake_redis_client):
  return RedisSessionService(client=fake_redis_client)


def _make_event(
    *,
    invocation_id: str = "inv-1",
    author: str = "user",
    state_delta: dict | None = None,
) -> Event:
  return Event(
      invocation_id=invocation_id,
      author=author,
      actions=EventActions(state_delta=state_delta or {}),
  )


@pytest.mark.asyncio
async def test_create_session_assigns_id_and_persists(service):
  session = await service.create_session(
      app_name="app", user_id="user-1"
  )
  assert session.id
  assert session.app_name == "app"
  assert session.user_id == "user-1"
  assert session.state == {}
  assert session.last_update_time > 0

  fetched = await service.get_session(
      app_name="app", user_id="user-1", session_id=session.id
  )
  assert fetched is not None
  assert fetched.id == session.id


@pytest.mark.asyncio
async def test_create_session_with_state_splits_scopes(service):
  initial_state = {
      "app:greeting": "hi",
      "user:nickname": "ada",
      "topic": "math",
      "temp:cache": "drop-me",
  }
  session = await service.create_session(
      app_name="app",
      user_id="user-1",
      state=initial_state,
  )
  # Session-scoped state survives; temp-scoped state is dropped.
  assert session.state["topic"] == "math"
  assert "temp:cache" not in session.state
  # App/user-scoped state is merged into the returned session.
  assert session.state["app:greeting"] == "hi"
  assert session.state["user:nickname"] == "ada"

  # User state is reachable independently of any session.
  user_state = await service.get_user_state(
      app_name="app", user_id="user-1"
  )
  assert user_state == {"nickname": "ada"}


@pytest.mark.asyncio
async def test_create_session_with_existing_id_raises(service):
  await service.create_session(
      app_name="app", user_id="u", session_id="fixed"
  )
  with pytest.raises(AlreadyExistsError):
    await service.create_session(
        app_name="app", user_id="u", session_id="fixed"
    )


@pytest.mark.asyncio
async def test_get_session_returns_none_when_missing(service):
  assert (
      await service.get_session(
          app_name="app", user_id="u", session_id="absent"
      )
      is None
  )


@pytest.mark.asyncio
async def test_append_event_persists_events_and_state(service):
  session = await service.create_session(app_name="app", user_id="u")
  event = _make_event(state_delta={"step": 1, "app:counter": 7})
  event.timestamp = time.time()

  await service.append_event(session, event)

  reloaded = await service.get_session(
      app_name="app", user_id="u", session_id=session.id
  )
  assert reloaded is not None
  assert len(reloaded.events) == 1
  assert reloaded.events[0].invocation_id == "inv-1"
  assert reloaded.state["step"] == 1
  assert reloaded.state["app:counter"] == 7


@pytest.mark.asyncio
async def test_append_event_temp_state_is_not_persisted(service):
  session = await service.create_session(app_name="app", user_id="u")
  event = _make_event(state_delta={"temp:ephemeral": "x"})
  event.timestamp = time.time()

  await service.append_event(session, event)
  # In-memory session may have temp state applied for the current invocation.
  assert session.state.get("temp:ephemeral") == "x"

  reloaded = await service.get_session(
      app_name="app", user_id="u", session_id=session.id
  )
  assert "temp:ephemeral" not in reloaded.state


@pytest.mark.asyncio
async def test_append_event_stale_marker_raises(service):
  session = await service.create_session(app_name="app", user_id="u")
  stale_session = await service.get_session(
      app_name="app", user_id="u", session_id=session.id
  )

  # First append using the original session succeeds and advances revision.
  ev1 = _make_event(state_delta={"v": 1})
  ev1.timestamp = time.time()
  await service.append_event(session, ev1)

  # The stale_session still holds the old revision marker.
  ev2 = _make_event(state_delta={"v": 2})
  ev2.timestamp = time.time()
  with pytest.raises(ValueError, match="modified in storage"):
    await service.append_event(stale_session, ev2)


@pytest.mark.asyncio
async def test_append_event_partial_event_is_skipped(service):
  session = await service.create_session(app_name="app", user_id="u")
  event = Event(invocation_id="inv", author="model", partial=True)
  await service.append_event(session, event)

  reloaded = await service.get_session(
      app_name="app", user_id="u", session_id=session.id
  )
  assert reloaded.events == []


@pytest.mark.asyncio
async def test_get_session_config_filters(service):
  session = await service.create_session(app_name="app", user_id="u")
  for i in range(3):
    e = _make_event(invocation_id=f"inv-{i}", state_delta={"i": i})
    e.timestamp = time.time() + i
    await service.append_event(session, e)

  config = GetSessionConfig(num_recent_events=1)
  trimmed = await service.get_session(
      app_name="app",
      user_id="u",
      session_id=session.id,
      config=config,
  )
  assert len(trimmed.events) == 1
  assert trimmed.events[0].invocation_id == "inv-2"


@pytest.mark.asyncio
async def test_list_sessions_returns_sessions_without_events(service):
  s1 = await service.create_session(app_name="app", user_id="u")
  s2 = await service.create_session(app_name="app", user_id="u")
  ev = _make_event(state_delta={"x": 1})
  ev.timestamp = time.time()
  await service.append_event(s1, ev)

  response = await service.list_sessions(app_name="app", user_id="u")
  ids = {s.id for s in response.sessions}
  assert ids == {s1.id, s2.id}
  for s in response.sessions:
    assert s.events == []


@pytest.mark.asyncio
async def test_list_sessions_across_users(service):
  s1 = await service.create_session(app_name="app", user_id="alice")
  s2 = await service.create_session(app_name="app", user_id="bob")

  response = await service.list_sessions(app_name="app")
  ids = {s.id for s in response.sessions}
  assert ids == {s1.id, s2.id}


@pytest.mark.asyncio
async def test_delete_session_removes_data(service):
  session = await service.create_session(app_name="app", user_id="u")
  await service.delete_session(
      app_name="app", user_id="u", session_id=session.id
  )
  assert (
      await service.get_session(
          app_name="app", user_id="u", session_id=session.id
      )
      is None
  )
  response = await service.list_sessions(app_name="app", user_id="u")
  assert response.sessions == []


@pytest.mark.asyncio
async def test_session_ttl_expires(fake_redis_client):
  service = RedisSessionService(
      client=fake_redis_client, session_ttl_seconds=60
  )
  session = await service.create_session(app_name="app", user_id="u")
  session_key = service._session_key("app", "u", session.id)
  ttl = await fake_redis_client.ttl(session_key)
  assert 0 < ttl <= 60


def test_init_requires_url_or_client():
  with pytest.raises(ValueError, match="redis_url"):
    RedisSessionService()
