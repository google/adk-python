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

"""Redis-backed session service implementation for ADK."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from typing import Any
from typing import cast
from typing import Optional

from ...errors.already_exists_error import AlreadyExistsError
from ...events.event import Event
from ...platform import uuid as platform_uuid
from ...sessions import _session_util
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from ...sessions.state import State
from ._config import RedisSessionServiceConfig

try:
  import redis.asyncio as redis_asyncio
except ImportError:
  redis_asyncio = None

logger = logging.getLogger("google_adk." + __name__)


class RedisSessionService(BaseSessionService):
  """Redis-backed session service for ADK.

  Stores session objects, events, and state in Redis using direct Redis
  connections (via redis-py asyncio).
  """

  def __init__(
      self,
      config: Optional[RedisSessionServiceConfig] = None,
      redis_client: Optional[Any] = None,
  ):
    """Initializes the RedisSessionService.

    Args:
      config: Optional RedisSessionServiceConfig instance.
      redis_client: Optional pre-configured redis.asyncio.Redis instance.
    """
    self.config = config or RedisSessionServiceConfig()
    self._redis_client = redis_client

  def _get_redis(self) -> Any:
    """Lazily initializes and returns the Redis client."""
    if self._redis_client is not None:
      return self._redis_client

    if redis_asyncio is None:
      raise ImportError(
          "RedisSessionService requires the 'redis' package. "
          "Install it with: pip install google-adk[redis] or pip install redis"
      )

    if self.config.uri:
      self._redis_client = redis_asyncio.from_url(
          self.config.uri, decode_responses=True
      )
    else:
      self._redis_client = redis_asyncio.Redis(
          host=self.config.host or "localhost",
          port=self.config.port or 6379,
          password=self.config.password,
          ssl=self.config.ssl,
          db=self.config.db,
          decode_responses=True,
      )
    return self._redis_client

  def _session_key(self, app_name: str, user_id: str, session_id: str) -> str:
    """Constructs the Redis key for a given session."""
    return f"{self.config.key_prefix}{app_name}:{user_id}:{session_id}"

  def _user_state_key(self, app_name: str, user_id: str) -> str:
    """Constructs the Redis key for user-scoped state."""
    return f"{self.config.key_prefix}user_state:{app_name}:{user_id}"

  def _app_state_key(self, app_name: str) -> str:
    """Constructs the Redis key for app-scoped state."""
    return f"{self.config.key_prefix}app_state:{app_name}"

  @staticmethod
  def _merge_state(
      app_state: Optional[dict[str, Any]],
      user_state: Optional[dict[str, Any]],
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    """Merge app, user, and session states into a single state dictionary."""
    merged_state = copy.deepcopy(session_state)
    for key, value in (app_state or {}).items():
      merged_state[State.APP_PREFIX + key] = value
    for key, value in (user_state or {}).items():
      merged_state[State.USER_PREFIX + key] = value
    return merged_state

  @staticmethod
  def _extract_session_only_state(state: dict[str, Any]) -> dict[str, Any]:
    """Extracts only session-scoped state keys, filtering out app:, user:, and temp:."""
    return {
        k: v
        for k, v in state.items()
        if not k.startswith(State.APP_PREFIX)
        and not k.startswith(State.USER_PREFIX)
        and not k.startswith(State.TEMP_PREFIX)
    }

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session in Redis.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      state: The initial state of the session.
      session_id: Optional client-provided session ID.

    Returns:
      The newly created Session instance.

    Raises:
      AlreadyExistsError: If a session with session_id already exists.
    """
    client = self._get_redis()
    sid = session_id or platform_uuid.new_uuid()
    key = self._session_key(app_name, user_id, sid)

    initial_state = state or {}
    state_deltas = _session_util.extract_state_delta(initial_state)
    app_state_delta = state_deltas.get("app", {})
    user_state_delta = state_deltas.get("user", {})
    session_state = state_deltas.get("session", {})

    app_key = self._app_state_key(app_name)
    user_key = self._user_state_key(app_name, user_id)

    app_raw, user_raw = await asyncio.gather(
        client.get(app_key),
        client.get(user_key),
    )
    current_app = json.loads(app_raw) if app_raw else {}
    current_user = json.loads(user_raw) if user_raw else {}

    if app_state_delta:
      current_app.update(app_state_delta)
      await client.set(
          app_key,
          json.dumps(current_app),
          ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
      )

    if user_state_delta:
      current_user.update(user_state_delta)
      await client.set(
          user_key,
          json.dumps(current_user),
          ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
      )

    now = time.time()
    # Save ONLY session-scoped state to Redis storage
    storage_session = Session(
        id=sid,
        app_name=app_name,
        user_id=user_id,
        state=session_state,
        events=[],
        last_update_time=now,
    )

    created = await client.set(
        key,
        storage_session.model_dump_json(),
        nx=True,
        ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
    )
    if not created:
      raise AlreadyExistsError(
          f"Session {sid} already exists for user {user_id} in app {app_name}."
      )

    # Return session with dynamically merged state (including app/user/temp state)
    merged_state = self._merge_state(current_app, current_user, session_state)
    for k, v in initial_state.items():
      if k.startswith(State.TEMP_PREFIX):
        merged_state[k] = v

    return Session(
        id=sid,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
        events=[],
        last_update_time=now,
    )

  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Retrieves a session from Redis."""
    client = self._get_redis()
    key = self._session_key(app_name, user_id, session_id)
    app_key = self._app_state_key(app_name)
    user_key = self._user_state_key(app_name, user_id)

    raw, app_raw, user_raw = await asyncio.gather(
        client.get(key),
        client.get(app_key),
        client.get(user_key),
    )
    if not raw:
      return None

    session = Session.model_validate_json(raw)
    app_state = json.loads(app_raw) if app_raw else {}
    user_state = json.loads(user_raw) if user_raw else {}
    session.state = self._merge_state(app_state, user_state, session.state)

    if config:
      if config.num_recent_events is not None:
        session.events = (
            session.events[-config.num_recent_events :]
            if config.num_recent_events
            else []
        )
      if config.after_timestamp is not None:
        session.events = [
            e for e in session.events if e.timestamp >= config.after_timestamp
        ]
    return session

  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists sessions matching the given app_name and optional user_id."""
    client = self._get_redis()
    pattern = (
        f"{self.config.key_prefix}{app_name}:{user_id}:*"
        if user_id
        else f"{self.config.key_prefix}{app_name}:*"
    )
    app_raw = await client.get(self._app_state_key(app_name))
    app_state = json.loads(app_raw) if app_raw else {}

    user_states_map: dict[str, dict[str, Any]] = {}
    if user_id:
      user_raw = await client.get(self._user_state_key(app_name, user_id))
      if user_raw:
        user_states_map[user_id] = json.loads(user_raw)

    sessions: list[Session] = []
    async for key in client.scan_iter(match=pattern):
      raw = await client.get(key)
      if raw:
        try:
          s = Session.model_validate_json(raw)
          u_id = s.user_id
          if u_id not in user_states_map:
            u_raw = await client.get(self._user_state_key(app_name, u_id))
            user_states_map[u_id] = json.loads(u_raw) if u_raw else {}
          s.state = self._merge_state(
              app_state, user_states_map.get(u_id, {}), s.state
          )
          sessions.append(s)
        except Exception as e:
          logger.warning("Failed to parse session at key %s: %s", key, e)

    # Sort descending by last_update_time
    sessions.sort(key=lambda s: s.last_update_time, reverse=True)
    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session from Redis."""
    client = self._get_redis()
    key = self._session_key(app_name, user_id, session_id)
    await client.delete(key)

  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    """Returns the user-scoped state for the given app and user."""
    client = self._get_redis()
    user_key = self._user_state_key(app_name, user_id)
    raw = await client.get(user_key)
    if not raw:
      return {}
    return cast(dict[str, Any], json.loads(raw))

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to the session and synchronizes state in Redis."""
    client = self._get_redis()
    event = await super().append_event(session, event)
    session.last_update_time = time.time()

    # Sync app and user state deltas to their respective keys
    if event.actions and event.actions.state_delta:
      deltas = _session_util.extract_state_delta(event.actions.state_delta)
      app_delta = deltas.get("app", {})
      user_delta = deltas.get("user", {})

      if app_delta:
        app_key = self._app_state_key(session.app_name)
        app_raw = await client.get(app_key)
        app_state = json.loads(app_raw) if app_raw else {}
        app_state.update(app_delta)
        await client.set(
            app_key,
            json.dumps(app_state),
            ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
        )

      if user_delta:
        user_key = self._user_state_key(session.app_name, session.user_id)
        user_raw = await client.get(user_key)
        user_state = json.loads(user_raw) if user_raw else {}
        user_state.update(user_delta)
        await client.set(
            user_key,
            json.dumps(user_state),
            ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
        )

    # Prepare storage copy with ONLY session-scoped state (excluding app:, user:, temp:)
    session_only_state = self._extract_session_only_state(session.state)
    storage_session = Session(
        id=session.id,
        app_name=session.app_name,
        user_id=session.user_id,
        state=session_only_state,
        events=session.events,
        last_update_time=session.last_update_time,
    )

    key = self._session_key(session.app_name, session.user_id, session.id)
    await client.set(
        key,
        storage_session.model_dump_json(),
        ex=self.config.ttl_seconds if self.config.ttl_seconds > 0 else None,
    )
    return event
