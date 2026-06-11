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

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from redis.asyncio import Redis

from ...errors.already_exists_error import AlreadyExistsError
from ...events.event import Event
from ...platform import time as platform_time
from ...platform import uuid as platform_uuid
from ...sessions import _session_util
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from ...sessions.state import State

logger = logging.getLogger("google_adk." + __name__)

_SessionLockKey = tuple[str, str, str]

DEFAULT_KEY_PREFIX = "adk:session"


class RedisSessionService(BaseSessionService):
  """Session service backed by Redis.

  Layout of keys (all under ``key_prefix``, defaults to ``adk:session``):

  - ``{prefix}:session:{app}:{user}:{session_id}`` — Hash holding session
    fields (``state`` as JSON, ``events`` as JSON list, ``last_update_time``,
    ``revision``). Optional TTL.
  - ``{prefix}:sessions_index:{app}:{user}`` — Set of session IDs for the
    user, used for fast listing.
  - ``{prefix}:app_state:{app}`` — Hash of app-scoped state (values JSON).
  - ``{prefix}:user_state:{app}:{user}`` — Hash of user-scoped state
    (values JSON).

  Concurrency: ``append_event`` uses an optimistic revision check combined
  with a ``WATCH``/``MULTI``/``EXEC`` transaction so concurrent writers from
  different processes are detected. An in-process lock additionally
  serializes appends from the same event loop.
  """

  def __init__(
      self,
      *,
      redis_url: Optional[str] = None,
      client: Optional[Redis] = None,
      key_prefix: str = DEFAULT_KEY_PREFIX,
      session_ttl_seconds: Optional[int] = None,
  ):
    """Initializes the Redis session service.

    Args:
      redis_url: Connection URL (e.g. ``redis://localhost:6379/0``). Ignored
        when ``client`` is supplied.
      client: Pre-built ``redis.asyncio.Redis`` client. Mutually exclusive
        with ``redis_url``.
      key_prefix: Prefix used for all Redis keys this service writes.
      session_ttl_seconds: Optional TTL applied to session-scoped keys on
        every write. ``None`` disables expiration.
    """
    if client is None and not redis_url:
      raise ValueError("Either 'client' or 'redis_url' must be provided.")

    if client is None:
      try:
        from redis.asyncio import Redis as _Redis
      except ImportError as e:
        raise ImportError(
            "RedisSessionService requires the 'redis' package. "
            "Install it with: pip install google-adk[redis]"
        ) from e
      client = _Redis.from_url(redis_url)

    self._client: Redis = client
    self._key_prefix = key_prefix
    self._session_ttl_seconds = session_ttl_seconds

    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

  # ---------------------------------------------------------------------------
  # Key helpers
  # ---------------------------------------------------------------------------

  def _session_key(self, app_name: str, user_id: str, session_id: str) -> str:
    return f"{self._key_prefix}:session:{app_name}:{user_id}:{session_id}"

  def _sessions_index_key(self, app_name: str, user_id: str) -> str:
    return f"{self._key_prefix}:sessions_index:{app_name}:{user_id}"

  def _app_state_key(self, app_name: str) -> str:
    return f"{self._key_prefix}:app_state:{app_name}"

  def _user_state_key(self, app_name: str, user_id: str) -> str:
    return f"{self._key_prefix}:user_state:{app_name}:{user_id}"

  def _users_index_key(self, app_name: str) -> str:
    return f"{self._key_prefix}:users_index:{app_name}"

  # ---------------------------------------------------------------------------
  # Concurrency helpers
  # ---------------------------------------------------------------------------

  @asynccontextmanager
  async def _with_session_lock(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> AsyncIterator[None]:
    """Serializes event appends for the same session within this process."""
    lock_key = (app_name, user_id, session_id)
    async with self._session_locks_guard:
      lock = self._session_locks.get(lock_key) or asyncio.Lock()
      self._session_locks[lock_key] = lock
      self._session_lock_ref_count[lock_key] = (
          self._session_lock_ref_count.get(lock_key, 0) + 1
      )

    try:
      async with lock:
        yield
    finally:
      async with self._session_locks_guard:
        remaining = self._session_lock_ref_count.get(lock_key, 0) - 1
        if remaining <= 0 and not lock.locked():
          self._session_lock_ref_count.pop(lock_key, None)
          self._session_locks.pop(lock_key, None)
        else:
          self._session_lock_ref_count[lock_key] = remaining

  # ---------------------------------------------------------------------------
  # State helpers
  # ---------------------------------------------------------------------------

  @staticmethod
  def _decode(value: Any) -> str:
    if isinstance(value, bytes):
      return value.decode("utf-8")
    return value

  async def _read_app_state(self, app_name: str) -> dict[str, Any]:
    raw = await self._client.hgetall(self._app_state_key(app_name))
    return {self._decode(k): json.loads(self._decode(v)) for k, v in raw.items()}

  async def _read_user_state(
      self, app_name: str, user_id: str
  ) -> dict[str, Any]:
    raw = await self._client.hgetall(self._user_state_key(app_name, user_id))
    return {self._decode(k): json.loads(self._decode(v)) for k, v in raw.items()}

  @staticmethod
  def _merge_state(
      app_state: dict[str, Any],
      user_state: dict[str, Any],
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    merged: dict[str, Any] = dict(session_state)
    for k, v in app_state.items():
      merged[State.APP_PREFIX + k] = v
    for k, v in user_state.items():
      merged[State.USER_PREFIX + k] = v
    return merged

  @staticmethod
  def _serialize_session_state(state: dict[str, Any]) -> dict[str, Any]:
    """Strips app/user/temp-prefixed keys to obtain session-only state."""
    return {
        k: v
        for k, v in state.items()
        if not k.startswith(State.APP_PREFIX)
        and not k.startswith(State.USER_PREFIX)
        and not k.startswith(State.TEMP_PREFIX)
    }

  # ---------------------------------------------------------------------------
  # Serialization
  # ---------------------------------------------------------------------------

  @staticmethod
  def _dump_session(
      session_state: dict[str, Any],
      events: list[Event],
      last_update_time: float,
      revision: int,
  ) -> dict[str, str]:
    return {
        "state": json.dumps(session_state),
        "events": json.dumps(
            [e.model_dump(mode="json", exclude_none=True) for e in events]
        ),
        "last_update_time": str(last_update_time),
        "revision": str(revision),
    }

  def _load_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      raw: dict[Any, Any],
  ) -> tuple[Session, int]:
    decoded = {self._decode(k): self._decode(v) for k, v in raw.items()}
    session_state = json.loads(decoded.get("state", "{}"))
    events_data = json.loads(decoded.get("events", "[]"))
    try:
      last_update_time = float(decoded.get("last_update_time", "0") or 0)
    except ValueError:
      last_update_time = 0.0
    try:
      revision = int(decoded.get("revision", "0") or 0)
    except ValueError:
      revision = 0
    events = [Event.model_validate(e) for e in events_data]
    session = Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=session_state,
        events=events,
        last_update_time=last_update_time,
    )
    return session, revision

  async def _apply_ttl(self, *keys: str) -> None:
    if not self._session_ttl_seconds:
      return
    for key in keys:
      await self._client.expire(key, self._session_ttl_seconds)

  # ---------------------------------------------------------------------------
  # BaseSessionService API
  # ---------------------------------------------------------------------------

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else platform_uuid.new_uuid()
    )

    session_key = self._session_key(app_name, user_id, session_id)
    if await self._client.exists(session_key):
      raise AlreadyExistsError(f"Session with id {session_id} already exists.")

    state_deltas = _session_util.extract_state_delta(state or {})
    app_state_delta = state_deltas["app"]
    user_state_delta = state_deltas["user"]
    session_state = state_deltas["session"]

    if app_state_delta:
      await self._client.hset(
          self._app_state_key(app_name),
          mapping={k: json.dumps(v) for k, v in app_state_delta.items()},
      )
    if user_state_delta:
      await self._client.hset(
          self._user_state_key(app_name, user_id),
          mapping={k: json.dumps(v) for k, v in user_state_delta.items()},
      )

    now = platform_time.get_time()
    revision = 1
    await self._client.hset(
        session_key,
        mapping=self._dump_session(session_state, [], now, revision),
    )
    await self._client.sadd(
        self._sessions_index_key(app_name, user_id), session_id
    )
    await self._client.sadd(self._users_index_key(app_name), user_id)
    await self._apply_ttl(
        session_key, self._sessions_index_key(app_name, user_id)
    )

    app_state = await self._read_app_state(app_name)
    user_state = await self._read_user_state(app_name, user_id)
    merged = self._merge_state(app_state, user_state, session_state)

    session = Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=merged,
        events=[],
        last_update_time=now,
    )
    session._storage_update_marker = str(revision)
    return session

  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    session_key = self._session_key(app_name, user_id, session_id)
    raw = await self._client.hgetall(session_key)
    if not raw:
      return None

    session, revision = self._load_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        raw=raw,
    )

    if config:
      if config.num_recent_events is not None:
        if config.num_recent_events == 0:
          session.events = []
        else:
          session.events = session.events[-config.num_recent_events :]
      if config.after_timestamp:
        i = len(session.events) - 1
        while i >= 0:
          if session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          session.events = session.events[i + 1 :]

    app_state = await self._read_app_state(app_name)
    user_state = await self._read_user_state(app_name, user_id)
    session.state = self._merge_state(app_state, user_state, session.state)
    session._storage_update_marker = str(revision) if revision > 0 else None
    return session

  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    if user_id is not None:
      user_ids = [user_id]
    else:
      raw_users = await self._client.smembers(self._users_index_key(app_name))
      user_ids = [self._decode(u) for u in raw_users]

    app_state = await self._read_app_state(app_name)
    sessions: list[Session] = []

    for uid in user_ids:
      raw_ids = await self._client.smembers(
          self._sessions_index_key(app_name, uid)
      )
      session_ids = [self._decode(sid) for sid in raw_ids]
      user_state = await self._read_user_state(app_name, uid)
      for sid in session_ids:
        raw = await self._client.hgetall(self._session_key(app_name, uid, sid))
        if not raw:
          # Index entry without backing hash (e.g. expired session). Skip.
          continue
        session, _ = self._load_session(
            app_name=app_name,
            user_id=uid,
            session_id=sid,
            raw=raw,
        )
        session.events = []
        session.state = self._merge_state(app_state, user_state, session.state)
        sessions.append(session)

    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    session_key = self._session_key(app_name, user_id, session_id)
    pipeline = self._client.pipeline()
    pipeline.delete(session_key)
    pipeline.srem(self._sessions_index_key(app_name, user_id), session_id)
    await pipeline.execute()

  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    return await self._read_user_state(app_name, user_id)

  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)

    state_delta = (
        event.actions.state_delta
        if event.actions and event.actions.state_delta
        else {}
    )
    state_deltas = _session_util.extract_state_delta(state_delta)
    app_updates = state_deltas["app"]
    user_updates = state_deltas["user"]
    session_updates = state_deltas["session"]

    session_key = self._session_key(
        session.app_name, session.user_id, session.id
    )
    app_state_key = self._app_state_key(session.app_name)
    user_state_key = self._user_state_key(session.app_name, session.user_id)

    async with self._with_session_lock(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    ):
      async with self._client.pipeline() as pipe:
        while True:
          await pipe.watch(session_key)
          raw = await pipe.hgetall(session_key)
          if not raw:
            await pipe.unwatch()
            raise ValueError(f"Session {session.id} not found.")

          decoded = {self._decode(k): self._decode(v) for k, v in raw.items()}
          try:
            current_revision = int(decoded.get("revision", "0") or 0)
          except ValueError:
            current_revision = 0

          if (
              session._storage_update_marker is not None
              and session._storage_update_marker != str(current_revision)
          ):
            await pipe.unwatch()
            raise ValueError(
                "The session has been modified in storage since it was loaded."
                " Please reload the session before appending more events."
            )

          stored_events_data = json.loads(decoded.get("events", "[]"))
          stored_session_state = json.loads(decoded.get("state", "{}"))

          stored_session_state.update(session_updates)
          for k, v in session_updates.items():
            session.state[k] = v

          stored_events_data.append(
              event.model_dump(mode="json", exclude_none=True)
          )
          session.last_update_time = event.timestamp
          new_revision = current_revision + 1
          session_only_state = self._serialize_session_state(
              stored_session_state
          )

          pipe.multi()
          pipe.hset(
              session_key,
              mapping={
                  "state": json.dumps(session_only_state),
                  "events": json.dumps(stored_events_data),
                  "last_update_time": str(session.last_update_time),
                  "revision": str(new_revision),
              },
          )
          if app_updates:
            pipe.hset(
                app_state_key,
                mapping={k: json.dumps(v) for k, v in app_updates.items()},
            )
          if user_updates:
            pipe.hset(
                user_state_key,
                mapping={k: json.dumps(v) for k, v in user_updates.items()},
            )
          if self._session_ttl_seconds:
            pipe.expire(session_key, self._session_ttl_seconds)
            pipe.expire(
                self._sessions_index_key(session.app_name, session.user_id),
                self._session_ttl_seconds,
            )

          try:
            await pipe.execute()
          except _watch_error_class():
            # Lost the race with another writer — retry the WATCH loop.
            continue
          session._storage_update_marker = str(new_revision)
          break

    await super().append_event(session, event)
    return event

  async def close(self) -> None:
    """Closes the underlying Redis client."""
    aclose = getattr(self._client, "aclose", None)
    if aclose is not None:
      await aclose()
    else:
      close = getattr(self._client, "close", None)
      if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
          await result


def _watch_error_class() -> type[Exception]:
  """Returns the redis WatchError class, lazily imported."""
  try:
    from redis.exceptions import WatchError

    return WatchError
  except ImportError:
    return Exception
