import json
import time
from typing import Any
from typing import Optional
import uuid

import redis
from typing_extensions import override

from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

DEFAULT_EXPIRATION = 60 * 60  # 1 hour


class RedisMemorySessionService(BaseSessionService):
  """A Redis-backed implementation of the session service."""

  def __init__(
      self,
      host="localhost",
      port=6379,
      db=0,
      uri=None,
      expire=DEFAULT_EXPIRATION,
  ):
    self.host = host
    self.port = port
    self.db = db
    self.uri = uri
    self.expire = expire

    self.cache = (
        redis.Redis.from_url(uri)
        if uri
        else redis.Redis(host=host, port=port, db=db)
    )

  @override
  def create_session(
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
        else str(uuid.uuid4())
    )
    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=state or {},
        last_update_time=time.time(),
    )

    sessions = self._load_sessions(app_name, user_id)
    sessions[session_id] = session.to_dict()
    self._save_sessions(app_name, user_id, sessions)

    return self._merge_state(app_name, user_id, session)

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    sessions = self._load_sessions(app_name, user_id)
    if session_id not in sessions:
      return None

    session = Session.from_dict(sessions[session_id])
    if config:
      if config.num_recent_events:
        session.events = session.events[-config.num_recent_events :]
      elif config.after_timestamp:
        i = len(session.events) - 1
        while i >= 0:
          if session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          session.events = session.events[i:]

    return self._merge_state(app_name, user_id, session)

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    sessions = self._load_sessions(app_name, user_id)
    session_objs = []

    for session_data in sessions.values():
      session = Session.from_dict(session_data)
      session.events = []
      session.state = {}
      session_objs.append(session)

    return ListSessionsResponse(sessions=session_objs)

  @override
  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    sessions = self._load_sessions(app_name, user_id)
    if session_id in sessions:
      del sessions[session_id]
      self._save_sessions(app_name, user_id, sessions)

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    await super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    if event.actions and event.actions.state_delta:
      for key, value in event.actions.state_delta.items():
        if key.startswith(State.APP_PREFIX):
          self.cache.hset(
              f"{State.APP_PREFIX}{session.app_name}",
              key.removeprefix(State.APP_PREFIX),
              json.dumps(value),
          )
        if key.startswith(State.USER_PREFIX):
          self.cache.hset(
              f"{State.USER_PREFIX}{session.app_name}:{session.user_id}",
              key.removeprefix(State.USER_PREFIX),
              json.dumps(value),
          )

    sessions = self._load_sessions(session.app_name, session.user_id)
    sessions[session.id] = session.to_dict()
    self._save_sessions(session.app_name, session.user_id, sessions)

    return event

  def _merge_state(
      self, app_name: str, user_id: str, session: Session
  ) -> Session:
    app_state = self.cache.hgetall(f"{State.APP_PREFIX}{app_name}")
    for k, v in app_state.items():
      session.state[State.APP_PREFIX + k.decode()] = json.loads(v.decode())

    user_state_key = f"{State.USER_PREFIX}{app_name}:{user_id}"
    user_state = self.cache.hgetall(user_state_key)
    for k, v in user_state.items():
      session.state[State.USER_PREFIX + k.decode()] = json.loads(v.decode())

    return session

  def _load_sessions(self, app_name: str, user_id: str) -> dict[str, dict]:
    key = f"{State.APP_PREFIX}{app_name}:{user_id}"
    raw = self.cache.get(key)
    if not raw:
      return {}
    return json.loads(raw.decode())

  def _save_sessions(
      self, app_name: str, user_id: str, sessions: dict[str, Any]
  ):
    key = f"{State.APP_PREFIX}{app_name}:{user_id}"
    self.cache.set(key, json.dumps(sessions))
    self.cache.expire(key, self.expire)
