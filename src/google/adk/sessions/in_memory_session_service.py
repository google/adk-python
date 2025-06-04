# Copyright 2025 Google LLC
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

import copy
import logging
import time
from typing import Any
from typing import Optional
import uuid

from typing_extensions import override

from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger('google_adk.' + __name__)


class InMemorySessionService(BaseSessionService):
  """An in-memory implementation of the session service."""

  def __init__(self):
    # A map from app name to a map from user ID to a map from session ID to
    # session.
    self.sessions: dict[str, dict[str, dict[str, Session]]] = {}
    # A map from app name to a map from user ID to a map from key to the value.
    self.user_state: dict[str, dict[str, dict[str, Any]]] = {}
    # A map from app name to a map from key to the value.
    self.app_state: dict[str, dict[str, Any]] = {}

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    return self._create_session_impl(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )

  def create_session_sync(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    logger.warning('Deprecated. Please migrate to the async method.')
    return self._create_session_impl(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )

  def _create_session_impl(
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

    if app_name not in self.sessions:
      self.sessions[app_name] = {}
    if user_id not in self.sessions[app_name]:
      self.sessions[app_name][user_id] = {}
    self.sessions[app_name][user_id][session_id] = session

    copied_session = copy.deepcopy(session)
    return self._merge_state(app_name, user_id, copied_session)

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    return self._get_session_impl(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        config=config,
    )

  def get_session_sync(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    logger.warning('Deprecated. Please migrate to the async method.')
    return self._get_session_impl(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        config=config,
    )

  def _get_session_impl(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    if app_name not in self.sessions:
      return None
    if user_id not in self.sessions[app_name]:
      return None
    if session_id not in self.sessions[app_name][user_id]:
      return None

    session = self.sessions[app_name][user_id].get(session_id)
    copied_session = copy.deepcopy(session)

    if config:
      if config.num_recent_events:
        copied_session.events = copied_session.events[
            -config.num_recent_events :
        ]
      if config.after_timestamp:
        i = len(copied_session.events) - 1
        while i >= 0:
          if copied_session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          copied_session.events = copied_session.events[i + 1 :]

    return self._merge_state(app_name, user_id, copied_session)

  def _merge_state(
      self, app_name: str, user_id: str, copied_session: Session
  ) -> Session:
    # Merge app state
    if app_name in self.app_state:
      for key in self.app_state[app_name].keys():
        copied_session.state[State.APP_PREFIX + key] = self.app_state[app_name][
            key
        ]

    if (
        app_name not in self.user_state
        or user_id not in self.user_state[app_name]
    ):
      return copied_session

    # Merge session state with user state.
    for key in self.user_state[app_name][user_id].keys():
      copied_session.state[State.USER_PREFIX + key] = self.user_state[app_name][
          user_id
      ][key]
    return copied_session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    return self._list_sessions_impl(app_name=app_name, user_id=user_id)

  def list_sessions_sync(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    logger.warning('Deprecated. Please migrate to the async method.')
    return self._list_sessions_impl(app_name=app_name, user_id=user_id)

  def _list_sessions_impl(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    empty_response = ListSessionsResponse()
    if app_name not in self.sessions:
      return empty_response
    if user_id not in self.sessions[app_name]:
      return empty_response

    sessions_without_events = []
    for session in self.sessions[app_name][user_id].values():
      copied_session = copy.deepcopy(session)
      copied_session.events = []
      copied_session.state = {}
      sessions_without_events.append(copied_session)
    return ListSessionsResponse(sessions=sessions_without_events)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    self._delete_session_impl(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  def delete_session_sync(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    logger.warning('Deprecated. Please migrate to the async method.')
    self._delete_session_impl(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  def _delete_session_impl(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    if (
        self._get_session_impl(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        is None
    ):
      return

    self.sessions[app_name][user_id].pop(session_id)

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    await super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    # Update the storage session
    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id

    def _warning(message: str) -> None:
      logger.warning(
          f'Failed to append event to session {session_id}: {message}'
      )

    if app_name not in self.sessions:
      _warning(f'app_name {app_name} not in sessions')
      return event
    if user_id not in self.sessions[app_name]:
      _warning(f'user_id {user_id} not in sessions[app_name]')
      return event
    if session_id not in self.sessions[app_name][user_id]:
      _warning(f'session_id {session_id} not in sessions[app_name][user_id]')
      return event

    if event.actions and event.actions.state_delta:
      for key in event.actions.state_delta:
        if key.startswith(State.APP_PREFIX):
          self.app_state.setdefault(app_name, {})[
              key.removeprefix(State.APP_PREFIX)
          ] = event.actions.state_delta[key]

        if key.startswith(State.USER_PREFIX):
          self.user_state.setdefault(app_name, {}).setdefault(user_id, {})[
              key.removeprefix(State.USER_PREFIX)
          ] = event.actions.state_delta[key]

    storage_session = self.sessions[app_name][user_id].get(session_id)
    await super().append_event(session=storage_session, event=event)

    storage_session.last_update_time = event.timestamp

    return event
