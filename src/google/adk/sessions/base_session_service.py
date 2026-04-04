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

import abc
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from .session import Session
from .state import State


class GetSessionConfig(BaseModel):
  """The configuration of getting a session.

  Attributes:
    num_recent_events: The limit of recent events to get for the session.
      Optional: if None, the filter is not applied; if greater than 0, returns
        at most given number of recent events; if 0, no events are returned.
    after_timestamp: The earliest timestamp of events to get for the session.
      Optional: if None, the filter is not applied; otherwise, returns events
        with timestamp >= the given time.
  """

  num_recent_events: Optional[int] = None
  after_timestamp: Optional[float] = None


class ListSessionsResponse(BaseModel):
  """The response of listing sessions.

  The events and states are not set within each Session object.
  """

  sessions: list[Session] = Field(default_factory=list)


class BaseSessionService(abc.ABC):
  """Base class for session services.

  The service provides a set of methods for managing sessions and events.
  """

  @property
  def _secret_state_cache(
      self,
  ) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Process-local cache for secret-scoped state.

    Keyed by (app_name, user_id, session_id).
    Lazily initialized to avoid requiring subclasses to call
    super().__init__().
    """
    try:
      return self.__secret_state_cache
    except AttributeError:
      self.__secret_state_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
      return self.__secret_state_cache

  @abc.abstractmethod
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session.

    Args:
      app_name: the name of the app.
      user_id: the id of the user.
      state: the initial state of the session.
      session_id: the client-provided id of the session. If not provided, a
        generated ID will be used.

    Returns:
      session: The newly created session instance.
    """

  @abc.abstractmethod
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Gets a session."""

  @abc.abstractmethod
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists all the sessions for a user.

    Args:
      app_name: The name of the app.
      user_id: The ID of the user. If not provided, lists all sessions for all
        users.

    Returns:
      A ListSessionsResponse containing the sessions.
    """

  @abc.abstractmethod
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session."""

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session object."""
    if event.partial:
      return event
    # Apply temp-scoped state to the in-memory session BEFORE trimming the
    # event delta, so that subsequent agents within the same invocation can
    # read temp values (e.g. output_key='temp:my_key' in SequentialAgent).
    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)
    # Apply secret-scoped state to in-memory session and process cache
    # BEFORE trimming, so the session retains secret values across turns.
    self._apply_secret_state(session, event)
    event = self._trim_secret_delta_state(event)
    self._update_session_state(session, event)
    session.events.append(event)
    return event

  def _apply_temp_state(self, session: Session, event: Event) -> None:
    """Applies temp-scoped state delta to the in-memory session state.

    Temp state is ephemeral: it lives in the session's in-memory state for
    the duration of the current invocation but is NOT persisted to storage
    (the event delta is trimmed separately by _trim_temp_delta_state).
    """
    if not event.actions or not event.actions.state_delta:
      return
    for key, value in event.actions.state_delta.items():
      if key.startswith(State.TEMP_PREFIX):
        session.state[key] = value

  def _trim_temp_delta_state(self, event: Event) -> Event:
    """Removes temporary state delta keys from the event.

    This prevents temp-scoped state from being persisted, while the
    in-memory session state (updated by _apply_temp_state) retains the
    values for the duration of the current invocation.
    """
    if not event.actions or not event.actions.state_delta:
      return event

    event.actions.state_delta = {
        key: value
        for key, value in event.actions.state_delta.items()
        if not key.startswith(State.TEMP_PREFIX)
    }
    return event

  def _apply_secret_state(self, session: Session, event: Event) -> None:
    """Applies secret-scoped state to in-memory session and process cache.

    Secret state survives across turns (via the process-local cache) but
    is never persisted to storage. The event delta is trimmed separately
    by _trim_secret_delta_state.
    """
    if not event.actions or not event.actions.state_delta:
      return
    cache_key = (session.app_name, session.user_id, session.id)
    for key, value in event.actions.state_delta.items():
      if key.startswith(State.SECRET_PREFIX):
        session.state[key] = value
        self._secret_state_cache.setdefault(cache_key, {})[key] = value

  def _trim_secret_delta_state(self, event: Event) -> Event:
    """Removes secret-scoped keys from event delta before persistence."""
    if not event.actions or not event.actions.state_delta:
      return event
    event.actions.state_delta = {
        key: value
        for key, value in event.actions.state_delta.items()
        if not key.startswith(State.SECRET_PREFIX)
    }
    return event

  def _seed_secret_state_on_create(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      state: Optional[dict[str, Any]],
  ) -> Optional[dict[str, Any]]:
    """Extracts secret-scoped keys from initial state into the cache.

    Returns the state dict with secret keys removed (for persistence)
    but seeds them in the process-local cache so get_session() can
    restore them.
    """
    if not state:
      return state
    secret_keys = {
        k: v for k, v in state.items() if k.startswith(State.SECRET_PREFIX)
    }
    if not secret_keys:
      return state
    cache_key = (app_name, user_id, session_id)
    self._secret_state_cache.setdefault(cache_key, {}).update(secret_keys)
    return {
        k: v for k, v in state.items() if not k.startswith(State.SECRET_PREFIX)
    }

  def _restore_secret_state(self, session: Session) -> None:
    """Merges cached secret state into an in-memory session."""
    cache_key = (session.app_name, session.user_id, session.id)
    secret_state = self._secret_state_cache.get(cache_key, {})
    for key, value in secret_state.items():
      session.state[key] = value

  def _evict_secret_state(
      self, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Removes cached secret state for a deleted session."""
    cache_key = (app_name, user_id, session_id)
    self._secret_state_cache.pop(cache_key, None)

  def _update_session_state(self, session: Session, event: Event) -> None:
    """Updates the session state based on the event."""
    if not event.actions or not event.actions.state_delta:
      return
    for key, value in event.actions.state_delta.items():
      session.state.update({key: value})
