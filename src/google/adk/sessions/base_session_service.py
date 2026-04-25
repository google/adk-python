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

from google.adk.platform import time as platform_time
from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from ..events.event_actions import EventActions
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

  def _update_session_state(self, session: Session, event: Event) -> None:
    """Updates the session state based on the event."""
    if not event.actions or not event.actions.state_delta:
      return
    for key, value in event.actions.state_delta.items():
      session.state.update({key: value})

  async def _record_initial_state_event(
      self, session: Session, state: Optional[dict[str, Any]]
  ) -> None:
    """Appends a synthetic event carrying the initial non-temp session state.

    Subclasses call this from `create_session` so that initial state flows
    through `append_event` (the single state-merging path) and so that
    `rewind_async` can restore session-scoped initial values for keys later
    overwritten or introduced by subsequent events.

    Args:
      session: The newly created session to attach the event to.
      state: The initial state dict supplied to `create_session`. Temp-prefixed
        keys are dropped because temp state is ephemeral and never persisted.
    """
    if not state:
      return
    state_delta = {
        k: v for k, v in state.items() if not k.startswith(State.TEMP_PREFIX)
    }
    if not state_delta:
      return
    # Round to microseconds so the timestamp roundtrips exactly through
    # storage backends that persist timestamps as datetime (microsecond
    # precision) — keeps in-memory and reloaded events comparable.
    timestamp = round(platform_time.get_time(), 6)
    initial_event = Event(
        author='user',
        timestamp=timestamp,
        actions=EventActions(state_delta=dict(state_delta)),
    )
    await self.append_event(session=session, event=initial_event)
