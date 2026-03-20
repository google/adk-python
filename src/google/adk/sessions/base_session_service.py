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
import base64
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from .session import Session
from .state import State

_MAX_EVENT_PAGE_SIZE = 1000


def _encode_event_page_token(offset: int) -> str:
  return base64.b64encode(str(offset).encode()).decode()


def _decode_event_page_token(token: Optional[str]) -> int:
  if not token:
    return 0
  try:
    return max(0, int(base64.b64decode(token).decode()))
  except (ValueError, TypeError):
    return 0


class EventPagination(BaseModel):
  """Pagination configuration for events in ``get_session``.

  When passed via ``GetSessionConfig.event_pagination``, only ``page_size``
  events are returned per call. When omitted, all events are returned.
  """

  page_size: int = 100
  """Maximum number of events per page (1–1000, clamped automatically)."""

  page_token: Optional[str] = None
  """Opaque token returned by a previous call to fetch the next page."""

  page_offset: Optional[int] = None
  """Optional 0-based starting offset. Takes precedence over ``page_token``."""

  @property
  def effective_page_size(self) -> int:
    return max(1, min(self.page_size, _MAX_EVENT_PAGE_SIZE))

  @property
  def offset(self) -> int:
    if self.page_offset is not None:
      return max(0, self.page_offset)
    return _decode_event_page_token(self.page_token)


class GetSessionConfig(BaseModel):
  """The configuration of getting a session."""

  num_recent_events: Optional[int] = None
  after_timestamp: Optional[float] = None
  event_pagination: Optional[EventPagination] = None


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
