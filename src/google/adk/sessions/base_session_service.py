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
import copy
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from .session import Session
from .state import State


class GetSessionConfig(BaseModel):
  """The configuration of getting a session."""

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

  @abc.abstractmethod
  async def clone_session(
      self,
      *,
      app_name: str,
      src_user_id: str,
      src_session_id: Optional[str] = None,
      new_user_id: Optional[str] = None,
      new_session_id: Optional[str] = None,
  ) -> Session:
    """Clones session(s) and their events to a new session.

    This method supports two modes:

    1. Single session clone: When `src_session_id` is provided, clones that
       specific session to the new session.

    2. All sessions clone: When `src_session_id` is NOT provided, finds all
       sessions for `src_user_id` and merges ALL their events into a single
       new session.

    Events are automatically deduplicated by event ID - only the first
    occurrence of each event ID is kept.

    Args:
      app_name: The name of the app.
      src_user_id: The source user ID whose session(s) to clone.
      src_session_id: The source session ID to clone. If not provided, all
        sessions for the source user will be merged into one new session.
      new_user_id: The user ID for the new session. If not provided, uses the
        same user_id as the source.
      new_session_id: The session ID for the new session. If not provided, a
        new ID will be auto-generated (UUID4).

    Returns:
      The newly created session with cloned events.

    Raises:
      ValueError: If no source sessions are found.
      AlreadyExistsError: If a session with new_session_id already exists.
    """

  def _prepare_sessions_for_cloning(
      self, source_sessions: list[Session]
  ) -> tuple[dict[str, Any], list[Event]]:
    """Prepares source sessions for cloning by merging states and deduplicating events.

    This is a shared helper method used by all clone_session implementations
    to ensure consistent behavior across different session service backends.

    The method:
    1. Sorts sessions by last_update_time for deterministic state merging
    2. Merges states from all sessions (later sessions overwrite earlier ones)
    3. Collects all events, sorts by timestamp, and deduplicates by event ID

    Args:
      source_sessions: List of source sessions to process.

    Returns:
      A tuple of (merged_state, deduplicated_events):
        - merged_state: Combined state from all sessions (deep copied)
        - deduplicated_events: Chronologically sorted, deduplicated events
    """
    # Sort sessions by update time for deterministic state merging
    # Use sorted() to avoid modifying the input list in-place
    sorted_sessions = sorted(source_sessions, key=lambda s: s.last_update_time)

    # Merge states from all source sessions
    merged_state: dict[str, Any] = {}
    for session in sorted_sessions:
      merged_state.update(copy.deepcopy(session.state))

    # Collect all events, sort by timestamp, then deduplicate
    # to ensure chronological "first occurrence wins"
    all_source_events: list[Event] = []
    for session in sorted_sessions:
      all_source_events.extend(session.events)
    all_source_events.sort(key=lambda e: e.timestamp)

    all_events: list[Event] = []
    seen_event_ids: set[str] = set()
    for event in all_source_events:
      if event.id not in seen_event_ids:
        seen_event_ids.add(event.id)
        all_events.append(event)

    return merged_state, all_events

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session object."""
    if event.partial:
      return event
    event = self._trim_temp_delta_state(event)
    self._update_session_state(session, event)
    session.events.append(event)
    return event

  def _trim_temp_delta_state(self, event: Event) -> Event:
    """Removes temporary state delta keys from the event."""
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
      if key.startswith(State.TEMP_PREFIX):
        continue
      session.state.update({key: value})
