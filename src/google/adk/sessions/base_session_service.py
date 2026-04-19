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

  ## Behavior Contracts

  All implementations of `BaseSessionService` must adhere to the following
  behavioral contracts. These contracts ensure consistent behavior across
  different storage backends (in-memory, SQLite, PostgreSQL, Redis, etc.).

  ### 1. Session Creation Atomicity

  `create_session` must be atomic with respect to `session_id`. If a session
  with the same `(app_name, user_id, session_id)` already exists:
  - The call **must** raise `AlreadyExistsError`
  - No partial state changes should be visible

  This ensures that concurrent session creations with the same ID are properly
  serialized, with exactly one succeeding and all others failing.

  ### 2. Event Appending Ordering Guarantees

  `append_event` must maintain the following ordering invariants:

  - **Append Order**: Events are appended to `session.events` in the order
    they are received. The list index reflects the order of appending.

  - **Timestamp Monotonicity**: Within a single session, the `timestamp` of
    newly appended events must be greater than or equal to the `last_update_time`
    of the session. After appending, `session.last_update_time` is updated to
    `event.timestamp`.

  - **Concurrency Control**: Subclass implementations **may or may not**
    serialize concurrent appends to the same session.

    - **Guaranteed serialization**: `DatabaseSessionService` uses per-session
      locks and storage revision markers. When two concurrent operations attempt
      to append using stale session objects, exactly one succeeds and the other
      raises `ValueError` with a message indicating the session was "modified
      in storage".

    - **Implementation-defined behavior**: `InMemorySessionService` and
      `SqliteSessionService` do NOT guarantee serialization under concurrent
      writes. Lost writes or unexpected behavior may occur.

    **Recommendation**: Use `DatabaseSessionService` for production workloads
    that require guaranteed concurrency control.

  ### 3. State Update Merge Semantics

  ⚠️ **CURRENTLY INCONSISTENT ACROSS IMPLEMENTATIONS**

  See `contributing/dev/session_state_merge_semantics.md` for detailed analysis.

  **Current Behavior (Bug)**:
  - `InMemorySessionService` and `DatabaseSessionService`: Use **shallow merge**
    (`dict.update()` and `dict |` operator). Nested dicts are REPLACED, not merged.
  - `SqliteSessionService`: Uses **RFC 7396 JSON Merge Patch** (recursive merge).
    Nested dicts are RECURSIVELY merged.

  **Expected Future Behavior**:
  All implementations should standardize on **RFC 7396 JSON Merge Patch** semantics,
  which is the industry standard and more intuitive for "partial updates".

  **RFC 7396 Rules**:
  - If patch value is `null` → delete key from target
  - If patch value is dict AND target value is dict → **recursively merge**
  - Otherwise → **replace**

  **RFC 7396 Example**:
  ```python
  # Initial state
  session.state = {"a": 1, "nested": {"inner_a": 1, "inner_b": 2}}

  # state_delta
  {"nested": {"inner_a": 100, "inner_c": 300}, "new_key": "added"}

  # Expected RFC 7396 Result
  {
      "a": 1,
      "nested": {
          "inner_a": 100,  # updated
          "inner_b": 2,    # PRESERVED! (recursive merge)
          "inner_c": 300   # added
      },
      "new_key": "added"
  }
  ```

  ### 4. Session Deletion Semantics

  `delete_session` must be idempotent and safe:

  - Deleting a non-existent session **must not** raise an exception
  - After a successful deletion, subsequent `get_session` calls for the same
    `(app_name, user_id, session_id)` must return `None`

  ### 5. Session Isolation

  Sessions are isolated by `(app_name, user_id, session_id)` tuple:

  - Two sessions with different `session_id` values have independent `state`
    and `events`. Modifying one does not affect the other.

  - Two sessions with the same `session_id` but different `user_id` are
    completely separate and do not share any state or events.

  - `list_sessions` only returns sessions matching the provided `app_name`
    and optional `user_id` filter.

  ### 6. Copy-on-Read Semantics

  Implementations should return copies of session objects from `get_session`
  and `list_sessions` to prevent:
  - Accidental in-memory modifications from affecting persisted state
  - Race conditions between concurrent readers

  This means modifications to a returned `Session` object will not be visible
  to other callers unless explicitly persisted via `append_event`.
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
