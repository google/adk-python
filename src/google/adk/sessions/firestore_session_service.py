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

"""Firestore-backed session service for Google ADK.

Provides persistent, serverless session storage using Google Cloud Firestore.
This is well-suited for production deployments on Cloud Run, Cloud Functions,
or any GCP environment where managing a SQL database is undesirable.

Firestore collection layout::

    adk_app_states/{app_name}
    adk_user_states/{app_name}_{user_id}
    adk_sessions/{session_id}
        -> subcollection: events/{event_id}

Requires the ``google-cloud-firestore`` package::

    pip install google-cloud-firestore
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any
from typing import Optional
import uuid

from typing_extensions import override

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

# Firestore collection names
_APP_STATES_COLLECTION = "adk_app_states"
_USER_STATES_COLLECTION = "adk_user_states"
_SESSIONS_COLLECTION = "adk_sessions"
_EVENTS_SUBCOLLECTION = "events"

# Firestore document field names
_FIELD_APP_NAME = "app_name"
_FIELD_USER_ID = "user_id"
_FIELD_STATE = "state"
_FIELD_CREATE_TIME = "create_time"
_FIELD_UPDATE_TIME = "update_time"
_FIELD_EVENT_DATA = "event_data"
_FIELD_TIMESTAMP = "timestamp"
_FIELD_INVOCATION_ID = "invocation_id"


def _user_state_doc_id(app_name: str, user_id: str) -> str:
  """Builds a deterministic document ID for a user state entry."""
  return f"{app_name}_{user_id}"


class FirestoreSessionService(BaseSessionService):
  """A session service backed by Google Cloud Firestore.

  This service stores sessions, events, and state in Firestore collections,
  providing serverless, persistent storage suitable for production use.

  Args:
    project: GCP project ID. If ``None``, uses Application Default Credentials.
    database: Firestore database ID. Defaults to ``"(default)"``.
    collection_prefix: Optional prefix for all collection names, useful for
      multi-tenant setups or testing isolation.
  """

  def __init__(
      self,
      *,
      project: Optional[str] = None,
      database: str = "(default)",
      collection_prefix: str = "",
  ):
    try:
      from google.cloud.firestore_v1 import AsyncClient
    except ImportError as e:
      raise ImportError(
          "FirestoreSessionService requires google-cloud-firestore. "
          "Install it with: pip install google-cloud-firestore"
      ) from e

    self._db = AsyncClient(project=project, database=database)
    self._prefix = collection_prefix

  # -- Collection helpers --------------------------------------------------

  def _col_app_states(self):
    return self._db.collection(f"{self._prefix}{_APP_STATES_COLLECTION}")

  def _col_user_states(self):
    return self._db.collection(f"{self._prefix}{_USER_STATES_COLLECTION}")

  def _col_sessions(self):
    return self._db.collection(f"{self._prefix}{_SESSIONS_COLLECTION}")

  def _events_col(self, session_id: str):
    """Returns the events subcollection for a given session."""
    return (
        self._col_sessions()
        .document(session_id)
        .collection(_EVENTS_SUBCOLLECTION)
    )

  # -- State helpers -------------------------------------------------------

  async def _get_app_state(self, app_name: str) -> dict[str, Any]:
    """Fetches the app-level state dict, returning empty dict if missing."""
    doc = await self._col_app_states().document(app_name).get()
    if doc.exists:
      return doc.to_dict().get(_FIELD_STATE, {})
    return {}

  async def _set_app_state(self, app_name: str, state: dict[str, Any]) -> None:
    await self._col_app_states().document(app_name).set(
        {_FIELD_STATE: state}, merge=True
    )

  async def _get_user_state(
      self, app_name: str, user_id: str
  ) -> dict[str, Any]:
    doc_id = _user_state_doc_id(app_name, user_id)
    doc = await self._col_user_states().document(doc_id).get()
    if doc.exists:
      return doc.to_dict().get(_FIELD_STATE, {})
    return {}

  async def _set_user_state(
      self, app_name: str, user_id: str, state: dict[str, Any]
  ) -> None:
    doc_id = _user_state_doc_id(app_name, user_id)
    await self._col_user_states().document(doc_id).set(
        {
            _FIELD_APP_NAME: app_name,
            _FIELD_USER_ID: user_id,
            _FIELD_STATE: state,
        },
        merge=True,
    )

  def _merge_state(
      self,
      app_state: dict[str, Any],
      user_state: dict[str, Any],
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    """Merges app, user, and session state into a single dict."""
    merged = copy.deepcopy(session_state)
    for key, value in app_state.items():
      merged[State.APP_PREFIX + key] = value
    for key, value in user_state.items():
      merged[State.USER_PREFIX + key] = value
    return merged

  # -- CRUD ----------------------------------------------------------------

  @override
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
        else str(uuid.uuid4())
    )

    # Check for duplicate
    existing = await self._col_sessions().document(session_id).get()
    if existing.exists:
      raise AlreadyExistsError(f"Session with id {session_id} already exists.")

    # Extract state deltas
    state_deltas = _session_util.extract_state_delta(state)
    app_state_delta = state_deltas["app"]
    user_state_delta = state_deltas["user"]
    session_state = state_deltas["session"]

    # Update app / user state
    if app_state_delta:
      current_app_state = await self._get_app_state(app_name)
      current_app_state.update(app_state_delta)
      await self._set_app_state(app_name, current_app_state)

    if user_state_delta:
      current_user_state = await self._get_user_state(app_name, user_id)
      current_user_state.update(user_state_delta)
      await self._set_user_state(app_name, user_id, current_user_state)

    now = time.time()
    # Store session document
    await self._col_sessions().document(session_id).set({
        _FIELD_APP_NAME: app_name,
        _FIELD_USER_ID: user_id,
        _FIELD_STATE: session_state,
        _FIELD_CREATE_TIME: now,
        _FIELD_UPDATE_TIME: now,
    })

    # Build merged state for response
    app_state = await self._get_app_state(app_name)
    user_state = await self._get_user_state(app_name, user_id)
    merged = self._merge_state(app_state, user_state, session_state)

    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged,
        last_update_time=now,
    )

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    doc = await self._col_sessions().document(session_id).get()
    if not doc.exists:
      return None

    data = doc.to_dict()
    if data.get(_FIELD_APP_NAME) != app_name:
      return None
    if data.get(_FIELD_USER_ID) != user_id:
      return None

    session_state = data.get(_FIELD_STATE, {})

    # Fetch events from subcollection
    events_query = self._events_col(session_id).order_by(_FIELD_TIMESTAMP)

    if config and config.after_timestamp:
      events_query = events_query.where(
          filter=self._db.field_filter(
              _FIELD_TIMESTAMP, ">=", config.after_timestamp
          )
      )

    event_docs = events_query.stream()
    events: list[Event] = []
    async for event_doc in event_docs:
      event_data = event_doc.to_dict()
      raw = event_data.get(_FIELD_EVENT_DATA, {})
      if raw:
        events.append(Event.model_validate(raw))

    if config and config.num_recent_events:
      events = events[-config.num_recent_events :]

    # Merge states
    app_state = await self._get_app_state(app_name)
    user_state = await self._get_user_state(app_name, user_id)
    merged = self._merge_state(app_state, user_state, session_state)

    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged,
        events=events,
        last_update_time=data.get(_FIELD_UPDATE_TIME, 0.0),
    )

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    query = self._col_sessions().where(
        filter=self._db.field_filter(_FIELD_APP_NAME, "==", app_name)
    )
    if user_id is not None:
      query = query.where(
          filter=self._db.field_filter(_FIELD_USER_ID, "==", user_id)
      )

    sessions: list[Session] = []
    async for doc in query.stream():
      data = doc.to_dict()
      session_state = data.get(_FIELD_STATE, {})
      sid = doc.id
      suid = data.get(_FIELD_USER_ID, "")

      app_state = await self._get_app_state(app_name)
      user_state = await self._get_user_state(app_name, suid)
      merged = self._merge_state(app_state, user_state, session_state)

      sessions.append(
          Session(
              app_name=app_name,
              user_id=suid,
              id=sid,
              state=merged,
              last_update_time=data.get(_FIELD_UPDATE_TIME, 0.0),
          )
      )

    return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    session_ref = self._col_sessions().document(session_id)
    doc = await session_ref.get()
    if not doc.exists:
      return

    # Delete all events in the subcollection first
    events_ref = session_ref.collection(_EVENTS_SUBCOLLECTION)
    async for event_doc in events_ref.stream():
      await event_doc.reference.delete()

    # Delete the session document
    await session_ref.delete()

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id

    session_ref = self._col_sessions().document(session_id)
    doc = await session_ref.get()
    if not doc.exists:
      logger.warning(
          "Cannot append event: session %s not found in Firestore.",
          session_id,
      )
      return event

    # Update in-memory session state via base class
    await super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    # Extract and apply state deltas
    if event.actions and event.actions.state_delta:
      state_deltas = _session_util.extract_state_delta(
          event.actions.state_delta
      )
      app_state_delta = state_deltas["app"]
      user_state_delta = state_deltas["user"]
      session_state_delta = state_deltas["session"]

      if app_state_delta:
        current_app_state = await self._get_app_state(app_name)
        current_app_state.update(app_state_delta)
        await self._set_app_state(app_name, current_app_state)

      if user_state_delta:
        current_user_state = await self._get_user_state(app_name, user_id)
        current_user_state.update(user_state_delta)
        await self._set_user_state(app_name, user_id, current_user_state)

      if session_state_delta:
        stored_data = doc.to_dict()
        stored_state = stored_data.get(_FIELD_STATE, {})
        stored_state.update(session_state_delta)
        await session_ref.update({_FIELD_STATE: stored_state})

    # Store event in subcollection
    event_data = event.model_dump(exclude_none=True, mode="json")
    await self._events_col(session_id).document(event.id).set({
        _FIELD_EVENT_DATA: event_data,
        _FIELD_TIMESTAMP: event.timestamp,
        _FIELD_INVOCATION_ID: event.invocation_id,
    })

    # Update session timestamp
    await session_ref.update({_FIELD_UPDATE_TIME: event.timestamp})

    return event

  async def close(self) -> None:
    """Closes the underlying Firestore client."""
    self._db.close()

  async def __aenter__(self) -> FirestoreSessionService:
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
