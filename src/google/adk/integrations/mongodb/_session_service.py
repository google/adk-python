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
import copy
import json
import logging
import time
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _client
from ...errors._stale_session_error import StaleSessionError
from ...errors.already_exists_error import AlreadyExistsError
from ...errors.session_not_found_error import SessionNotFoundError
from ...events.event import Event
from ...features import experimental
from ...features import FeatureName
from ...platform import uuid as platform_uuid
from ...sessions import _session_util
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from ...sessions.state import State

if TYPE_CHECKING:
  from pymongo import MongoClient

logger = logging.getLogger("google_adk." + __name__)

_STALE_SESSION_ERROR_MESSAGE = (
    "The session has been modified in storage since it was loaded. "
    "Please reload the session before appending more events."
)

DEFAULT_SESSIONS_COLLECTION = "sessions"
DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_APP_STATE_COLLECTION = "app_states"
DEFAULT_USER_STATE_COLLECTION = "user_states"

_SessionLockKey = tuple[str, str, str]


def _dumps_state(state: dict[str, Any]) -> str:
  """Serializes a state bucket, coercing non-JSON values first."""
  return json.dumps(_session_util.make_json_safe_state(state))


def _loads_state(raw: Any) -> dict[str, Any]:
  """Parses a stored state bucket (JSON string or legacy dict)."""
  if isinstance(raw, str):
    return json.loads(raw)
  return dict(raw or {})


@experimental(FeatureName.MONGODB_SESSION_SERVICE)
class MongoDbSessionService(BaseSessionService):
  """Session service that uses MongoDB as the backend.

  Document layout within the configured database:

    - `<sessions_collection>`: one document per session, keyed by
      `<app_name>/<user_id>/<session_id>`, holding the session-scoped state
      (JSON-encoded), timestamps, and an optimistic-concurrency `revision`.
    - `<events_collection>`: one document per event, keyed by
      `<app_name>/<user_id>/<session_id>/<event_id>`, holding the full
      serialized event under `event_data`.
    - `<app_state_collection>`: one document per app, keyed by `<app_name>`.
    - `<user_state_collection>`: one document per user, keyed by
      `<app_name>/<user_id>`.

  State buckets are stored JSON-encoded so state keys containing characters
  that MongoDB forbids in document fields (e.g. `.`, `$`) round-trip safely.

  Example:
      ```python
      session_service = MongoDbSessionService(
          connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
          database_name="my_app",
      )
      runner = Runner(
          agent=agent, app_name="my_app", session_service=session_service
      )
      ```
  """

  def __init__(
      self,
      *,
      database_name: str,
      connection_string: str | None = None,
      mongo_client: MongoClient | None = None,
      sessions_collection: str = DEFAULT_SESSIONS_COLLECTION,
      events_collection: str = DEFAULT_EVENTS_COLLECTION,
      app_state_collection: str = DEFAULT_APP_STATE_COLLECTION,
      user_state_collection: str = DEFAULT_USER_STATE_COLLECTION,
  ):
    """Initializes the MongoDB session service.

    Args:
      database_name: The MongoDB database used to store sessions, events and
        shared state.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this service. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`).
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      sessions_collection: Collection name for session documents.
      events_collection: Collection name for event documents.
      app_state_collection: Collection name for app state documents.
      user_state_collection: Collection name for user state documents.
    """
    if mongo_client is not None and connection_string is not None:
      raise ValueError(
          "Only one of `connection_string` and `mongo_client` may be provided."
      )
    if mongo_client is not None:
      self._client = mongo_client
      self._owns_client = False
    elif connection_string is not None:
      self._client = _client.get_mongo_client(connection_string)
      self._owns_client = True
    else:
      raise ValueError(
          "Either `connection_string` or `mongo_client` must be provided."
      )
    self._database_name = database_name
    self.sessions_collection = sessions_collection
    self.events_collection = events_collection
    self.app_state_collection = app_state_collection
    self.user_state_collection = user_state_collection

    # Per-session locks used to serialize append_event calls in this process.
    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

  def _sessions(self):
    return self._client[self._database_name][self.sessions_collection]

  def _events(self):
    return self._client[self._database_name][self.events_collection]

  def _app_states(self):
    return self._client[self._database_name][self.app_state_collection]

  def _user_states(self):
    return self._client[self._database_name][self.user_state_collection]

  @staticmethod
  def _session_key(app_name: str, user_id: str, session_id: str) -> str:
    return f"{app_name}/{user_id}/{session_id}"

  @staticmethod
  def _user_key(app_name: str, user_id: str) -> str:
    return f"{app_name}/{user_id}"

  @asynccontextmanager
  async def _with_session_lock(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> AsyncGenerator[None]:
    """Serializes event appends for the same session within this process."""
    lock_key = (app_name, user_id, session_id)
    async with self._session_locks_guard:
      lock = self._session_locks.get(lock_key)
      if lock is None:
        lock = asyncio.Lock()
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

  @staticmethod
  def _merge_state(
      app_state: dict[str, Any] | None,
      user_state: dict[str, Any] | None,
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    """Merges app, user, and session states into a single state dictionary."""
    merged_state = copy.deepcopy(session_state)
    for key, value in (app_state or {}).items():
      merged_state[State.APP_PREFIX + key] = value
    for key, value in (user_state or {}).items():
      merged_state[State.USER_PREFIX + key] = value
    return merged_state

  def _read_app_state(self, app_name: str) -> dict[str, Any]:
    doc = self._app_states().find_one({"_id": app_name})
    return _loads_state(doc.get("state")) if doc else {}

  def _read_user_state(self, app_name: str, user_id: str) -> dict[str, Any]:
    doc = self._user_states().find_one(
        {"_id": self._user_key(app_name, user_id)}
    )
    return _loads_state(doc.get("state")) if doc else {}

  def _merge_state_bucket(
      self, collection: Any, doc_id: str, delta: dict[str, Any]
  ) -> dict[str, Any]:
    """Merges delta into a stored state bucket and returns the merged state."""
    existing = collection.find_one({"_id": doc_id})
    merged = _loads_state(existing.get("state")) if existing else {}
    merged.update(delta)
    collection.update_one(
        {"_id": doc_id}, {"$set": {"state": _dumps_state(merged)}}, upsert=True
    )
    return merged

  def _to_session(
      self,
      doc: dict[str, Any],
      merged_state: dict[str, Any],
      events: list[Event],
  ) -> Session:
    session = Session(
        id=doc["id"],
        app_name=doc["app_name"],
        user_id=doc["user_id"],
        state=merged_state,
        events=events,
        last_update_time=doc.get("update_time", 0.0),
    )
    session._storage_update_marker = str(doc.get("revision", 0))
    return session

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: dict[str, Any] | None = None,
      session_id: str | None = None,
  ) -> Session:
    """Creates a new session in MongoDB."""

    def _create() -> tuple[str, dict[str, Any]]:
      sid = session_id or platform_uuid.new_uuid()
      state_deltas = _session_util.extract_state_delta(state or {})

      app_state = (
          self._merge_state_bucket(
              self._app_states(), app_name, state_deltas["app"]
          )
          if state_deltas["app"]
          else self._read_app_state(app_name)
      )
      user_state = (
          self._merge_state_bucket(
              self._user_states(),
              self._user_key(app_name, user_id),
              state_deltas["user"],
          )
          if state_deltas["user"]
          else self._read_user_state(app_name, user_id)
      )

      now = time.time()
      doc = {
          "_id": self._session_key(app_name, user_id, sid),
          "id": sid,
          "app_name": app_name,
          "user_id": user_id,
          "state": _dumps_state(state_deltas["session"]),
          "create_time": now,
          "update_time": now,
          "revision": 0,
      }
      try:
        self._sessions().insert_one(doc)
      except Exception as exc:
        if exc.__class__.__name__ == "DuplicateKeyError":
          raise AlreadyExistsError(f"Session {sid} already exists.") from exc
        raise
      merged = self._merge_state(app_state, user_state, state_deltas["session"])
      return sid, merged

    sid, merged_state = await asyncio.to_thread(_create)
    session = Session(
        id=sid,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
        events=[],
        last_update_time=time.time(),
    )
    session._storage_update_marker = "0"
    return session

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: GetSessionConfig | None = None,
  ) -> Session | None:
    """Gets a session from MongoDB."""

    def _get() -> Session | None:
      doc = self._sessions().find_one(
          {"_id": self._session_key(app_name, user_id, session_id)}
      )
      if not doc:
        return None

      # A requested count of zero asks for no event history at all (callers
      # use it to probe whether a session exists), so skip the events query.
      events: list[Event] = []
      if config is None or config.num_recent_events != 0:
        query: dict[str, Any] = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
        }
        if config and config.after_timestamp is not None:
          query["timestamp"] = {"$gte": config.after_timestamp}

        cursor = self._events().find(query).sort("timestamp", -1)
        if config and config.num_recent_events is not None:
          cursor = cursor.limit(config.num_recent_events)
        event_docs = list(cursor)
        event_docs.reverse()  # restore chronological order
        events = [
            Event.model_validate(event_doc["event_data"])
            for event_doc in event_docs
        ]

      merged = self._merge_state(
          self._read_app_state(app_name),
          self._read_user_state(app_name, user_id),
          _loads_state(doc.get("state")),
      )
      return self._to_session(doc, merged, events)

    return await asyncio.to_thread(_get)

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: str | None = None
  ) -> ListSessionsResponse:
    """Lists sessions from MongoDB, oldest update first."""

    def _list() -> list[Session]:
      query: dict[str, Any] = {"app_name": app_name}
      if user_id:
        query["user_id"] = user_id
      docs = list(self._sessions().find(query))

      app_state = self._read_app_state(app_name)
      user_ids = {doc["user_id"] for doc in docs}
      user_states = {
          uid: self._read_user_state(app_name, uid) for uid in user_ids
      }

      sessions = [
          self._to_session(
              doc,
              self._merge_state(
                  app_state,
                  user_states.get(doc["user_id"], {}),
                  _loads_state(doc.get("state")),
              ),
              [],
          )
          for doc in docs
      ]
      sessions.sort(key=lambda s: (s.last_update_time, s.user_id, s.id))
      return sessions

    return ListSessionsResponse(sessions=await asyncio.to_thread(_list))

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session and its events from MongoDB."""

    def _delete() -> None:
      self._events().delete_many(
          {"app_name": app_name, "user_id": user_id, "session_id": session_id}
      )
      self._sessions().delete_one(
          {"_id": self._session_key(app_name, user_id, session_id)}
      )

    await asyncio.to_thread(_delete)

  @override
  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    """Returns the user-scoped state for the given app and user."""
    return await asyncio.to_thread(self._read_user_state, app_name, user_id)

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session in MongoDB."""
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

    async with self._with_session_lock(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    ):

      def _append() -> int:
        if state_deltas["app"]:
          self._merge_state_bucket(
              self._app_states(), session.app_name, state_deltas["app"]
          )
        if state_deltas["user"]:
          self._merge_state_bucket(
              self._user_states(),
              self._user_key(session.app_name, session.user_id),
              state_deltas["user"],
          )

        session_only_state = {
            key: value
            for key, value in session.state.items()
            if not key.startswith(State.APP_PREFIX)
            and not key.startswith(State.USER_PREFIX)
            and not key.startswith(State.TEMP_PREFIX)
        }
        session_only_state.update(state_deltas["session"])

        session_doc_id = self._session_key(
            session.app_name, session.user_id, session.id
        )
        current = self._sessions().find_one({"_id": session_doc_id})
        if not current:
          raise SessionNotFoundError(f"Session {session.id} not found.")
        current_revision = current.get("revision", 0)
        if session._storage_update_marker is not None and (
            session._storage_update_marker != str(current_revision)
        ):
          raise StaleSessionError(_STALE_SESSION_ERROR_MESSAGE)

        # The revision filter makes the update a no-op when a concurrent
        # writer bumped the revision between our read and write.
        updated = self._sessions().find_one_and_update(
            {"_id": session_doc_id, "revision": current_revision},
            {
                "$set": {
                    "state": _dumps_state(session_only_state),
                    "update_time": event.timestamp,
                },
                "$inc": {"revision": 1},
            },
            return_document=True,
        )
        if updated is None:
          raise StaleSessionError(_STALE_SESSION_ERROR_MESSAGE)

        # Upsert keeps event ingestion idempotent across retries.
        self._events().replace_one(
            {"_id": f"{session_doc_id}/{event.id}"},
            {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "timestamp": event.timestamp,
                "event_data": event.model_dump(exclude_none=True, mode="json"),
            },
            upsert=True,
        )
        return int(updated.get("revision", current_revision + 1))

      new_revision = await asyncio.to_thread(_append)
      session._storage_update_marker = str(new_revision)
      session.last_update_time = event.timestamp

    await super().append_event(session, event)
    return event

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
