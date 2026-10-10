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

# Recommended secondary indexes, created only when `ensure_indexes=True`.
# Reads and writes are otherwise keyed by `_id`, which MongoDB indexes
# implicitly; these cover the two query patterns that are not `_id` lookups.
_EVENTS_HISTORY_INDEX_NAME = "app_user_session_ts"
_EVENTS_HISTORY_INDEX_KEYS = [
    ("app_name", 1),
    ("user_id", 1),
    ("session_id", 1),
    ("timestamp", -1),
]
_SESSIONS_LIST_INDEX_NAME = "app_user"
_SESSIONS_LIST_INDEX_KEYS = [("app_name", 1), ("user_id", 1)]

_SessionLockKey = tuple[str, str, str]


def _is_transaction_unsupported_error(exc: Exception) -> bool:
  """Returns True when the deployment cannot run multi-document transactions.

  mongomock raises NotImplementedError for start_session, and a standalone
  mongod fails the first transactional operation with IllegalOperation
  (error code 20). Both conditions are permanent for the life of the client,
  so callers may cache the verdict and stop attempting transactions.
  """
  if isinstance(exc, NotImplementedError):
    return True
  if getattr(exc, "code", None) == 20:
    return True
  message = str(exc).lower()
  return "transaction" in message and (
      "replica set" in message or "not supported" in message
  )


def _dumps_state(state: dict[str, Any]) -> str:
  """Serializes a state bucket, coercing non-JSON values first."""
  return json.dumps(_session_util.make_json_safe_state(state))


def _loads_state(raw: Any) -> dict[str, Any]:
  """Parses a stored state bucket (JSON string or legacy dict)."""
  if isinstance(raw, str):
    return json.loads(raw)
  return dict(raw or {})


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

  Writes that span multiple documents (state buckets, the session revision
  bump, and the event insert) run inside a multi-document transaction so
  they commit or abort together, and so concurrent writers to shared app or
  user state lose to a retry instead of silently overwriting each other.
  Transactions require a replica set, sharded cluster, or Atlas deployment;
  on deployments without them (standalone mongod) the service falls back to
  sequential writes and logs a warning.

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
      ensure_indexes: bool = False,
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
      ensure_indexes: When True, create the recommended secondary indexes on
        first use: `app_user_session_ts` on the events collection (the
        app/user/session/timestamp lookup behind `get_session`), and
        `app_user` on the sessions collection (the app/user filter behind
        `list_sessions`). Index creation is idempotent, so this is safe to
        leave on, but it requires the `createIndex` privilege. All other
        reads and writes are keyed by `_id`, which MongoDB indexes
        implicitly.
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
    self._connection_string = connection_string
    self._database_name = database_name
    self.sessions_collection = sessions_collection
    self.events_collection = events_collection
    self.app_state_collection = app_state_collection
    self.user_state_collection = user_state_collection
    self._ensure_indexes = ensure_indexes
    self._indexes_ensured = False

    # Per-session locks used to serialize append_event calls in this process.
    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

    # Set to False after the deployment proves it cannot run multi-document
    # transactions (standalone mongod, mongomock), so later writes skip the
    # transaction attempt instead of paying for a failure each time.
    self._transactions_supported = True

  def __getstate__(self) -> dict[str, Any]:
    """Drops the unpicklable client and locks so the service can be pickled.

    Agent Engine packages apps with cloudpickle; the MongoClient (sockets,
    locks, background threads) and the asyncio locks cannot cross that
    boundary. The client is rebuilt from the connection string on restore and
    the per-session lock table starts empty.
    """
    state = _client.drop_client_for_pickle(
        self.__dict__,
        owns_client=self._owns_client,
        owner="MongoDbSessionService",
    )
    state["_session_locks"] = {}
    state["_session_lock_ref_count"] = {}
    state["_session_locks_guard"] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    self.__dict__.update(state)
    self._client = _client.get_mongo_client(self._connection_string)
    self._session_locks_guard = asyncio.Lock()

  def _sessions(self):
    return self._client[self._database_name][self.sessions_collection]

  def _events(self):
    return self._client[self._database_name][self.events_collection]

  def _app_states(self):
    return self._client[self._database_name][self.app_state_collection]

  def _user_states(self):
    return self._client[self._database_name][self.user_state_collection]

  def _ensure_indexes_once(self) -> None:
    """Creates the recommended secondary indexes on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. It must run outside the write transactions:
    `createIndexes` is not allowed inside a multi-document transaction.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    self._events().create_index(
        _EVENTS_HISTORY_INDEX_KEYS, name=_EVENTS_HISTORY_INDEX_NAME
    )
    self._sessions().create_index(
        _SESSIONS_LIST_INDEX_KEYS, name=_SESSIONS_LIST_INDEX_NAME
    )
    self._indexes_ensured = True

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

  def _run_in_transaction(self, work):
    """Runs `work(mongo_session)` as one atomic unit when possible.

    Multi-document transactions make the state-bucket merges, the session
    revision bump, and the event write commit or abort together, and a
    concurrent conflicting writer loses to a write-conflict retry instead of
    silently overwriting. Transactions require a replica set, sharded
    cluster, or Atlas deployment. Where they are unavailable (a standalone
    mongod, or mongomock in tests) the work runs directly without a session,
    preserving the pre-transaction behavior.

    Args:
      work: Callable taking the pymongo client session (or None when
        transactions are unsupported) and returning the work's result.
    """
    if not self._transactions_supported:
      return work(None)
    try:
      mongo_session = self._client.start_session()
    except Exception as exc:
      if not _is_transaction_unsupported_error(exc):
        raise
      self._transactions_supported = False
      logger.warning(
          "MongoDB deployment does not support sessions/transactions (%s); "
          "writes will not be atomic across documents.",
          exc,
      )
      return work(None)
    with mongo_session:
      try:
        # with_transaction retries the callback on TransientTransactionError
        # and re-commits on UnknownTransactionCommitResult; other errors
        # (StaleSessionError, SessionNotFoundError, ...) abort and propagate.
        return mongo_session.with_transaction(work)
      except Exception as exc:
        if not _is_transaction_unsupported_error(exc):
          raise
        self._transactions_supported = False
        logger.warning(
            "MongoDB deployment does not support transactions (%s); "
            "writes will not be atomic across documents.",
            exc,
        )
        return work(None)

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

  def _read_app_state(
      self, app_name: str, mongo_session: Any = None
  ) -> dict[str, Any]:
    doc = self._app_states().find_one({"_id": app_name}, session=mongo_session)
    return _loads_state(doc.get("state")) if doc else {}

  def _read_user_state(
      self, app_name: str, user_id: str, mongo_session: Any = None
  ) -> dict[str, Any]:
    doc = self._user_states().find_one(
        {"_id": self._user_key(app_name, user_id)}, session=mongo_session
    )
    return _loads_state(doc.get("state")) if doc else {}

  def _merge_state_bucket(
      self,
      collection: Any,
      doc_id: str,
      delta: dict[str, Any],
      mongo_session: Any = None,
  ) -> dict[str, Any]:
    """Merges delta into a stored state bucket and returns the merged state.

    The read-modify-write is only safe against lost updates when it runs
    inside a transaction (`mongo_session` set); concurrent writers otherwise
    race on the read and the last write wins.
    """
    existing = collection.find_one({"_id": doc_id}, session=mongo_session)
    merged = _loads_state(existing.get("state")) if existing else {}
    merged.update(delta)
    collection.update_one(
        {"_id": doc_id},
        {"$set": {"state": _dumps_state(merged)}},
        upsert=True,
        session=mongo_session,
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
    await asyncio.to_thread(self._ensure_indexes_once)

    def _create(mongo_session) -> tuple[str, dict[str, Any]]:
      sid = session_id or platform_uuid.new_uuid()
      state_deltas = _session_util.extract_state_delta(state or {})

      app_state = (
          self._merge_state_bucket(
              self._app_states(),
              app_name,
              state_deltas["app"],
              mongo_session,
          )
          if state_deltas["app"]
          else self._read_app_state(app_name, mongo_session)
      )
      user_state = (
          self._merge_state_bucket(
              self._user_states(),
              self._user_key(app_name, user_id),
              state_deltas["user"],
              mongo_session,
          )
          if state_deltas["user"]
          else self._read_user_state(app_name, user_id, mongo_session)
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
        self._sessions().insert_one(doc, session=mongo_session)
      except Exception as exc:
        if exc.__class__.__name__ == "DuplicateKeyError":
          raise AlreadyExistsError(f"Session {sid} already exists.") from exc
        raise
      merged = self._merge_state(app_state, user_state, state_deltas["session"])
      return sid, merged

    sid, merged_state = await asyncio.to_thread(
        self._run_in_transaction, _create
    )
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
    await asyncio.to_thread(self._ensure_indexes_once)

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
    await asyncio.to_thread(self._ensure_indexes_once)

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
    await asyncio.to_thread(self._ensure_indexes_once)

    def _delete(mongo_session) -> None:
      self._events().delete_many(
          {"app_name": app_name, "user_id": user_id, "session_id": session_id},
          session=mongo_session,
      )
      self._sessions().delete_one(
          {"_id": self._session_key(app_name, user_id, session_id)},
          session=mongo_session,
      )

    await asyncio.to_thread(self._run_in_transaction, _delete)

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

    await asyncio.to_thread(self._ensure_indexes_once)
    async with self._with_session_lock(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    ):

      def _append(mongo_session) -> int:
        if state_deltas["app"]:
          self._merge_state_bucket(
              self._app_states(),
              session.app_name,
              state_deltas["app"],
              mongo_session,
          )
        if state_deltas["user"]:
          self._merge_state_bucket(
              self._user_states(),
              self._user_key(session.app_name, session.user_id),
              state_deltas["user"],
              mongo_session,
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
        current = self._sessions().find_one(
            {"_id": session_doc_id}, session=mongo_session
        )
        if not current:
          raise SessionNotFoundError(f"Session {session.id} not found.")
        current_revision = current.get("revision", 0)
        if session._storage_update_marker is not None and (
            session._storage_update_marker != str(current_revision)
        ):
          raise StaleSessionError(_STALE_SESSION_ERROR_MESSAGE)

        # The revision filter makes the update a no-op when a concurrent
        # writer bumped the revision between our read and write. Inside a
        # transaction the conflict instead aborts with
        # TransientTransactionError, and the driver's retry re-reads the
        # bumped revision and lands here.
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
            session=mongo_session,
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
            session=mongo_session,
        )
        return int(updated.get("revision", current_revision + 1))

      new_revision = await asyncio.to_thread(self._run_in_transaction, _append)
      session._storage_update_marker = str(new_revision)
      session.last_update_time = event.timestamp

    await super().append_event(session, event)
    return event

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
