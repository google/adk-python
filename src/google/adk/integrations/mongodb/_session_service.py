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
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
import copy
import json
import logging
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar

from typing_extensions import override

from . import _client
from ...errors._stale_session_error import StaleSessionError
from ...errors.already_exists_error import AlreadyExistsError
from ...errors.session_not_found_error import SessionNotFoundError
from ...events.event import Event
from ...features import experimental
from ...features import FeatureName
from ...platform import time as platform_time
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
    ("revision", -1),
]
_SESSIONS_LIST_INDEX_NAME = "app_user"
_SESSIONS_LIST_INDEX_KEYS = [("app_name", 1), ("user_id", 1)]
# IndexOptionsConflict (an equivalent index exists under another name) and
# IndexKeySpecsConflict (the name exists with other keys) repeat on every
# retry until the index is dropped by hand.
_INDEX_CONFLICT_ERROR_CODES = frozenset({85, 86})

_SessionLockKey = tuple[str, str, str]
_T = TypeVar("_T")


def _is_transaction_unsupported_error(exc: Exception) -> bool:
  """Returns True when the deployment cannot run multi-document transactions.

  In-memory test fakes raise NotImplementedError for start_session, and a
  standalone mongod fails the first transactional operation with
  IllegalOperation (error code 20). Both conditions are permanent for the life
  of the client, so callers may cache the verdict and stop attempting
  transactions.
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
    return dict(json.loads(raw))
  return dict(raw or {})


@experimental(FeatureName.MONGODB_SESSION_SERVICE)
class MongoDbSessionService(BaseSessionService):
  """Session service that uses MongoDB as the backend.

  Document layout within the configured database:

    - `<sessions_collection>`: one document per session, keyed by
      `{"app_name": <app_name>, "user_id": <user_id>, "session_id":
      <session_id>}`, holding the session-scoped state (JSON-encoded),
      timestamps, and an optimistic-concurrency `revision`.
    - `<events_collection>`: one document per event, keyed by
      `{"app_name": <app_name>, "user_id": <user_id>, "session_id":
      <session_id>, "event_id": <event_id>}`, holding the full JSON-encoded
      event under `event_data`.
    - `<app_state_collection>`: one document per app, keyed by `<app_name>`.
    - `<user_state_collection>`: one document per user, keyed by
      `{"app_name": <app_name>, "user_id": <user_id>}`.

  State buckets and `event_data` are stored JSON-encoded so state keys
  containing characters that MongoDB forbids in document fields (e.g. `.`, `$`)
  round-trip safely.

  Writes that span multiple documents (state buckets, the session revision
  bump, and the event insert) run inside a multi-document transaction so
  they commit or abort together, and so concurrent writers to shared app or
  user state lose to a retry instead of silently overwriting each other.
  Transactions require a replica set, sharded cluster, or Atlas deployment;
  on deployments without them (standalone mongod) the service falls back to
  sequential writes and logs a warning.

  The service takes a synchronous PyMongo `MongoClient`, so one client can be
  shared with `MongoDbToolset`, and runs each PyMongo call in a worker thread
  via `asyncio.to_thread`. Each call holds a thread from the default executor
  while it waits on the server, and cancelling the awaiting task does not
  stop a call already running there; set `timeoutMS` on the client to bound
  it.

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
      mongo_client: MongoClient[Any] | None = None,
      sessions_collection: str = DEFAULT_SESSIONS_COLLECTION,
      events_collection: str = DEFAULT_EVENTS_COLLECTION,
      app_state_collection: str = DEFAULT_APP_STATE_COLLECTION,
      user_state_collection: str = DEFAULT_USER_STATE_COLLECTION,
      ensure_indexes: bool = True,
  ):
    """Initializes the MongoDB session service.

    Args:
      database_name: The MongoDB database used to store sessions, events and
        shared state.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this service. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`). PyMongo sets no operation time
        limit by default; add `timeoutMS=<ms>` to the URI to bound each
        operation.
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      sessions_collection: Collection name for session documents.
      events_collection: Collection name for event documents.
      app_state_collection: Collection name for app state documents.
      user_state_collection: Collection name for user state documents.
      ensure_indexes: Create the recommended secondary indexes on first use
        (default True): `app_user_session_ts` on the events collection (the
        app/user/session/timestamp lookup behind `get_session`), and
        `app_user` on the sessions collection (the app/user filter behind
        `list_sessions`). Index creation is idempotent. An index-creation
        error never fails a session call: it logs a warning and reads run
        unindexed. A missing `createIndex` privilege or a conflicting
        existing index is not retried; other errors are retried on the next
        call. Pass False to skip the attempt entirely. All other reads
        and writes are keyed by `_id`, which MongoDB indexes implicitly.
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
    self._sessions_collection = sessions_collection
    self._events_collection = events_collection
    self._app_state_collection = app_state_collection
    self._user_state_collection = user_state_collection
    self._ensure_indexes = ensure_indexes
    self._indexes_ensured = False

    # Per-session locks used to serialize append_event calls in this process.
    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

    # Set to False after the deployment proves it cannot run multi-document
    # transactions (standalone mongod, in-memory fakes), so later writes skip
    # the transaction attempt instead of paying for a failure each time.
    self._transactions_supported = True

  def _sessions(self) -> Any:
    return self._client[self._database_name][self._sessions_collection]

  def _events(self) -> Any:
    return self._client[self._database_name][self._events_collection]

  def _app_states(self) -> Any:
    return self._client[self._database_name][self._app_state_collection]

  def _user_states(self) -> Any:
    return self._client[self._database_name][self._user_state_collection]

  async def _ensure_indexes_once(self) -> None:
    """Creates the recommended secondary indexes on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. It must run outside the write transactions:
    `createIndexes` is not allowed inside a multi-document transaction. An
    index error never fails the session call: it logs a warning and reads
    fall back to unindexed scans. A missing `createIndex` privilege or a
    conflicting existing index is not retried; any other error is retried on
    the next operation.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    await asyncio.to_thread(self._create_indexes)

  def _create_indexes(self) -> None:
    done = True
    for collection, keys, name in (
        (
            self._events(),
            _EVENTS_HISTORY_INDEX_KEYS,
            _EVENTS_HISTORY_INDEX_NAME,
        ),
        (
            self._sessions(),
            _SESSIONS_LIST_INDEX_KEYS,
            _SESSIONS_LIST_INDEX_NAME,
        ),
    ):
      try:
        collection.create_index(keys, name=name)
      # Indexes only speed up reads, so no index error may fail a session
      # call. pymongo stays an optional import, so match the error by shape.
      except Exception as exc:
        permanent = _client.is_index_privilege_error(exc) or (
            getattr(exc, "code", None) in _INDEX_CONFLICT_ERROR_CODES
        )
        # Leave the flag unset after a transient failure so a later
        # operation retries.
        done = done and permanent
        logger.warning(
            "MongoDbSessionService could not create the recommended index"
            " %r; reads that need it fall back to unindexed scans.%s Pass"
            " ensure_indexes=False to skip index creation. Details: %s",
            name,
            "" if permanent else " Will retry on the next operation.",
            exc,
        )
    self._indexes_ensured = done

  @staticmethod
  def _session_key(
      app_name: str, user_id: str, session_id: str
  ) -> dict[str, str]:
    return {
        "app_name": app_name,
        "user_id": user_id,
        "session_id": session_id,
    }

  @staticmethod
  def _user_key(app_name: str, user_id: str) -> dict[str, str]:
    return {"app_name": app_name, "user_id": user_id}

  @staticmethod
  def _event_key(
      app_name: str, user_id: str, session_id: str, event_id: str
  ) -> dict[str, str]:
    return {
        "app_name": app_name,
        "user_id": user_id,
        "session_id": session_id,
        "event_id": event_id,
    }

  @asynccontextmanager
  async def _with_session_lock(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> AsyncIterator[None]:
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

  def _run_in_transaction(self, work: Callable[[Any], _T]) -> _T:
    """Runs `work(mongo_session)` as one atomic unit when possible.

    Multi-document transactions make the state-bucket merges, the session
    revision bump, and the event write commit or abort together, and a
    concurrent conflicting writer loses to a write-conflict retry instead of
    silently overwriting. Transactions require a replica set, sharded
    cluster, or Atlas deployment. Where they are unavailable (a standalone
    mongod, or an in-memory fake in tests) the work runs directly without a
    session, preserving the pre-transaction behavior.

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
      doc_id: str | dict[str, str],
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
    await self._ensure_indexes_once()

    def _create(
        mongo_session: Any,
    ) -> tuple[str, dict[str, Any], float]:
      sid = session_id or platform_uuid.new_uuid()
      state_deltas = _session_util.extract_json_safe_state_delta(state or {})

      now = platform_time.get_time()
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
      merged = self._merge_state(app_state, user_state, state_deltas["session"])
      return sid, merged, now

    sid, merged_state, now = await asyncio.to_thread(
        self._run_in_transaction, _create
    )
    session = Session(
        id=sid,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
        events=[],
        last_update_time=now,
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
    await self._ensure_indexes_once()

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

        cursor = (
            self._events()
            .find(query)
            .sort([("timestamp", -1), ("revision", -1)])
        )
        if config and config.num_recent_events is not None:
          cursor = cursor.limit(config.num_recent_events)
        event_docs = list(cursor)
        event_docs.reverse()  # restore chronological order
        events = [
            Event.model_validate_json(event_doc["event_data"])
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
    await self._ensure_indexes_once()

    def _list() -> list[Session]:
      query: dict[str, Any] = {"app_name": app_name}
      if user_id is not None:
        query["user_id"] = user_id
      docs = list(self._sessions().find(query))

      app_state = self._read_app_state(app_name)
      user_ids = {doc["user_id"] for doc in docs}
      user_docs = self._user_states().find(
          {"_id": {"$in": [self._user_key(app_name, uid) for uid in user_ids]}}
      )
      user_states = {
          doc["_id"]["user_id"]: _loads_state(doc.get("state"))
          for doc in user_docs
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
    await self._ensure_indexes_once()

    def _delete(mongo_session: Any) -> None:
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
    state_deltas = _session_util.extract_json_safe_state_delta(state_delta)

    await self._ensure_indexes_once()
    async with self._with_session_lock(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    ):

      def _append(mongo_session: Any) -> int:
        session_doc_id = self._session_key(
            session.app_name, session.user_id, session.id
        )
        current = self._sessions().find_one(
            {"_id": session_doc_id}, session=mongo_session
        )
        if not current:
          raise SessionNotFoundError(f"Session {session.id} not found.")
        current_revision = current.get("revision", 0)
        if session._storage_update_marker is not None:
          if session._storage_update_marker != str(current_revision):
            raise StaleSessionError(_STALE_SESSION_ERROR_MESSAGE)
        elif current.get("update_time", 0.0) > session.last_update_time:
          raise StaleSessionError(_STALE_SESSION_ERROR_MESSAGE)

        event_doc_id = self._event_key(
            session.app_name, session.user_id, session.id, event.id
        )
        # Checked before any write: without a transaction, a duplicate caught
        # by the insert below would leave the revision bump and state merges.
        if self._events().find_one(
            {"_id": event_doc_id}, session=mongo_session
        ):
          raise AlreadyExistsError(f"Event {event.id} already exists.")

        session_only_state = _loads_state(current.get("state"))
        session_only_state.update(state_deltas["session"])

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

        new_revision = int(updated.get("revision", current_revision + 1))

        self._events().insert_one(
            {
                "_id": event_doc_id,
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "timestamp": event.timestamp,
                "revision": new_revision,
                "event_data": event.model_dump_json(exclude_none=True),
            },
            session=mongo_session,
        )
        return new_revision

      new_revision = await asyncio.to_thread(self._run_in_transaction, _append)
      session._storage_update_marker = str(new_revision)
      session.last_update_time = event.timestamp

    return self._commit_event_to_session(session, event)

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
