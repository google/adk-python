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
from datetime import datetime
from datetime import timezone
import logging
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import TypeAlias
from typing import TypeVar

from google.adk.platform import time as platform_time
from sqlalchemy import delete
from sqlalchemy import event
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy.engine import Connection
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as DatabaseSessionFactory
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from typing_extensions import override

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .migration import _schema_check_utils
from .schemas.v0 import Base as BaseV0
from .schemas.v0 import StorageAppState as StorageAppStateV0
from .schemas.v0 import StorageEvent as StorageEventV0
from .schemas.v0 import StorageSession as StorageSessionV0
from .schemas.v0 import StorageUserState as StorageUserStateV0
from .schemas.v1 import Base as BaseV1
from .schemas.v1 import StorageAppState as StorageAppStateV1
from .schemas.v1 import StorageEvent as StorageEventV1
from .schemas.v1 import StorageMetadata
from .schemas.v1 import StorageSession as StorageSessionV1
from .schemas.v1 import StorageUserState as StorageUserStateV1
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

_STALE_SESSION_ERROR_MESSAGE = (
    "The session has been modified in storage since it was loaded. "
    "Please reload the session before appending more events."
)

_SQLITE_DIALECT = "sqlite"
_MARIADB_DIALECT = "mariadb"
_MYSQL_DIALECT = "mysql"
_POSTGRESQL_DIALECT = "postgresql"
# Tuple key order for in-process per-session lock maps:
# (app_name, user_id, session_id).
_SessionLockKey: TypeAlias = tuple[str, str, str]
_StorageStateT = TypeVar(
    "_StorageStateT",
    StorageAppStateV0,
    StorageAppStateV1,
    StorageUserStateV0,
    StorageUserStateV1,
)


async def _select_required_state(
    *,
    sql_session: DatabaseSessionFactory,
    state_model: type[_StorageStateT],
    predicates: tuple[Any, ...],
    use_row_level_locking: bool,
    missing_message: str,
) -> _StorageStateT:
  """Returns a state row, raising if the row is missing."""
  stmt = select(state_model).filter(*predicates)
  if use_row_level_locking:
    stmt = stmt.with_for_update()
  result = await sql_session.execute(stmt)
  state_row = result.scalars().one_or_none()
  if state_row is None:
    raise ValueError(missing_message)
  return state_row


async def _get_or_create_state(
    *,
    sql_session: DatabaseSessionFactory,
    state_model: type[_StorageStateT],
    primary_key: Any,
    defaults: dict[str, Any],
) -> _StorageStateT:
  """Returns an existing state row or creates one, handling concurrent inserts.

  Uses a SAVEPOINT so that an IntegrityError from a racing INSERT does not
  invalidate the outer transaction.
  """
  row = await sql_session.get(state_model, primary_key)
  if row is not None:
    return row
  try:
    async with sql_session.begin_nested():
      row = state_model(**defaults)
      sql_session.add(row)
    return row
  except IntegrityError:
    # Another concurrent caller inserted the row first.
    # The savepoint was rolled back, so re-fetch the winner's row.
    row = await sql_session.get(state_model, primary_key)
    if row is None:
      raise
    return row


def _set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA foreign_keys=ON")
  cursor.close()


def _ensure_schema_indexes_exist(
    connection: Connection, metadata: MetaData
) -> None:
  """Ensures indexes declared in metadata exist for existing tables."""
  logger.debug("Ensuring schema indexes exist for metadata tables.")
  for table in metadata.sorted_tables:
    for index in sorted(table.indexes, key=lambda item: item.name or ""):
      index.create(bind=connection, checkfirst=True)


def _setup_database_schema(connection: Connection, metadata: MetaData) -> None:
  """Ensures tables and indexes declared in metadata exist."""
  metadata.create_all(bind=connection)
  _ensure_schema_indexes_exist(connection, metadata)


def _merge_state(
    app_state: dict[str, Any],
    user_state: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
  """Merge app, user, and session states into a single state dictionary."""
  merged_state = copy.deepcopy(session_state)
  for key in app_state.keys():
    merged_state[State.APP_PREFIX + key] = app_state[key]
  for key in user_state.keys():
    merged_state[State.USER_PREFIX + key] = user_state[key]
  return merged_state


class _SchemaClasses:
  """A helper class to hold schema classes based on version."""

  def __init__(self, version: str):
    if version == _schema_check_utils.LATEST_SCHEMA_VERSION:
      self.StorageSession = StorageSessionV1
      self.StorageAppState = StorageAppStateV1
      self.StorageUserState = StorageUserStateV1
      self.StorageEvent = StorageEventV1
    else:
      self.StorageSession = StorageSessionV0
      self.StorageAppState = StorageAppStateV0
      self.StorageUserState = StorageUserStateV0
      self.StorageEvent = StorageEventV0


class DatabaseSessionService(BaseSessionService):
  """A session service that uses a database for storage."""

  def __init__(self, db_url: str, **kwargs: Any):
    """Initializes the database session service with a database URL."""
    # 1. Create DB engine for db connection
    # 2. Create all tables based on schema
    # 3. Initialize all properties
    try:
      engine_kwargs = dict(kwargs)
      url = make_url(db_url)
      if (
          url.get_backend_name() == _SQLITE_DIALECT
          and url.database == ":memory:"
      ):
        engine_kwargs.setdefault("poolclass", StaticPool)
        connect_args = dict(engine_kwargs.get("connect_args", {}))
        connect_args.setdefault("check_same_thread", False)
        engine_kwargs["connect_args"] = connect_args
      elif url.get_backend_name() != _SQLITE_DIALECT:
        engine_kwargs.setdefault("pool_pre_ping", True)

      db_engine = create_async_engine(db_url, **engine_kwargs)
      if db_engine.dialect.name == _SQLITE_DIALECT:
        # Set sqlite pragma to enable foreign keys constraints
        event.listen(db_engine.sync_engine, "connect", _set_sqlite_pragma)

    except Exception as e:
      if isinstance(e, ArgumentError):
        raise ValueError(
            f"Invalid database URL format or argument '{db_url}'."
        ) from e
      if isinstance(e, ImportError):
        raise ValueError(
            f"Database related module not found for URL '{db_url}'."
        ) from e
      raise ValueError(
          f"Failed to create database engine for URL '{db_url}'"
      ) from e

    self.db_engine: AsyncEngine = db_engine
    # DB session factory method
    self.database_session_factory: async_sessionmaker[
        DatabaseSessionFactory
    ] = async_sessionmaker(bind=self.db_engine, expire_on_commit=False)
    read_only_engine = self.db_engine.execution_options(read_only=True)
    self._read_only_database_session_factory: async_sessionmaker[
        DatabaseSessionFactory
    ] = async_sessionmaker(bind=read_only_engine, expire_on_commit=False)

    # Flag to indicate if tables are created
    self._tables_created = False

    # Lock to ensure thread-safe table creation
    self._table_creation_lock = asyncio.Lock()

    # The current database schema version in use, "None" if not yet checked
    self._db_schema_version: Optional[str] = None

    # Per-session locks used to serialize append_event calls in this process.
    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

  def _get_schema_classes(self) -> _SchemaClasses:
    return _SchemaClasses(self._db_schema_version)

  def _get_database_session_factory(
      self, *, read_only: bool = False
  ) -> async_sessionmaker[DatabaseSessionFactory]:
    if read_only:
      return self._read_only_database_session_factory
    return self.database_session_factory

  @asynccontextmanager
  async def _rollback_on_exception_session(
      self,
      *,
      read_only: bool = False,
  ) -> AsyncIterator[DatabaseSessionFactory]:
    """Yields a database session with guaranteed rollback on errors.

    On normal exit the caller is responsible for committing; on any exception
    the transaction is explicitly rolled back before the error propagates,
    preventing connection-pool exhaustion from lingering invalid transactions.
    """
    session_factory = self._get_database_session_factory(read_only=read_only)
    async with session_factory() as sql_session:
      try:
        yield sql_session
      except BaseException:
        await sql_session.rollback()
        raise

  def _supports_row_level_locking(self) -> bool:
    return self.db_engine.dialect.name in (
        _MARIADB_DIALECT,
        _MYSQL_DIALECT,
        _POSTGRESQL_DIALECT,
    )

  @asynccontextmanager
  async def _with_session_lock(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> AsyncIterator[None]:
    """Serializes event appends for the same session within this process."""
    # Use one lock per logical ADK session to prevent concurrent append_event
    # writes from racing in the same process.
    lock_key = (app_name, user_id, session_id)
    async with self._session_locks_guard:
      lock = self._session_locks.get(lock_key)
      if lock is None:
        lock = asyncio.Lock()
        self._session_locks[lock_key] = lock
      # Reference counting keeps lock objects alive while they are in use by
      # concurrent tasks and allows cleanup once all waiters complete.
      self._session_lock_ref_count[lock_key] = (
          self._session_lock_ref_count.get(lock_key, 0) + 1
      )

    try:
      async with lock:
        yield
    finally:
      async with self._session_locks_guard:
        remaining = self._session_lock_ref_count.get(lock_key, 0) - 1
        # Remove lock bookkeeping after the last waiter exits.
        if remaining <= 0 and not lock.locked():
          self._session_lock_ref_count.pop(lock_key, None)
          self._session_locks.pop(lock_key, None)
        else:
          self._session_lock_ref_count[lock_key] = remaining

  async def _prepare_tables(self):
    """Ensure database tables are ready for use.

    This method is called lazily before each database operation. It checks the
    DB schema version to use and creates the tables (including setting the
    schema version metadata) if needed.
    """
    # Early return if tables are already created
    if self._tables_created:
      return

    async with self._table_creation_lock:
      # Double-check after acquiring the lock
      if self._tables_created:
        return

      # Check the database schema version and set the _db_schema_version
      if self._db_schema_version is None:
        try:
          async with self.db_engine.connect() as conn:
            self._db_schema_version = await conn.run_sync(
                _schema_check_utils.get_db_schema_version_from_connection
            )
        except Exception as e:
          logger.error("Failed to inspect database tables: %s", e)
          raise

      async with self.db_engine.begin() as conn:
        if self._db_schema_version == _schema_check_utils.LATEST_SCHEMA_VERSION:
          # Uncomment to recreate DB every time
          # await conn.run_sync(BaseV1.metadata.drop_all)
          logger.debug("Using V1 schema tables...")
          await conn.run_sync(_setup_database_schema, BaseV1.metadata)
        else:
          # await conn.run_sync(BaseV0.metadata.drop_all)
          logger.debug("Using V0 schema tables...")
          await conn.run_sync(_setup_database_schema, BaseV0.metadata)

      if self._db_schema_version == _schema_check_utils.LATEST_SCHEMA_VERSION:
        async with self._rollback_on_exception_session() as sql_session:
          # Check if schema version is set, if not, set it to the latest
          # version
          stmt = select(StorageMetadata).where(
              StorageMetadata.key == _schema_check_utils.SCHEMA_VERSION_KEY
          )
          result = await sql_session.execute(stmt)
          metadata = result.scalars().first()
          if not metadata:
            metadata = StorageMetadata(
                key=_schema_check_utils.SCHEMA_VERSION_KEY,
                value=_schema_check_utils.LATEST_SCHEMA_VERSION,
            )
            sql_session.add(metadata)
            await sql_session.commit()

      self._tables_created = True

  async def _session_matches_storage_revision(
      self,
      *,
      sql_session: DatabaseSessionFactory,
      schema: _SchemaClasses,
      session: Session,
  ) -> bool:
    """Returns whether a marker-less session still matches stored events."""
    if not session.events:
      stmt = (
          select(schema.StorageEvent.id)
          .filter(schema.StorageEvent.app_name == session.app_name)
          .filter(schema.StorageEvent.session_id == session.id)
          .filter(schema.StorageEvent.user_id == session.user_id)
          .limit(1)
      )
      result = await sql_session.execute(stmt)
      return result.scalar_one_or_none() is None

    stmt = (
        select(schema.StorageEvent.id)
        .filter(schema.StorageEvent.app_name == session.app_name)
        .filter(schema.StorageEvent.session_id == session.id)
        .filter(schema.StorageEvent.user_id == session.user_id)
        .order_by(
            schema.StorageEvent.timestamp.desc(), schema.StorageEvent.id.desc()
        )
        .limit(1)
    )
    result = await sql_session.execute(stmt)
    latest_storage_event_id = result.scalar_one_or_none()
    return latest_storage_event_id == session.events[-1].id

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    # 1. Populate states.
    # 2. Build storage session object
    # 3. Add the object to the table
    # 4. Build the session object with generated id
    # 5. Return the session
    await self._prepare_tables()
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session() as sql_session:
      if session_id and await sql_session.get(
          schema.StorageSession, (app_name, user_id, session_id)
      ):
        raise AlreadyExistsError(
            f"Session with id {session_id} already exists."
        )
      # Get or create state rows, handling concurrent insert races.
      storage_app_state = await _get_or_create_state(
          sql_session=sql_session,
          state_model=schema.StorageAppState,
          primary_key=app_name,
          defaults={"app_name": app_name, "state": {}},
      )
      storage_user_state = await _get_or_create_state(
          sql_session=sql_session,
          state_model=schema.StorageUserState,
          primary_key=(app_name, user_id),
          defaults={"app_name": app_name, "user_id": user_id, "state": {}},
      )

      # Extract state deltas
      state_deltas = _session_util.extract_state_delta(state)
      app_state_delta = state_deltas["app"]
      user_state_delta = state_deltas["user"]
      session_state = state_deltas["session"]

      # Apply state delta
      if app_state_delta:
        storage_app_state.state = storage_app_state.state | app_state_delta
      if user_state_delta:
        storage_user_state.state = storage_user_state.state | user_state_delta

      # Store the session
      now = datetime.fromtimestamp(platform_time.get_time(), tz=timezone.utc)
      is_sqlite = self.db_engine.dialect.name == _SQLITE_DIALECT
      is_postgresql = self.db_engine.dialect.name == _POSTGRESQL_DIALECT
      if is_sqlite or is_postgresql:
        now = now.replace(tzinfo=None)

      storage_session = schema.StorageSession(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=session_state,
          create_time=now,
          update_time=now,
      )
      sql_session.add(storage_session)
      await sql_session.commit()

      # Merge states for response
      merged_state = _merge_state(
          storage_app_state.state, storage_user_state.state, session_state
      )
      session = storage_session.to_session(
          state=merged_state, is_sqlite=is_sqlite
      )
    return session

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    await self._prepare_tables()
    # 1. Get the storage session entry from session table
    # 2. Get all the events based on session id and filtering config
    # 3. Convert and return the session
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session(
        read_only=True
    ) as sql_session:
      storage_session = await sql_session.get(
          schema.StorageSession, (app_name, user_id, session_id)
      )
      if storage_session is None:
        return None

      stmt = (
          select(schema.StorageEvent)
          .filter(schema.StorageEvent.app_name == app_name)
          .filter(schema.StorageEvent.session_id == storage_session.id)
          .filter(schema.StorageEvent.user_id == user_id)
      )

      if config and config.after_timestamp:
        after_dt = datetime.fromtimestamp(config.after_timestamp)
        stmt = stmt.filter(schema.StorageEvent.timestamp >= after_dt)

      stmt = stmt.order_by(schema.StorageEvent.timestamp.desc())

      if config and config.num_recent_events:
        stmt = stmt.limit(config.num_recent_events)

      result = await sql_session.execute(stmt)
      storage_events = result.scalars().all()

      # Fetch states from storage
      storage_app_state = await sql_session.get(
          schema.StorageAppState, (app_name)
      )
      storage_user_state = await sql_session.get(
          schema.StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state

      # Merge states
      merged_state = _merge_state(app_state, user_state, session_state)

      # Convert storage session to session
      events = [e.to_event() for e in reversed(storage_events)]
      is_sqlite = self.db_engine.dialect.name == _SQLITE_DIALECT
      session = storage_session.to_session(
          state=merged_state, events=events, is_sqlite=is_sqlite
      )
    return session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    await self._prepare_tables()
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session(
        read_only=True
    ) as sql_session:
      stmt = select(schema.StorageSession).filter(
          schema.StorageSession.app_name == app_name
      )
      if user_id is not None:
        stmt = stmt.filter(schema.StorageSession.user_id == user_id)

      result = await sql_session.execute(stmt)
      results = result.scalars().all()

      # Fetch app state from storage
      storage_app_state = await sql_session.get(
          schema.StorageAppState, (app_name)
      )
      app_state = storage_app_state.state if storage_app_state else {}

      # Fetch user state(s) from storage
      user_states_map = {}
      if user_id is not None:
        storage_user_state = await sql_session.get(
            schema.StorageUserState, (app_name, user_id)
        )
        if storage_user_state:
          user_states_map[user_id] = storage_user_state.state
      else:
        user_state_stmt = select(schema.StorageUserState).filter(
            schema.StorageUserState.app_name == app_name
        )
        user_state_result = await sql_session.execute(user_state_stmt)
        all_user_states_for_app = user_state_result.scalars().all()
        for storage_user_state in all_user_states_for_app:
          user_states_map[storage_user_state.user_id] = storage_user_state.state

      sessions = []
      is_sqlite = self.db_engine.dialect.name == _SQLITE_DIALECT
      for storage_session in results:
        session_state = storage_session.state
        user_state = user_states_map.get(storage_session.user_id, {})
        merged_state = _merge_state(app_state, user_state, session_state)
        sessions.append(
            storage_session.to_session(state=merged_state, is_sqlite=is_sqlite)
        )
      return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, app_name: str, user_id: str, session_id: str
  ) -> None:
    await self._prepare_tables()
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session() as sql_session:
      stmt = delete(schema.StorageSession).where(
          schema.StorageSession.app_name == app_name,
          schema.StorageSession.user_id == user_id,
          schema.StorageSession.id == session_id,
      )
      await sql_session.execute(stmt)
      await sql_session.commit()

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    await self._prepare_tables()
    if event.partial:
      return event

    # Apply temp state to in-memory session before trimming, so that
    # subsequent agents within the same invocation can read temp values.
    self._apply_temp_state(session, event)
    # Trim temp state before persisting
    event = self._trim_temp_delta_state(event)

    # 1. Validate the session has not gone stale.
    # 2. Update session attributes based on event config.
    # 3. Store the new event.
    schema = self._get_schema_classes()
    is_sqlite = self.db_engine.dialect.name == _SQLITE_DIALECT
    use_row_level_locking = self._supports_row_level_locking()

    state_delta = (
        event.actions.state_delta
        if event.actions and event.actions.state_delta
        else {}
    )
    state_deltas = _session_util.extract_state_delta(state_delta)
    has_app_delta = bool(state_deltas["app"])
    has_user_delta = bool(state_deltas["user"])

    async with self._with_session_lock(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    ):
      async with self._rollback_on_exception_session() as sql_session:
        storage_session_stmt = (
            select(schema.StorageSession)
            .filter(schema.StorageSession.app_name == session.app_name)
            .filter(schema.StorageSession.user_id == session.user_id)
            .filter(schema.StorageSession.id == session.id)
        )
        if use_row_level_locking:
          storage_session_stmt = storage_session_stmt.with_for_update()
        storage_session_result = await sql_session.execute(storage_session_stmt)
        storage_session = storage_session_result.scalars().one_or_none()
        if storage_session is None:
          raise ValueError(f"Session {session.id} not found.")
        storage_update_time = storage_session.get_update_timestamp(is_sqlite)
        storage_update_marker = storage_session.get_update_marker()

        storage_app_state = await _select_required_state(
            sql_session=sql_session,
            state_model=schema.StorageAppState,
            predicates=(schema.StorageAppState.app_name == session.app_name,),
            use_row_level_locking=use_row_level_locking and has_app_delta,
            missing_message=(
                "App state missing for app_name="
                f"{session.app_name!r}. Session state tables should be "
                "initialized by create_session."
            ),
        )
        storage_user_state = await _select_required_state(
            sql_session=sql_session,
            state_model=schema.StorageUserState,
            predicates=(
                schema.StorageUserState.app_name == session.app_name,
                schema.StorageUserState.user_id == session.user_id,
            ),
            use_row_level_locking=use_row_level_locking and has_user_delta,
            missing_message=(
                "User state missing for app_name="
                f"{session.app_name!r}, user_id={session.user_id!r}. "
                "Session state tables should be initialized by "
                "create_session."
            ),
        )

        if session._storage_update_marker is not None:
          # Sessions loaded by DatabaseSessionService carry an exact storage
          # revision marker, so stale-writer detection can use that marker
          # instead of relying on rounded timestamps.
          if session._storage_update_marker != storage_update_marker:
            raise ValueError(_STALE_SESSION_ERROR_MESSAGE)
          # Keep the float timestamp synchronized with the exact storage value
          # so tiny round-trip differences do not trigger false stale checks on
          # the next append.
          session.last_update_time = storage_update_time
        elif storage_update_time > session.last_update_time:
          # Backward-compatible fallback for marker-less session objects, such
          # as older in-memory sessions or manually constructed Session values.
          # Only reject when storage has actually advanced beyond the in-memory
          # revision represented by session.events.
          if not await self._session_matches_storage_revision(
              sql_session=sql_session, schema=schema, session=session
          ):
            raise ValueError(_STALE_SESSION_ERROR_MESSAGE)
          session.last_update_time = storage_update_time
        session._storage_update_marker = storage_update_marker

        # Merge pre-extracted state deltas into storage.
        if has_app_delta:
          storage_app_state.state = (
              storage_app_state.state | state_deltas["app"]
          )
        if has_user_delta:
          storage_user_state.state = (
              storage_user_state.state | state_deltas["user"]
          )
        if state_deltas["session"]:
          storage_session.state = (
              storage_session.state | state_deltas["session"]
          )

        if is_sqlite:
          update_time = datetime.fromtimestamp(
              event.timestamp, timezone.utc
          ).replace(tzinfo=None)
        else:
          update_time = datetime.fromtimestamp(event.timestamp)
        storage_session.update_time = update_time
        sql_session.add(schema.StorageEvent.from_event(session, event))

        await sql_session.commit()

        # Update timestamp with commit time
        session.last_update_time = storage_session.get_update_timestamp(
            is_sqlite
        )
        session._storage_update_marker = storage_session.get_update_marker()

    # Also update the in-memory session
    await super().append_event(session=session, event=event)
    return event

  async def close(self) -> None:
    """Disposes the SQLAlchemy engine and closes pooled connections."""
    await self.db_engine.dispose()

  async def __aenter__(self) -> DatabaseSessionService:
    """Enters the async context manager and returns this service."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exits the async context manager and closes the service."""
    await self.close()
