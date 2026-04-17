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
import copy
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import TypeAlias
from typing import TypeVar

from google.adk.platform import time as platform_time
from sqlalchemy import delete
from sqlalchemy import event
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as DatabaseSessionFactory
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm.exc import StaleDataError
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


def _set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA foreign_keys=ON")
  cursor.close()


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
    try:
      engine_kwargs = dict(kwargs)
      url = make_url(db_url)
      if url.get_backend_name() == _SQLITE_DIALECT:
        engine_kwargs.setdefault("poolclass", StaticPool)
        connect_args = dict(engine_kwargs.get("connect_args", {}))
        connect_args.setdefault("check_same_thread", False)
        connect_args.setdefault("isolation_level", "IMMEDIATE")
        engine_kwargs["connect_args"] = connect_args
      elif url.get_backend_name() != _SQLITE_DIALECT:
        engine_kwargs.setdefault("pool_pre_ping", True)

      db_engine = create_async_engine(db_url, **engine_kwargs)
      if db_engine.dialect.name == _SQLITE_DIALECT:
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

    self.database_session_factory: async_sessionmaker[
        DatabaseSessionFactory
    ] = async_sessionmaker(bind=self.db_engine, expire_on_commit=False)

    self._tables_created = False
    self._table_creation_lock = asyncio.Lock()
    self._db_schema_version: Optional[str] = None
    self._session_locks: dict[_SessionLockKey, asyncio.Lock] = {}
    self._session_lock_ref_count: dict[_SessionLockKey, int] = {}
    self._session_locks_guard = asyncio.Lock()

  def _get_schema_classes(self) -> _SchemaClasses:
    return _SchemaClasses(self._db_schema_version)

  @asynccontextmanager
  async def _rollback_on_exception_session(
      self,
  ) -> AsyncIterator[DatabaseSessionFactory]:
    async with self.database_session_factory() as sql_session:
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

  async def _prepare_tables(self):
    if self._tables_created:
      return

    async with self._table_creation_lock:
      if self._tables_created:
        return

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
          logger.debug("Using V1 schema tables...")
          await conn.run_sync(BaseV1.metadata.create_all)
        else:
          logger.debug("Using V0 schema tables...")
          await conn.run_sync(BaseV0.metadata.create_all)

      if self._db_schema_version == _schema_check_utils.LATEST_SCHEMA_VERSION:
        async with self._rollback_on_exception_session() as sql_session:
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

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    await self._prepare_tables()
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session() as sql_session:
      if session_id and await sql_session.get(
          schema.StorageSession, (app_name, user_id, session_id)
      ):
        raise AlreadyExistsError(
            f"Session with id {session_id} already exists."
        )
      storage_app_state = await sql_session.get(
          schema.StorageAppState, (app_name)
      )
      storage_user_state = await sql_session.get(
          schema.StorageUserState, (app_name, user_id)
      )

      if not storage_app_state:
        storage_app_state = schema.StorageAppState(app_name=app_name, state={})
        sql_session.add(storage_app_state)
      if not storage_user_state:
        storage_user_state = schema.StorageUserState(
            app_name=app_name, user_id=user_id, state={}
        )
        sql_session.add(storage_user_state)

      state_deltas = _session_util.extract_state_delta(state)
      app_state_delta = state_deltas["app"]
      user_state_delta = state_deltas["user"]
      session_state = state_deltas["session"]

      if app_state_delta:
        storage_app_state.state = storage_app_state.state | app_state_delta
      if user_state_delta:
        storage_user_state.state = storage_user_state.state | user_state_delta

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
          event_sequence=0,
      )
      sql_session.add(storage_session)
      await sql_session.commit()

      merged_state = _merge_state(
          storage_app_state.state, storage_user_state.state, session_state
      )
      session = storage_session.to_session(
          state=merged_state,
          is_sqlite=is_sqlite,
          event_sequence=storage_session.event_sequence,
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
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session() as sql_session:
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

      storage_app_state = await sql_session.get(
          schema.StorageAppState, (app_name)
      )
      storage_user_state = await sql_session.get(
          schema.StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state
      event_sequence = storage_session.event_sequence

      merged_state = _merge_state(app_state, user_state, session_state)

      events = [e.to_event() for e in reversed(storage_events)]
      is_sqlite = self.db_engine.dialect.name == _SQLITE_DIALECT
      session = storage_session.to_session(
          state=merged_state,
          events=events,
          is_sqlite=is_sqlite,
          event_sequence=event_sequence,
      )
    return session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    await self._prepare_tables()
    schema = self._get_schema_classes()
    async with self._rollback_on_exception_session() as sql_session:
      stmt = select(schema.StorageSession).filter(
          schema.StorageSession.app_name == app_name
      )
      if user_id is not None:
        stmt = stmt.filter(schema.StorageSession.user_id == user_id)

      result = await sql_session.execute(stmt)
      results = result.scalars().all()

      storage_app_state = await sql_session.get(
          schema.StorageAppState, (app_name)
      )
      app_state = storage_app_state.state if storage_app_state else {}

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
        event_sequence = storage_session.event_sequence
        sessions.append(
            storage_session.to_session(
                state=merged_state,
                is_sqlite=is_sqlite,
                event_sequence=event_sequence,
            )
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

    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)

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

        storage_session.event_sequence = session.event_sequence

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

        try:
          await sql_session.commit()
          session.event_sequence = storage_session.event_sequence
        except StaleDataError as e:
          raise ValueError(
              "The session has been modified in storage. "
              "Please reload the session before appending events."
          ) from e

        session.last_update_time = storage_session.get_update_timestamp(
            is_sqlite
        )

    await super().append_event(session=session, event=event)
    return event

  async def close(self) -> None:
    await self.db_engine.dispose()

  async def __aenter__(self) -> DatabaseSessionService:
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
