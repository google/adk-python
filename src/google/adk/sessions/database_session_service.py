# Copyright 2025 Google LLC
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
from datetime import datetime
import logging
from typing import Any
from typing import Optional

from sqlalchemy import delete
from sqlalchemy import event
from sqlalchemy import select
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as DatabaseSessionFactory
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import MetaData
from typing_extensions import override
from tzlocal import get_localzone

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .schemas.v0 import Base as BaseV0
from .schemas.v0 import StorageAppState as StorageAppStateV0
from .schemas.v0 import StorageEvent as StorageEventV0
from .schemas.v0 import StorageSession as StorageSessionV0
from .schemas.v0 import StorageUserState as StorageUserStateV0
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)


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


class DatabaseSessionService(BaseSessionService):
  """A session service that uses a database for storage."""

  def __init__(self, db_url: str, **kwargs: Any):
    """Initializes the database session service with a database URL."""
    # 1. Create DB engine for db connection
    # 2. Create all tables based on schema
    # 3. Initialize all properties
    try:
      db_engine = create_async_engine(db_url, **kwargs)
      if db_engine.dialect.name == "sqlite":
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

    # Get the local timezone
    local_timezone = get_localzone()
    logger.info("Local timezone: %s", local_timezone)

    self.db_engine: AsyncEngine = db_engine
    self.metadata: MetaData = MetaData()

    # DB session factory method
    self.database_session_factory: async_sessionmaker[
        DatabaseSessionFactory
    ] = async_sessionmaker(bind=self.db_engine, expire_on_commit=False)

    # Flag to indicate if tables are created
    self._tables_created = False
    # Lock to ensure thread-safe table creation
    self._table_creation_lock = asyncio.Lock()

  async def _ensure_tables_created(self):
    """Ensure database tables are created. This is called lazily."""
    if self._tables_created:
      return

    async with self._table_creation_lock:
      # Double-check after acquiring the lock
      if not self._tables_created:
        async with self.db_engine.begin() as conn:
          # Uncomment to recreate DB every time
          # await conn.run_sync(BaseV0.metadata.drop_all)
          await conn.run_sync(BaseV0.metadata.create_all)
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
    # 1. Populate states.
    # 2. Build storage session object
    # 3. Add the object to the table
    # 4. Build the session object with generated id
    # 5. Return the session
    await self._ensure_tables_created()
    async with self.database_session_factory() as sql_session:
      StorageSession = StorageSessionV0
      StorageAppState = StorageAppStateV0
      StorageUserState = StorageUserStateV0

      if session_id and await sql_session.get(
          StorageSession, (app_name, user_id, session_id)
      ):
        raise AlreadyExistsError(
            f"Session with id {session_id} already exists."
        )
      # Fetch app and user states from storage
      storage_app_state = await sql_session.get(StorageAppState, (app_name))
      storage_user_state = await sql_session.get(
          StorageUserState, (app_name, user_id)
      )

      # Create state tables if not exist
      if not storage_app_state:
        storage_app_state = StorageAppState(app_name=app_name, state={})
        sql_session.add(storage_app_state)
      if not storage_user_state:
        storage_user_state = StorageUserState(
            app_name=app_name, user_id=user_id, state={}
        )
        sql_session.add(storage_user_state)

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
      storage_session = StorageSession(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=session_state,
      )
      sql_session.add(storage_session)
      await sql_session.commit()

      await sql_session.refresh(storage_session)

      # Merge states for response
      merged_state = _merge_state(
          storage_app_state.state, storage_user_state.state, session_state
      )
      session = storage_session.to_session(state=merged_state)
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
    await self._ensure_tables_created()
    # 1. Get the storage session entry from session table
    # 2. Get all the events based on session id and filtering config
    # 3. Convert and return the session
    async with self.database_session_factory() as sql_session:
      StorageSession = StorageSessionV0
      StorageEvent = StorageEventV0
      StorageAppState = StorageAppStateV0
      StorageUserState = StorageUserStateV0

      storage_session = await sql_session.get(
          StorageSession, (app_name, user_id, session_id)
      )
      if storage_session is None:
        return None

      stmt = (
          select(StorageEvent)
          .filter(StorageEvent.app_name == app_name)
          .filter(StorageEvent.session_id == storage_session.id)
          .filter(StorageEvent.user_id == user_id)
      )

      if config and config.after_timestamp:
        after_dt = datetime.fromtimestamp(config.after_timestamp)
        stmt = stmt.filter(StorageEvent.timestamp >= after_dt)

      stmt = stmt.order_by(StorageEvent.timestamp.desc())

      if config and config.num_recent_events:
        stmt = stmt.limit(config.num_recent_events)

      result = await sql_session.execute(stmt)
      storage_events = result.scalars().all()

      # Fetch states from storage
      storage_app_state = await sql_session.get(StorageAppState, (app_name))
      storage_user_state = await sql_session.get(
          StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state

      # Merge states
      merged_state = _merge_state(app_state, user_state, session_state)

      # Convert storage session to session
      events = [e.to_event() for e in reversed(storage_events)]
      session = storage_session.to_session(state=merged_state, events=events)
    return session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    await self._ensure_tables_created()
    async with self.database_session_factory() as sql_session:
      StorageSession = StorageSessionV0
      StorageAppState = StorageAppStateV0
      StorageUserState = StorageUserStateV0

      stmt = select(StorageSession).filter(StorageSession.app_name == app_name)
      if user_id is not None:
        stmt = stmt.filter(StorageSession.user_id == user_id)

      result = await sql_session.execute(stmt)
      results = result.scalars().all()

      # Fetch app state from storage
      storage_app_state = await sql_session.get(StorageAppState, (app_name))
      app_state = storage_app_state.state if storage_app_state else {}

      # Fetch user state(s) from storage
      user_states_map = {}
      if user_id is not None:
        storage_user_state = await sql_session.get(
            StorageUserState, (app_name, user_id)
        )
        if storage_user_state:
          user_states_map[user_id] = storage_user_state.state
      else:
        user_state_stmt = select(StorageUserState).filter(
            StorageUserState.app_name == app_name
        )
        user_state_result = await sql_session.execute(user_state_stmt)
        all_user_states_for_app = user_state_result.scalars().all()
        for storage_user_state in all_user_states_for_app:
          user_states_map[storage_user_state.user_id] = storage_user_state.state

      sessions = []
      for storage_session in results:
        session_state = storage_session.state
        user_state = user_states_map.get(storage_session.user_id, {})
        merged_state = _merge_state(app_state, user_state, session_state)
        sessions.append(storage_session.to_session(state=merged_state))
      return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, app_name: str, user_id: str, session_id: str
  ) -> None:
    await self._ensure_tables_created()
    async with self.database_session_factory() as sql_session:
      StorageSession = StorageSessionV0

      stmt = delete(StorageSession).where(
          StorageSession.app_name == app_name,
          StorageSession.user_id == user_id,
          StorageSession.id == session_id,
      )
      await sql_session.execute(stmt)
      await sql_session.commit()

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    await self._ensure_tables_created()
    if event.partial:
      return event

    # Trim temp state before persisting
    event = self._trim_temp_delta_state(event)

    # 1. Check if timestamp is stale
    # 2. Update session attributes based on event config
    # 3. Store event to table
    async with self.database_session_factory() as sql_session:
      StorageSession = StorageSessionV0
      StorageEvent = StorageEventV0
      StorageAppState = StorageAppStateV0
      StorageUserState = StorageUserStateV0

      storage_session = await sql_session.get(
          StorageSession, (session.app_name, session.user_id, session.id)
      )

      if storage_session.update_timestamp_tz > session.last_update_time:
        raise ValueError(
            "The last_update_time provided in the session object"
            f" {datetime.fromtimestamp(session.last_update_time):'%Y-%m-%d %H:%M:%S'} is"
            " earlier than the update_time in the storage_session"
            f" {datetime.fromtimestamp(storage_session.update_timestamp_tz):'%Y-%m-%d %H:%M:%S'}."
            " Please check if it is a stale session."
        )

      # Fetch states from storage
      storage_app_state = await sql_session.get(
          StorageAppState, (session.app_name)
      )
      storage_user_state = await sql_session.get(
          StorageUserState, (session.app_name, session.user_id)
      )

      # Extract state delta
      if event.actions and event.actions.state_delta:
        state_deltas = _session_util.extract_state_delta(
            event.actions.state_delta
        )
        app_state_delta = state_deltas["app"]
        user_state_delta = state_deltas["user"]
        session_state_delta = state_deltas["session"]
        # Merge state and update storage
        if app_state_delta:
          storage_app_state.state = storage_app_state.state | app_state_delta
        if user_state_delta:
          storage_user_state.state = storage_user_state.state | user_state_delta
        if session_state_delta:
          storage_session.state = storage_session.state | session_state_delta

      if storage_session._dialect_name == "sqlite":
        update_time = datetime.utcfromtimestamp(event.timestamp)
      else:
        update_time = datetime.fromtimestamp(event.timestamp)
      storage_session.update_time = update_time
      sql_session.add(StorageEvent.from_event(session, event))

      await sql_session.commit()
      await sql_session.refresh(storage_session)

      # Update timestamp with commit time
      session.last_update_time = storage_session.update_timestamp_tz

    # Also update the in-memory session
    await super().append_event(session=session, event=event)
    return event
