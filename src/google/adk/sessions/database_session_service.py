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
from datetime import timezone
import json
import logging
from typing import Any
from typing import Optional
import uuid

from sqlalchemy import delete
from sqlalchemy import Dialect
from sqlalchemy import event
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy import inspect
from sqlalchemy import select
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as DatabaseSessionFactory
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import DateTime
from sqlalchemy.types import String
from sqlalchemy.types import TypeDecorator
from typing_extensions import override
from tzlocal import get_localzone

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .migration import _schema_check
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256


class DynamicJSON(TypeDecorator):
  """A JSON-like type that uses JSONB on PostgreSQL and TEXT with JSON serialization for other databases."""

  impl = Text  # Default implementation is TEXT

  def load_dialect_impl(self, dialect: Dialect):
    if dialect.name == "postgresql":
      return dialect.type_descriptor(postgresql.JSONB)
    if dialect.name == "mysql":
      # Use LONGTEXT for MySQL to address the data too long issue
      return dialect.type_descriptor(mysql.LONGTEXT)
    return dialect.type_descriptor(Text)  # Default to Text for other dialects

  def process_bind_param(self, value, dialect: Dialect):
    if value is not None:
      if dialect.name == "postgresql":
        return value  # JSONB handles dict directly
      return json.dumps(value)  # Serialize to JSON string for TEXT
    return value

  def process_result_value(self, value, dialect: Dialect):
    if value is not None:
      if dialect.name == "postgresql":
        return value  # JSONB returns dict directly
      else:
        return json.loads(value)  # Deserialize from JSON string for TEXT
    return value


class PreciseTimestamp(TypeDecorator):
  """Represents a timestamp precise to the microsecond."""

  impl = DateTime
  cache_ok = True

  def load_dialect_impl(self, dialect):
    if dialect.name == "mysql":
      return dialect.type_descriptor(mysql.DATETIME(fsp=6))
    return self.impl


class Base(DeclarativeBase):
  """Base class for database tables."""

  pass


class StorageMetadata(Base):
  """Represents internal metadata stored in the database."""

  __tablename__ = "adk_internal_metadata"
  key: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  value: Mapped[str] = mapped_column(String(DEFAULT_MAX_VARCHAR_LENGTH))


class StorageSession(Base):
  """Represents a session stored in the database."""

  __tablename__ = "sessions"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH),
      primary_key=True,
      default=lambda: str(uuid.uuid4()),
  )

  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )

  create_time: Mapped[datetime] = mapped_column(
      PreciseTimestamp, default=func.now()
  )
  update_time: Mapped[datetime] = mapped_column(
      PreciseTimestamp, default=func.now(), onupdate=func.now()
  )

  storage_events: Mapped[list[StorageEvent]] = relationship(
      "StorageEvent",
      back_populates="storage_session",
  )

  def __repr__(self):
    return f"<StorageSession(id={self.id}, update_time={self.update_time})>"

  @property
  def _dialect_name(self) -> Optional[str]:
    session = inspect(self).session
    return session.bind.dialect.name if session else None

  @property
  def update_timestamp_tz(self) -> datetime:
    """Returns the time zone aware update timestamp."""
    if self._dialect_name == "sqlite":
      # SQLite does not support timezone. SQLAlchemy returns a naive datetime
      # object without timezone information. We need to convert it to UTC
      # manually.
      return self.update_time.replace(tzinfo=timezone.utc).timestamp()
    return self.update_time.timestamp()

  def to_session(
      self,
      state: dict[str, Any] | None = None,
      events: list[Event] | None = None,
  ) -> Session:
    """Converts the storage session to a session object."""
    if state is None:
      state = {}
    if events is None:
      events = []

    return Session(
        app_name=self.app_name,
        user_id=self.user_id,
        id=self.id,
        state=state,
        events=events,
        last_update_time=self.update_timestamp_tz,
    )


class StorageEvent(Base):
  """Represents an event stored in the database."""

  __tablename__ = "events"

  id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  session_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )

  invocation_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_VARCHAR_LENGTH))
  timestamp: Mapped[PreciseTimestamp] = mapped_column(
      PreciseTimestamp, default=func.now()
  )
  event_data: Mapped[dict[str, Any]] = mapped_column(DynamicJSON)

  storage_session: Mapped[StorageSession] = relationship(
      "StorageSession",
      back_populates="storage_events",
  )

  __table_args__ = (
      ForeignKeyConstraint(
          ["app_name", "user_id", "session_id"],
          ["sessions.app_name", "sessions.user_id", "sessions.id"],
          ondelete="CASCADE",
      ),
  )

  @classmethod
  def from_event(cls, session: Session, event: Event) -> StorageEvent:
    """Creates a StorageEvent from an Event."""
    return StorageEvent(
        id=event.id,
        invocation_id=event.invocation_id,
        session_id=session.id,
        app_name=session.app_name,
        user_id=session.user_id,
        timestamp=datetime.fromtimestamp(event.timestamp),
        event_data=event.model_dump(exclude_none=True, mode="json"),
    )

  def to_event(self) -> Event:
    """Converts the StorageEvent to an Event."""
    return Event.model_validate({
        **self.event_data,
        "id": self.id,
        "invocation_id": self.invocation_id,
        "timestamp": self.timestamp.timestamp(),
    })


class StorageAppState(Base):
  """Represents an app state stored in the database."""

  __tablename__ = "app_states"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )
  update_time: Mapped[datetime] = mapped_column(
      PreciseTimestamp, default=func.now(), onupdate=func.now()
  )


class StorageUserState(Base):
  """Represents a user state stored in the database."""

  __tablename__ = "user_states"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )
  update_time: Mapped[datetime] = mapped_column(
      PreciseTimestamp, default=func.now(), onupdate=func.now()
  )


def set_sqlite_pragma(dbapi_connection, connection_record):
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA foreign_keys=ON")
  cursor.close()


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
        event.listen(db_engine.sync_engine, "connect", set_sqlite_pragma)

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
        # Check schema version BEFORE creating tables.
        # This prevents creating metadata table on a v0.1 DB.
        async with self.database_session_factory() as sql_session:
          version, is_v01 = await sql_session.run_sync(
              _schema_check.get_version_and_v01_status_sync
          )

          if is_v01:
            raise RuntimeError(
                "Database schema appears to be v0.1, but"
                f" {_schema_check.CURRENT_SCHEMA_VERSION} is required. Please"
                " migrate the database using 'adk migrate session'."
            )
          elif version and version < _schema_check.CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version is {version}, but current version is"
                f" {_schema_check.CURRENT_SCHEMA_VERSION}. Please migrate"
                " the database to the latest version using 'adk migrate"
                " session'."
            )

        async with self.db_engine.begin() as conn:
          # Uncomment to recreate DB every time
          # await conn.run_sync(Base.metadata.drop_all)
          await conn.run_sync(Base.metadata.create_all)

        # If we are here, DB is either new or >= current version.
        # If new or without metadata row, stamp it as current version.
        async with self.database_session_factory() as sql_session:
          metadata = await sql_session.get(
              StorageMetadata, _schema_check.SCHEMA_VERSION_KEY
          )
          if not metadata:
            sql_session.add(
                StorageMetadata(
                    key=_schema_check.SCHEMA_VERSION_KEY,
                    value=_schema_check.CURRENT_SCHEMA_VERSION,
                )
            )
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
    # 1. Populate states.
    # 2. Build storage session object
    # 3. Add the object to the table
    # 4. Build the session object with generated id
    # 5. Return the session
    await self._ensure_tables_created()
    async with self.database_session_factory() as sql_session:

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
        update_time = datetime.fromtimestamp(
            event.timestamp, timezone.utc
        ).replace(tzinfo=None)
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
