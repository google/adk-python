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

import copy
from datetime import datetime
from datetime import timezone
import json
import logging
import pickle
from typing import Any
from typing import Optional
import uuid

from sqlalchemy import Boolean
from sqlalchemy import delete
from sqlalchemy import Dialect
from sqlalchemy import event
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session as DatabaseSessionFactory
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData
from sqlalchemy.types import DateTime
from sqlalchemy.types import PickleType
from sqlalchemy.types import String
from sqlalchemy.types import TypeDecorator
from typing_extensions import override
from tzlocal import get_localzone

from . import _session_util
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
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


class DynamicPickleType(TypeDecorator):
  """Represents a type that can be pickled."""

  impl = PickleType

  def load_dialect_impl(self, dialect):
    if dialect.name == "spanner+spanner":
      from google.cloud.sqlalchemy_spanner.sqlalchemy_spanner import SpannerPickleType

      return dialect.type_descriptor(SpannerPickleType)
    return self.impl

  def process_bind_param(self, value, dialect):
    """Ensures the pickled value is a bytes object before passing it to the database dialect."""
    if value is not None:
      if dialect.name == "spanner+spanner":
        return pickle.dumps(value)
    return value

  def process_result_value(self, value, dialect):
    """Ensures the raw bytes from the database are unpickled back into a Python object."""
    if value is not None:
      if dialect.name == "spanner+spanner":
        return pickle.loads(value)
    return value


class Base(DeclarativeBase):
  """Base class for database tables."""

  pass


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
  author: Mapped[str] = mapped_column(String(DEFAULT_MAX_VARCHAR_LENGTH))
  actions: Mapped[MutableDict[str, Any]] = mapped_column(DynamicPickleType)
  long_running_tool_ids_json: Mapped[Optional[str]] = mapped_column(
      Text, nullable=True
  )
  branch: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  timestamp: Mapped[PreciseTimestamp] = mapped_column(
      PreciseTimestamp, default=func.now()
  )

  # === Fileds from llm_response.py ===
  content: Mapped[dict[str, Any]] = mapped_column(DynamicJSON, nullable=True)
  grounding_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  custom_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )

  partial: Mapped[bool] = mapped_column(Boolean, nullable=True)
  turn_complete: Mapped[bool] = mapped_column(Boolean, nullable=True)
  error_code: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  error_message: Mapped[str] = mapped_column(String(1024), nullable=True)
  interrupted: Mapped[bool] = mapped_column(Boolean, nullable=True)

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

  @property
  def long_running_tool_ids(self) -> set[str]:
    return (
        set(json.loads(self.long_running_tool_ids_json))
        if self.long_running_tool_ids_json
        else set()
    )

  @long_running_tool_ids.setter
  def long_running_tool_ids(self, value: set[str]):
    if value is None:
      self.long_running_tool_ids_json = None
    else:
      self.long_running_tool_ids_json = json.dumps(list(value))

  @classmethod
  def from_event(cls, session: Session, event: Event) -> StorageEvent:
    storage_event = StorageEvent(
        id=event.id,
        invocation_id=event.invocation_id,
        author=event.author,
        branch=event.branch,
        actions=event.actions,
        session_id=session.id,
        app_name=session.app_name,
        user_id=session.user_id,
        timestamp=datetime.fromtimestamp(event.timestamp),
        long_running_tool_ids=event.long_running_tool_ids,
        partial=event.partial,
        turn_complete=event.turn_complete,
        error_code=event.error_code,
        error_message=event.error_message,
        interrupted=event.interrupted,
    )
    if event.content:
      storage_event.content = event.content.model_dump(
          exclude_none=True, mode="json"
      )
    if event.grounding_metadata:
      storage_event.grounding_metadata = event.grounding_metadata.model_dump(
          exclude_none=True, mode="json"
      )
    if event.custom_metadata:
      storage_event.custom_metadata = event.custom_metadata
    return storage_event

  def to_event(self) -> Event:
    return Event(
        id=self.id,
        invocation_id=self.invocation_id,
        author=self.author,
        branch=self.branch,
        actions=self.actions,
        timestamp=self.timestamp.timestamp(),
        content=_session_util.decode_content(self.content),
        long_running_tool_ids=self.long_running_tool_ids,
        partial=self.partial,
        turn_complete=self.turn_complete,
        error_code=self.error_code,
        error_message=self.error_message,
        interrupted=self.interrupted,
        grounding_metadata=_session_util.decode_grounding_metadata(
            self.grounding_metadata
        ),
        custom_metadata=self.custom_metadata,
    )


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
      db_engine = create_engine(db_url, **kwargs)

      if db_engine.dialect.name == "sqlite":
        # Set sqlite pragma to enable foreign keys constraints
        event.listen(db_engine, "connect", set_sqlite_pragma)

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

    self.db_engine: Engine = db_engine
    self.metadata: MetaData = MetaData()
    self.inspector = inspect(self.db_engine)

    # DB session factory method
    self.database_session_factory: sessionmaker[DatabaseSessionFactory] = (
        sessionmaker(bind=self.db_engine)
    )

    # Uncomment to recreate DB every time
    # Base.metadata.drop_all(self.db_engine)
    Base.metadata.create_all(self.db_engine)

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

    with self.database_session_factory() as sql_session:

      # Fetch app and user states from storage
      storage_app_state = sql_session.get(StorageAppState, (app_name))
      storage_user_state = sql_session.get(
          StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}

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
      app_state_delta, user_state_delta, session_state = _extract_state_delta(
          state
      )

      # Apply state delta
      app_state.update(app_state_delta)
      user_state.update(user_state_delta)

      # Store app and user state
      if app_state_delta:
        storage_app_state.state = app_state
      if user_state_delta:
        storage_user_state.state = user_state

      # Store the session
      storage_session = StorageSession(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=session_state,
      )
      sql_session.add(storage_session)
      sql_session.commit()

      sql_session.refresh(storage_session)

      # Merge states for response
      merged_state = _merge_state(app_state, user_state, session_state)
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
    # 1. Get the storage session entry from session table
    # 2. Get all the events based on session id and filtering config
    # 3. Convert and return the session
    with self.database_session_factory() as sql_session:
      storage_session = sql_session.get(
          StorageSession, (app_name, user_id, session_id)
      )
      if storage_session is None:
        return None

      if config and config.after_timestamp:
        after_dt = datetime.fromtimestamp(config.after_timestamp)
        timestamp_filter = StorageEvent.timestamp >= after_dt
      else:
        timestamp_filter = True

      storage_events = (
          sql_session.query(StorageEvent)
          .filter(StorageEvent.app_name == app_name)
          .filter(StorageEvent.session_id == storage_session.id)
          .filter(StorageEvent.user_id == user_id)
          .filter(timestamp_filter)
          .order_by(StorageEvent.timestamp.desc())
          .limit(
              config.num_recent_events
              if config and config.num_recent_events
              else None
          )
          .all()
      )

      # Fetch states from storage
      storage_app_state = sql_session.get(StorageAppState, (app_name))
      storage_user_state = sql_session.get(
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
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    with self.database_session_factory() as sql_session:
      results = (
          sql_session.query(StorageSession)
          .filter(StorageSession.app_name == app_name)
          .filter(StorageSession.user_id == user_id)
          .all()
      )

      # Fetch states from storage
      storage_app_state = sql_session.get(StorageAppState, (app_name))
      storage_user_state = sql_session.get(
          StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}

      sessions = []
      for storage_session in results:
        session_state = storage_session.state
        merged_state = _merge_state(app_state, user_state, session_state)

        sessions.append(storage_session.to_session(state=merged_state))
      return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, app_name: str, user_id: str, session_id: str
  ) -> None:
    with self.database_session_factory() as sql_session:
      stmt = delete(StorageSession).where(
          StorageSession.app_name == app_name,
          StorageSession.user_id == user_id,
          StorageSession.id == session_id,
      )
      sql_session.execute(stmt)
      sql_session.commit()

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    # 1. Check if timestamp is stale
    # 2. Update session attributes based on event config
    # 3. Store event to table
    with self.database_session_factory() as sql_session:
      storage_session = sql_session.get(
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
      storage_app_state = sql_session.get(StorageAppState, (session.app_name))
      storage_user_state = sql_session.get(
          StorageUserState, (session.app_name, session.user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state

      # Extract state delta
      app_state_delta = {}
      user_state_delta = {}
      session_state_delta = {}
      if event.actions:
        if event.actions.state_delta:
          app_state_delta, user_state_delta, session_state_delta = (
              _extract_state_delta(event.actions.state_delta)
          )

      # Merge state and update storage
      if app_state_delta:
        app_state.update(app_state_delta)
        storage_app_state.state = app_state
      if user_state_delta:
        user_state.update(user_state_delta)
        storage_user_state.state = user_state
      if session_state_delta:
        session_state.update(session_state_delta)
        storage_session.state = session_state

      sql_session.add(StorageEvent.from_event(session, event))

      sql_session.commit()
      sql_session.refresh(storage_session)

      # Update timestamp with commit time
      session.last_update_time = storage_session.update_timestamp_tz

    # Also update the in-memory session
    await super().append_event(session=session, event=event)
    return event


def _extract_state_delta(state: dict[str, Any]):
  app_state_delta = {}
  user_state_delta = {}
  session_state_delta = {}
  if state:
    for key in state.keys():
      if key.startswith(State.APP_PREFIX):
        app_state_delta[key.removeprefix(State.APP_PREFIX)] = state[key]
      elif key.startswith(State.USER_PREFIX):
        user_state_delta[key.removeprefix(State.USER_PREFIX)] = state[key]
      elif not key.startswith(State.TEMP_PREFIX):
        session_state_delta[key] = state[key]
  return app_state_delta, user_state_delta, session_state_delta


def _merge_state(app_state, user_state, session_state):
  # Merge states for response
  merged_state = copy.deepcopy(session_state)
  for key in app_state.keys():
    merged_state[State.APP_PREFIX + key] = app_state[key]
  for key in user_state.keys():
    merged_state[State.USER_PREFIX + key] = user_state[key]
  return merged_state
