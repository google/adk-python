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

"""The v1 database schema for the DatabaseSessionService.

This module defines SQLAlchemy models for storing session and event data
in a relational database with the "events" table using JSON
serialization for Event data.

See https://github.com/google/adk-python/discussions/3605 for more details.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional
import uuid

from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import String

from ...events.event import Event
from ..session import Session
from .shared import DEFAULT_MAX_KEY_LENGTH
from .shared import DEFAULT_MAX_VARCHAR_LENGTH
from .shared import DynamicJSON
from .shared import PreciseTimestamp


class Base(DeclarativeBase):
  """Base class for v1 database tables."""

  pass


class StorageMetadata(Base):
  """Represents ADK internal metadata stored in the database.

  This table is used to store internal information like the schema version.
  The DatabaseSessionService will populate and utilize this table to manage
  database compatibility and migrations.
  """

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
      # Deleting a session will now automatically delete its associated events
      cascade="all, delete-orphan",
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
  # The event_data uses JSON serialization to store the Event data, replacing
  # various fields previously used.
  event_data: Mapped[dict[str, Any]] = mapped_column(DynamicJSON, nullable=True)

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
