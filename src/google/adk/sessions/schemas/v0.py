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
"""V0 database schema for ADK versions from 1.19.0 to 1.21.0.

This module defines SQLAlchemy models for storing session and event data
in a relational database with the EventActions object using pickle
serialization. To migrate from the schemas in earlier ADK versions to this
v0 schema, see
https://github.com/google/adk-python/blob/main/docs/upgrading_from_1_22_0.md.

The latest schema is defined in `v1.py`. That module uses JSON serialization
for the EventActions data as well as other fields in the `events` table. See
https://github.com/google/adk-python/discussions/3605 for more details.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
import json
import pickle
from typing import Any
from typing import Optional
import uuid

from google.genai import types
from sqlalchemy import Boolean
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import PickleType
from sqlalchemy.types import String
from sqlalchemy.types import TypeDecorator

from .. import _session_util
from ...events.event import Event
from ...events.event_actions import EventActions
from ..session import Session
from .shared import DEFAULT_MAX_KEY_LENGTH
from .shared import DEFAULT_MAX_VARCHAR_LENGTH
from .shared import DynamicJSON
from .shared import PreciseTimestamp


class DynamicPickleType(TypeDecorator):
  """Represents a type that can be pickled."""

  impl = PickleType

  def load_dialect_impl(self, dialect):
    if dialect.name == "mysql":
      return dialect.type_descriptor(mysql.LONGBLOB)
    if dialect.name == "spanner+spanner":
      from google.cloud.sqlalchemy_spanner.sqlalchemy_spanner import SpannerPickleType

      return dialect.type_descriptor(SpannerPickleType)
    return self.impl

  def process_bind_param(self, value, dialect):
    """Ensures the pickled value is a bytes object before passing it to the database dialect."""
    if value is not None:
      if dialect.name in ("spanner+spanner", "mysql"):
        return pickle.dumps(value)
    return value

  def process_result_value(self, value, dialect):
    """Ensures the raw bytes from the database are unpickled back into a Python object."""
    if value is not None:
      if dialect.name in ("spanner+spanner", "mysql"):
        return pickle.loads(value)
    return value


class Base(DeclarativeBase):
  """Base class for v0 database tables."""

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

  # === Fields from llm_response.py ===
  content: Mapped[dict[str, Any]] = mapped_column(DynamicJSON, nullable=True)
  grounding_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  custom_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  usage_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  citation_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )

  partial: Mapped[bool] = mapped_column(Boolean, nullable=True)
  turn_complete: Mapped[bool] = mapped_column(Boolean, nullable=True)
  error_code: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  error_message: Mapped[str] = mapped_column(Text, nullable=True)
  interrupted: Mapped[bool] = mapped_column(Boolean, nullable=True)
  input_transcription: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  output_transcription: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )

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
    if event.usage_metadata:
      storage_event.usage_metadata = event.usage_metadata.model_dump(
          exclude_none=True, mode="json"
      )
    if event.citation_metadata:
      storage_event.citation_metadata = event.citation_metadata.model_dump(
          exclude_none=True, mode="json"
      )
    if event.input_transcription:
      storage_event.input_transcription = event.input_transcription.model_dump(
          exclude_none=True, mode="json"
      )
    if event.output_transcription:
      storage_event.output_transcription = (
          event.output_transcription.model_dump(exclude_none=True, mode="json")
      )
    return storage_event

  def to_event(self) -> Event:
    return Event(
        id=self.id,
        invocation_id=self.invocation_id,
        author=self.author,
        branch=self.branch,
        # This is needed as previous ADK version pickled actions might not have
        # value defined in the current version of the EventActions model.
        actions=EventActions().model_copy(update=self.actions.model_dump()),
        timestamp=self.timestamp.timestamp(),
        long_running_tool_ids=self.long_running_tool_ids,
        partial=self.partial,
        turn_complete=self.turn_complete,
        error_code=self.error_code,
        error_message=self.error_message,
        interrupted=self.interrupted,
        custom_metadata=self.custom_metadata,
        content=_session_util.decode_model(self.content, types.Content),
        grounding_metadata=_session_util.decode_model(
            self.grounding_metadata, types.GroundingMetadata
        ),
        usage_metadata=_session_util.decode_model(
            self.usage_metadata, types.GenerateContentResponseUsageMetadata
        ),
        citation_metadata=_session_util.decode_model(
            self.citation_metadata, types.CitationMetadata
        ),
        input_transcription=_session_util.decode_model(
            self.input_transcription, types.Transcription
        ),
        output_transcription=_session_util.decode_model(
            self.output_transcription, types.Transcription
        ),
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
