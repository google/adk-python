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

"""SQLAlchemy ORM schema for DatabaseMemoryService tables."""

from __future__ import annotations

from typing import Any
from typing import Optional

from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.types import String

from ...sessions.schemas.shared import DynamicJSON
from ...sessions.schemas.shared import PreciseTimestamp

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256


class Base(DeclarativeBase):
  """Declarative base for memory schema tables."""

  pass


class StorageMemoryEntry(Base):
  """ORM model for the adk_memory_entries table."""

  __tablename__ = "adk_memory_entries"

  id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=False, index=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=False, index=True
  )
  session_id: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=True
  )
  event_id: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=True
  )
  author: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=True
  )
  timestamp: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  content_json: Mapped[Any] = mapped_column(DynamicJSON, nullable=True)
  search_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  custom_metadata: Mapped[Any] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), nullable=True
  )
  created_at: Mapped[Any] = mapped_column(
      PreciseTimestamp, server_default=func.now()
  )

  __table_args__ = (
      Index("ix_memory_entries_app_user", "app_name", "user_id"),
      Index("ix_memory_entries_session", "app_name", "user_id", "session_id"),
  )


class StorageScratchpadKV(Base):
  """ORM model for the adk_scratchpad_kv table.

  Composite PK: (app_name, user_id, session_id, key).
  Use session_id='' as a sentinel for user-level (non-session) KV.
  """

  __tablename__ = "adk_scratchpad_kv"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  session_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  key: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  value_json: Mapped[Any] = mapped_column(DynamicJSON, nullable=False)
  updated_at: Mapped[Any] = mapped_column(
      PreciseTimestamp,
      server_default=func.now(),
      onupdate=func.now(),
  )


class StorageScratchpadLog(Base):
  """ORM model for the adk_scratchpad_log table.

  Append-only. id is autoincrement int to preserve insertion order.
  Use session_id='' as a sentinel for user-level (non-session) log.
  """

  __tablename__ = "adk_scratchpad_log"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=False
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=False
  )
  session_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=False
  )
  tag: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=True, index=True
  )
  agent_name: Mapped[Optional[str]] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), nullable=True
  )
  content: Mapped[str] = mapped_column(Text, nullable=False)
  extra_json: Mapped[Optional[Any]] = mapped_column(DynamicJSON, nullable=True)
  created_at: Mapped[Any] = mapped_column(
      PreciseTimestamp, server_default=func.now()
  )

  __table_args__ = (
      Index(
          "ix_scratchpad_log_scope",
          "app_name",
          "user_id",
          "session_id",
      ),
  )
