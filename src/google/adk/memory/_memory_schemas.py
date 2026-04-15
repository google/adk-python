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

"""SQLAlchemy ORM models for DatabaseMemoryService."""

from __future__ import annotations

from typing import Any
from typing import Optional

from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.types import String

from ..sessions.schemas.shared import DynamicJSON

_MAX_KEY_LENGTH = 128


class Base(DeclarativeBase):
  """Base class for memory database tables."""

  pass


class StorageMemoryEntry(Base):
  """Represents a single memory entry stored in the database."""

  __tablename__ = "adk_memory_entries"

  id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

  app_name: Mapped[str] = mapped_column(String(_MAX_KEY_LENGTH), index=True)
  user_id: Mapped[str] = mapped_column(String(_MAX_KEY_LENGTH), index=True)

  keywords: Mapped[str] = mapped_column(Text)

  author: Mapped[Optional[str]] = mapped_column(
      String(_MAX_KEY_LENGTH), nullable=True
  )
  content: Mapped[dict[str, Any]] = mapped_column(DynamicJSON)
  timestamp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

  __table_args__ = (Index("idx_memory_app_user", "app_name", "user_id"),)

  def __repr__(self) -> str:
    return (
        f"<StorageMemoryEntry(id={self.id}, app_name={self.app_name!r}, "
        f"user_id={self.user_id!r})>"
    )
