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

from datetime import datetime
from datetime import timezone
import json

from sqlalchemy import Dialect
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import DateTime
from sqlalchemy.types import TypeDecorator

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256

# Dialects that store TIMESTAMP values as UTC-naive datetimes and therefore
# require us to reattach UTC tzinfo on read and strip it on write.
_NAIVE_UTC_DIALECTS = frozenset({"sqlite", "postgresql"})


def update_timestamp_from_dt(dt: datetime, dialect_name: str) -> float:
  """Converts a DB-returned datetime to a POSIX timestamp.

  SQLite and PostgreSQL store naive datetimes that represent UTC values.
  All other dialects return timezone-aware datetimes directly.
  """
  if dialect_name in _NAIVE_UTC_DIALECTS:
    return dt.replace(tzinfo=timezone.utc).timestamp()
  return dt.timestamp()


def update_time_from_timestamp(posix_ts: float, dialect_name: str) -> datetime:
  """Converts a POSIX timestamp to the datetime format expected by the DB.

  SQLite and PostgreSQL require a UTC-naive datetime; every other dialect
  accepts (and prefers) a UTC-aware datetime.
  """
  dt = datetime.fromtimestamp(posix_ts, timezone.utc)
  if dialect_name in _NAIVE_UTC_DIALECTS:
    return dt.replace(tzinfo=None)
  return dt


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
