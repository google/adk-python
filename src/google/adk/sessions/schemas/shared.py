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

import datetime
import json
from typing import Any
from typing import Callable
from typing import cast

from sqlalchemy import Dialect
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import DateTime
from sqlalchemy.types import TypeDecorator
from sqlalchemy.types import TypeEngine

from ...utils import _json_utils

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256


def timestamp_to_utc_datetime(timestamp: float) -> datetime.datetime:
  """Converts a POSIX timestamp to a naive UTC database value."""
  return datetime.datetime.fromtimestamp(
      timestamp, tz=datetime.timezone.utc
  ).replace(tzinfo=None)


def utc_datetime_to_timestamp(value: datetime.datetime) -> float:
  """Converts a database UTC datetime to a POSIX timestamp."""
  if value.tzinfo is None:
    value = value.replace(tzinfo=datetime.timezone.utc)
  return value.timestamp()


class DynamicJSON(TypeDecorator[dict[str, Any]]):  # type: ignore[misc]
  """A JSON-like type that uses JSONB on PostgreSQL and TEXT with JSON serialization for other databases."""

  impl = Text  # Default implementation is TEXT
  # Behavior depends only on the dialect, which the compiled cache already
  # keys on, so statements using this type are safe to cache.
  cache_ok = True

  def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
    if dialect.name == "postgresql":
      return dialect.type_descriptor(postgresql.JSONB())
    if dialect.name == "mysql":
      # Use LONGTEXT for MySQL to address the data too long issue
      return dialect.type_descriptor(mysql.LONGTEXT())
    return dialect.type_descriptor(Text())  # Default to Text for other dialects

  def process_bind_param(
      self, value: dict[str, Any] | None, dialect: Dialect
  ) -> dict[str, Any] | str | None:
    if value is not None:
      if dialect.name == "postgresql":
        return value  # JSONB handles dict directly
      return json.dumps(value)  # Serialize to JSON string for TEXT
    return value

  def process_result_value(
      self, value: object | None, dialect: Dialect
  ) -> dict[str, Any] | None:
    if value is None:
      return None
    decoded: object = value
    if dialect.name != "postgresql":
      decoded = _json_utils.safe_json_loads(
          cast("str | bytes | bytearray", value), context="session state"
      )
    return cast("dict[str, Any]", decoded)


class PreciseTimestamp(TypeDecorator[datetime.datetime]):  # type: ignore[misc]
  """Represents a timestamp precise to the microsecond."""

  impl = DateTime
  cache_ok = True

  def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
    if dialect.name == "mysql":
      return dialect.type_descriptor(mysql.DATETIME(fsp=6))
    return self.impl_instance

  def result_processor(
      self, dialect: Dialect, coltype: object
  ) -> Callable[[object], datetime.datetime | None]:
    impl_processor = self.impl_instance.result_processor(dialect, coltype)

    def process(value: object) -> datetime.datetime | None:
      if value is None:
        return None
      if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
      if impl_processor:
        value = impl_processor(value)
      return cast(datetime.datetime, value)

    return process
