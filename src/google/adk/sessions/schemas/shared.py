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

import json

from sqlalchemy import Dialect
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import DateTime
from sqlalchemy.types import TypeDecorator

from google.adk.utils import serialization_utils

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


class JsonEncodedType(DynamicJSON):
  """A JSON-encoded type with hybrid support for secure legacy pickles.

  New data is always stored as JSON. When reading, it first attempts to
  decode JSON. If that fails and the value is binary, it attempts to
  deserialize using serialization_utils.secure_loads (HMAC-verified).
  """

  def process_result_value(self, value, dialect: Dialect):
    if value is None:
      return None

    # Try JSON first (for new data or PostgreSQL JSONB)
    if dialect.name == "postgresql":
      return value

    if isinstance(value, str):
      try:
        return json.loads(value)
      except json.JSONDecodeError:
        # If it's a string that's not JSON, it might be a corrupted entry
        # or an unexpected format. Logic continues to check for binary.
        pass

    # If JSON failed, check if it's binary legacy data (HMAC signed)
    if isinstance(value, bytes):
      try:
        return serialization_utils.secure_loads(value)
      except serialization_utils.SecurityError:
        # If both JSON and secure_loads fail, re-raise or handle as Error
        raise

    return super().process_result_value(value, dialect)


class PreciseTimestamp(TypeDecorator):
  """Represents a timestamp precise to the microsecond."""

  impl = DateTime
  cache_ok = True

  def load_dialect_impl(self, dialect):
    if dialect.name == "mysql":
      return dialect.type_descriptor(mysql.DATETIME(fsp=6))
    return self.impl
