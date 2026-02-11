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
"""Database schema version check utility."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect
from sqlalchemy import text

logger = logging.getLogger("google_adk." + __name__)

SCHEMA_VERSION_KEY = "schema_version"
SCHEMA_VERSION_0_PICKLE = "0"
SCHEMA_VERSION_1_JSON = "1"
LATEST_SCHEMA_VERSION = SCHEMA_VERSION_1_JSON


def _get_schema_version_impl(inspector, connection) -> str:
  """Gets DB schema version using inspector and connection."""
  if inspector.has_table("adk_internal_metadata"):
    try:
      key_col = inspector.dialect.identifier_preparer.quote("key")
      result = connection.execute(
          text(
              f"SELECT value FROM adk_internal_metadata WHERE {key_col} = :key"
          ),
          {"key": SCHEMA_VERSION_KEY},
      ).fetchone()
      if result:
        return result[0]
      else:
        raise ValueError(
            "Schema version not found in adk_internal_metadata. The database"
            " might be malformed."
        )
    except Exception as e:
      logger.error(
          "Failed to query schema version from adk_internal_metadata: %s.",
          e,
      )
      raise

  # Metadata table doesn't exist, check for v0 schema.
  # V0 schema has an 'events' table with an 'actions' column.
  if inspector.has_table("events"):
    try:
      cols = {c["name"] for c in inspector.get_columns("events")}
      if "actions" in cols and "event_data" not in cols:
        logger.warning(
            "The database is using the legacy v0 schema, which uses Pickle to"
            " serialize event actions. The v0 schema will not be supported"
            " going forward and will be deprecated in a few rollouts. Please"
            " migrate to the v1 schema which uses JSON serialization for event"
            " data. You can use `adk migrate session` command to migrate your"
            " database."
        )
        return SCHEMA_VERSION_0_PICKLE
    except Exception as e:
      logger.error("Failed to inspect 'events' table columns: %s", e)
      raise
  # New database, use the latest schema.
  return LATEST_SCHEMA_VERSION


def get_db_schema_version_from_connection(connection) -> str:
  """Gets DB schema version from a DB connection."""
  inspector = inspect(connection)
  return _get_schema_version_impl(inspector, connection)


def to_sync_url(db_url: str) -> str:
  """Removes '+driver' from SQLAlchemy URL.

  This is useful when you need to use a synchronous SQLAlchemy engine with
  a database URL that specifies an async driver (e.g., postgresql+asyncpg://
  or sqlite+aiosqlite://).

  Args:
    db_url: The database URL, potentially with a driver specification.

  Returns:
    The database URL with the driver specification removed (e.g.,
    'postgresql+asyncpg://host/db' becomes 'postgresql://host/db').

  Examples:
    >>> to_sync_url('postgresql+asyncpg://localhost/mydb')
    'postgresql://localhost/mydb'
    >>> to_sync_url('sqlite+aiosqlite:///path/to/db.sqlite')
    'sqlite:///path/to/db.sqlite'
    >>> to_sync_url('mysql://localhost/mydb')  # No driver, returns unchanged
    'mysql://localhost/mydb'
  """
  if "://" in db_url:
    scheme, _, rest = db_url.partition("://")
    if "+" in scheme:
      dialect = scheme.split("+", 1)[0]
      return f"{dialect}://{rest}"
  return db_url


def get_db_schema_version(db_url: str) -> str:
  """Reads schema version from DB.

  Checks metadata table first and then falls back to table structure.

  Args:
    db_url: The database URL.

  Returns:
    The detected schema version as a string. Returns `LATEST_SCHEMA_VERSION`
    if it's a new database.
  """
  engine = None
  try:
    engine = create_sync_engine(to_sync_url(db_url))
    with engine.connect() as connection:
      inspector = inspect(connection)
      return _get_schema_version_impl(inspector, connection)
  except Exception:
    logger.warning(
        "Failed to get schema version from database %s.",
        db_url,
    )
    raise
  finally:
    if engine:
      engine.dispose()
