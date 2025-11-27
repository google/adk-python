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
"""Database schema version check utility."""

from __future__ import annotations

import logging

import sqlalchemy
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect
from sqlalchemy import text

logger = logging.getLogger("google_adk." + __name__)

SCHEMA_VERSION_KEY = "schema_version"
SCHEMA_VERSION_0_1_PICKLE = "0.1"
SCHEMA_VERSION_1_0_JSON = "1.0"
CURRENT_SCHEMA_VERSION = "1.0"


def _to_sync_url(db_url: str) -> str:
  """Removes +driver from SQLAlchemy URL."""
  if "://" in db_url:
    scheme, _, rest = db_url.partition("://")
    if "+" in scheme:
      dialect = scheme.split("+", 1)[0]
      return f"{dialect}://{rest}"
  return db_url


def get_version_and_v01_status_sync(
    sess: sqlalchemy.orm.Session,
) -> tuple[str | None, bool]:
  """Returns (version, is_v01) inspecting the database."""
  inspector = sqlalchemy.inspect(sess.get_bind())
  if inspector.has_table("adk_internal_metadata"):
    try:
      result = sess.execute(
          text("SELECT value FROM adk_internal_metadata WHERE key = :key"),
          {"key": SCHEMA_VERSION_KEY},
      ).fetchone()
      # If table exists, with or without key, it's 1.0 or newer.
      return (result[0] if result else SCHEMA_VERSION_1_0_JSON), False
    except Exception as e:
      logger.warning(
          "Could not read from adk_internal_metadata: %s. Assuming v1.0.",
          e,
      )
      return SCHEMA_VERSION_1_0_JSON, False

  if inspector.has_table("events"):
    try:
      cols = {c["name"] for c in inspector.get_columns("events")}
      if "actions" in cols and "event_data" not in cols:
        return None, True  # 0.1 schema
    except Exception as e:
      logger.warning("Could not inspect 'events' table columns: %s", e)
  return None, False  # New DB


def get_db_schema_version(db_url: str) -> str | None:
  """Reads schema version from DB.

  Checks metadata table first, falls back to table structure for 0.1 vs 1.0.
  """
  engine = None
  try:
    engine = create_sync_engine(_to_sync_url(db_url))
    inspector = inspect(engine)

    if inspector.has_table("adk_internal_metadata"):
      with engine.connect() as connection:
        result = connection.execute(
            text("SELECT value FROM adk_internal_metadata WHERE key = :key"),
            parameters={"key": SCHEMA_VERSION_KEY},
        ).fetchone()
        # If table exists, with or without key, it's 1.0 or newer.
        return result[0] if result else SCHEMA_VERSION_1_0_JSON

    # Metadata table doesn't exist, check for 0.1 schema.
    # 0.1 schema has an 'events' table with an 'actions' column.
    if inspector.has_table("events"):
      try:
        cols = {c["name"] for c in inspector.get_columns("events")}
        if "actions" in cols and "event_data" not in cols:
          return SCHEMA_VERSION_0_1_PICKLE
      except Exception as e:
        logger.warning("Could not inspect 'events' table columns: %s", e)

    # If no metadata table and not identifiable as 0.1,
    # assume it is a new/empty DB requiring schema 1.0.
    return SCHEMA_VERSION_1_0_JSON
  except Exception as e:
    logger.info(
        "Could not determine schema version by inspecting database: %s."
        " Assuming v1.0.",
        e,
    )
    return SCHEMA_VERSION_1_0_JSON
  finally:
    if engine:
      engine.dispose()
