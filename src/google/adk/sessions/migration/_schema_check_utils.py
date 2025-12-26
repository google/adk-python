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
      result = connection.execute(
          text("SELECT value FROM adk_internal_metadata WHERE key = :key"),
          {"key": SCHEMA_VERSION_KEY},
      ).fetchone()
      if result:
        return result[0]
      else:
        return LATEST_SCHEMA_VERSION
    except Exception as e:
      logger.warning(
          "Failed to query schema version from adk_internal_metadata,"
          " assuming the latest schema: %s.",
          e,
      )
      return LATEST_SCHEMA_VERSION
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
            " data. The migration command and script will be provided soon."
        )
        return SCHEMA_VERSION_0_PICKLE
    except Exception as e:
      logger.warning("Failed to inspect 'events' table columns: %s", e)
      return LATEST_SCHEMA_VERSION
  # New database, assume the latest schema.
  return LATEST_SCHEMA_VERSION


def get_db_schema_version_from_connection(connection) -> str:
  """Gets DB schema version from a DB connection."""
  inspector = inspect(connection)
  return _get_schema_version_impl(inspector, connection)
