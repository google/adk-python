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
"""Migration script to convert TIMESTAMP to TIMESTAMPTZ for PostgreSQL.

Starting from ADK v1.24.0, DatabaseSessionService creates timezone-aware
datetime objects (with tzinfo=UTC). When using PostgreSQL with asyncpg,
this causes a conflict if existing timestamp columns are defined as
TIMESTAMP WITHOUT TIME ZONE, resulting in:

  asyncpg.exceptions.DataError: can't subtract offset-naive and
  offset-aware datetimes

This migration alters all timestamp columns in ADK tables to use
TIMESTAMP WITH TIME ZONE. It is safe to run on existing data as
PostgreSQL will interpret existing naive timestamps as being in the
server's timezone (typically UTC).

Usage:
  python -m google.adk.sessions.migration.migrate_postgresql_timestamptz \
    --db_url postgresql+asyncpg://user:pass@host:port/dbname
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy import text

from . import _schema_check_utils

logger = logging.getLogger("google_adk." + __name__)

# Columns to migrate: (table_name, column_name)
_TIMESTAMP_COLUMNS = [
    ("sessions", "create_time"),
    ("sessions", "update_time"),
    ("events", "timestamp"),
    ("app_states", "update_time"),
    ("user_states", "update_time"),
]


def migrate(db_url: str) -> None:
  """Migrates TIMESTAMP columns to TIMESTAMP WITH TIME ZONE for PostgreSQL.

  Args:
    db_url: The database URL (sync or async format).
  """
  sync_url = _schema_check_utils.to_sync_url(db_url)
  engine = create_engine(sync_url)

  try:
    with engine.begin() as conn:
      # Only run on PostgreSQL
      if engine.dialect.name != "postgresql":
        logger.info(
            "Skipping TIMESTAMPTZ migration: not a PostgreSQL database"
            " (dialect=%s).",
            engine.dialect.name,
        )
        return

      migrated = 0
      for table_name, column_name in _TIMESTAMP_COLUMNS:
        # Check if table exists
        result = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = :table_name "
                "AND column_name = :column_name"
            ),
            {"table_name": table_name, "column_name": column_name},
        ).fetchone()

        if result is None:
          logger.debug(
              "Skipping %s.%s: column not found.", table_name, column_name
          )
          continue

        if result[0] == "timestamp with time zone":
          logger.debug(
              "Skipping %s.%s: already TIMESTAMP WITH TIME ZONE.",
              table_name,
              column_name,
          )
          continue

        logger.info(
            "Migrating %s.%s from %s to TIMESTAMP WITH TIME ZONE.",
            table_name,
            column_name,
            result[0],
        )
        conn.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ALTER COLUMN {column_name} "
                f"TYPE TIMESTAMP WITH TIME ZONE"
            )
        )
        migrated += 1

      if migrated > 0:
        logger.info(
            "Successfully migrated %d column(s) to TIMESTAMP WITH TIME ZONE.",
            migrated,
        )
      else:
        logger.info("No columns needed migration.")

  finally:
    engine.dispose()


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Migrate PostgreSQL TIMESTAMP columns to TIMESTAMP WITH TIME ZONE"
          " for ADK DatabaseSessionService."
      )
  )
  parser.add_argument(
      "--db_url",
      required=True,
      help="Database URL (e.g., postgresql+asyncpg://user:pass@host:port/db)",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  migrate(args.db_url)


if __name__ == "__main__":
  main()
