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
"""Migration script from SQLAlchemy SQLite to the new SQLite JSON schema."""

from __future__ import annotations

import argparse
from datetime import timezone
import json
import logging
import sqlite3
import sys

from google.adk.sessions import database_session_service as dss
from google.adk.sessions import sqlite_session_service as sss
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("google_adk." + __name__)


def migrate(source_db_url: str, dest_db_path: str):
  """Migrates data from a SQLAlchemy-based SQLite DB to the new schema."""
  logger.info(f"Connecting to source database: {source_db_url}")
  try:
    engine = create_engine(source_db_url)
    dss.Base.metadata.create_all(engine)  # Ensure tables exist for inspection
    SourceSession = sessionmaker(bind=engine)
    source_session = SourceSession()
  except Exception as e:
    logger.error(f"Failed to connect to source database: {e}")
    sys.exit(1)

  logger.info(f"Connecting to destination database: {dest_db_path}")
  try:
    dest_conn = sqlite3.connect(dest_db_path)
    dest_cursor = dest_conn.cursor()
    dest_cursor.execute(sss.PRAGMA_FOREIGN_KEYS)
    dest_cursor.executescript(sss.CREATE_SCHEMA_SQL)
  except Exception as e:
    logger.error(f"Failed to connect to destination database: {e}")
    sys.exit(1)

  try:
    # Migrate app_states
    logger.info("Migrating app_states...")
    app_states = source_session.query(dss.StorageAppState).all()
    for item in app_states:
      dest_cursor.execute(
          "INSERT INTO app_states (app_name, state, update_time) VALUES (?,"
          " ?, ?)",
          (
              item.app_name,
              json.dumps(item.state),
              item.update_time.replace(tzinfo=timezone.utc).timestamp(),
          ),
      )
    logger.info(f"Migrated {len(app_states)} app_states.")

    # Migrate user_states
    logger.info("Migrating user_states...")
    user_states = source_session.query(dss.StorageUserState).all()
    for item in user_states:
      dest_cursor.execute(
          "INSERT INTO user_states (app_name, user_id, state, update_time)"
          " VALUES (?, ?, ?, ?)",
          (
              item.app_name,
              item.user_id,
              json.dumps(item.state),
              item.update_time.replace(tzinfo=timezone.utc).timestamp(),
          ),
      )
    logger.info(f"Migrated {len(user_states)} user_states.")

    # Migrate sessions
    logger.info("Migrating sessions...")
    sessions = source_session.query(dss.StorageSession).all()
    for item in sessions:
      dest_cursor.execute(
          "INSERT INTO sessions (app_name, user_id, id, state, create_time,"
          " update_time) VALUES (?, ?, ?, ?, ?, ?)",
          (
              item.app_name,
              item.user_id,
              item.id,
              json.dumps(item.state),
              item.create_time.replace(tzinfo=timezone.utc).timestamp(),
              item.update_time.replace(tzinfo=timezone.utc).timestamp(),
          ),
      )
    logger.info(f"Migrated {len(sessions)} sessions.")

    # Migrate events
    logger.info("Migrating events...")
    events = source_session.query(dss.StorageEvent).all()
    for item in events:
      try:
        event_obj = item.to_event()
        event_data = event_obj.model_dump_json(exclude_none=True)
        dest_cursor.execute(
            "INSERT INTO events (id, app_name, user_id, session_id,"
            " invocation_id, timestamp, event_data) VALUES (?, ?, ?, ?, ?,"
            " ?, ?)",
            (
                event_obj.id,
                item.app_name,
                item.user_id,
                item.session_id,
                event_obj.invocation_id,
                event_obj.timestamp,
                event_data,
            ),
        )
      except Exception as e:
        logger.warning(f"Failed to migrate event {item.id}: {e}")
    logger.info(f"Migrated {len(events)} events.")

    dest_conn.commit()
    logger.info("Migration completed successfully.")

  except Exception as e:
    logger.error(f"An error occurred during migration: {e}", exc_info=True)
    dest_conn.rollback()
    sys.exit(1)
  finally:
    source_session.close()
    dest_conn.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Migrate ADK sessions from an existing SQLAlchemy-based "
          "SQLite database to a new SQLite database with JSON events."
      )
  )
  parser.add_argument(
      "--source_db_path",
      required=True,
      help="Path to the source SQLite database file (e.g., /path/to/old.db)",
  )
  parser.add_argument(
      "--dest_db_path",
      required=True,
      help=(
          "Path to the destination SQLite database file (e.g., /path/to/new.db)"
      ),
  )
  args = parser.parse_args()

  source_url = f"sqlite:///{args.source_db_path}"
  migrate(source_url, args.dest_db_path)
