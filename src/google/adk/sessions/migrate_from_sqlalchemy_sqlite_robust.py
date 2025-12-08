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
"""Robust migration script from SQLAlchemy SQLite to the new SQLite JSON schema.

This version handles old database schemas by using raw SQL queries instead of
relying on ORM models that expect current schema.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import logging
import pickle
import sqlite3
import sys
from typing import Any

from google.adk.sessions import sqlite_session_service as sss
from google.genai import types

logger = logging.getLogger("google_adk." + __name__)


def get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
  """Get the set of column names for a table."""
  cursor.execute(f"PRAGMA table_info({table_name})")
  return {row[1] for row in cursor.fetchall()}


def convert_timestamp_to_float(timestamp_value: Any) -> float:
  """Convert various timestamp formats to float (seconds since epoch)."""
  if isinstance(timestamp_value, (int, float)):
    return float(timestamp_value)
  elif isinstance(timestamp_value, str):
    # Try parsing as ISO format
    try:
      dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
      return dt.timestamp()
    except ValueError:
      # Try as timestamp string
      return float(timestamp_value)
  elif isinstance(timestamp_value, datetime):
    return timestamp_value.timestamp()
  else:
    raise ValueError(f"Cannot convert timestamp: {timestamp_value}")


def unpickle_if_needed(value: Any) -> Any:
  """Unpickle value if it's bytes, otherwise return as-is."""
  if isinstance(value, bytes):
    try:
      return pickle.loads(value)
    except Exception:
      return value
  return value


def parse_json_if_needed(value: Any) -> Any:
  """Parse JSON string if needed, otherwise return as-is."""
  if isinstance(value, str):
    try:
      return json.loads(value)
    except Exception:
      return value
  return value


def build_event_json(row: dict[str, Any], available_columns: set[str]) -> str:
  """Build the Event JSON from a database row, handling missing columns gracefully."""
  # Core fields that should always exist
  event_dict = {
      "id": row["id"],
      "invocation_id": row["invocation_id"],
      "author": row["author"],
      "timestamp": convert_timestamp_to_float(row["timestamp"]),
  }
  
  # Optional fields - only include if they exist and are not None
  optional_fields = {
      "branch": "branch",
      "partial": "partial", 
      "turn_complete": "turn_complete",
      "error_code": "error_code",
      "error_message": "error_message",
      "interrupted": "interrupted",
  }
  
  for json_key, col_name in optional_fields.items():
    if col_name in available_columns and row.get(col_name) is not None:
      event_dict[json_key] = row[col_name]
  
  # Handle actions (might be pickled)
  if "actions" in available_columns and row.get("actions") is not None:
    actions_value = unpickle_if_needed(row["actions"])
    if actions_value:
      # Convert to dict if it's a model
      if hasattr(actions_value, "model_dump"):
        event_dict["actions"] = actions_value.model_dump(exclude_none=True)
      elif isinstance(actions_value, dict):
        event_dict["actions"] = actions_value
  
  # Handle long_running_tool_ids
  if "long_running_tool_ids_json" in available_columns:
    lrt_json = row.get("long_running_tool_ids_json")
    if lrt_json:
      try:
        lrt_list = json.loads(lrt_json) if isinstance(lrt_json, str) else lrt_json
        if lrt_list:
          event_dict["long_running_tool_ids"] = lrt_list
      except Exception:
        pass
  
  # Handle JSON/JSONB fields (content, grounding_metadata, etc.)
  json_fields = [
      "content",
      "grounding_metadata", 
      "custom_metadata",
      "usage_metadata",
      "citation_metadata",
      "input_transcription",
      "output_transcription",
  ]
  
  for field_name in json_fields:
    if field_name in available_columns and row.get(field_name) is not None:
      field_value = parse_json_if_needed(row[field_name])
      if field_value:
        event_dict[field_name] = field_value
  
  return json.dumps(event_dict)


def migrate(source_db_path: str, dest_db_path: str):
  """Migrates data from a SQLAlchemy-based SQLite DB to the new schema."""
  logger.info(f"Connecting to source database: {source_db_path}")
  
  try:
    source_conn = sqlite3.connect(source_db_path)
    source_conn.row_factory = sqlite3.Row
    source_cursor = source_conn.cursor()
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
    # Get available columns for each table
    app_states_cols = get_table_columns(source_cursor, "app_states")
    user_states_cols = get_table_columns(source_cursor, "user_states")
    sessions_cols = get_table_columns(source_cursor, "sessions")
    events_cols = get_table_columns(source_cursor, "events")
    
    logger.info(f"Source database events table has {len(events_cols)} columns")
    
    # Migrate app_states
    logger.info("Migrating app_states...")
    source_cursor.execute("SELECT * FROM app_states")
    app_states = source_cursor.fetchall()
    
    for row in app_states:
      state = parse_json_if_needed(row["state"])
      update_time = convert_timestamp_to_float(row["update_time"])
      
      dest_cursor.execute(
          "INSERT INTO app_states (app_name, state, update_time) VALUES (?, ?, ?)",
          (row["app_name"], json.dumps(state), update_time),
      )
    logger.info(f"Migrated {len(app_states)} app_states.")
    
    # Migrate user_states
    logger.info("Migrating user_states...")
    source_cursor.execute("SELECT * FROM user_states")
    user_states = source_cursor.fetchall()
    
    for row in user_states:
      state = parse_json_if_needed(row["state"])
      update_time = convert_timestamp_to_float(row["update_time"])
      
      dest_cursor.execute(
          "INSERT INTO user_states (app_name, user_id, state, update_time) VALUES (?, ?, ?, ?)",
          (row["app_name"], row["user_id"], json.dumps(state), update_time),
      )
    logger.info(f"Migrated {len(user_states)} user_states.")
    
    # Migrate sessions
    logger.info("Migrating sessions...")
    source_cursor.execute("SELECT * FROM sessions")
    sessions = source_cursor.fetchall()
    
    for row in sessions:
      state = parse_json_if_needed(row["state"])
      create_time = convert_timestamp_to_float(row["create_time"])
      update_time = convert_timestamp_to_float(row["update_time"])
      
      dest_cursor.execute(
          "INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time) VALUES (?, ?, ?, ?, ?, ?)",
          (
              row["app_name"],
              row["user_id"],
              row["id"],
              json.dumps(state),
              create_time,
              update_time,
          ),
      )
    logger.info(f"Migrated {len(sessions)} sessions.")
    
    # Migrate events
    logger.info("Migrating events...")
    source_cursor.execute("SELECT * FROM events")
    events = source_cursor.fetchall()
    
    migrated_count = 0
    failed_count = 0
    
    for row in events:
      try:
        # Convert row to dict for easier access
        row_dict = dict(row)
        
        # Build event JSON handling missing columns
        event_data = build_event_json(row_dict, events_cols)
        
        # Parse to validate and get values
        event_json = json.loads(event_data)
        
        dest_cursor.execute(
            "INSERT INTO events (id, app_name, user_id, session_id, invocation_id, timestamp, event_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_json["id"],
                row_dict["app_name"],
                row_dict["user_id"],
                row_dict["session_id"],
                event_json["invocation_id"],
                event_json["timestamp"],
                event_data,
            ),
        )
        migrated_count += 1
        
      except Exception as e:
        logger.warning(f"Failed to migrate event {row_dict.get('id', 'unknown')}: {e}")
        failed_count += 1
    
    logger.info(f"Migrated {migrated_count} events ({failed_count} failed).")
    
    dest_conn.commit()
    logger.info("Migration completed successfully.")
    
  except Exception as e:
    logger.error(f"An error occurred during migration: {e}", exc_info=True)
    dest_conn.rollback()
    sys.exit(1)
  finally:
    source_conn.close()
    dest_conn.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Migrate ADK sessions from an existing SQLAlchemy-based "
          "SQLite database to a new SQLite database with JSON events. "
          "This version handles old database schemas gracefully."
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
      help="Path to the destination SQLite database file (e.g., /path/to/new.db)",
  )
  args = parser.parse_args()
  
  # Set up logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
  
  migrate(args.source_db_path, args.dest_db_path)
