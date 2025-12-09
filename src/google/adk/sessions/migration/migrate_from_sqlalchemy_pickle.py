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
"""Migration script from SQLAlchemy DB with Pickle Events to JSON schema."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import logging
import pickle
import sys
from typing import Any
from typing import Optional

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _session_util
from google.adk.sessions import database_session_service as dss
from google.adk.sessions.migration import _schema_check
from google.genai import types
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy import create_engine
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy import text
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import PickleType
from sqlalchemy.types import String
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger("google_adk." + __name__)


# --- Old Schema Definitions ---
class DynamicPickleType(TypeDecorator):
  """Represents a type that can be pickled."""

  impl = PickleType

  def load_dialect_impl(self, dialect):
    if dialect.name == "mysql":
      return dialect.type_descriptor(mysql.LONGBLOB)
    if dialect.name == "spanner+spanner":
      from google.cloud.sqlalchemy_spanner.sqlalchemy_spanner import SpannerPickleType

      return dialect.type_descriptor(SpannerPickleType)
    return self.impl

  def process_bind_param(self, value, dialect):
    """Ensures the pickled value is a bytes object before passing it to the database dialect."""
    if value is not None:
      if dialect.name in ("spanner+spanner", "mysql"):
        return pickle.dumps(value)
    return value

  def process_result_value(self, value, dialect):
    """Ensures the raw bytes from the database are unpickled back into a Python object."""
    if value is not None:
      if dialect.name in ("spanner+spanner", "mysql"):
        return pickle.loads(value)
    return value


class OldBase(DeclarativeBase):
  pass


class OldStorageSession(OldBase):
  __tablename__ = "sessions"
  app_name: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(dss.DynamicJSON), default={}
  )
  create_time: Mapped[datetime] = mapped_column(
      dss.PreciseTimestamp, default=func.now()
  )
  update_time: Mapped[datetime] = mapped_column(
      dss.PreciseTimestamp, default=func.now(), onupdate=func.now()
  )


class OldStorageEvent(OldBase):
  """Old storage event with pickle."""

  __tablename__ = "events"
  id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  app_name: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  session_id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  invocation_id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_VARCHAR_LENGTH)
  )
  author: Mapped[str] = mapped_column(String(dss.DEFAULT_MAX_VARCHAR_LENGTH))
  actions: Mapped[Any] = mapped_column(DynamicPickleType)
  long_running_tool_ids_json: Mapped[Optional[str]] = mapped_column(
      Text, nullable=True
  )
  branch: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  timestamp: Mapped[datetime] = mapped_column(
      dss.PreciseTimestamp, default=func.now()
  )
  content: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  grounding_metadata: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  custom_metadata: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  usage_metadata: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  citation_metadata: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  partial: Mapped[bool] = mapped_column(Boolean, nullable=True)
  turn_complete: Mapped[bool] = mapped_column(Boolean, nullable=True)
  error_code: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  error_message: Mapped[str] = mapped_column(String(1024), nullable=True)
  interrupted: Mapped[bool] = mapped_column(Boolean, nullable=True)
  input_transcription: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  output_transcription: Mapped[dict[str, Any]] = mapped_column(
      dss.DynamicJSON, nullable=True
  )
  __table_args__ = (
      ForeignKeyConstraint(
          ["app_name", "user_id", "session_id"],
          ["sessions.app_name", "sessions.user_id", "sessions.id"],
          ondelete="CASCADE",
      ),
  )

  @property
  def long_running_tool_ids(self) -> set[str]:
    return (
        set(json.loads(self.long_running_tool_ids_json))
        if self.long_running_tool_ids_json
        else set()
    )


def _to_datetime_obj(val: Any) -> datetime | Any:
  """Converts string to datetime if needed."""
  if isinstance(val, str):
    try:
      return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
      try:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
      except ValueError:
        pass  # return as is if not matching format
  return val


def _row_to_event(row: dict) -> Event:
  """Converts event row (dict) to event object, handling missing columns and deserializing."""

  actions_val = row.get("actions")
  actions = None
  if actions_val is not None:
    try:
      if isinstance(actions_val, bytes):
        actions = pickle.loads(actions_val)
      else:  # for spanner - it might return object directly
        actions = actions_val
    except Exception as e:
      logger.warning(
          f"Failed to unpickle actions for event {row.get('id')}: {e}"
      )
      actions = None

  if actions and hasattr(actions, "model_dump"):
    actions = EventActions().model_copy(update=actions.model_dump())
  elif isinstance(actions, dict):
    actions = EventActions(**actions)
  else:
    actions = EventActions()

  def _safe_json_load(val):
    data = None
    if isinstance(val, str):
      try:
        data = json.loads(val)
      except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON for event {row.get('id')}")
        return None
    elif isinstance(val, dict):
      data = val  # for postgres JSONB
    return data

  content_dict = _safe_json_load(row.get("content"))
  grounding_metadata_dict = _safe_json_load(row.get("grounding_metadata"))
  custom_metadata_dict = _safe_json_load(row.get("custom_metadata"))
  usage_metadata_dict = _safe_json_load(row.get("usage_metadata"))
  citation_metadata_dict = _safe_json_load(row.get("citation_metadata"))
  input_transcription_dict = _safe_json_load(row.get("input_transcription"))
  output_transcription_dict = _safe_json_load(row.get("output_transcription"))

  long_running_tool_ids_json = row.get("long_running_tool_ids_json")
  long_running_tool_ids = set()
  if long_running_tool_ids_json:
    try:
      long_running_tool_ids = set(json.loads(long_running_tool_ids_json))
    except json.JSONDecodeError:
      logger.warning(
          "Failed to decode long_running_tool_ids_json for event"
          f" {row.get('id')}"
      )
      long_running_tool_ids = set()

  event_id = row.get("id")
  if not event_id:
    raise ValueError("Event must have an id.")
  timestamp = _to_datetime_obj(row.get("timestamp"))
  if not timestamp:
    raise ValueError(f"Event {event_id} must have a timestamp.")

  return Event(
      id=event_id,
      invocation_id=row.get("invocation_id", ""),
      author=row.get("author", "agent"),
      branch=row.get("branch"),
      actions=actions,
      timestamp=timestamp.replace(tzinfo=timezone.utc).timestamp(),
      long_running_tool_ids=long_running_tool_ids,
      partial=row.get("partial"),
      turn_complete=row.get("turn_complete"),
      error_code=row.get("error_code"),
      error_message=row.get("error_message"),
      interrupted=row.get("interrupted"),
      custom_metadata=custom_metadata_dict,
      content=_session_util.decode_model(content_dict, types.Content),
      grounding_metadata=_session_util.decode_model(
          grounding_metadata_dict, types.GroundingMetadata
      ),
      usage_metadata=_session_util.decode_model(
          usage_metadata_dict, types.GenerateContentResponseUsageMetadata
      ),
      citation_metadata=_session_util.decode_model(
          citation_metadata_dict, types.CitationMetadata
      ),
      input_transcription=_session_util.decode_model(
          input_transcription_dict, types.Transcription
      ),
      output_transcription=_session_util.decode_model(
          output_transcription_dict, types.Transcription
      ),
  )


def _get_state_dict(state_val: Any) -> dict:
  """Safely load dict from JSON string or return dict if already dict."""
  if isinstance(state_val, dict):
    return state_val
  if isinstance(state_val, str):
    try:
      return json.loads(state_val)
    except json.JSONDecodeError:
      logger.warning(
          "Failed to parse state JSON string, defaulting to empty dict."
      )
      return {}
  return {}


class OldStorageAppState(OldBase):
  __tablename__ = "app_states"
  app_name: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(dss.DynamicJSON), default={}
  )
  update_time: Mapped[datetime] = mapped_column(
      dss.PreciseTimestamp, default=func.now(), onupdate=func.now()
  )


class OldStorageUserState(OldBase):
  __tablename__ = "user_states"
  app_name: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(dss.DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(dss.DynamicJSON), default={}
  )
  update_time: Mapped[datetime] = mapped_column(
      dss.PreciseTimestamp, default=func.now(), onupdate=func.now()
  )


# --- Migration Logic ---
def migrate(source_db_url: str, dest_db_url: str):
  """Migrates data from old pickle schema to new JSON schema."""
  logger.info(f"Connecting to source database: {source_db_url}")
  try:
    source_engine = create_engine(source_db_url)
    SourceSession = sessionmaker(bind=source_engine)
  except Exception as e:
    logger.error(f"Failed to connect to source database: {e}")
    raise RuntimeError(f"Failed to connect to source database: {e}") from e

  logger.info(f"Connecting to destination database: {dest_db_url}")
  try:
    dest_engine = create_engine(dest_db_url)
    dss.Base.metadata.create_all(dest_engine)
    DestSession = sessionmaker(bind=dest_engine)
  except Exception as e:
    logger.error(f"Failed to connect to destination database: {e}")
    raise RuntimeError(f"Failed to connect to destination database: {e}") from e

  with SourceSession() as source_session, DestSession() as dest_session:
    dest_session.merge(
        dss.StorageMetadata(
            key=_schema_check.SCHEMA_VERSION_KEY,
            value=_schema_check.SCHEMA_VERSION_1_0_JSON,
        )
    )
    dest_session.commit()
    try:
      inspector = sqlalchemy.inspect(source_engine)

      logger.info("Migrating app_states...")
      if inspector.has_table("app_states"):
        rows = (
            source_session.execute(text("SELECT * FROM app_states"))
            .mappings()
            .all()
        )
        for row in rows:
          dest_session.merge(
              dss.StorageAppState(
                  app_name=row["app_name"],
                  state=_get_state_dict(row.get("state")),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        dest_session.commit()
        logger.info(f"Migrated {len(rows)} app_states.")
      else:
        logger.info("No 'app_states' table found in source db.")

      logger.info("Migrating user_states...")
      if inspector.has_table("user_states"):
        rows = (
            source_session.execute(text("SELECT * FROM user_states"))
            .mappings()
            .all()
        )
        for row in rows:
          dest_session.merge(
              dss.StorageUserState(
                  app_name=row["app_name"],
                  user_id=row["user_id"],
                  state=_get_state_dict(row.get("state")),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        dest_session.commit()
        logger.info(f"Migrated {len(rows)} user_states.")
      else:
        logger.info("No 'user_states' table found in source db.")

      logger.info("Migrating sessions...")
      if inspector.has_table("sessions"):
        rows = (
            source_session.execute(text("SELECT * FROM sessions"))
            .mappings()
            .all()
        )
        for row in rows:
          dest_session.merge(
              dss.StorageSession(
                  app_name=row["app_name"],
                  user_id=row["user_id"],
                  id=row["id"],
                  state=_get_state_dict(row.get("state")),
                  create_time=_to_datetime_obj(row["create_time"]),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        dest_session.commit()
        logger.info(f"Migrated {len(rows)} sessions.")
      else:
        logger.info("No 'sessions' table found in source db.")

      logger.info("Migrating events...")
      events = []
      if inspector.has_table("events"):
        rows = (
            source_session.execute(text("SELECT * FROM events"))
            .mappings()
            .all()
        )
        for row in rows:
          try:
            event_obj = _row_to_event(dict(row))
            new_event = dss.StorageEvent(
                id=event_obj.id,
                app_name=row["app_name"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                invocation_id=event_obj.invocation_id,
                timestamp=datetime.fromtimestamp(
                    event_obj.timestamp, timezone.utc
                ).replace(tzinfo=None),
                event_data=event_obj.model_dump(mode="json", exclude_none=True),
            )
            dest_session.merge(new_event)
            events.append(new_event)
          except Exception as e:
            logger.warning(
                f"Failed to migrate event row {row.get('id', 'N/A')}: {e}"
            )
        dest_session.commit()
        logger.info(f"Migrated {len(events)} events.")
      else:
        logger.info("No 'events' table found in source database.")

      logger.info("Migration completed successfully.")
    except Exception as e:
      logger.error(f"An error occurred during migration: {e}", exc_info=True)
      dest_session.rollback()
      raise RuntimeError(f"An error occurred during migration: {e}") from e


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Migrate ADK sessions from SQLAlchemy Pickle format to JSON format."
      )
  )
  parser.add_argument(
      "--source_db_url", required=True, help="SQLAlchemy URL of source database"
  )
  parser.add_argument(
      "--dest_db_url",
      required=True,
      help="SQLAlchemy URL of destination database",
  )
  args = parser.parse_args()
  try:
    migrate(args.source_db_url, args.dest_db_url)
  except Exception as e:
    logger.error(f"Migration failed: {e}")
    sys.exit(1)
