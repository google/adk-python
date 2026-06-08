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
"""Migration script from SQLAlchemy DB with Pickle Events to JSON schema."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import io
import json
import logging
import pickle
import sys
from typing import Any

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _session_util
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.schemas import v1
from google.genai import types
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("google_adk." + __name__)

_ALLOWED_PICKLE_GLOBALS: set[tuple[str, str]] = {
    # Builtin containers/primitives.
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "set"),
    ("builtins", "tuple"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("datetime", "datetime"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    # Expected pickled payload for v0 session schema events.
    ("fastapi.openapi.models", "APIKey"),
    ("fastapi.openapi.models", "APIKeyIn"),
    ("fastapi.openapi.models", "HTTPBase"),
    ("fastapi.openapi.models", "HTTPBearer"),
    ("fastapi.openapi.models", "OAuth2"),
    ("fastapi.openapi.models", "OAuthFlow"),
    ("fastapi.openapi.models", "OAuthFlowAuthorizationCode"),
    ("fastapi.openapi.models", "OAuthFlowClientCredentials"),
    ("fastapi.openapi.models", "OAuthFlowImplicit"),
    ("fastapi.openapi.models", "OAuthFlowPassword"),
    ("fastapi.openapi.models", "OAuthFlows"),
    ("fastapi.openapi.models", "OpenIdConnect"),
    ("fastapi.openapi.models", "SecurityBase"),
    ("fastapi.openapi.models", "SecurityScheme"),
    ("fastapi.openapi.models", "SecuritySchemeType"),
    ("google.adk.auth.auth_credential", "AuthCredential"),
    ("google.adk.auth.auth_credential", "AuthCredentialTypes"),
    ("google.adk.auth.auth_credential", "HttpAuth"),
    ("google.adk.auth.auth_credential", "HttpCredentials"),
    ("google.adk.auth.auth_credential", "OAuth2Auth"),
    ("google.adk.auth.auth_credential", "ServiceAccountCredential"),
    ("google.adk.auth.auth_schemes", "CustomAuthScheme"),
    ("google.adk.auth.auth_schemes", "ExtendedOAuth2"),
    ("google.adk.auth.auth_schemes", "OAuthGrantType"),
    ("google.adk.auth.auth_schemes", "OpenIdConnectWithConfig"),
    ("google.adk.auth.auth_tool", "AuthConfig"),
    ("google.adk.events.event_actions", "EventActions"),
    ("google.adk.events.event_actions", "EventCompaction"),
    ("google.adk.events.ui_widget", "UiWidget"),
    ("google.adk.tools.tool_confirmation", "ToolConfirmation"),
    ("google.genai.types", "Blob"),
    ("google.genai.types", "CodeExecutionResult"),
    ("google.genai.types", "Content"),
    ("google.genai.types", "ExecutableCode"),
    ("google.genai.types", "FileData"),
    ("google.genai.types", "FunctionCall"),
    ("google.genai.types", "FunctionResponse"),
    ("google.genai.types", "FunctionResponseBlob"),
    ("google.genai.types", "FunctionResponseFileData"),
    ("google.genai.types", "FunctionResponsePart"),
    ("google.genai.types", "Part"),
    ("google.genai.types", "PartMediaResolution"),
    ("google.genai.types", "VideoMetadata"),
}


class _RestrictedUnpickler(pickle.Unpickler):
  """Restricted unpickler for migrating legacy v0 schema actions.

  The v0 session schema stored `EventActions` as a pickled blob. During
  migration we treat the raw bytes read from the source DB as untrusted input
  and only allow the minimum set of safe globals needed to reconstruct
  `EventActions`.
  """

  def find_class(self, module: str, name: str) -> Any:  # noqa: ANN001
    if (module, name) in _ALLOWED_PICKLE_GLOBALS:
      return super().find_class(module, name)
    raise pickle.UnpicklingError(
        f"Blocked global during migration unpickle: {module}.{name}"
    )


def _restricted_pickle_loads(
    data: bytes, *, allow_unsafe_unpickling: bool = False
) -> Any:
  """Load a pickle payload using the restricted unpickler by default."""
  if allow_unsafe_unpickling:
    return pickle.loads(data)
  return _RestrictedUnpickler(io.BytesIO(data)).load()


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


def _row_to_event(
    row: dict[str, Any], *, allow_unsafe_unpickling: bool = False
) -> Event:
  """Converts event row (dict) to event object, handling missing columns and deserializing."""

  actions_val = row.get("actions")
  actions = None
  if actions_val is not None:
    try:
      if isinstance(actions_val, bytes):
        actions = _restricted_pickle_loads(
            actions_val, allow_unsafe_unpickling=allow_unsafe_unpickling
        )
      else:  # for spanner - it might return object directly
        actions = actions_val
    except Exception as e:
      logger.warning(
          f"Failed to unpickle actions for event {row.get('id')}: {e}"
      )
      actions = None

  if actions and hasattr(actions, "model_dump"):
    actions = EventActions().model_validate(actions.model_dump())
  elif isinstance(actions, dict):
    actions = EventActions(**actions)
  else:
    actions = EventActions()

  def _safe_json_load(val: Any) -> dict[str, Any] | None:
    if isinstance(val, str):
      try:
        data = json.loads(val)
      except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON for event {row.get('id')}")
        return None
    elif isinstance(val, dict):
      return val  # for postgres JSONB
    else:
      return None

    if isinstance(data, dict):
      return data
    logger.warning(
        f"Expected JSON object for event {row.get('id')}, got"
        f" {type(data).__name__}."
    )
    return None

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


def _get_state_dict(state_val: Any) -> dict[str, Any]:
  """Safely load dict from JSON string or return dict if already dict."""
  if isinstance(state_val, dict):
    return state_val
  if isinstance(state_val, str):
    try:
      data = json.loads(state_val)
    except json.JSONDecodeError:
      logger.warning(
          "Failed to parse state JSON string, defaulting to empty dict."
      )
      return {}
    if isinstance(data, dict):
      return data
    logger.warning("State JSON was not an object, defaulting to empty dict.")
    return {}
  return {}


# --- Migration Logic ---
def migrate(
    source_db_url: str,
    dest_db_url: str,
    allow_unsafe_unpickling: bool = False,
) -> None:
  """Migrates data from old pickle schema to new JSON schema."""
  # Convert async driver URLs to sync URLs for SQLAlchemy's synchronous engine.
  # This allows users to provide URLs like 'postgresql+asyncpg://...' and have
  # them automatically converted to 'postgresql://...' for migration.
  source_sync_url = _schema_check_utils.to_sync_url(source_db_url)
  dest_sync_url = _schema_check_utils.to_sync_url(dest_db_url)

  logger.info(f"Connecting to source database: {source_db_url}")
  if allow_unsafe_unpickling:
    logger.warning(
        "Unsafe pickle migration mode is enabled. Only use this with a trusted"
        " source database."
    )
  try:
    source_engine = create_engine(source_sync_url)
    SourceSession = sessionmaker(bind=source_engine)
  except Exception as e:
    logger.error(f"Failed to connect to source database: {e}")
    raise RuntimeError(f"Failed to connect to source database: {e}") from e

  logger.info(f"Connecting to destination database: {dest_db_url}")
  try:
    dest_engine = create_engine(dest_sync_url)
    v1.Base.metadata.create_all(dest_engine)
    DestSession = sessionmaker(bind=dest_engine)
  except Exception as e:
    logger.error(f"Failed to connect to destination database: {e}")
    raise RuntimeError(f"Failed to connect to destination database: {e}") from e

  with SourceSession() as source_session, DestSession() as dest_session:
    try:
      dest_session.merge(
          v1.StorageMetadata(
              key=_schema_check_utils.SCHEMA_VERSION_KEY,
              value=_schema_check_utils.SCHEMA_VERSION_1_JSON,
          )
      )
      logger.info("Created metadata table in destination database.")

      inspector = sqlalchemy.inspect(source_engine)

      logger.info("Migrating app_states...")
      if inspector.has_table("app_states"):
        num_rows = 0
        for row in source_session.execute(
            text("SELECT * FROM app_states")
        ).mappings():
          num_rows += 1
          dest_session.merge(
              v1.StorageAppState(
                  app_name=row["app_name"],
                  state=_get_state_dict(row.get("state")),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        logger.info(f"Migrated {num_rows} app_states.")
      else:
        logger.info("No 'app_states' table found in source db.")

      logger.info("Migrating user_states...")
      if inspector.has_table("user_states"):
        num_rows = 0
        for row in source_session.execute(
            text("SELECT * FROM user_states")
        ).mappings():
          num_rows += 1
          dest_session.merge(
              v1.StorageUserState(
                  app_name=row["app_name"],
                  user_id=row["user_id"],
                  state=_get_state_dict(row.get("state")),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        logger.info(f"Migrated {num_rows} user_states.")
      else:
        logger.info("No 'user_states' table found in source db.")

      logger.info("Migrating sessions...")
      if inspector.has_table("sessions"):
        num_rows = 0
        for row in source_session.execute(
            text("SELECT * FROM sessions")
        ).mappings():
          num_rows += 1
          dest_session.merge(
              v1.StorageSession(
                  app_name=row["app_name"],
                  user_id=row["user_id"],
                  id=row["id"],
                  state=_get_state_dict(row.get("state")),
                  create_time=_to_datetime_obj(row["create_time"]),
                  update_time=_to_datetime_obj(row["update_time"]),
              )
          )
        logger.info(f"Migrated {num_rows} sessions.")
      else:
        logger.info("No 'sessions' table found in source db.")

      logger.info("Migrating events...")
      num_rows = 0
      if inspector.has_table("events"):
        for row in source_session.execute(
            text("SELECT * FROM events")
        ).mappings():
          try:
            event_obj = _row_to_event(
                dict(row),
                allow_unsafe_unpickling=allow_unsafe_unpickling,
            )
            new_event = v1.StorageEvent(
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
            num_rows += 1
          except Exception as e:
            logger.warning(
                f"Failed to migrate event row {row.get('id', 'N/A')}: {e}"
            )
        logger.info(f"Migrated {num_rows} events.")
      else:
        logger.info("No 'events' table found in source database.")

      dest_session.commit()
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
  parser.add_argument(
      "--allow_unsafe_unpickling",
      "--allow-unsafe-unpickling",
      action="store_true",
      help=(
          "Allow legacy pickle payloads to use Python's unsafe pickle loader."
          " Only use this with a trusted source database."
      ),
  )
  args = parser.parse_args()
  try:
    migrate(
        args.source_db_url,
        args.dest_db_url,
        allow_unsafe_unpickling=args.allow_unsafe_unpickling,
    )
  except Exception as e:
    logger.error(f"Migration failed: {e}")
    sys.exit(1)
