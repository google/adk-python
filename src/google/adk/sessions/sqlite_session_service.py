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

from contextlib import asynccontextmanager
import copy
import json
import logging
import os
import sqlite3
from typing import Any
from typing import Optional
from urllib.parse import unquote
from urllib.parse import urlparse

import aiosqlite
from google.adk.platform import time as platform_time
from google.adk.platform import uuid as platform_uuid
from typing_extensions import override

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..errors.version_mismatch_error import VersionMismatchError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

SCHEMA_VERSION = 1

SCHEMA_VERSION_KEY = "schema_version"

PRAGMA_FOREIGN_KEYS = "PRAGMA foreign_keys = ON"

APP_STATES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_states (
    app_name TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    update_time REAL NOT NULL
);
"""

USER_STATES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_states (
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL,
    update_time REAL NOT NULL,
    PRIMARY KEY (app_name, user_id)
);
"""

SESSIONS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    id TEXT NOT NULL,
    state TEXT NOT NULL,
    create_time REAL NOT NULL,
    update_time REAL NOT NULL,
    PRIMARY KEY (app_name, user_id, id)
);
"""

EVENTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    invocation_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_data TEXT NOT NULL,
    PRIMARY KEY (app_name, user_id, session_id, id),
    FOREIGN KEY (app_name, user_id, session_id) REFERENCES sessions(app_name, user_id, id) ON DELETE CASCADE
);
"""

METADATA_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CREATE_SCHEMA_SQL = "\n".join([
    APP_STATES_TABLE_SCHEMA,
    USER_STATES_TABLE_SCHEMA,
    SESSIONS_TABLE_SCHEMA,
    EVENTS_TABLE_SCHEMA,
    METADATA_TABLE_SCHEMA,
])


def _get_default_db_path() -> str:
  """Returns the default database path based on environment variables and XDG Base Directory Specification.

  Priority order (highest to lowest):
    1. ADK_HOME/sessions.db (if ADK_HOME environment variable is set)
    2. XDG_DATA_HOME/adk/sessions.db (if XDG_DATA_HOME environment variable is set)
    3. ~/.local/share/adk/sessions.db (default XDG data directory)
    4. ~/.adk/sessions.db (legacy fallback for backward compatibility)

  Returns:
    The default database path as a string.
  """
  if "ADK_HOME" in os.environ:
    adk_home = os.path.expanduser(os.environ["ADK_HOME"])
    return os.path.join(adk_home, "sessions.db")

  if "XDG_DATA_HOME" in os.environ:
    xdg_data_home = os.path.expanduser(os.environ["XDG_DATA_HOME"])
    return os.path.join(xdg_data_home, "adk", "sessions.db")

  home = os.path.expanduser("~")
  xdg_default = os.path.join(home, ".local", "share", "adk", "sessions.db")

  legacy_path = os.path.join(home, ".adk", "sessions.db")
  if os.path.exists(legacy_path):
    return legacy_path

  return xdg_default


def _parse_db_path(db_path: str) -> tuple[str, str, bool]:
  """Normalizes a SQLite db path from a URL or filesystem path.

  Returns:
    A tuple of:
      - filesystem path (for `os.path.exists` and user-facing messages)
      - value to pass to sqlite/aiosqlite connect
      - whether to pass `uri=True` to sqlite/aiosqlite connect

  Notes:
    When a SQLAlchemy-style SQLite URL is provided, this follows SQLAlchemy's
    conventions:
      - `sqlite:///relative.db` is a path relative to the current working dir.
      - `sqlite:////absolute.db` is an absolute filesystem path.
  """
  if not db_path.startswith(("sqlite:", "sqlite+aiosqlite:")):
    return db_path, db_path, False

  parsed = urlparse(db_path)
  raw_path = unquote(parsed.path)
  if not raw_path:
    return db_path, db_path, False

  normalized_path = raw_path
  if normalized_path.startswith("//"):
    normalized_path = normalized_path[1:]
  elif normalized_path.startswith("/"):
    normalized_path = normalized_path[1:]

  if parsed.query:
    # sqlite3 only treats the filename as a URI when it starts with `file:`.
    return normalized_path, f"file:{normalized_path}?{parsed.query}", True

  return normalized_path, normalized_path, False


class SqliteSessionService(BaseSessionService):
  """A session service that uses an SQLite database for storage via aiosqlite.

  Event data is stored as JSON to allow for schema flexibility as event
  fields evolve.

  State Merge Semantics:
    This implementation uses SQLite's json_patch function (RFC 7396) for
    state updates, which performs a RECURSIVE merge of nested dictionaries.
    This differs from DatabaseSessionService which uses a shallow dict.update()
    merge. For example:

    - Existing state: {"nested": {"a": 1, "b": 2}}
    - State delta: {"nested": {"b": 3, "c": 4}}
    - SqliteSessionService result: {"nested": {"a": 1, "b": 3, "c": 4}}
    - DatabaseSessionService result: {"nested": {"b": 3, "c": 4}}

    When using nested state dictionaries, be aware of this semantic difference.

  Schema Versioning:
    This service tracks schema version in the 'metadata' table. If the
    database schema version is older than the expected version, a
    VersionMismatchError will be raised, and a migration script should be run.
  """

  def __init__(self, db_path: str = None):
    """Initializes the SQLite session service with a database path.

    Args:
      db_path: The path to the SQLite database file. If not provided,
        the default path is determined by the following priority order:
        1. ADK_HOME/sessions.db (if ADK_HOME environment variable is set)
        2. XDG_DATA_HOME/adk/sessions.db (if XDG_DATA_HOME environment variable is set)
        3. ~/.local/share/adk/sessions.db (default XDG data directory)
        4. ~/.adk/sessions.db (legacy fallback, only if the file already exists)
        The directory will be created if it doesn't exist.

    Raises:
      VersionMismatchError: If the database schema version is incompatible
        with the current code.
      RuntimeError: If the database uses an old schema that requires migration.
    """
    if db_path is None:
      db_path = _get_default_db_path()

    self._db_path, self._db_connect_path, self._db_connect_uri = _parse_db_path(
        db_path
    )

    db_dir = os.path.dirname(self._db_path)
    if db_dir and not os.path.exists(db_dir):
      os.makedirs(db_dir)

    if self._is_migration_needed():
      raise RuntimeError(
          f"Database {self._db_path} seems to use an old schema."
          " Please run the migration command to"
          " migrate it to the new schema. Example: `python -m"
          " google.adk.sessions.migration.migrate_from_sqlalchemy_sqlite"
          f" --source_db_path {self._db_path} --dest_db_path"
          f" {self._db_path}.new` then backup {self._db_path} and rename"
          f" {self._db_path}.new to {self._db_path}."
      )

    self._check_schema_version()

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    if session_id:
      session_id = session_id.strip()
    if not session_id:
      session_id = platform_uuid.new_uuid()
    now = platform_time.get_time()

    async with self._get_db_connection() as db:
      # Check if session_id already exists
      async with db.execute(
          "SELECT 1 FROM sessions WHERE app_name=? AND user_id=? AND id=?",
          (app_name, user_id, session_id),
      ) as cursor:
        if await cursor.fetchone():
          raise AlreadyExistsError(
              f"Session with id {session_id} already exists."
          )

      # Extract state deltas
      state_deltas = _session_util.extract_state_delta(state)
      app_state_delta = state_deltas["app"]
      user_state_delta = state_deltas["user"]
      session_state = state_deltas["session"]

      # Apply state delta and update/insert states atomically
      if app_state_delta:
        await self._upsert_app_state(db, app_name, app_state_delta, now)
      if user_state_delta:
        await self._upsert_user_state(
            db, app_name, user_id, user_state_delta, now
        )

      # Fetch current state after upserts
      storage_app_state = await self._get_app_state(db, app_name)
      storage_user_state = await self._get_user_state(db, app_name, user_id)

      # Store the session
      await db.execute(
          """
          INSERT INTO sessions (app_name, user_id, id, state, create_time, update_time)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
              app_name,
              user_id,
              session_id,
              json.dumps(session_state),
              now,
              now,
          ),
      )
      await db.commit()

      # Merge states for response
      merged_state = _merge_state(
          storage_app_state, storage_user_state, session_state
      )
      return Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=merged_state,
          events=[],
          last_update_time=now,
      )

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    async with self._get_db_connection() as db:
      async with db.execute(
          "SELECT state, update_time FROM sessions WHERE app_name=? AND"
          " user_id=? AND id=?",
          (app_name, user_id, session_id),
      ) as cursor:
        session_row = await cursor.fetchone()
        if session_row is None:
          return None
        session_state = json.loads(session_row["state"])
        last_update_time = session_row["update_time"]

      # Build events query
      query_parts = [
          "SELECT event_data FROM events",
          "WHERE app_name=? AND user_id=? AND session_id=?",
      ]
      params: list[Any] = [app_name, user_id, session_id]

      if config and config.after_timestamp:
        query_parts.append("AND timestamp >= ?")
        params.append(config.after_timestamp)

      query_parts.append("ORDER BY timestamp DESC")

      if config and config.num_recent_events is not None:
        query_parts.append("LIMIT ?")
        params.append(config.num_recent_events)

      if config and config.num_recent_events == 0:
        event_rows = []
      else:
        event_rows = await db.execute_fetchall(" ".join(query_parts), params)
      storage_events_data = [row["event_data"] for row in event_rows]

      # Fetch states from storage
      app_state = await self._get_app_state(db, app_name)
      user_state = await self._get_user_state(db, app_name, user_id)

      # Merge states
      merged_state = _merge_state(app_state, user_state, session_state)

      # Deserialize events and reverse to chronological order
      events = [
          Event.model_validate_json(event_data)
          for event_data in reversed(storage_events_data)
      ]

      return Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=merged_state,
          events=events,
          last_update_time=last_update_time,
      )

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    sessions_list = []
    async with self._get_db_connection() as db:
      # Fetch sessions
      if user_id:
        session_rows = await db.execute_fetchall(
            "SELECT id, user_id, state, update_time FROM sessions WHERE"
            " app_name=? AND user_id=?",
            (app_name, user_id),
        )
      else:
        session_rows = await db.execute_fetchall(
            "SELECT id, user_id, state, update_time FROM sessions WHERE"
            " app_name=?",
            (app_name,),
        )

      # Fetch app state
      app_state = await self._get_app_state(db, app_name)

      # Fetch user states
      user_states_map = {}
      if user_id:
        user_state = await self._get_user_state(db, app_name, user_id)
        if user_state:
          user_states_map[user_id] = user_state
      else:
        async with db.execute(
            "SELECT user_id, state FROM user_states WHERE app_name=?",
            (app_name,),
        ) as cursor:
          async for row in cursor:
            user_states_map[row["user_id"]] = json.loads(row["state"])

      # Build session list
      for row in session_rows:
        session_user_id = row["user_id"]
        session_state = json.loads(row["state"])
        user_state = user_states_map.get(session_user_id, {})
        merged_state = _merge_state(app_state, user_state, session_state)
        sessions_list.append(
            Session(
                app_name=app_name,
                user_id=session_user_id,
                id=row["id"],
                state=merged_state,
                events=[],
                last_update_time=row["update_time"],
            )
        )
    return ListSessionsResponse(sessions=sessions_list)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    async with self._get_db_connection() as db:
      await db.execute(
          "DELETE FROM sessions WHERE app_name=? AND user_id=? AND id=?",
          (app_name, user_id, session_id),
      )
      await db.commit()

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    # Apply temp state to in-memory session before trimming, so that
    # subsequent agents within the same invocation can read temp values.
    self._apply_temp_state(session, event)
    # Trim temp state before persisting
    event = self._trim_temp_delta_state(event)
    event_timestamp = event.timestamp

    async with self._get_db_connection() as db:
      # Check for stale session
      async with db.execute(
          "SELECT update_time FROM sessions WHERE app_name=? AND user_id=? AND"
          " id=?",
          (session.app_name, session.user_id, session.id),
      ) as cursor:
        row = await cursor.fetchone()
        if row is None:
          raise ValueError(f"Session {session.id} not found.")
        storage_update_time = row["update_time"]
        if storage_update_time > session.last_update_time:
          raise ValueError(
              "The last_update_time provided in the session object is"
              " earlier than the update_time in storage."
              " Please check if it is a stale session."
          )

      # Apply state delta if present
      has_session_state_delta = False
      if event.actions and event.actions.state_delta:
        state_deltas = _session_util.extract_state_delta(
            event.actions.state_delta
        )
        app_state_delta = state_deltas["app"]
        user_state_delta = state_deltas["user"]
        session_state_delta = state_deltas["session"]

        if app_state_delta:
          await self._upsert_app_state(
              db, session.app_name, app_state_delta, event_timestamp
          )
        if user_state_delta:
          await self._upsert_user_state(
              db,
              session.app_name,
              session.user_id,
              user_state_delta,
              event_timestamp,
          )
        if session_state_delta:
          await self._update_session_state_in_db(
              db,
              session.app_name,
              session.user_id,
              session.id,
              session_state_delta,
              event_timestamp,
          )
          has_session_state_delta = True

      # Insert event and update session timestamp
      await db.execute(
          """
          INSERT INTO events (id, app_name, user_id, session_id, invocation_id, timestamp, event_data)
          VALUES (?, ?, ?, ?, ?, ?, ?)
          """,
          (
              event.id,
              session.app_name,
              session.user_id,
              session.id,
              event.invocation_id,
              event.timestamp,
              event.model_dump_json(exclude_none=True),
          ),
      )
      if not has_session_state_delta:
        await db.execute(
            "UPDATE sessions SET update_time=? WHERE app_name=? AND user_id=?"
            " AND id=?",
            (
                event_timestamp,
                session.app_name,
                session.user_id,
                session.id,
            ),
        )
      await db.commit()

      # Update timestamp based on event time
      session.last_update_time = event_timestamp

    # Also update the in-memory session
    await super().append_event(session=session, event=event)
    return event

  @asynccontextmanager
  async def _get_db_connection(self):
    """Connects to the db and performs initial setup.

    This method:
    1. Connects to the SQLite database
    2. Enables foreign keys
    3. Creates all tables if they don't exist
    4. Initializes schema version in metadata table if not already set
    """
    async with aiosqlite.connect(
        self._db_connect_path, uri=self._db_connect_uri
    ) as db:
      db.row_factory = aiosqlite.Row
      await db.execute(PRAGMA_FOREIGN_KEYS)
      await db.executescript(CREATE_SCHEMA_SQL)

      await db.execute(
          """
          INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)
          """,
          (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
      )
      await db.commit()

      yield db

  async def _get_state(
      self, db: aiosqlite.Connection, query: str, params: tuple
  ) -> dict[str, Any]:
    """Fetches and deserializes a JSON state column from a single row."""
    async with db.execute(query, params) as cursor:
      row = await cursor.fetchone()
      return json.loads(row["state"]) if row else {}

  async def _get_app_state(
      self, db: aiosqlite.Connection, app_name: str
  ) -> dict[str, Any]:
    return await self._get_state(
        db, "SELECT state FROM app_states WHERE app_name=?", (app_name,)
    )

  async def _get_user_state(
      self, db: aiosqlite.Connection, app_name: str, user_id: str
  ) -> dict[str, Any]:
    return await self._get_state(
        db,
        "SELECT state FROM user_states WHERE app_name=? AND user_id=?",
        (app_name, user_id),
    )

  async def _get_session_state(
      self,
      db: aiosqlite.Connection,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> dict[str, Any]:
    return await self._get_state(
        db,
        "SELECT state FROM sessions WHERE app_name=? AND user_id=? AND id=?",
        (app_name, user_id, session_id),
    )

  async def _upsert_app_state(
      self, db: aiosqlite.Connection, app_name: str, delta: dict, now: float
  ) -> None:
    """Atomically inserts or updates app state using json_patch."""
    await db.execute(
        """
        INSERT INTO app_states (app_name, state, update_time) VALUES (?, ?, ?)
        ON CONFLICT(app_name) DO UPDATE SET state=json_patch(state, excluded.state), update_time=excluded.update_time
        """,
        (app_name, json.dumps(delta), now),
    )

  async def _upsert_user_state(
      self,
      db: aiosqlite.Connection,
      app_name: str,
      user_id: str,
      delta: dict,
      now: float,
  ) -> None:
    """Atomically inserts or updates user state using json_patch."""
    await db.execute(
        """
        INSERT INTO user_states (app_name, user_id, state, update_time) VALUES (?, ?, ?, ?)
        ON CONFLICT(app_name, user_id) DO UPDATE SET state=json_patch(state, excluded.state), update_time=excluded.update_time
        """,
        (app_name, user_id, json.dumps(delta), now),
    )

  async def _update_session_state_in_db(
      self,
      db: aiosqlite.Connection,
      app_name: str,
      user_id: str,
      session_id: str,
      delta: dict,
      now: float,
  ) -> None:
    """Atomically updates session state using json_patch."""
    await db.execute(
        "UPDATE sessions SET state=json_patch(state, ?), update_time=? WHERE"
        " app_name=? AND user_id=? AND id=?",
        (
            json.dumps(delta),
            now,
            app_name,
            user_id,
            session_id,
        ),
    )

  def _is_migration_needed(self) -> bool:
    """Checks if migration to new schema is needed."""
    if not os.path.exists(self._db_path):
      return False
    try:
      with sqlite3.connect(
          self._db_connect_path, uri=self._db_connect_uri
      ) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' and name='events'"
        )
        if not cursor.fetchone():
          return False

        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]
        if "event_data" in columns:
          return False
        else:
          return True
    except sqlite3.Error as e:
      raise RuntimeError(
          f"Error accessing database {self._db_path}: {e}"
      ) from e

  def _check_schema_version(self) -> None:
    """Checks if the database schema version is compatible with the current code.

    This method:
    1. If the database doesn't exist yet, does nothing (version will be
       initialized on first connection).
    2. If the database exists:
       - First checks if metadata table exists and schema_version is set
       - If schema_version exists but doesn't match SCHEMA_VERSION,
         raises VersionMismatchError.
       - If metadata table or schema_version doesn't exist, attempts to
         initialize them. If the database is read-only (e.g., opened with
         mode=ro), this initialization is skipped and the version check
         is deferred until actual database operations are performed.

    Raises:
      VersionMismatchError: If the database schema version is incompatible.
    """
    if not os.path.exists(self._db_path):
      return

    try:
      with sqlite3.connect(
          self._db_connect_path, uri=self._db_connect_uri
      ) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        )
        if not cursor.fetchone():
          try:
            cursor.execute(METADATA_TABLE_SCHEMA)
            cursor.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
            )
            conn.commit()
          except sqlite3.OperationalError:
            return
          return

        cursor.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (SCHEMA_VERSION_KEY,),
        )
        result = cursor.fetchone()
        if result is None:
          try:
            cursor.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
            )
            conn.commit()
          except sqlite3.OperationalError:
            return
          return

        stored_version = int(result[0])
        if stored_version != SCHEMA_VERSION:
          raise VersionMismatchError(
              expected_version=SCHEMA_VERSION,
              actual_version=stored_version,
          )

    except sqlite3.Error as e:
      if "readonly" in str(e).lower():
        return
      raise RuntimeError(
          f"Error checking schema version for database {self._db_path}: {e}"
      ) from e


def _merge_state(app_state, user_state, session_state):
  """Merges app, user, and session states into a single dictionary."""
  merged_state = copy.deepcopy(session_state)
  for key, value in app_state.items():
    merged_state[State.APP_PREFIX + key] = value
  for key, value in user_state.items():
    merged_state[State.USER_PREFIX + key] = value
  return merged_state
