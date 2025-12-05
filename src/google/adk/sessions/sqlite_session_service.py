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
from __future__ import annotations

from contextlib import asynccontextmanager
import copy
import json
import logging
import os
import sqlite3
import time
from typing import Any
from typing import Optional
import uuid

import aiosqlite
from typing_extensions import override

from . import _session_util
from ..errors.already_exists_error import AlreadyExistsError
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

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
CREATE_SCHEMA_SQL = "\n".join([
    APP_STATES_TABLE_SCHEMA,
    USER_STATES_TABLE_SCHEMA,
    SESSIONS_TABLE_SCHEMA,
    EVENTS_TABLE_SCHEMA,
])


class SqliteSessionService(BaseSessionService):
  """A session service that uses an SQLite database for storage via aiosqlite.

  Event data is stored as JSON to allow for schema flexibility as event
  fields evolve.
  """

  def __init__(self, db_path: str):
    """Initializes the SQLite session service with a database path."""
    self._db_path = db_path

    if self._is_migration_needed():
      raise RuntimeError(
          f"Database {db_path} seems to use an old schema."
          " Please run the migration command to"
          " migrate it to the new schema. Example: `python -m"
          " google.adk.sessions.migrate_from_sqlalchemy_sqlite"
          f" --source_db_path {db_path} --dest_db_path"
          f" {db_path}.new` then backup {db_path} and rename"
          f" {db_path}.new to {db_path}."
      )

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
      session_id = str(uuid.uuid4())
    now = time.time()

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

      if config and config.num_recent_events:
        query_parts.append("LIMIT ?")
        params.append(config.num_recent_events)

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
    """Connects to the db and performs initial setup."""
    async with aiosqlite.connect(self._db_path) as db:
      db.row_factory = aiosqlite.Row
      await db.execute(PRAGMA_FOREIGN_KEYS)
      await db.executescript(CREATE_SCHEMA_SQL)
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
      with sqlite3.connect(self._db_path) as conn:
        cursor = conn.cursor()
        # Check if events table exists
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' and name='events'"
        )
        if not cursor.fetchone():
          return False  # No events table, so no migration needed.

        # If events table exists, check for event_data column
        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]
        if "event_data" in columns:
          return False  # New schema: event_data column exists.
        else:
          return (
              True  # Old schema: events table exists, but no event_data column.
          )
    except sqlite3.Error as e:
      raise RuntimeError(
          f"Error accessing database {self._db_path}: {e}"
      ) from e


def _merge_state(app_state, user_state, session_state):
  """Merges app, user, and session states into a single dictionary."""
  merged_state = copy.deepcopy(session_state)
  for key, value in app_state.items():
    merged_state[State.APP_PREFIX + key] = value
  for key, value in user_state.items():
    merged_state[State.USER_PREFIX + key] = value
  return merged_state
