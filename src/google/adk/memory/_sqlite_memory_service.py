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

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import asynccontextmanager
import hashlib
import itertools
from pathlib import Path
import re
import sqlite3
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING
import unicodedata
from urllib.parse import unquote
from urllib.parse import urlparse

import aiosqlite
from google.adk.platform import time as platform_time
from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session


_SCHEMA_VERSION = "1"
_DEFAULT_MAX_RESULTS = 50
_DEFAULT_MAX_EVENT_BYTES = 262_144
_DEFAULT_BUSY_TIMEOUT_MS = 3_000
_UNKNOWN_SESSION_ID = "__unknown_session_id__"

_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS adk_memory_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  author TEXT NOT NULL,
  timestamp REAL NOT NULL,
  content_json TEXT NOT NULL,
  search_text TEXT NOT NULL,
  UNIQUE(app_name, user_id, session_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_events_scope
ON memory_events(app_name, user_id, timestamp DESC);
"""

_CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_events_fts
USING fts5(
  search_text,
  content='memory_events',
  content_rowid='id',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_events_ai
AFTER INSERT ON memory_events BEGIN
  INSERT INTO memory_events_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_events_ad
AFTER DELETE ON memory_events BEGIN
  INSERT INTO memory_events_fts(memory_events_fts, rowid, search_text)
  VALUES('delete', old.id, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_events_au
AFTER UPDATE OF search_text ON memory_events BEGIN
  INSERT INTO memory_events_fts(memory_events_fts, rowid, search_text)
  VALUES('delete', old.id, old.search_text);
  INSERT INTO memory_events_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;
"""


def _parse_db_path(db_path: str | Path) -> tuple[str, str, bool]:
  """Normalizes a SQLite db path from a URL or filesystem path.

  Returns:
    A tuple of:
      - filesystem path (for Path operations, user messages, or ':memory:')
      - value to pass to sqlite/aiosqlite connect
      - whether to pass uri=True to sqlite/aiosqlite connect

  Notes:
    When a SQLAlchemy-style SQLite URL is provided, this follows SQLAlchemy's
    conventions:
      - `sqlite:///relative.db` is a path relative to the current working dir.
      - `sqlite:////absolute.db` is an absolute filesystem path.
  """
  db_path_str = str(db_path).strip()
  if not db_path_str:
    raise ValueError("db_path must be set.")

  if not db_path_str.startswith(("sqlite:", "sqlite+aiosqlite:")):
    return db_path_str, db_path_str, False

  parsed = urlparse(db_path_str)
  raw_path = unquote(parsed.path)
  if not raw_path:
    return db_path_str, db_path_str, False

  normalized_path = raw_path
  if normalized_path.startswith("/"):
    normalized_path = normalized_path[1:]

  if parsed.query:
    # sqlite3 only treats the filename as a URI when it starts with `file:`.
    return normalized_path, f"file:{normalized_path}?{parsed.query}", True

  return normalized_path, normalized_path, False


class SqliteMemoryService(BaseMemoryService):
  """A persistent, local memory service backed by SQLite.

  When using an in-memory database (``db_path=":memory:"`` or ``sqlite:///:memory:``),
  the underlying connection remains open for the lifetime of the service.
  Callers should call :meth:`close` explicitly to release the database
  connection and worker thread.

  Events without an explicit ``id`` use a deterministic hash of their
  author, timestamp, and content as an ID, ensuring idempotent re-ingestion.
  """

  def __init__(
      self,
      db_path: str | Path,
      *,
      fts: Literal["auto", "on", "off"] = "auto",
      max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
      max_results: int = _DEFAULT_MAX_RESULTS,
  ) -> None:
    """Initializes a SqliteMemoryService.

    Args:
      db_path: The SQLite database path or sqlite:/// URI.
      fts: "auto", "on", or "off" for FTS5 usage.
      max_event_bytes: Maximum bytes of search index text stored per event.
        Note that this truncates search index text only, not stored or
        returned content.
      max_results: Maximum number of memories returned per query.
    """
    self._db_path, self._db_connect_path, self._db_connect_uri = _parse_db_path(
        db_path
    )

    if self._db_path != ":memory:":
      path = Path(self._db_path)
      if path.exists() and path.is_dir():
        raise ValueError(f"db_path {self._db_path} is a directory.")
      path.parent.mkdir(parents=True, exist_ok=True)

    if fts not in ("auto", "on", "off"):
      raise ValueError("fts must be one of: auto, on, off.")
    if max_event_bytes <= 0:
      raise ValueError("max_event_bytes must be positive.")
    if max_results <= 0:
      raise ValueError("max_results must be positive.")

    self._fts_mode = fts
    self._max_event_bytes = max_event_bytes
    self._max_results = max_results
    self._schema_ready = False
    self._initialized = False
    self._fts_available: bool | None = None
    self._schema_lock = asyncio.Lock()
    self._memory_conn: aiosqlite.Connection | None = None

  @asynccontextmanager
  async def _get_db_connection(self) -> AsyncIterator[aiosqlite.Connection]:
    if self._db_path == ":memory:":
      if self._memory_conn is None:
        async with self._schema_lock:
          if self._memory_conn is None:
            conn = await aiosqlite.connect(":memory:")
            try:
              conn.row_factory = aiosqlite.Row
              await _apply_pragmas(conn)
              await self._ensure_schema(conn)
            except BaseException:
              await conn.close()
              raise
            self._memory_conn = conn
      yield self._memory_conn
    else:
      async with aiosqlite.connect(
          self._db_connect_path, uri=self._db_connect_uri
      ) as db:
        db.row_factory = aiosqlite.Row
        await _apply_pragmas(db)
        if not self._schema_ready:
          async with self._schema_lock:
            if not self._schema_ready:
              await self._ensure_schema(db)
              self._schema_ready = True
        yield db

  async def _ensure_schema(self, db: aiosqlite.Connection) -> None:
    if not self._initialized:
      is_readonly = self._db_connect_uri and "mode=ro" in self._db_connect_path
      if not is_readonly:
        try:
          await db.executescript(_CREATE_SCHEMA_SQL)
          await db.execute(
              "INSERT OR IGNORE INTO adk_memory_metadata (key, value) VALUES"
              " (?, ?)",
              ("schema_version", _SCHEMA_VERSION),
          )
          async with db.execute(
              "SELECT value FROM adk_memory_metadata WHERE key ="
              " 'schema_version'"
          ) as cursor:
            version_row = await cursor.fetchone()
            if version_row and version_row[0] != _SCHEMA_VERSION:
              raise RuntimeError(
                  f"Unsupported schema version: {version_row[0]}. Expected"
                  f" {_SCHEMA_VERSION}."
              )
          self._fts_available = await _setup_fts(db, self._fts_mode)
          await db.commit()
        except (sqlite3.OperationalError, aiosqlite.OperationalError) as exc:
          if "readonly" in str(exc).lower().replace("-", ""):
            is_readonly = True
          else:
            raise

      if is_readonly:
        async with db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND"
            " name='adk_memory_metadata'"
        ) as cursor:
          has_metadata = await cursor.fetchone() is not None
        if has_metadata:
          async with db.execute(
              "SELECT value FROM adk_memory_metadata WHERE key ="
              " 'schema_version'"
          ) as cursor:
            version_row = await cursor.fetchone()
            if version_row and version_row[0] != _SCHEMA_VERSION:
              raise RuntimeError(
                  f"Unsupported schema version: {version_row[0]}. Expected"
                  f" {_SCHEMA_VERSION}."
              )
        async with db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND"
            " name='memory_events_fts'"
        ) as cursor:
          fts_table_exists = await cursor.fetchone() is not None
        if self._fts_mode == "on" and not fts_table_exists:
          raise RuntimeError(
              "FTS5 table memory_events_fts not found in read-only database."
          )
        self._fts_available = fts_table_exists and self._fts_mode != "off"

      self._initialized = True

  async def close(self) -> None:
    """Closes any persistent resources."""
    async with self._schema_lock:
      if self._memory_conn is not None:
        await self._memory_conn.close()
        self._memory_conn = None
        self._initialized = False
        self._schema_ready = False

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self.add_events_to_memory(
        app_name=session.app_name,
        user_id=session.user_id,
        events=session.events,
        session_id=session.id,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    _ = custom_metadata
    scoped_session_id = session_id or _UNKNOWN_SESSION_ID

    events_to_insert: list[tuple[str, str, str, str, str, float, str, str]] = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      search_text = _extract_event_text(event)
      if not search_text:
        continue
      search_text = unicodedata.normalize("NFC", search_text).lower()
      search_text = _split_script_runs(search_text)
      search_text = _truncate_bytes(search_text, self._max_event_bytes)
      event_timestamp = float(
          event.timestamp
          if event.timestamp is not None
          else platform_time.get_time()
      )
      author = event.author or ""
      content_json = event.content.model_dump_json(exclude_none=True)
      if event.id:
        event_id = event.id
      else:
        payload = f"{author}:{event_timestamp}:{content_json}".encode("utf-8")
        event_id = hashlib.sha256(payload).hexdigest()[:32]
      events_to_insert.append((
          app_name,
          user_id,
          scoped_session_id,
          event_id,
          author,
          event_timestamp,
          content_json,
          search_text,
      ))

    if not events_to_insert:
      return

    async with self._get_db_connection() as db:
      for row_data in events_to_insert:
        await db.execute(
            """
            INSERT INTO memory_events (
              app_name, user_id, session_id, event_id,
              author, timestamp, content_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(app_name, user_id, session_id, event_id) DO UPDATE SET
              author=excluded.author,
              timestamp=excluded.timestamp,
              content_json=excluded.content_json,
              search_text=excluded.search_text
            """,
            row_data,
        )
      await db.commit()

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    if not query or not query.strip():
      return SearchMemoryResponse()

    tokens = _extract_tokens(query)
    if not tokens:
      return SearchMemoryResponse()

    async with self._get_db_connection() as db:
      # self._fts_available is initialized during _ensure_schema within _get_db_connection.
      if self._fts_available:
        fts_query = " OR ".join(_escape_fts_token(token) for token in tokens)
        async with db.execute(
            """
            SELECT
              memory_events.id,
              memory_events.session_id,
              memory_events.event_id,
              memory_events.author,
              memory_events.timestamp,
              memory_events.content_json
            FROM memory_events_fts
            JOIN memory_events ON memory_events.id = memory_events_fts.rowid
            WHERE memory_events.app_name = ?
              AND memory_events.user_id = ?
              AND memory_events_fts.search_text MATCH ?
            ORDER BY bm25(memory_events_fts), memory_events.timestamp DESC
            LIMIT ?
            """,
            (app_name, user_id, fts_query, self._max_results),
        ) as cursor:
          rows = await cursor.fetchall()
      else:
        like_clauses = " OR ".join(
            ["search_text LIKE ? ESCAPE '\\'"] * len(tokens)
        )
        score_cases = " + ".join(
            ["(CASE WHEN search_text LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"]
            * len(tokens)
        )
        like_patterns = [f"%{_escape_like_query(token)}%" for token in tokens]
        params: list[Any] = [
            app_name,
            user_id,
            *like_patterns,
            *like_patterns,
            self._max_results,
        ]

        async with db.execute(
            f"""
            SELECT
              id, session_id, event_id, author, timestamp, content_json
            FROM memory_events
            WHERE app_name = ?
              AND user_id = ?
              AND ({like_clauses})
            ORDER BY ({score_cases}) DESC, timestamp DESC
            LIMIT ?
            """,
            params,
        ) as cursor:
          rows = await cursor.fetchall()

    return SearchMemoryResponse(
        memories=[_row_to_memory_entry(row) for row in rows]
    )


async def _apply_pragmas(db: aiosqlite.Connection) -> None:
  await db.execute(f"PRAGMA busy_timeout={_DEFAULT_BUSY_TIMEOUT_MS}")
  try:
    await db.execute("PRAGMA journal_mode=WAL")
  except (sqlite3.OperationalError, aiosqlite.OperationalError):
    pass
  try:
    await db.execute("PRAGMA synchronous=NORMAL")
  except (sqlite3.OperationalError, aiosqlite.OperationalError):
    pass


async def _setup_fts(db: aiosqlite.Connection, fts_mode: str) -> bool:
  if fts_mode == "off":
    return False
  try:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND"
        " name='memory_events_fts'"
    )
    fts_table_existed = await cursor.fetchone() is not None
    await cursor.close()
    await db.executescript(_CREATE_FTS_SQL)
    if not fts_table_existed:
      await db.execute(
          "INSERT INTO memory_events_fts(memory_events_fts) VALUES('rebuild')"
      )
  except (sqlite3.OperationalError, aiosqlite.OperationalError) as exc:
    if "no such module" in str(exc).lower():
      if fts_mode == "on":
        raise RuntimeError(
            "FTS5 is not available in this SQLite build."
        ) from exc
      return False
    raise
  return True


def _truncate_bytes(text: str, max_bytes: int) -> str:
  encoded = text.encode("utf-8")
  if len(encoded) <= max_bytes:
    return text
  return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _extract_event_text(event: Event) -> str:
  if not event.content or not event.content.parts:
    return ""
  parts = []
  for part in event.content.parts:
    if not part.text:
      continue
    if getattr(part, "thought", False):
      continue
    text = part.text.replace("\n", " ").strip()
    if text:
      parts.append(text)
  if not parts:
    return ""
  return " ".join(parts)


def _is_latin(c: str) -> bool:
  if c.isascii():
    return c.isalnum() or c == "_"
  return unicodedata.name(c, "").startswith("LATIN")


def _split_script_runs(text: str) -> str:
  if text.isascii():
    return text
  chunks = []
  for word in text.split():
    if word.isascii():
      chunks.append(word)
    else:
      for _, group in itertools.groupby(word, _is_latin):
        chunks.append("".join(group))
  return " ".join(chunks)


def _extract_tokens(query: str) -> list[str]:
  normalized = unicodedata.normalize("NFC", query)
  normalized = _split_script_runs(normalized)
  seen = set()
  tokens = []
  for word in re.findall(r"[\w%]+", normalized):
    lowered = word.lower()
    if lowered not in seen:
      seen.add(lowered)
      tokens.append(lowered)
  return tokens


def _escape_fts_token(token: str) -> str:
  escaped = token.replace('"', '""')
  return f'"{escaped}"'


def _escape_like_query(word: str) -> str:
  return word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_memory_entry(row: aiosqlite.Row) -> MemoryEntry:
  content = types.Content.model_validate_json(row["content_json"])
  return MemoryEntry(
      id=str(row["id"]),
      content=content,
      author=row["author"] or None,
      timestamp=_utils.format_timestamp(row["timestamp"]),
      custom_metadata={
          "session_id": row["session_id"],
          "event_id": row["event_id"],
      },
  )
