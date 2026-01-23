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

import asyncio
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from typing import Awaitable
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)


_SCHEMA_VERSION = "1"
_DEFAULT_MAX_RESULTS = 50
_DEFAULT_MAX_SESSION_BYTES = 262_144
_DEFAULT_MAX_SNIPPET_CHARS = 1_000
_DEFAULT_BUSY_TIMEOUT_MS = 3_000

_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  session_json TEXT,
  search_text TEXT NOT NULL,
  extracted_json TEXT,
  metadata_json TEXT,
  content_sha256 TEXT NOT NULL,
  UNIQUE(app_name, user_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_scope_updated
ON sessions(app_name, user_id, updated_at_ms DESC);
"""

_CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
USING fts5(
  search_text,
  content='sessions',
  content_rowid='id',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS sessions_ai
AFTER INSERT ON sessions BEGIN
  INSERT INTO sessions_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad
AFTER DELETE ON sessions BEGIN
  INSERT INTO sessions_fts(sessions_fts, rowid, search_text)
  VALUES('delete', old.id, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au
AFTER UPDATE OF search_text ON sessions BEGIN
  INSERT INTO sessions_fts(sessions_fts, rowid, search_text)
  VALUES('delete', old.id, old.search_text);
  INSERT INTO sessions_fts(rowid, search_text)
  VALUES (new.id, new.search_text);
END;
"""


@dataclass(frozen=True)
class ExtractionResult:
  """Result of extracting memory content from a session."""

  summary_text: str
  entries: Optional[list[str]] = None
  metadata: Optional[dict[str, Any]] = None


class MemoryExtractor(Protocol):
  """Protocol for extracting memory content from a session."""

  def __call__(
      self, session: Session
  ) -> ExtractionResult | Awaitable[ExtractionResult]:
    ...


class _DefaultExtractor:
  """Deterministic extractor with no external dependencies."""

  def __call__(self, session: Session) -> ExtractionResult:
    del session
    return ExtractionResult(summary_text="")


class SqliteMemoryService(BaseMemoryService):
  """A persistent, local memory service backed by SQLite."""

  def __init__(
      self,
      *,
      db_path: str | Path,
      fts: str = "auto",
      extractor: Optional[MemoryExtractor] = None,
      max_session_bytes: int = _DEFAULT_MAX_SESSION_BYTES,
      store_full_events: bool = True,
      summary_policy: str = "raw_plus_summary",
      max_search_snippet_chars: int = _DEFAULT_MAX_SNIPPET_CHARS,
      max_results: int = _DEFAULT_MAX_RESULTS,
  ):
    """Initializes a SqliteMemoryService.

    Args:
      db_path: The SQLite database path.
      fts: "auto", "on", or "off" for FTS5 usage.
      extractor: Optional extractor for summaries and metadata.
      max_session_bytes: Maximum bytes to store per session snapshot.
      store_full_events: Whether to persist the full session JSON.
      summary_policy: "none", "extractor_only", or "raw_plus_summary".
      max_search_snippet_chars: Maximum chars returned per memory entry.
      max_results: Maximum number of memories returned per query.
    """
    self._db_path = str(db_path)
    if not self._db_path:
      raise ValueError("db_path must be set.")

    if self._db_path != ":memory:":
      path = Path(self._db_path)
      if path.exists() and path.is_dir():
        raise ValueError(f"db_path {self._db_path} is a directory.")
      path.parent.mkdir(parents=True, exist_ok=True)

    if fts not in ("auto", "on", "off"):
      raise ValueError("fts must be one of: auto, on, off.")
    if summary_policy not in ("none", "extractor_only", "raw_plus_summary"):
      raise ValueError(
          "summary_policy must be one of: none, extractor_only, "
          "raw_plus_summary."
      )
    if max_session_bytes <= 0:
      raise ValueError("max_session_bytes must be positive.")
    if max_search_snippet_chars <= 0:
      raise ValueError("max_search_snippet_chars must be positive.")
    if max_results <= 0:
      raise ValueError("max_results must be positive.")

    self._fts_mode = fts
    self._extractor = extractor or _DefaultExtractor()
    self._max_session_bytes = max_session_bytes
    self._store_full_events = store_full_events
    self._summary_policy = summary_policy
    self._max_search_snippet_chars = max_search_snippet_chars
    self._max_results = max_results
    self._initialized = False
    self._fts_available: Optional[bool] = None
    self._init_lock = threading.Lock()

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    raw_text = _build_raw_text(session.events)
    extraction = await _run_extractor(self._extractor, session)
    search_text = _build_search_text(raw_text, extraction, self._summary_policy)
    session_json = (
        _serialize_session(session) if self._store_full_events else ""
    )
    extracted_json = _serialize_extracted(extraction)
    metadata_json = _serialize_metadata(extraction)

    if not search_text and not session_json:
      return

    _enforce_max_bytes(
        self._max_session_bytes,
        session_json,
        search_text,
        extracted_json,
        metadata_json,
    )
    content_sha256 = _hash_payload(
        session_json,
        search_text,
        extracted_json,
        metadata_json,
    )
    now_ms = _now_ms()

    await asyncio.to_thread(
        self._upsert_session,
        session.app_name,
        session.user_id,
        session.id,
        now_ms,
        session_json,
        search_text,
        extracted_json,
        metadata_json,
        content_sha256,
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    if not query or not query.strip():
      return SearchMemoryResponse()

    rows = await asyncio.to_thread(
        self._search_sessions, app_name, user_id, query
    )
    response = SearchMemoryResponse()
    for row in rows:
      response.memories.append(
          _row_to_memory_entry(row, query, self._max_search_snippet_chars)
      )
    return response

  def _upsert_session(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      now_ms: int,
      session_json: str,
      search_text: str,
      extracted_json: str,
      metadata_json: str,
      content_sha256: str,
  ) -> None:
    with self._open_connection() as conn:
      cursor = conn.execute(
          """
          SELECT id, content_sha256, created_at_ms
          FROM sessions
          WHERE app_name=? AND user_id=? AND session_id=?
          """,
          (app_name, user_id, session_id),
      )
      row = cursor.fetchone()
      if row and row["content_sha256"] == content_sha256:
        return
      if row:
        conn.execute(
            """
            UPDATE sessions
            SET updated_at_ms=?, session_json=?, search_text=?,
                extracted_json=?, metadata_json=?, content_sha256=?
            WHERE app_name=? AND user_id=? AND session_id=?
            """,
            (
                now_ms,
                session_json or None,
                search_text,
                extracted_json or None,
                metadata_json or None,
                content_sha256,
                app_name,
                user_id,
                session_id,
            ),
        )
      else:
        conn.execute(
            """
            INSERT INTO sessions (
              app_name, user_id, session_id,
              created_at_ms, updated_at_ms,
              session_json, search_text, extracted_json, metadata_json,
              content_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                app_name,
                user_id,
                session_id,
                now_ms,
                now_ms,
                session_json or None,
                search_text,
                extracted_json or None,
                metadata_json or None,
                content_sha256,
            ),
        )
      conn.commit()

  def _search_sessions(
      self, app_name: str, user_id: str, query: str
  ) -> list[sqlite3.Row]:
    with self._open_connection() as conn:
      if self._fts_available:
        fts_query = _prepare_fts_query(query)
        cursor = conn.execute(
            """
            SELECT s.id, s.session_id, s.search_text, s.extracted_json,
                   s.metadata_json, s.updated_at_ms
            FROM sessions_fts f
            JOIN sessions s ON s.id = f.rowid
            WHERE s.app_name=? AND s.user_id=? AND f.search_text MATCH ?
            ORDER BY bm25(sessions_fts)
            LIMIT ?
            """,
            (app_name, user_id, fts_query, self._max_results),
        )
      else:
        like_query = f"%{query}%"
        cursor = conn.execute(
            """
            SELECT id, session_id, search_text, extracted_json,
                   metadata_json, updated_at_ms
            FROM sessions
            WHERE app_name=? AND user_id=? AND search_text LIKE ?
            ORDER BY updated_at_ms DESC
            LIMIT ?
            """,
            (app_name, user_id, like_query, self._max_results),
        )
      return cursor.fetchall()

  def _open_connection(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self._db_path)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    self._ensure_schema(conn)
    return conn

  def _ensure_schema(self, conn: sqlite3.Connection) -> None:
    with self._init_lock:
      if not self._initialized:
        conn.executescript(_CREATE_SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", _SCHEMA_VERSION),
        )
        self._fts_available = _setup_fts(conn, self._fts_mode)
        conn.commit()
        self._initialized = True


def _apply_pragmas(conn: sqlite3.Connection) -> None:
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA synchronous=NORMAL")
  conn.execute(f"PRAGMA busy_timeout={_DEFAULT_BUSY_TIMEOUT_MS}")


def _setup_fts(conn: sqlite3.Connection, fts_mode: str) -> bool:
  if fts_mode == "off":
    return False
  try:
    conn.executescript(_CREATE_FTS_SQL)
  except sqlite3.OperationalError as exc:
    if fts_mode == "on":
      raise RuntimeError("FTS5 is not available in this SQLite build.") from exc
    return False
  return True


def _serialize_session(session: Session) -> str:
  payload = session.model_dump(exclude_none=True, mode="json")
  return _stable_json_dumps(payload)


def _stable_json_dumps(payload: Any) -> str:
  return json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
  )


async def _run_extractor(
    extractor: MemoryExtractor, session: Session
) -> ExtractionResult:
  result = extractor(session)
  if inspect.isawaitable(result):
    return await result
  return result


def _serialize_extracted(extraction: ExtractionResult) -> str:
  if not extraction:
    return ""
  payload: dict[str, Any] = {}
  if extraction.summary_text:
    payload["summary"] = extraction.summary_text
  if extraction.entries:
    payload["entries"] = extraction.entries
  if not payload:
    return ""
  return _stable_json_dumps(payload)


def _serialize_metadata(extraction: ExtractionResult) -> str:
  if not extraction or not extraction.metadata:
    return ""
  return _stable_json_dumps(extraction.metadata)


def _hash_payload(
    session_json: str,
    search_text: str,
    extracted_json: str,
    metadata_json: str,
) -> str:
  payload = {
      "session_json": session_json,
      "search_text": search_text,
      "extracted_json": extracted_json,
      "metadata_json": metadata_json,
  }
  digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8"))
  return digest.hexdigest()


def _enforce_max_bytes(
    max_bytes: int,
    session_json: str,
    search_text: str,
    extracted_json: str,
    metadata_json: str,
) -> None:
  total_bytes = 0
  for item in (session_json, search_text, extracted_json, metadata_json):
    if item:
      total_bytes += len(item.encode("utf-8"))
    raise ValueError(f"Session payload is too large ({total_bytes} bytes), exceeding the limit of {max_bytes} bytes.")


def _build_raw_text(events: list[Event]) -> str:
  lines = []
  for event in events:
    text = _extract_event_text(event)
    if text:
      lines.append(text)
  return "\n".join(lines).strip()


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
  joined = " ".join(parts)
  if event.author:
    return f"{event.author}: {joined}"
  return joined


def _build_search_text(
    raw_text: str,
    extraction: ExtractionResult,
    summary_policy: str,
) -> str:
  extra_lines = []
  if extraction:
    if extraction.summary_text:
      extra_lines.append(extraction.summary_text)
    if extraction.entries:
      extra_lines.extend(extraction.entries)
  extra_text = "\n".join(extra_lines).strip()

  if summary_policy == "none":
    return raw_text
  if summary_policy == "extractor_only":
    return extra_text
  if summary_policy == "raw_plus_summary":
    if extra_text and raw_text:
      return f"{raw_text}\n{extra_text}"
    return extra_text or raw_text
  raise ValueError(f"Unsupported summary_policy: {summary_policy}")


def _prepare_fts_query(query: str) -> str:
  escaped = query.replace('"', '""')
  return f'"{escaped}"'


def _row_to_memory_entry(
    row: sqlite3.Row, query: str, max_snippet_chars: int
) -> MemoryEntry:
  search_text = row["search_text"] or ""
  snippet = _build_snippet(search_text, query, max_snippet_chars)
  content = types.Content(
      role="user",
      parts=[types.Part(text=snippet)],
  )
  metadata: dict[str, Any] = {
      "session_id": row["session_id"],
      "updated_at_ms": row["updated_at_ms"],
  }
  if row["metadata_json"]:
    try:
      metadata["metadata"] = json.loads(row["metadata_json"])
    except json.JSONDecodeError:
      logger.warning(
          "Failed to decode metadata_json for session_id %s.",
          row["session_id"],
      )
  if row["extracted_json"]:
    try:
      metadata["extracted"] = json.loads(row["extracted_json"])
    except json.JSONDecodeError:
      logger.warning(
          "Failed to decode extracted_json for session_id %s.",
          row["session_id"],
      )
  return MemoryEntry(
      id=str(row["id"]),
      content=content,
      custom_metadata=metadata,
      author="memory",
      timestamp=_utils.format_timestamp(row["updated_at_ms"] / 1000.0),
  )


def _build_snippet(text: str, query: str, max_chars: int) -> str:
  if not text:
    return ""
  if len(text) <= max_chars:
    return text
  lower_text = text.lower()
  lower_query = query.lower()
  index = lower_text.find(lower_query)
  if index == -1:
    return text[:max_chars].rstrip()
  start = max(index - max_chars // 3, 0)
  end = min(start + max_chars, len(text))
  return text[start:end].rstrip()


def _now_ms() -> int:
  return int(time.time() * 1000)
