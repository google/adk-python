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
import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING

import aiosqlite
from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


def _prepare_fts_query(query: str) -> str:
  """Prepares a query for FTS5 by escaping special characters.

  FTS5 has special characters like quotes that need to be escaped.
  We also wrap each word with quotes for exact word matching.
  """
  # Remove special FTS5 characters and split into words
  import re

  words = re.findall(r'\w+', query.lower())
  if not words:
    return ''
  # Join words with OR to match any of the words
  return ' OR '.join(f'"{word}"' for word in words)


class SqliteMemoryService(BaseMemoryService):
  """An async SQLite-based memory service for persistent storage with FTS5 search.

  This implementation provides persistent storage of memory entries in a SQLite
  database using aiosqlite for async operations with SQLite FTS5 (Full-Text Search)
  for efficient and scalable search functionality.

  This service is suitable for development and production use where persistent
  memory with fast text search is required.
  """

  def __init__(self, db_path: str = 'memory.db'):
    """Initializes a SqliteMemoryService.

    Args:
        db_path: Path to the SQLite database file. Defaults to 'memory.db'
            in the current directory.
    """
    self._db_path = Path(db_path)
    self._initialized = False
    self._init_lock = None

  async def _ensure_initialized(self):
    """Ensures the database is initialized exactly once using lazy initialization."""
    if self._initialized:
      return

    # Lazy creation of the lock to avoid event loop issues in Python 3.9
    if self._init_lock is None:
      self._init_lock = asyncio.Lock()

    async with self._init_lock:
      # Double-check after acquiring lock
      if self._initialized:
        return

      try:
        await self._init_db()
        self._initialized = True
      except aiosqlite.Error as e:
        logger.error(f'Failed to initialize database: {e}')
        # Don't mark as initialized if init failed
        raise

  async def _init_db(self):
    """Initializes the SQLite database with required tables and FTS5 virtual table."""
    # Create directory if it doesn't exist
    self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(self._db_path) as db:
      # Create memory_entries table
      await db.execute("""
          CREATE TABLE IF NOT EXISTS memory_entries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              app_name TEXT NOT NULL,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              author TEXT,
              timestamp REAL NOT NULL,
              content_json TEXT NOT NULL,
              text_content TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(app_name, user_id, session_id, timestamp, author)
          )
      """)

      # Create FTS5 virtual table for efficient full-text search
      await db.execute("""
          CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_fts USING fts5(
              app_name,
              user_id,
              text_content,
              content='memory_entries',
              content_rowid='id'
          )
      """)

      # Create triggers to keep FTS5 table in sync with main table
      await db.execute("""
          CREATE TRIGGER IF NOT EXISTS memory_entries_ai AFTER INSERT ON memory_entries BEGIN
            INSERT INTO memory_entries_fts(rowid, app_name, user_id, text_content)
            VALUES (new.id, new.app_name, new.user_id, new.text_content);
          END
      """)

      await db.execute("""
          CREATE TRIGGER IF NOT EXISTS memory_entries_ad AFTER DELETE ON memory_entries BEGIN
            INSERT INTO memory_entries_fts(memory_entries_fts, rowid, app_name, user_id, text_content)
            VALUES('delete', old.id, old.app_name, old.user_id, old.text_content);
          END
      """)

      await db.execute("""
          CREATE TRIGGER IF NOT EXISTS memory_entries_au AFTER UPDATE ON memory_entries BEGIN
            INSERT INTO memory_entries_fts(memory_entries_fts, rowid, app_name, user_id, text_content)
            VALUES('delete', old.id, old.app_name, old.user_id, old.text_content);
            INSERT INTO memory_entries_fts(rowid, app_name, user_id, text_content)
            VALUES (new.id, new.app_name, new.user_id, new.text_content);
          END
      """)

      # Create index for timestamp ordering on main table
      await db.execute("""
          CREATE INDEX IF NOT EXISTS idx_memory_timestamp 
          ON memory_entries(timestamp DESC)
      """)

      await db.commit()

  @override
  async def add_session_to_memory(self, session: Session):
    """Adds a session to the SQLite memory store.

    A session may be added multiple times during its lifetime. Duplicate
    entries are ignored based on unique constraints.

    Args:
        session: The session to add to memory.
    """
    await self._ensure_initialized()

    async with aiosqlite.connect(self._db_path) as db:
      for event in session.events:
        if not event.content or not event.content.parts:
          continue

        # Extract text content for search
        text_parts = [part.text for part in event.content.parts if part.text]
        if not text_parts:
          continue

        text_content = ' '.join(text_parts)
        content_json = json.dumps(
            event.content.model_dump(exclude_none=True, mode='json')
        )

        try:
          await db.execute(
              """
              INSERT OR IGNORE INTO memory_entries 
              (app_name, user_id, session_id, author, timestamp, 
               content_json, text_content)
              VALUES (?, ?, ?, ?, ?, ?, ?)
          """,
              (
                  session.app_name,
                  session.user_id,
                  session.id,
                  event.author,
                  event.timestamp,
                  content_json,
                  text_content,
              ),
          )
        except aiosqlite.Error as e:
          logger.error(f'Database error adding memory entry: {e}')
          continue
        except (json.JSONEncodeError, ValueError) as e:
          logger.error(f'Error encoding memory entry content: {e}')
          continue

      await db.commit()
      logger.info(
          f'Added session {session.id} to memory for user {session.user_id}'
      )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for memories using SQLite FTS5 full-text search.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing matching memories ordered by timestamp.
    """
    await self._ensure_initialized()

    if not query.strip():
      return SearchMemoryResponse(memories=[])

    # Prepare query for FTS5
    fts_query = _prepare_fts_query(query)
    if not fts_query:
      return SearchMemoryResponse(memories=[])

    memories = []
    try:
      async with aiosqlite.connect(self._db_path) as db:
        # Use FTS5 for efficient full-text search
        async with db.execute(
            """
            SELECT m.author, m.timestamp, m.content_json
            FROM memory_entries_fts f
            JOIN memory_entries m ON f.rowid = m.id
            WHERE f.memory_entries_fts MATCH ? 
              AND f.app_name = ? 
              AND f.user_id = ?
            ORDER BY m.timestamp DESC
        """,
            (fts_query, app_name, user_id),
        ) as cursor:
          async for author, timestamp, content_json in cursor:
            try:
              content_dict = json.loads(content_json)
              content = types.Content(**content_dict)

              memory_entry = MemoryEntry(
                  content=content,
                  author=author,
                  timestamp=_utils.format_timestamp(timestamp),
              )
              memories.append(memory_entry)
            except (json.JSONDecodeError, ValueError) as e:
              logger.error(f'Error parsing memory entry content: {e}')
              continue

    except aiosqlite.Error as e:
      logger.error(f'Database error during search: {e}')
      # Return empty response on database errors
      return SearchMemoryResponse(memories=[])

    return SearchMemoryResponse(memories=memories)

  async def clear_memory(self, app_name: str = None, user_id: str = None):
    """Clears memory entries from the database.

    Args:
        app_name: If specified, only clear entries for this app.
        user_id: If specified, only clear entries for this user (requires app_name).
    """
    await self._ensure_initialized()

    try:
      async with aiosqlite.connect(self._db_path) as db:
        if user_id and not app_name:
          raise ValueError(
              'When clearing memory by user_id, app_name must also be provided.'
          )

        if app_name and user_id:
          await db.execute(
              'DELETE FROM memory_entries WHERE app_name = ? AND user_id = ?',
              (app_name, user_id),
          )
        elif app_name:
          await db.execute(
              'DELETE FROM memory_entries WHERE app_name = ?', (app_name,)
          )
        else:
          await db.execute('DELETE FROM memory_entries')

        await db.commit()
        logger.info('Cleared memory entries from database')
    except aiosqlite.Error as e:
      logger.error(f'Database error clearing memory: {e}')
      raise

  async def get_memory_stats(self) -> Dict[str, Any]:
    """Returns statistics about the memory database.

    Returns:
        Dictionary containing database statistics.
    """
    await self._ensure_initialized()

    try:
      async with aiosqlite.connect(self._db_path) as db:
        # Total entries
        async with db.execute('SELECT COUNT(*) FROM memory_entries') as cursor:
          row = await cursor.fetchone()
          total_entries = row[0] if row else 0

        # Entries per app
        entries_per_app = {}
        async with db.execute("""
            SELECT app_name, COUNT(*) 
            FROM memory_entries 
            GROUP BY app_name
        """) as cursor:
          async for app_name, count in cursor:
            entries_per_app[app_name] = count

        # Database file size
        db_size_bytes = (
            self._db_path.stat().st_size if self._db_path.exists() else 0
        )

        return {
            'total_entries': total_entries,
            'entries_per_app': entries_per_app,
            'database_file_size_bytes': db_size_bytes,
            'database_path': str(self._db_path),
        }
    except aiosqlite.Error as e:
      logger.error(f'Database error getting stats: {e}')
      return {
          'total_entries': 0,
          'entries_per_app': {},
          'database_file_size_bytes': 0,
          'database_path': str(self._db_path),
          'error': str(e),
      }
