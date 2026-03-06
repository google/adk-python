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

"""SQL-backed memory service with scratchpad support for ADK agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import asynccontextmanager
import logging
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.genai import types
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry
from .memory_search_backend import KeywordSearchBackend
from .memory_search_backend import MemorySearchBackend
from .schemas.memory_schema import Base
from .schemas.memory_schema import StorageMemoryEntry
from .schemas.memory_schema import StorageScratchpadKV
from .schemas.memory_schema import StorageScratchpadLog

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

_SQLITE_DIALECT = "sqlite"


class DatabaseMemoryService(BaseMemoryService):
  """A durable, SQL-backed memory service for any SQLAlchemy-supported DB.

  Works with SQLite, PostgreSQL, MySQL, MariaDB, and Spanner. Exposes a
  scratchpad (KV store + append-log) for agents to use as intermediate
  working memory during task execution.

  Usage::

      from google.adk.memory import DatabaseMemoryService

      # SQLite (no external DB needed):
      svc = DatabaseMemoryService("sqlite+aiosqlite:///:memory:")

      # PostgreSQL:
      svc = DatabaseMemoryService(
          "postgresql+asyncpg://user:pass@host/dbname"
      )
  """

  def __init__(
      self,
      db_url: str,
      search_backend: Optional[MemorySearchBackend] = None,
      **kwargs: Any,
  ):
    """Initialises the service and creates a DB engine.

    Args:
      db_url: SQLAlchemy async connection URL.
      search_backend: Optional custom search backend. Defaults to
        KeywordSearchBackend.
      **kwargs: Extra keyword arguments forwarded to
        sqlalchemy.ext.asyncio.create_async_engine.

    Raises:
      ValueError: If the db_url is invalid or the required DB driver is
        not installed.
    """
    try:
      engine_kwargs = dict(kwargs)
      url = make_url(db_url)
      backend = url.get_backend_name()
      if backend == _SQLITE_DIALECT and url.database == ":memory:":
        engine_kwargs.setdefault("poolclass", StaticPool)
        connect_args = dict(engine_kwargs.get("connect_args", {}))
        connect_args.setdefault("check_same_thread", False)
        engine_kwargs["connect_args"] = connect_args
      elif backend != _SQLITE_DIALECT:
        engine_kwargs.setdefault("pool_pre_ping", True)

      self.db_engine: AsyncEngine = create_async_engine(db_url, **engine_kwargs)
    except ArgumentError as exc:
      raise ValueError(
          f"Invalid database URL format or argument '{db_url}'."
      ) from exc
    except ImportError as exc:
      raise ValueError(
          f"Database-related module not found for URL '{db_url}'."
      ) from exc

    self._session_factory: async_sessionmaker[AsyncSession] = (
        async_sessionmaker(bind=self.db_engine, expire_on_commit=False)
    )
    self._search_backend: MemorySearchBackend = (
        search_backend or KeywordSearchBackend()
    )
    self._tables_created = False
    self._table_creation_lock = asyncio.Lock()

  # ---------------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------------

  @asynccontextmanager
  async def _session(self) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession; roll back on exception."""
    async with self._session_factory() as session:
      try:
        yield session
        await session.commit()
      except Exception:
        await session.rollback()
        raise

  async def _prepare_tables(self) -> None:
    """Lazy, double-checked table initialisation."""
    if self._tables_created:
      return
    async with self._table_creation_lock:
      if self._tables_created:
        return
      async with self.db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
      self._tables_created = True

  @staticmethod
  def _extract_search_text(content: types.Content) -> str:
    """Join all text parts of a Content into a single searchable string."""
    if not content or not content.parts:
      return ""
    return " ".join(part.text for part in content.parts if part.text)

  @staticmethod
  def _should_skip_event(event: Event) -> bool:
    """Return True if the event has no usable text content."""
    if not event.content or not event.content.parts:
      return True
    return not any(part.text for part in event.content.parts if part.text)

  # ---------------------------------------------------------------------------
  # BaseMemoryService implementation
  # ---------------------------------------------------------------------------

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Idempotently ingest all events from a session.

    Deletes any existing rows for this session, then re-inserts from scratch.

    Args:
      session: The session whose events should be stored in memory.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      # Delete existing rows for this session so re-ingest is idempotent.
      await sql.execute(
          delete(StorageMemoryEntry).where(
              StorageMemoryEntry.app_name == session.app_name,
              StorageMemoryEntry.user_id == session.user_id,
              StorageMemoryEntry.session_id == session.id,
          )
      )
      for event in session.events:
        if self._should_skip_event(event):
          continue
        content_dict = event.content.model_dump(mode="json", exclude_none=True)
        sql.add(
            StorageMemoryEntry(
                id=str(uuid.uuid4()),
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                event_id=event.id,
                author=event.author,
                timestamp=_utils.format_timestamp(event.timestamp),
                content_json=content_dict,
                search_text=self._extract_search_text(event.content),
                custom_metadata={},
            )
        )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: Optional[str] = None,
      custom_metadata: Optional[Mapping[str, object]] = None,
  ) -> None:
    """Delta-insert events; skips duplicate event_id within the same session.

    Args:
      app_name: The application name for memory scope.
      user_id: The user ID for memory scope.
      events: The events to add to memory.
      session_id: Optional session ID for memory scope/partitioning.
      custom_metadata: Optional metadata attached to each stored entry.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      # Fetch existing event IDs for this session to avoid duplicates.
      stmt = select(StorageMemoryEntry.event_id).where(
          StorageMemoryEntry.app_name == app_name,
          StorageMemoryEntry.user_id == user_id,
          StorageMemoryEntry.session_id == session_id,
          StorageMemoryEntry.event_id.isnot(None),
      )
      result = await sql.execute(stmt)
      existing_event_ids = {row[0] for row in result.fetchall()}

      meta = dict(custom_metadata) if custom_metadata else {}
      for event in events:
        if self._should_skip_event(event):
          continue
        if event.id and event.id in existing_event_ids:
          continue
        content_dict = event.content.model_dump(mode="json", exclude_none=True)
        sql.add(
            StorageMemoryEntry(
                id=str(uuid.uuid4()),
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                event_id=event.id,
                author=event.author,
                timestamp=_utils.format_timestamp(event.timestamp),
                content_json=content_dict,
                search_text=self._extract_search_text(event.content),
                custom_metadata=meta,
            )
        )
        if event.id:
          existing_event_ids.add(event.id)

  @override
  async def add_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memories: Sequence[MemoryEntry],
      custom_metadata: Optional[Mapping[str, object]] = None,
  ) -> None:
    """Directly insert MemoryEntry objects (not tied to session events).

    Args:
      app_name: The application name for memory scope.
      user_id: The user ID for memory scope.
      memories: Explicit memory items to add.
      custom_metadata: Optional metadata attached to each stored entry.
    """
    await self._prepare_tables()
    meta = dict(custom_metadata) if custom_metadata else {}
    async with self._session() as sql:
      for entry in memories:
        entry_id = entry.id or str(uuid.uuid4())
        content_dict = entry.content.model_dump(mode="json", exclude_none=True)
        sql.add(
            StorageMemoryEntry(
                id=entry_id,
                app_name=app_name,
                user_id=user_id,
                session_id=None,
                event_id=None,
                author=entry.author,
                timestamp=entry.timestamp,
                content_json=content_dict,
                search_text=self._extract_search_text(entry.content),
                custom_metadata={**entry.custom_metadata, **meta},
            )
        )

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Search stored memories using the configured search backend.

    Args:
      app_name: The name of the application.
      user_id: The id of the user.
      query: The query to search for.

    Returns:
      A SearchMemoryResponse containing the matching memories.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      rows = await self._search_backend.search(
          sql_session=sql,
          app_name=app_name,
          user_id=user_id,
          query=query,
      )
      memories = []
      for row in rows:
        try:
          content = types.Content.model_validate(row.content_json)
        except Exception:  # pylint: disable=broad-except
          logger.warning(
              "Skipping memory entry %s: invalid content JSON", row.id
          )
          continue
        memories.append(
            MemoryEntry(
                id=row.id,
                content=content,
                author=row.author,
                timestamp=row.timestamp,
                custom_metadata=row.custom_metadata or {},
            )
        )
    return SearchMemoryResponse(memories=memories)

  # ---------------------------------------------------------------------------
  # Scratchpad KV methods
  # ---------------------------------------------------------------------------

  async def set_scratchpad(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
      key: str,
      value: Any,
  ) -> None:
    """Write a key-value pair to the scratchpad.

    Overwrites any existing value for the same composite key.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level (non-session) KV.
      key: The key to write.
      value: The JSON-serialisable value to store.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      existing = await sql.get(
          StorageScratchpadKV, (app_name, user_id, session_id, key)
      )
      if existing is not None:
        existing.value_json = value
      else:
        sql.add(
            StorageScratchpadKV(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                key=key,
                value_json=value,
            )
        )

  async def get_scratchpad(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
      key: str,
  ) -> Any | None:
    """Read a value from the scratchpad.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level (non-session) KV.
      key: The key to read.

    Returns:
      The stored value, or None if the key does not exist.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      row = await sql.get(
          StorageScratchpadKV, (app_name, user_id, session_id, key)
      )
      return row.value_json if row is not None else None

  async def delete_scratchpad(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
      key: str,
  ) -> None:
    """Delete a key-value pair from the scratchpad. No-op if not found.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level (non-session) KV.
      key: The key to delete.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      await sql.execute(
          delete(StorageScratchpadKV).where(
              StorageScratchpadKV.app_name == app_name,
              StorageScratchpadKV.user_id == user_id,
              StorageScratchpadKV.session_id == session_id,
              StorageScratchpadKV.key == key,
          )
      )

  async def list_scratchpad_keys(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
  ) -> list[str]:
    """Return all keys present in the scratchpad for the given scope.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level (non-session) KV.

    Returns:
      A list of key strings.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      result = await sql.execute(
          select(StorageScratchpadKV.key).where(
              StorageScratchpadKV.app_name == app_name,
              StorageScratchpadKV.user_id == user_id,
              StorageScratchpadKV.session_id == session_id,
          )
      )
      return [row[0] for row in result.fetchall()]

  # ---------------------------------------------------------------------------
  # Scratchpad log methods
  # ---------------------------------------------------------------------------

  async def append_log(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
      content: str,
      tag: Optional[str] = None,
      agent_name: Optional[str] = None,
      extra: Optional[Any] = None,
  ) -> None:
    """Append an entry to the append-only scratchpad log.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level log.
      content: The text content to log.
      tag: Optional category label for filtering.
      agent_name: Optional name of the agent appending this entry.
      extra: Optional JSON-serialisable extra data.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      sql.add(
          StorageScratchpadLog(
              app_name=app_name,
              user_id=user_id,
              session_id=session_id,
              tag=tag,
              agent_name=agent_name,
              content=content,
              extra_json=extra,
          )
      )

  async def get_log(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str = "",
      tag: Optional[str] = None,
      limit: int = 50,
  ) -> list[dict]:
    """Read the most recent log entries, optionally filtered by tag.

    Args:
      app_name: Application name scope.
      user_id: User ID scope.
      session_id: Session ID scope. Use '' for user-level log.
      tag: Optional tag to filter results by.
      limit: Maximum number of entries to return.

    Returns:
      A list of dicts with keys: id, tag, agent_name, content, extra.
    """
    await self._prepare_tables()
    async with self._session() as sql:
      stmt = (
          select(StorageScratchpadLog)
          .where(
              StorageScratchpadLog.app_name == app_name,
              StorageScratchpadLog.user_id == user_id,
              StorageScratchpadLog.session_id == session_id,
          )
          .order_by(StorageScratchpadLog.id.desc())
          .limit(limit)
      )
      if tag is not None:
        stmt = stmt.where(StorageScratchpadLog.tag == tag)
      result = await sql.execute(stmt)
      rows = result.scalars().all()
      return [
          {
              "id": r.id,
              "tag": r.tag,
              "agent_name": r.agent_name,
              "content": r.content,
              "extra": r.extra_json,
          }
          for r in reversed(rows)
      ]
