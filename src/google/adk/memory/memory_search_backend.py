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

"""Memory search backends for DatabaseMemoryService."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import re
from typing import TYPE_CHECKING

from sqlalchemy import or_
from sqlalchemy import select

from .schemas.memory_schema import StorageMemoryEntry

if TYPE_CHECKING:
  from sqlalchemy.ext.asyncio import AsyncSession

_ILIKE_DIALECTS = frozenset({"postgresql", "mysql", "mariadb"})


class MemorySearchBackend(ABC):
  """Abstract base class for memory search strategies."""

  @abstractmethod
  async def search(
      self,
      *,
      sql_session: AsyncSession,
      app_name: str,
      user_id: str,
      query: str,
      limit: int = 10,
  ) -> Sequence[StorageMemoryEntry]:
    """Search for memory entries matching the query.

    Args:
      sql_session: The active async SQLAlchemy session.
      app_name: Application name scope.
      user_id: User ID scope.
      query: Natural-language or keyword query string.
      limit: Maximum number of results to return.

    Returns:
      A sequence of matching StorageMemoryEntry rows.
    """


class KeywordSearchBackend(MemorySearchBackend):
  """LIKE/ILIKE keyword search on the search_text column.

  Strategy:
    1. Tokenise the query into individual words.
    2. Try an AND predicate (all tokens must appear) — return if found.
    3. Fall back to OR (any token matches) if AND yields nothing.

  Uses ILIKE on PostgreSQL/MySQL/MariaDB and LIKE on SQLite
  (case-insensitive by default collation).
  """

  async def search(
      self,
      *,
      sql_session: AsyncSession,
      app_name: str,
      user_id: str,
      query: str,
      limit: int = 10,
  ) -> Sequence[StorageMemoryEntry]:
    """Search for memory entries using LIKE/ILIKE keyword matching."""
    if not query or not query.strip():
      return []

    tokens = [
        cleaned
        for raw in query.split()
        if raw.strip()
        for cleaned in [re.sub(r"[^\w]", "", raw).lower()]
        if cleaned
    ]
    if not tokens:
      return []

    # Determine dialect via the engine bound to this session.
    dialect_name = sql_session.get_bind().dialect.name
    use_ilike = dialect_name in _ILIKE_DIALECTS

    def _like_expr(token: str):
      pattern = f"%{token}%"
      col = StorageMemoryEntry.search_text
      return col.ilike(pattern) if use_ilike else col.like(pattern)

    base_stmt = (
        select(StorageMemoryEntry)
        .where(
            StorageMemoryEntry.app_name == app_name,
            StorageMemoryEntry.user_id == user_id,
            StorageMemoryEntry.search_text.isnot(None),
        )
        .limit(limit)
    )

    # AND predicate: all tokens must match.
    and_stmt = base_stmt.where(*[_like_expr(t) for t in tokens])
    result = await sql_session.execute(and_stmt)
    rows = result.scalars().all()
    if rows:
      return rows

    # OR fallback: any token matches.
    or_stmt = base_stmt.where(or_(*[_like_expr(t) for t in tokens]))
    result = await sql_session.execute(or_stmt)
    return result.scalars().all()
