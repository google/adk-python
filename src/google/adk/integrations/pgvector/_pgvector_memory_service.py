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

"""PostgreSQL/pgvector-backed memory service implementation for ADK."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import json
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry
from ._config import PgVectorMemoryServiceConfig

if TYPE_CHECKING:
  from ...events.event import Event
  from ...sessions.session import Session

try:
  from psycopg_pool import AsyncConnectionPool
except ImportError:
  AsyncConnectionPool = None

try:
  from pgvector.psycopg import register_vector_async
except ImportError:
  register_vector_async = None

logger = logging.getLogger("google_adk." + __name__)

# An async callable that turns a batch of texts into their embedding vectors.
Embedder = Callable[[Sequence[str]], Awaitable[Sequence[Sequence[float]]]]


def _content_text(content: Optional[types.Content]) -> str:
  """Joins the text parts of a content into a single string."""
  if not content or not content.parts:
    return ""
  return " ".join(part.text for part in content.parts if part.text)


class PgVectorMemoryService(BaseMemoryService):
  """A memory service backed by PostgreSQL with the pgvector extension.

  Events are embedded with a ``google-genai`` embedding model and stored, with
  their vectors, in a PostgreSQL table you own. ``search_memory`` returns the
  memories whose embeddings are closest to the query embedding by cosine
  distance, using a pgvector HNSW index. Unlike ``VertexAiMemoryBankService``
  and ``VertexAiRagMemoryService`` the store runs entirely on your own database,
  so it suits on-premise, air-gapped, and data-residency-constrained
  deployments; unlike ``InMemoryMemoryService`` it persists across restarts and
  ranks by semantic similarity rather than keyword overlap.

  Memories are scoped by ``(app_name, user_id)``. Ingesting the same event more
  than once updates the existing row in place rather than creating duplicates.

  Example::

      from google.adk.integrations.pgvector import PgVectorMemoryService
      from google.adk.integrations.pgvector import PgVectorMemoryServiceConfig

      memory_service = PgVectorMemoryService(
          PgVectorMemoryServiceConfig(
              dsn="postgresql://user:password@localhost:5432/adk",
          )
      )

  The connection pool and the embedding function can both be injected, which
  makes the service usable with a pre-tuned pool or a non-Google embedding
  provider, and testable without a live database.
  """

  def __init__(
      self,
      config: Optional[PgVectorMemoryServiceConfig] = None,
      *,
      connection_pool: Optional[Any] = None,
      genai_client: Optional[Any] = None,
      embedder: Optional[Embedder] = None,
  ):
    """Initializes the PgVectorMemoryService.

    Args:
      config: Configuration for the service. Defaults to
        ``PgVectorMemoryServiceConfig()``; a ``dsn`` is required unless
        ``connection_pool`` is supplied.
      connection_pool: Optional pre-configured ``psycopg_pool``
        ``AsyncConnectionPool``. When omitted, one is created lazily from
        ``config.dsn``.
      genai_client: Optional ``google.genai.Client`` used for the default
        embedder. Ignored when ``embedder`` is supplied.
      embedder: Optional async callable that embeds a batch of texts. When
        omitted, texts are embedded with ``config.embedding_model`` through the
        ``google-genai`` client.
    """
    self.config = config or PgVectorMemoryServiceConfig()
    self._pool = connection_pool
    self._owns_pool = connection_pool is None
    self._pool_opened = False
    self._genai_client = genai_client
    self._embedder = embedder
    self._schema_ready = False
    self._schema_lock = asyncio.Lock()

  # --- Public BaseMemoryService API ----------------------------------------

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self._add_events(
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
    await self._add_events(
        app_name=app_name,
        user_id=user_id,
        events=events,
        session_id=session_id,
        custom_metadata=custom_metadata,
    )

  @override
  async def add_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    shared_metadata = dict(custom_metadata) if custom_metadata else {}
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for memory in memories:
      text = _content_text(memory.content)
      if not text:
        continue
      metadata = {**shared_metadata, **(memory.custom_metadata or {})}
      key = memory.id or _content_text(memory.content)
      rows.append({
          "id": self._entry_id(app_name, user_id, None, key),
          "app_name": app_name,
          "user_id": user_id,
          "session_id": None,
          "author": memory.author,
          "timestamp": memory.timestamp,
          "text": text,
          "content": memory.content,
          "custom_metadata": metadata,
      })
      texts.append(text)
    await self._embed_and_upsert(rows, texts)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    if not query or not query.strip():
      return SearchMemoryResponse()

    embedding = (await self._embed([query]))[0]
    pool = await self._ensure_pool()
    async with pool.connection() as conn:
      await self._prepare_connection(conn)
      cursor = await conn.execute(
          "SELECT author, timestamp, content, custom_metadata, id,"
          f" embedding <=> %s::vector AS distance FROM {self._table}"
          " WHERE app_name = %s AND user_id = %s"
          " ORDER BY distance LIMIT %s",
          (self._to_vector(embedding), app_name, user_id, self.config.top_k),
      )
      rows = await cursor.fetchall()

    threshold = self.config.distance_threshold
    memories: list[MemoryEntry] = []
    for author, timestamp, content, custom_metadata, entry_id, distance in rows:
      if (
          threshold is not None
          and distance is not None
          and distance > threshold
      ):
        continue
      memories.append(
          MemoryEntry(
              content=types.Content.model_validate(_as_dict(content)),
              author=author,
              timestamp=timestamp,
              custom_metadata=_as_dict(custom_metadata),
              id=entry_id,
          )
      )
    return SearchMemoryResponse(memories=memories)

  # --- Internals ------------------------------------------------------------

  @property
  def _table(self) -> str:
    return self.config.table_name

  async def _add_events(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    shared_metadata = dict(custom_metadata) if custom_metadata else {}
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for event in events:
      text = _content_text(event.content)
      if not text:
        continue
      key = event.id or f"{session_id}:{event.timestamp}:{text}"
      rows.append({
          "id": self._entry_id(app_name, user_id, session_id, key),
          "app_name": app_name,
          "user_id": user_id,
          "session_id": session_id,
          "author": event.author,
          "timestamp": _utils.format_timestamp(event.timestamp),
          "text": text,
          "content": event.content,
          "custom_metadata": shared_metadata,
      })
      texts.append(text)
    await self._embed_and_upsert(rows, texts)

  async def _embed_and_upsert(
      self, rows: list[dict[str, Any]], texts: list[str]
  ) -> None:
    if not rows:
      return
    embeddings = await self._embed(texts)
    pool = await self._ensure_pool()
    async with pool.connection() as conn:
      await self._prepare_connection(conn)
      for row, embedding in zip(rows, embeddings):
        await conn.execute(
            f"INSERT INTO {self._table} (id, app_name, user_id, session_id,"
            " author, timestamp, text, content, custom_metadata, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,"
            " %s::vector)"
            " ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text,"
            " content = EXCLUDED.content,"
            " custom_metadata = EXCLUDED.custom_metadata,"
            " embedding = EXCLUDED.embedding",
            (
                row["id"],
                row["app_name"],
                row["user_id"],
                row["session_id"],
                row["author"],
                row["timestamp"],
                row["text"],
                _content_json(row["content"]),
                json.dumps(row["custom_metadata"]),
                self._to_vector(embedding),
            ),
        )

  def _entry_id(
      self,
      app_name: str,
      user_id: str,
      session_id: str | None,
      key: str,
  ) -> str:
    """Builds a stable, collision-resistant id for a memory row.

    The id is derived from the memory's scope and a per-event key so that
    re-ingesting the same event updates its row in place instead of inserting a
    duplicate.
    """
    digest = hashlib.sha256(
        "\x00".join([app_name, user_id, session_id or "", key]).encode("utf-8")
    ).hexdigest()
    return digest

  async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
    embedder = self._embedder or self._default_embed
    vectors = await embedder(list(texts))
    return [[float(v) for v in vector] for vector in vectors]

  async def _default_embed(self, texts: Sequence[str]) -> list[list[float]]:
    """Embeds texts with the configured google-genai embedding model."""
    from google.genai import Client  # pylint: disable=import-outside-toplevel

    client = self._genai_client or Client()
    config = types.EmbedContentConfig()
    if self.config.embedding_dimension:
      config.output_dimensionality = self.config.embedding_dimension
    try:
      response = await client.aio.models.embed_content(
          model=self.config.embedding_model,
          contents=list(texts),
          config=config,
      )
    except Exception as ex:
      raise RuntimeError(f"Failed to embed content: {ex!r}") from ex
    return [list(embedding.values) for embedding in response.embeddings]

  def _to_vector(self, embedding: Sequence[float]) -> Any:
    """Returns the embedding in the form the driver stores as a pgvector value.

    pgvector adapts a plain list of floats, so the list is passed through as-is.
    Keeping this in one place lets a different adapter (for example a numpy
    array) be swapped in without touching the queries.
    """
    return list(embedding)

  def _get_pool(self) -> Any:
    """Lazily creates and returns the connection pool."""
    if self._pool is not None:
      return self._pool
    if AsyncConnectionPool is None:
      raise ImportError(
          "PgVectorMemoryService requires the psycopg connection pool. Install"
          " the optional dependencies with `pip install"
          ' "google-adk[pgvector]"`.'
      )
    if not self.config.dsn:
      raise ValueError(
          "PgVectorMemoryServiceConfig.dsn is required when a connection_pool"
          " is not provided."
      )
    self._pool = AsyncConnectionPool(self.config.dsn, open=False)
    return self._pool

  async def _ensure_pool(self) -> Any:
    """Returns the pool, opening it once if this service created it.

    An injected pool is assumed to be managed (opened and closed) by the
    caller, mirroring how the Redis integration accepts a pre-configured
    client.
    """
    pool = self._get_pool()
    if self._owns_pool and not self._pool_opened:
      await pool.open()
      self._pool_opened = True
    return pool

  async def close(self) -> None:
    """Closes the connection pool if this service created it."""
    if self._owns_pool and self._pool is not None and self._pool_opened:
      await self._pool.close()
      self._pool_opened = False

  async def _prepare_connection(self, conn: Any) -> None:
    """Registers the vector type and makes sure the schema exists."""
    if register_vector_async is not None:
      await register_vector_async(conn)
    if self._schema_ready:
      return
    async with self._schema_lock:
      if self._schema_ready:
        return
      await self._ensure_schema(conn)
      self._schema_ready = True

  async def _ensure_schema(self, conn: Any) -> None:
    """Creates the pgvector extension, table, and indexes if they are absent."""
    dimension = self.config.embedding_dimension
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(
        f"CREATE TABLE IF NOT EXISTS {self._table} ("
        " id text PRIMARY KEY,"
        " app_name text NOT NULL,"
        " user_id text NOT NULL,"
        " session_id text,"
        " author text,"
        " timestamp text,"
        " text text NOT NULL,"
        " content jsonb NOT NULL,"
        " custom_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,"
        f" embedding vector({dimension}) NOT NULL,"
        " created_at timestamptz NOT NULL DEFAULT now()"
        ")"
    )
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {self._table}_app_user_idx"
        f" ON {self._table} (app_name, user_id)"
    )
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx"
        f" ON {self._table} USING hnsw (embedding vector_cosine_ops)"
        f" WITH (m = {self.config.hnsw_m},"
        f" ef_construction = {self.config.hnsw_ef_construction})"
    )


def _content_json(content: types.Content) -> str:
  """Serializes a content to a JSON string for a jsonb column."""
  return json.dumps(content.model_dump(mode="json", exclude_none=True))


def _as_dict(value: Any) -> dict[str, Any]:
  """Normalizes a jsonb column value, which a driver may hand back as text."""
  if isinstance(value, str):
    return json.loads(value)
  return value or {}
