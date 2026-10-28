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
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import logging
import re
import threading
from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _client
from . import _search_tool
from ...features import experimental
from ...features import FeatureName
from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry

if TYPE_CHECKING:
  from google.genai import Client
  from pymongo import MongoClient

  from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MEMORIES_COLLECTION = "memories"

# Retrieval modes for the `retrieval_mode` constructor argument.
KEYWORD_RETRIEVAL = "keyword"
VECTOR_RETRIEVAL = "vector"
AUTO_EMBEDDING_RETRIEVAL = "auto"
_RETRIEVAL_MODES = frozenset(
    {KEYWORD_RETRIEVAL, VECTOR_RETRIEVAL, AUTO_EMBEDDING_RETRIEVAL}
)

_DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-005"
_DEFAULT_EMBEDDING_DIMENSIONS = 768  # native size of text-embedding-005
_DEFAULT_AUTO_EMBEDDING_MODEL = "voyage-4"
_DEFAULT_MEMORY_VECTOR_INDEX_NAME = "memory_vector_index"
_DEFAULT_NUM_CANDIDATES = 100

# Default retention for stored memories, in days. Each memory document carries
# an `expire_at` date this many days past ingest, and the `memory_ttl` TTL
# index (created when `ensure_indexes=True`) lets MongoDB's TTL monitor delete
# documents once `expire_at` passes. Pass `memory_ttl_days=None` to keep
# memories indefinitely.
_DEFAULT_MEMORY_TTL_DAYS = 7

# Recommended secondary index, created only when `ensure_indexes=True`. It
# covers the `search_memory` lookup (app/user equality plus a multikey `$in`
# on keywords); writes are keyed by `_id`, which MongoDB indexes implicitly.
_MEMORIES_RECALL_INDEX_NAME = "app_user_keywords"
_MEMORIES_RECALL_INDEX_KEYS = [("app_name", 1), ("user_id", 1), ("keywords", 1)]

# TTL index on the `expire_at` date field: MongoDB's TTL monitor deletes a
# memory document once its `expire_at` passes. Created for every retrieval
# mode when `ensure_indexes=True` and a TTL is configured.
_MEMORIES_TTL_INDEX_NAME = "memory_ttl"

# Compact English stop-word list ignored when extracting keywords. Kept
# intentionally short; callers can pass their own set via `stop_words`.
DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


@experimental(FeatureName.MONGODB_MEMORY_SERVICE)
class MongoDbMemoryService(BaseMemoryService):
  """Memory service that uses MongoDB as the backend.

  Events ingested from sessions are stored as memory documents (one per
  event) in a single collection. `retrieval_mode` selects how
  `search_memory` recalls them:

    - `"keyword"` (default): each document carries the lowercase keywords
      extracted from its text; recall matches documents whose keyword array
      intersects the query's keywords, ranked by overlap (most recent first
      within a tie) and capped at `max_results`. Works on any MongoDB
      deployment — no embedding model, credentials, or search index needed.
    - `"vector"`: texts are embedded at ingest with the Google embedding
      model (`embedding_model_name`) through the genai client, and recall
      runs `$vectorSearch` with the embedded query. Requires a deployment
      with vector search (Atlas or self-managed 8.0+) and Google credentials.
    - `"auto"`: Atlas Automated Embedding (Preview). Documents carry plain
      text only; an `autoEmbed` vector index over the `text` field makes
      Atlas embed documents on write and queries on read with a Voyage AI
      model, so no Google credentials are needed. Freshly ingested memories
      become recallable once Atlas has embedded them (typically seconds).

  Example:
      ```python
      memory_service = MongoDbMemoryService(
          connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
          database_name="my_app",
      )
      runner = Runner(
          agent=agent, app_name="my_app", memory_service=memory_service
      )
      ```
  """

  def __init__(
      self,
      *,
      database_name: str,
      connection_string: str | None = None,
      mongo_client: MongoClient | None = None,
      memories_collection: str = DEFAULT_MEMORIES_COLLECTION,
      stop_words: set[str] | None = None,
      ensure_indexes: bool = True,
      max_results: int = 20,
      retrieval_mode: str = KEYWORD_RETRIEVAL,
      genai_client: Client | None = None,
      embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL_NAME,
      embedding_output_dimensionality: int | None = None,
      auto_embedding_model_name: str = _DEFAULT_AUTO_EMBEDDING_MODEL,
      vector_index_name: str = _DEFAULT_MEMORY_VECTOR_INDEX_NAME,
      memory_ttl_days: int | None = _DEFAULT_MEMORY_TTL_DAYS,
  ):
    """Initializes the MongoDB memory service.

    Args:
      database_name: The MongoDB database used to store memories.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this service. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`).
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      memories_collection: Collection name for memory documents.
      stop_words: Words to ignore when extracting keywords (keyword mode).
        Defaults to a standard English stop-word list.
      ensure_indexes: Create the recommended index for the retrieval mode on
        first use (default True): the `app_user_keywords` multikey index in
        keyword mode, or the `memory_vector_index` vector search index in
        vector/auto modes. Index creation is idempotent. Without the
        `createIndex`/`createSearchIndexes` privilege a warning is logged
        once and reads fall back to unindexed behavior; pass False to skip
        the attempt entirely. Vector search indexes build asynchronously —
        recall works once the index reports READY.
      max_results: Maximum number of memories a search returns (keyword mode:
        ranked by overlap and recency; vector modes: nearest neighbors).
      retrieval_mode: "keyword" (default), "vector", or "auto" — see the
        class docstring. The mode must match how the stored memories were
        ingested: memories ingested in one mode are invisible to the others.
      genai_client: The `google.genai` client used to embed memory texts and
        queries in vector mode. Defaults to a client built from the ambient
        environment. Unused in keyword and auto modes.
      embedding_model_name: Google embedding model for vector mode. Must be
        the model the stored embeddings were produced with.
      embedding_output_dimensionality: Optional shortened embedding length;
        must match the vector index's `numDimensions` when set.
      auto_embedding_model_name: Voyage AI model the auto mode's `autoEmbed`
        index embeds with.
      vector_index_name: Name of the vector search index used in vector/auto
        modes.
      memory_ttl_days: Number of days a memory is retained before MongoDB's
        TTL monitor deletes it, measured from ingest (default 7). Each memory
        document stores an `expire_at` date this many days in the future, and
        when `ensure_indexes=True` a `memory_ttl` TTL index is created on that
        field. Pass None to keep memories indefinitely (no `expire_at`, no TTL
        index). Deletion runs in the background, so expired memories may
        linger briefly past their `expire_at`.

    Raises:
      ValueError: If `max_results` is not positive, `retrieval_mode` is
        unknown, or `memory_ttl_days` is not positive.
    """
    if max_results < 1:
      raise ValueError("max_results must be a positive integer.")
    if memory_ttl_days is not None and memory_ttl_days < 1:
      raise ValueError(
          "memory_ttl_days must be a positive integer or None (no expiry)."
      )
    if retrieval_mode not in _RETRIEVAL_MODES:
      raise ValueError(
          f"retrieval_mode must be one of {sorted(_RETRIEVAL_MODES)}, got"
          f" {retrieval_mode!r}."
      )
    if mongo_client is not None and connection_string is not None:
      raise ValueError(
          "Only one of `connection_string` and `mongo_client` may be provided."
      )
    if mongo_client is not None:
      self._client = mongo_client
      self._owns_client = False
    elif connection_string is not None:
      self._client = _client.get_mongo_client(connection_string)
      self._owns_client = True
    else:
      raise ValueError(
          "Either `connection_string` or `mongo_client` must be provided."
      )
    self._database_name = database_name
    self.memories_collection = memories_collection
    self.stop_words = (
        stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    )
    self._ensure_indexes = ensure_indexes
    self._indexes_ensured = False
    self._max_results = max_results
    self._retrieval_mode = retrieval_mode
    self._genai_client = genai_client
    self._embedding_model_name = embedding_model_name
    self._embedding_output_dimensionality = embedding_output_dimensionality
    self._auto_embedding_model_name = auto_embedding_model_name
    self._vector_index_name = vector_index_name
    self._memory_ttl_days = memory_ttl_days
    # Serializes first-use index creation across the worker threads that
    # `asyncio.to_thread` runs `_ensure_indexes_once` on.
    self._index_lock = threading.Lock()

  def _memories(self):
    return self._client[self._database_name][self.memories_collection]

  def _ensure_indexes_once(self) -> None:
    """Creates the recommended secondary indexes on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. A deployment without the `createIndex`
    privilege (e.g. a least-privilege runtime user) logs a warning once and
    falls back to unindexed reads rather than failing every operation.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    with self._index_lock:
      # Re-check under the lock: a concurrent first call may have done it.
      if self._indexes_ensured:
        return
      try:
        if self._retrieval_mode == KEYWORD_RETRIEVAL:
          self._memories().create_index(
              _MEMORIES_RECALL_INDEX_KEYS, name=_MEMORIES_RECALL_INDEX_NAME
          )
        else:
          self._create_vector_index()
        self._create_ttl_index()
      except (
          Exception
      ) as exc:  # pymongo stays an optional import; match by shape.
        if not _client.is_index_privilege_error(exc):
          # Unexpected failure: leave the flag unset so a later operation
          # retries after the deployment recovers.
          raise
        logger.warning(
            "MongoDbMemoryService could not create the recommended index"
            " (createIndex privilege missing); reads will fall back to"
            " unindexed behavior. Pass ensure_indexes=False to silence this,"
            " or grant the privilege. Details: %s",
            exc,
        )
      self._indexes_ensured = True

  def _create_ttl_index(self) -> None:
    """Creates the TTL index that expires memories after `memory_ttl_days`.

    The index targets the `expire_at` date field written at ingest. With
    `expireAfterSeconds=0` MongoDB's TTL monitor deletes a document as soon
    as its `expire_at` passes, so the per-document date carries the retention
    (re-ingesting a session refreshes it). No-op when no TTL is configured.
    """
    if self._memory_ttl_days is None:
      return
    self._memories().create_index(
        [("expire_at", 1)],
        name=_MEMORIES_TTL_INDEX_NAME,
        expireAfterSeconds=0,
    )

  def _create_vector_index(self) -> None:
    """Creates the vector search index for vector/auto retrieval modes."""
    # pymongo is imported lazily here: it stays an optional dependency until
    # a MongoDB-backed service is actually constructed.
    from pymongo.operations import SearchIndexModel  # pylint: disable=import-outside-toplevel

    # createSearchIndexes rejects a collection that does not exist yet, which
    # is exactly the state of a fresh deployment on its first ingest.
    database = self._client[self._database_name]
    if self.memories_collection not in database.list_collection_names():
      try:
        database.create_collection(self.memories_collection)
      except Exception as exc:
        # CollectionInvalid (code 48): another process created it first.
        if getattr(exc, "code", None) != 48:
          raise

    fields: list[dict[str, Any]] = []
    if self._retrieval_mode == VECTOR_RETRIEVAL:
      fields.append({
          "type": "vector",
          "path": "embedding",
          "numDimensions": (
              self._embedding_output_dimensionality
              or _DEFAULT_EMBEDDING_DIMENSIONS
          ),
          "similarity": "cosine",
      })
    else:  # auto mode: Atlas embeds the plain `text` field server-side.
      fields.append({
          "type": "autoEmbed",
          "modality": "text",
          "path": "text",
          "model": self._auto_embedding_model_name,
      })
    fields += [
        {"type": "filter", "path": "app_name"},
        {"type": "filter", "path": "user_id"},
    ]
    self._memories().create_search_index(
        model=SearchIndexModel(
            definition={"fields": fields},
            name=self._vector_index_name,
            type="vectorSearch",
        )
    )

  def _extract_keywords(self, text: str) -> set[str]:
    """Extracts lowercase keywords from text, ignoring stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in self.stop_words}

  async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
    """Embeds memory texts in one batch call with the configured model."""
    # google.genai is imported lazily here: it stays an optional dependency
    # until vector mode is actually used.
    from google.genai import Client  # pylint: disable=import-outside-toplevel
    from google.genai import types as genai_types  # pylint: disable=import-outside-toplevel

    config = genai_types.EmbedContentConfig()
    if self._embedding_output_dimensionality:
      config.output_dimensionality = self._embedding_output_dimensionality
    try:
      response = await (
          self._genai_client or Client()
      ).aio.models.embed_content(
          model=self._embedding_model_name,
          contents=list(texts),
          config=config,
      )
    except Exception as ex:
      raise RuntimeError(f"Failed to embed memory texts: {ex!r}") from ex
    if not response.embeddings:
      raise ValueError(
          f"Model {self._embedding_model_name} returned no embeddings."
      )
    return [[float(v) for v in e.values] for e in response.embeddings]

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Ingests the session's text events into the memory collection.

    Ingestion is idempotent: memory documents are keyed by
    `<app_name>/<user_id>/<session_id>/<event_id>`, so re-adding a session
    overwrites its memories rather than duplicating them. In vector mode the
    texts are embedded with one batch call before the write; in auto mode
    plain text is stored and Atlas embeds it when the autoEmbed index
    processes the document.
    """
    await asyncio.to_thread(self._ensure_indexes_once)

    pairs: list[tuple[Any, str]] = []
    for event in session.events:
      if not event.content or not event.content.parts:
        continue
      text = " ".join([part.text for part in event.content.parts if part.text])
      if not text:
        continue
      pairs.append((event, text))
    if not pairs:
      return

    embeddings: list[list[float]] | None = None
    if self._retrieval_mode == VECTOR_RETRIEVAL:
      embeddings = await self._embed_texts([text for _, text in pairs])

    # One expiry for the whole batch, so every memory from this session ages
    # out together. Re-ingesting a session refreshes expire_at (upsert).
    expire_at = None
    if self._memory_ttl_days is not None:
      expire_at = datetime.now(timezone.utc) + timedelta(
          days=self._memory_ttl_days
      )

    def _add() -> None:
      # pymongo is imported lazily here: it stays an optional dependency
      # until a MongoDB-backed service is actually constructed.
      from pymongo import ReplaceOne  # pylint: disable=import-outside-toplevel

      # One bulk_write round-trip, not one replace_one per event.
      operations = []
      for i, (event, text) in enumerate(pairs):
        document = {
            "app_name": session.app_name,
            "user_id": session.user_id,
            "session_id": session.id,
            "author": event.author,
            "text": text,
            "content": event.content.model_dump(exclude_none=True, mode="json"),
            "timestamp": event.timestamp,
        }
        if expire_at is not None:
          document["expire_at"] = expire_at
        if self._retrieval_mode == KEYWORD_RETRIEVAL:
          keywords = self._extract_keywords(text)
          if not keywords:
            continue
          document["keywords"] = sorted(keywords)
        elif embeddings is not None:
          document["embedding"] = embeddings[i]
        memory_id = (
            f"{session.app_name}/{session.user_id}/{session.id}/{event.id}"
        )
        operations.append(ReplaceOne({"_id": memory_id}, document, upsert=True))
      if operations:
        self._memories().bulk_write(operations, ordered=False)

    await asyncio.to_thread(_add)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches memory for events matching the query.

    Keyword mode matches the query's keywords; vector mode embeds the query
    and runs `$vectorSearch`; auto mode sends the query text to the autoEmbed
    index. All modes cap results at `max_results`.
    """
    await asyncio.to_thread(self._ensure_indexes_once)

    if self._retrieval_mode == KEYWORD_RETRIEVAL:
      keywords = self._extract_keywords(query)
      if not keywords:
        return SearchMemoryResponse()
      docs = await asyncio.to_thread(
          self._keyword_search, app_name, user_id, keywords
      )
    else:
      docs = await self._vector_search(app_name, user_id, query)

    seen = set()
    memories = []
    for doc in docs:
      try:
        from google.genai import types

        content = types.Content.model_validate(doc["content"])
        entry = MemoryEntry(
            id=doc.get("_id"),
            content=content,
            author=doc.get("author"),
            timestamp=_utils.format_timestamp(doc.get("timestamp", 0.0)),
        )
      except Exception as exc:
        logger.warning(f"Failed to parse memory entry: {exc}")
        continue
      content_text = (
          " ".join([part.text for part in content.parts if part.text])
          if content.parts
          else ""
      )
      key = (entry.author, content_text, entry.timestamp)
      if key not in seen:
        seen.add(key)
        memories.append(entry)

    return SearchMemoryResponse(memories=memories)

  def _keyword_search(
      self, app_name: str, user_id: str, keywords: set[str]
  ) -> list[dict[str, Any]]:
    """Keyword recall: overlap-ranked, capped — all server-side."""
    # Ranking and the limit run server-side: a common keyword can match a
    # large share of the memories, and without `$limit` every one of them
    # crosses the wire.
    pipeline = [
        {
            "$match": {
                "app_name": app_name,
                "user_id": user_id,
                "keywords": {"$in": sorted(keywords)},
            }
        },
        {
            "$addFields": {
                "overlap": {
                    "$size": {
                        "$filter": {
                            "input": "$keywords",
                            "as": "keyword",
                            "cond": {"$in": ["$$keyword", sorted(keywords)]},
                        }
                    }
                }
            }
        },
        {"$sort": {"overlap": -1, "timestamp": -1}},
        {"$limit": self._max_results},
    ]
    return list(self._memories().aggregate(pipeline))

  async def _vector_search(
      self, app_name: str, user_id: str, query: str
  ) -> list[dict[str, Any]]:
    """Vector recall: `$vectorSearch` with a pre-filter on app and user."""
    stage: dict[str, Any] = {
        "index": self._vector_index_name,
        "numCandidates": max(self._max_results * 10, _DEFAULT_NUM_CANDIDATES),
        "limit": self._max_results,
        "filter": {"app_name": app_name, "user_id": user_id},
    }
    if self._retrieval_mode == AUTO_EMBEDDING_RETRIEVAL:
      # Atlas embeds the query text with the autoEmbed index's model.
      stage["path"] = "text"
      stage["query"] = {"text": query}
    else:
      stage["path"] = "embedding"
      stage["queryVector"] = await _search_tool._embed_query(
          query=query,
          model_name=self._embedding_model_name,
          output_dimensionality=self._embedding_output_dimensionality,
          genai_client=self._genai_client,
      )

    def _run() -> list[dict[str, Any]]:
      return list(
          self._memories().aggregate([
              {"$vectorSearch": stage},
              {"$project": {"embedding": 0}},
          ])
      )

    return await asyncio.to_thread(_run)

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
