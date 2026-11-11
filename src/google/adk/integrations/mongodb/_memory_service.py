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

"""MongoDB-backed memory service for long-term event recall."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import itertools
import logging
import re
import threading
from typing import Any
from typing import get_args
from typing import Literal
from typing import TYPE_CHECKING
import unicodedata

from google.genai import types
from pydantic import ValidationError
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
  from pymongo import ReplaceOne
  from pymongo.collection import Collection

  from ...events.event import Event
  from ...sessions.session import Session
else:
  try:
    from pymongo import ReplaceOne
  except ImportError:

    class ReplaceOne:
      """Fallback when pymongo is not installed."""

      def __init__(
          self,
          filter: Mapping[str, Any],  # pylint: disable=redefined-builtin
          replacement: dict[str, Any],
          upsert: bool = False,
      ) -> None:
        self._filter = filter
        self._doc = replacement
        self._upsert = upsert


logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MEMORIES_COLLECTION = "memories"
_DEFAULT_MAX_RESULTS = 50
_UNKNOWN_SESSION_ID = "__unknown_session_id__"

_RetrievalMode = Literal["keyword", "vector", "auto"]
_KEYWORD_RETRIEVAL = "keyword"
_VECTOR_RETRIEVAL = "vector"
_AUTO_EMBEDDING_RETRIEVAL = "auto"
_RETRIEVAL_MODES = frozenset(get_args(_RetrievalMode))

_DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-005"
_DEFAULT_EMBEDDING_DIMENSIONS = 768  # native size of text-embedding-005
_DEFAULT_AUTO_EMBEDDING_MODEL = "voyage-4"
_DEFAULT_MEMORY_VECTOR_INDEX_NAME = "memory_vector_index"
_DEFAULT_NUM_CANDIDATES = 100
# Atlas rejects a `$vectorSearch` whose `numCandidates` exceeds this.
_MAX_NUM_CANDIDATES = 10_000
# Texts per embed_content request; the embedding APIs cap inputs per request.
_EMBED_BATCH_SIZE = 100
# Server error code for creating a search index whose name is already taken.
_INDEX_ALREADY_EXISTS_CODE = 68

# Recommended secondary index, created on first use unless
# `ensure_indexes=False`. It covers the keyword-mode `search_memory` lookup
# (app/user equality plus a multikey `$in` on keywords); writes are keyed by
# `_id`, which MongoDB indexes implicitly.
_MEMORIES_RECALL_INDEX_NAME = "app_user_keywords"
_MEMORIES_RECALL_INDEX_KEYS = [("app_name", 1), ("user_id", 1), ("keywords", 1)]

# TTL index on the `expire_at` date field: MongoDB's TTL monitor deletes a
# memory document once its `expire_at` passes. Created for every retrieval
# mode when `ensure_indexes=True` and a TTL is configured.
_MEMORIES_TTL_INDEX_NAME = "memory_ttl"

# Compact English stop-word list ignored when extracting keywords. Kept
# intentionally short; callers can pass their own set via `stop_words`.
DEFAULT_STOP_WORDS = frozenset({
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
})


def _is_latin(c: str) -> bool:
  if c.isascii():
    return c.isalnum() or c == "_"
  return unicodedata.name(c, "").startswith("LATIN")


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
      with vector search (Atlas or self-managed 8.2+) and Google credentials.
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
          agent=agent,
          app_name="my_app",
          session_service=InMemorySessionService(),
          memory_service=memory_service,
      )
      ```
  """

  def __init__(
      self,
      *,
      database_name: str,
      connection_string: str | None = None,
      mongo_client: MongoClient[dict[str, Any]] | None = None,
      memories_collection: str = DEFAULT_MEMORIES_COLLECTION,
      stop_words: set[str] | None = None,
      max_results: int = _DEFAULT_MAX_RESULTS,
      ensure_indexes: bool = True,
      retrieval_mode: _RetrievalMode = "keyword",
      genai_client: Client | None = None,
      embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL_NAME,
      embedding_output_dimensionality: int | None = None,
      auto_embedding_model_name: str = _DEFAULT_AUTO_EMBEDDING_MODEL,
      vector_index_name: str = _DEFAULT_MEMORY_VECTOR_INDEX_NAME,
      memory_ttl_days: int | None = None,
  ) -> None:
    """Initializes the MongoDB memory service.

    Args:
      database_name: The MongoDB database used to store memories.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this service. Requires the `pymongo` package (`pip
        install google-adk[mongodb]`).
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      memories_collection: Collection name for memory documents.
      stop_words: Words to ignore when extracting keywords (keyword mode).
        Defaults to a standard English stop-word list.
      max_results: Maximum number of memories returned per query (keyword
        mode: ranked by overlap and recency; vector modes: nearest neighbors).
      ensure_indexes: Create the recommended indexes for the retrieval mode on
        first use (default True): the `app_user_keywords` multikey index in
        keyword mode or the `memory_vector_index` vector search index in
        vector/auto modes, plus the `memory_ttl` TTL index when
        `memory_ttl_days` is set. Index creation is idempotent. Without the
        `createIndex`/`createSearchIndexes` privilege a warning is logged once
        and reads fall back to unindexed behavior; pass False to skip the
        attempt entirely. Vector search indexes build asynchronously — recall
        works once the index reports READY.
      retrieval_mode: "keyword" (default), "vector", or "auto" — see the class
        docstring. The mode must match how the stored memories were
        ingested: memories ingested in one mode are invisible to the others.
      genai_client: The `google.genai` client used to embed memory texts and
        queries in vector mode. Defaults to a client built from the ambient
        environment. Unused in keyword and auto modes.
      embedding_model_name: Google embedding model for vector mode. Must be the
        model the stored embeddings were produced with.
      embedding_output_dimensionality: Optional shortened embedding length. When
        unset, 768 (the native size of text-embedding-005) is used both for the
        index's `numDimensions` and to validate ingested vectors, so set this
        whenever `embedding_model_name` produces a different size.
      auto_embedding_model_name: Voyage AI model the auto mode's `autoEmbed`
        index embeds with.
      vector_index_name: Name of the vector search index used in vector/auto
        modes.
      memory_ttl_days: Number of days a memory is retained before MongoDB's TTL
        monitor deletes it, measured from ingest. Each memory document stores an
        `expire_at` date this many days in the future, and when
        `ensure_indexes=True` a `memory_ttl` TTL index is created on that field.
        Defaults to None, which keeps memories indefinitely (no `expire_at`, no
        TTL index). Deletion runs in the background, so expired memories may
        linger briefly past their `expire_at`.

    Raises:
      ValueError: If `max_results` or `memory_ttl_days` is not positive, or
        `retrieval_mode` is unknown.
    """
    if max_results <= 0:
      raise ValueError("max_results must be positive.")
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
      self._client: MongoClient[dict[str, Any]] = mongo_client
      self._owns_client = False
    elif connection_string is not None:
      self._client = _client.get_mongo_client(connection_string)
      self._owns_client = True
    else:
      raise ValueError(
          "Either `connection_string` or `mongo_client` must be provided."
      )
    self._database_name = database_name
    self._memories_collection = memories_collection
    self._stop_words = (
        {unicodedata.normalize("NFC", word).lower() for word in stop_words}
        if stop_words is not None
        else set(DEFAULT_STOP_WORDS)
    )
    self._max_results = max_results
    self._ensure_indexes = ensure_indexes
    self._indexes_ensured = False
    self._retrieval_mode = retrieval_mode
    self._genai_client = genai_client
    self._embedding_model_name = embedding_model_name
    self._embedding_output_dimensionality = embedding_output_dimensionality
    self._auto_embedding_model_name = auto_embedding_model_name
    self._vector_index_name = vector_index_name
    self._memory_ttl_days = memory_ttl_days
    # Serializes first-use index creation across the worker threads that
    # `asyncio.to_thread` runs `_ensure_indexes_once` on (vector index
    # creation is not concurrency-safe on a cold collection).
    self._index_lock = threading.Lock()

  def _memories(self) -> Collection[dict[str, Any]]:
    return self._client[self._database_name][self._memories_collection]

  @staticmethod
  def _memory_key(
      app_name: str, user_id: str, session_id: str, event_id: str
  ) -> dict[str, str]:
    return {
        "app_name": app_name,
        "user_id": user_id,
        "session_id": session_id,
        "event_id": event_id,
    }

  def _ensure_indexes_once(self) -> None:
    """Creates the recommended secondary indexes on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. A missing `createIndex` privilege logs one
    warning and falls back to unindexed reads; any other failure propagates
    and is retried on the next operation.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    with self._index_lock:
      # Re-check under the lock: a concurrent first call may have done it.
      if self._indexes_ensured:
        return
      if self._retrieval_mode == _KEYWORD_RETRIEVAL:
        self._create_index(
            lambda: self._memories().create_index(
                _MEMORIES_RECALL_INDEX_KEYS, name=_MEMORIES_RECALL_INDEX_NAME
            )
        )
      else:
        self._create_index(self._create_vector_index)
      self._create_index(self._create_ttl_index)
      self._indexes_ensured = True

  def _create_index(self, create: Callable[[], object]) -> None:
    """Runs one index-creation call, tolerating per-index conflicts.

    Code 85 (IndexOptionsConflict) means an equivalent index already exists
    under another name/options, so reads are still indexed; a missing
    `createIndex` privilege warns once and falls back to unindexed reads.
    Either way the remaining indexes are still attempted. Any other failure
    propagates so the next operation retries (the guard flag stays unset).
    """
    # pymongo is an optional import, so the error is matched by shape.
    try:
      create()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if getattr(exc, "code", None) == 85:
        return
      if not _client.is_index_privilege_error(exc):
        raise
      logger.warning(
          "MongoDbMemoryService could not create a recommended index"
          " (createIndex or createSearchIndexes privilege missing). Without it"
          " keyword reads fall back to unindexed scans, vector search returns"
          " nothing, and memories do not expire. Create the index as an admin"
          " and pass ensure_indexes=False, or grant the privilege. Details: %s",
          exc,
      )

  def _create_ttl_index(self) -> None:
    """Creates the TTL index that expires memories after `memory_ttl_days`.

    The index targets the `expire_at` date field written at ingest. With
    `expireAfterSeconds=0` MongoDB's TTL monitor deletes a document as soon
    as its `expire_at` passes, so the per-document date carries the retention
    (re-ingesting events refreshes it). No-op when no TTL is configured.
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
    # createSearchIndexes rejects a collection that does not exist yet, which
    # is exactly the state of a fresh deployment on its first ingest.
    database = self._client[self._database_name]
    if not database.list_collection_names(
        filter={"name": self._memories_collection}
    ):
      try:
        # check_exists=False leaves the existence check to the server, which
        # reports a concurrent create as NamespaceExists (code 48).
        database.create_collection(
            self._memories_collection, check_exists=False
        )
      except Exception as exc:
        if getattr(exc, "code", None) != 48:
          raise

    if self._retrieval_mode == _VECTOR_RETRIEVAL:
      search_field: dict[str, Any] = {
          "type": "vector",
          "path": "embedding",
          "numDimensions": (
              self._embedding_output_dimensionality
              or _DEFAULT_EMBEDDING_DIMENSIONS
          ),
          "similarity": "cosine",
      }
    else:  # auto mode: Atlas embeds the plain `text` field server-side.
      search_field = {
          "type": "autoEmbed",
          "modality": "text",
          "path": "text",
          "model": self._auto_embedding_model_name,
      }
    # Unlike create_index, create_search_index is not idempotent: it fails
    # when the index exists, which is every start after the first.
    existing = next(
        iter(self._memories().list_search_indexes(self._vector_index_name)),
        None,
    )
    if existing is not None:
      self._check_vector_index(existing, search_field)
      return
    try:
      self._memories().create_search_index(
          model={
              "definition": {
                  "fields": [
                      search_field,
                      {"type": "filter", "path": "app_name"},
                      {"type": "filter", "path": "user_id"},
                  ]
              },
              "name": self._vector_index_name,
              "type": "vectorSearch",
          }
      )
    except Exception as exc:
      # Another process created it between the check and the create.
      if getattr(exc, "code", None) != _INDEX_ALREADY_EXISTS_CODE:
        raise

  def _check_vector_index(
      self, existing: Mapping[str, Any], search_field: dict[str, Any]
  ) -> None:
    """Raises if an existing vector index does not match this service.

    Atlas silently skips vectors not sized for the index, so reusing a
    mismatched index would lose every memory without an error.
    """
    definition = (
        existing.get("latestDefinition") or existing.get("definition") or {}
    )
    actual = next(
        (
            field
            for field in definition.get("fields", [])
            if field.get("path") == search_field["path"]
        ),
        {},
    )
    mismatched = [
        f"{key}={actual.get(key)!r} (expected {search_field[key]!r})"
        for key in ("type", "numDimensions", "model")
        if key in search_field and actual.get(key) != search_field[key]
    ]
    if mismatched:
      raise ValueError(
          f"Vector index {self._vector_index_name!r} does not match this"
          f" service's configuration: {', '.join(mismatched)} on"
          f" {search_field['path']!r}. Drop and recreate the index, or pass a"
          " different vector_index_name."
      )

  def _extract_keywords(self, text: str) -> set[str]:
    """Extracts lowercase keywords from text, ignoring stop words.

    A token mixing Latin and non-Latin scripts also yields each of its
    single-script runs, so "MongoDB" matches "使用MongoDB存储", the same way
    the in-memory and SQLite memory services tokenize.
    """
    normalized = unicodedata.normalize("NFC", text)
    words: set[str] = set()
    for word in re.findall(r"\w+", normalized):
      words.add(word.lower())
      if not word.isascii():
        # Split before lowercasing: "İ".lower() is "i" plus U+0307, which is
        # not Latin, so splitting afterwards stores a bare combining mark.
        words.update(
            "".join(group).lower()
            for _, group in itertools.groupby(word, _is_latin)
        )
    return words - self._stop_words

  async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
    """Embeds memory texts in batches with the configured model."""
    # google.genai is imported lazily here: it stays an optional dependency
    # until vector mode is actually used.
    from google.genai import Client  # pylint: disable=import-outside-toplevel
    from google.genai import types as genai_types  # pylint: disable=import-outside-toplevel

    config = genai_types.EmbedContentConfig()
    if self._embedding_output_dimensionality:
      config.output_dimensionality = self._embedding_output_dimensionality
    client = self._genai_client or Client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
      batch = texts[start : start + _EMBED_BATCH_SIZE]
      try:
        response = await client.aio.models.embed_content(
            model=self._embedding_model_name,
            contents=list(batch),
            config=config,
        )
      except Exception as ex:
        raise RuntimeError(f"Failed to embed memory texts: {ex!r}") from ex
      if not response.embeddings or len(response.embeddings) != len(batch):
        raise ValueError(
            f"Model {self._embedding_model_name} returned"
            f" {len(response.embeddings or [])} embeddings for"
            f" {len(batch)} texts."
        )
      vectors.extend(
          [float(v) for v in e.values or []] for e in response.embeddings
      )
    return vectors

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Ingests the session's text events into the memory collection."""
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
    """Ingests text events into the memory collection.

    Ingestion is idempotent: memory documents are keyed by a compound `_id`
    of `(app_name, user_id, session_id, event_id)`, so re-adding events
    overwrites their memories rather than duplicating them. In vector mode
    the texts are embedded with one batch call before the write; in auto mode
    plain text is stored and Atlas embeds it when the autoEmbed index
    processes the document.
    """
    del custom_metadata  # Not supported by MongoDB memory service.
    scoped_session_id = session_id or _UNKNOWN_SESSION_ID

    entries: list[tuple[Event, types.Content, str, list[Any]]] = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      visible_parts = [part for part in event.content.parts if not part.thought]
      text = " ".join([part.text for part in visible_parts if part.text])
      if not text:
        continue
      entries.append((event, event.content, text, visible_parts))
    if not entries:
      return

    embeddings: list[list[float]] | None = None
    if self._retrieval_mode == _VECTOR_RETRIEVAL:
      embeddings = await self._embed_texts([text for _, _, text, _ in entries])
      expected = (
          self._embedding_output_dimensionality or _DEFAULT_EMBEDDING_DIMENSIONS
      )
      if any(len(vector) != expected for vector in embeddings):
        # Atlas silently skips vectors whose length differs from the index's
        # numDimensions, so a mismatch would lose memories without an error.
        raise ValueError(
            f"Model {self._embedding_model_name} returned embeddings that are"
            f" not {expected}-dimensional. Set embedding_output_dimensionality"
            " to the model's output size and the vector index's numDimensions."
        )

    # One expiry for the whole batch, so every memory ingested together ages
    # out together. Re-ingesting events refreshes expire_at (upsert).
    expire_at = None
    if self._memory_ttl_days is not None:
      expire_at = datetime.now(timezone.utc) + timedelta(
          days=self._memory_ttl_days
      )

    operations: list[ReplaceOne[dict[str, Any]]] = []
    for i, (event, content, text, visible_parts) in enumerate(entries):
      document: dict[str, Any] = {
          "app_name": app_name,
          "user_id": user_id,
          "session_id": scoped_session_id,
          "author": event.author,
          "text": text,
          "content": (
              content.model_copy(update={"parts": visible_parts}).model_dump(
                  exclude_none=True, mode="json"
              )
          ),
          "timestamp": event.timestamp,
      }
      if expire_at is not None:
        document["expire_at"] = expire_at
      if self._retrieval_mode == _KEYWORD_RETRIEVAL:
        keywords = self._extract_keywords(text)
        if not keywords:
          continue
        document["keywords"] = sorted(keywords)
      elif embeddings is not None:
        document["embedding"] = embeddings[i]
      operations.append(
          ReplaceOne(
              {
                  "_id": self._memory_key(
                      app_name, user_id, scoped_session_id, event.id
                  )
              },
              document,
              upsert=True,
          )
      )

    def _write() -> None:
      self._ensure_indexes_once()
      if operations:
        self._memories().bulk_write(operations, ordered=False)

    await asyncio.to_thread(_write)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches memory for events matching the query.

    Keyword mode matches the query's keywords; vector mode embeds the query
    and runs `$vectorSearch`; auto mode sends the query text to the autoEmbed
    index. All modes cap results at `max_results`.
    """
    if not query.strip():
      return SearchMemoryResponse()
    query_vector: list[float] | None = None
    if self._retrieval_mode == _VECTOR_RETRIEVAL:
      # Embed in async context; the genai client's async transport needs a
      # running loop, which a worker thread would not have.
      query_vector = await _search_tool._embed_query(
          query=query,
          model_name=self._embedding_model_name,
          output_dimensionality=self._embedding_output_dimensionality,
          genai_client=self._genai_client,
      )

    # Keyword mode: a query with no usable keywords matches nothing, so skip
    # the aggregate — and index creation — entirely.
    query_keywords: list[str] | None = None
    if self._retrieval_mode == _KEYWORD_RETRIEVAL:
      keywords = self._extract_keywords(query)
      if not keywords:
        return SearchMemoryResponse()
      query_keywords = sorted(keywords)

    def _search() -> list[dict[str, Any]]:
      self._ensure_indexes_once()
      if self._retrieval_mode == _KEYWORD_RETRIEVAL:
        return self._keyword_search(app_name, user_id, query_keywords or [])
      return self._vector_search_docs(app_name, user_id, query, query_vector)

    docs = await asyncio.to_thread(_search)

    seen: set[tuple[str | None, str, str | None]] = set()
    memories: list[MemoryEntry] = []
    for doc in docs:
      try:
        content = types.Content.model_validate(doc["content"])
      except (KeyError, ValidationError) as exc:
        logger.warning("Failed to parse memory entry: %s", exc)
        continue
      raw_id = doc.get("_id")
      event_id = raw_id.get("event_id") if isinstance(raw_id, dict) else raw_id
      entry = MemoryEntry(
          id=event_id if isinstance(event_id, str) else None,
          content=content,
          author=doc.get("author"),
          timestamp=_utils.format_timestamp(doc.get("timestamp", 0.0)),
      )
      content_text = (
          " ".join([
              part.text
              for part in content.parts
              if part.text and not part.thought
          ])
          if content.parts
          else ""
      )
      key = (entry.author, content_text, entry.timestamp)
      if key in seen:
        continue
      seen.add(key)
      memories.append(entry)

    return SearchMemoryResponse(memories=memories)

  def _keyword_search(
      self, app_name: str, user_id: str, query_keywords: list[str]
  ) -> list[dict[str, Any]]:
    """Keyword recall: overlap-ranked, capped — all server-side."""
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "app_name": app_name,
                "user_id": user_id,
                "keywords": {"$in": query_keywords},
            }
        },
        {
            "$addFields": {
                "_score": {
                    "$size": {"$setIntersection": ["$keywords", query_keywords]}
                }
            }
        },
        {"$sort": {"_score": -1, "timestamp": -1}},
        {"$limit": self._max_results},
    ]
    return list(self._memories().aggregate(pipeline))

  def _vector_search_docs(
      self,
      app_name: str,
      user_id: str,
      query: str,
      query_vector: list[float] | None,
  ) -> list[dict[str, Any]]:
    """Vector recall: `$vectorSearch` with a pre-filter on app and user."""
    stage: dict[str, Any] = {
        "index": self._vector_index_name,
        "numCandidates": min(
            max(self._max_results * 10, _DEFAULT_NUM_CANDIDATES),
            _MAX_NUM_CANDIDATES,
        ),
        "limit": min(self._max_results, _MAX_NUM_CANDIDATES),
        "filter": {"app_name": app_name, "user_id": user_id},
    }
    if self._retrieval_mode == _AUTO_EMBEDDING_RETRIEVAL:
      # Atlas embeds the query text with the autoEmbed index's model.
      stage["path"] = "text"
      stage["query"] = {"text": query}
    else:
      stage["path"] = "embedding"
      stage["queryVector"] = query_vector

    return list(
        self._memories().aggregate([
            {"$vectorSearch": stage},
            {"$project": {"embedding": 0}},
        ])
    )

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
