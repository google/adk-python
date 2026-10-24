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
from collections.abc import Mapping
from collections.abc import Sequence
import itertools
import logging
import re
from typing import Any
from typing import TYPE_CHECKING
import unicodedata

from google.genai import types
from pydantic import ValidationError
from typing_extensions import override

from . import _client
from ...features import experimental
from ...features import FeatureName
from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry

if TYPE_CHECKING:
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

# Recommended secondary index, created on first use unless
# `ensure_indexes=False`. It covers the `search_memory` lookup (app/user
# equality plus a multikey `$in` on keywords); writes are keyed by `_id`, which
# MongoDB indexes implicitly.
_MEMORIES_RECALL_INDEX_NAME = "app_user_keywords"
_MEMORIES_RECALL_INDEX_KEYS = [("app_name", 1), ("user_id", 1), ("keywords", 1)]

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
  event) in a single collection, each carrying the lowercase keywords
  extracted from its text content. `search_memory` matches documents whose
  keyword array intersects the query's keywords, ranked by keyword overlap.

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
  ) -> None:
    """Initializes the MongoDB memory service.

    Args:
      database_name: The MongoDB database used to store memories.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this service. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`).
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      memories_collection: Collection name for memory documents.
      stop_words: Words to ignore when extracting keywords. Defaults to a
        standard English stop-word list.
      max_results: Maximum number of memories returned per query.
      ensure_indexes: Create the recommended secondary index on first use:
        `app_user_keywords` on the memories collection (the app/user/keywords
        lookup behind `search_memory`). Index creation is idempotent. Without
        the `createIndex` privilege, a warning is logged once and reads fall
        back to unindexed scans; pass False to skip the attempt entirely.
    """
    if max_results <= 0:
      raise ValueError("max_results must be positive.")
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
    """Creates the recommended secondary index on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. A missing `createIndex` privilege logs one
    warning and falls back to unindexed reads; any other failure propagates
    and is retried on the next operation.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    try:
      self._memories().create_index(
          _MEMORIES_RECALL_INDEX_KEYS, name=_MEMORIES_RECALL_INDEX_NAME
      )
    # pymongo is an optional import, so the error is matched by shape.
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Code 85 (IndexOptionsConflict): an index on these keys already exists
      # under another name or with other options, so reads are still indexed.
      if getattr(exc, "code", None) == 85:
        self._indexes_ensured = True
        return
      if not _client.is_index_privilege_error(exc):
        raise
      logger.warning(
          "MongoDbMemoryService could not create the recommended index"
          " (createIndex privilege missing); reads will fall back to"
          " unindexed scans. Pass ensure_indexes=False to silence this, or"
          " grant the privilege. Details: %s",
          exc,
      )
    self._indexes_ensured = True

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
    overwrites their memories rather than duplicating them.
    """
    del custom_metadata  # Not supported by MongoDB memory service.
    scoped_session_id = session_id or _UNKNOWN_SESSION_ID
    operations: list[ReplaceOne[dict[str, Any]]] = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      visible_parts = [part for part in event.content.parts if not part.thought]
      text = " ".join([part.text for part in visible_parts if part.text])
      if not text:
        continue
      keywords = self._extract_keywords(text)
      if not keywords:
        continue
      operations.append(
          ReplaceOne(
              {
                  "_id": self._memory_key(
                      app_name, user_id, scoped_session_id, event.id
                  )
              },
              {
                  "app_name": app_name,
                  "user_id": user_id,
                  "session_id": scoped_session_id,
                  "author": event.author,
                  "keywords": sorted(keywords),
                  "content": (
                      event.content.model_copy(
                          update={"parts": visible_parts}
                      ).model_dump(exclude_none=True, mode="json")
                  ),
                  "timestamp": event.timestamp,
              },
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
    """Searches memory for events matching the query's keywords."""
    keywords = self._extract_keywords(query)
    if not keywords:
      return SearchMemoryResponse()

    query_keywords = sorted(keywords)
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

    def _search() -> list[dict[str, Any]]:
      self._ensure_indexes_once()
      return list(self._memories().aggregate(pipeline))

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

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
