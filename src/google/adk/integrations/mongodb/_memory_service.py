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
import logging
import re
from typing import Any
from typing import TYPE_CHECKING

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

  from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MEMORIES_COLLECTION = "memories"

# Recommended secondary index, created only when `ensure_indexes=True`. It
# covers the `search_memory` lookup (app/user equality plus a multikey `$in`
# on keywords); writes are keyed by `_id`, which MongoDB indexes implicitly.
_MEMORIES_RECALL_INDEX_NAME = "app_user_keywords"
_MEMORIES_RECALL_INDEX_KEYS = [("app_name", 1), ("user_id", 1), ("keywords", 1)]

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
  event) in a single collection, each carrying the lowercase keywords
  extracted from its text content. `search_memory` matches documents whose
  keyword array intersects the query's keywords, ranks them by keyword
  overlap (most recent first within a tie), and returns at most
  `max_results` entries — a common keyword in a large memory store can
  otherwise return everything.

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
      stop_words: Words to ignore when extracting keywords. Defaults to a
        standard English stop-word list.
      ensure_indexes: Create the recommended secondary index on first use
        (default True): `app_user_keywords` on the memories collection (the
        app/user/keywords lookup behind `search_memory`). Index creation is
        idempotent. Without the `createIndex` privilege a warning is logged
        once and reads fall back to unindexed scans; pass False to skip the
        attempt entirely.
      max_results: Maximum number of memories a search returns, ranked by
        keyword overlap and recency. Bounds what a query with a common
        keyword can return.

    Raises:
      ValueError: If `max_results` is not positive.
    """
    if max_results < 1:
      raise ValueError("max_results must be a positive integer.")
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

  def _memories(self):
    return self._client[self._database_name][self.memories_collection]

  def _ensure_indexes_once(self) -> None:
    """Creates the recommended secondary index on first use, if enabled.

    Index creation is idempotent on the server, and the guard flag only
    skips work in this process. A deployment without the `createIndex`
    privilege (e.g. a least-privilege runtime user) logs a warning once and
    falls back to unindexed reads rather than failing every operation.
    """
    if not self._ensure_indexes or self._indexes_ensured:
      return
    try:
      self._memories().create_index(
          _MEMORIES_RECALL_INDEX_KEYS, name=_MEMORIES_RECALL_INDEX_NAME
      )
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
          " unindexed scans. Pass ensure_indexes=False to silence this, or"
          " grant the privilege. Details: %s",
          exc,
      )
    self._indexes_ensured = True

  def _extract_keywords(self, text: str) -> set[str]:
    """Extracts lowercase keywords from text, ignoring stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in self.stop_words}

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Ingests the session's text events into the memory collection.

    Ingestion is idempotent: memory documents are keyed by
    `<app_name>/<user_id>/<session_id>/<event_id>`, so re-adding a session
    overwrites its memories rather than duplicating them.
    """
    await asyncio.to_thread(self._ensure_indexes_once)

    def _add() -> None:
      for event in session.events:
        if not event.content or not event.content.parts:
          continue
        text = " ".join(
            [part.text for part in event.content.parts if part.text]
        )
        if not text:
          continue
        keywords = self._extract_keywords(text)
        if not keywords:
          continue
        memory_id = (
            f"{session.app_name}/{session.user_id}/{session.id}/{event.id}"
        )
        self._memories().replace_one(
            {"_id": memory_id},
            {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "author": event.author,
                "keywords": sorted(keywords),
                "content": event.content.model_dump(
                    exclude_none=True, mode="json"
                ),
                "timestamp": event.timestamp,
            },
            upsert=True,
        )

    await asyncio.to_thread(_add)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches memory for events matching the query's keywords."""
    await asyncio.to_thread(self._ensure_indexes_once)

    keywords = self._extract_keywords(query)
    if not keywords:
      return SearchMemoryResponse()

    def _search() -> list[dict[str, Any]]:
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

    docs = await asyncio.to_thread(_search)

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

  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this service."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
