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

"""Milvus-backed memory service for cross-session conversation memory."""

from __future__ import annotations

import logging
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..features import experimental
from ..features import FeatureName
from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

try:
  from pymilvus import DataType
  from pymilvus import MilvusClient
except ImportError:
  MilvusClient = None
  DataType = None


def _should_skip_event(content: types.Content) -> bool:
  """Returns True if the event has no user-readable content."""
  if not content or not content.parts:
    return True
  for part in content.parts:
    if part.text or part.inline_data or part.file_data:
      return False
  return True


def _extract_event_text(content: types.Content) -> str:
  """Extracts and joins all text parts from event content."""
  if not content or not content.parts:
    return ""
  return " ".join(part.text for part in content.parts if part.text)


@experimental(FeatureName.MILVUS_MEMORY_SERVICE)
class MilvusMemoryService(BaseMemoryService):
  """Memory service backed by Milvus vector database.

  Stores session events as vector-embedded text in a Milvus collection,
  enabling semantic search across past conversations.

  Supports all Milvus deployment modes:
  - Milvus Lite (local file path, e.g., ``"./memory.db"``)
  - Milvus Server (e.g., ``"http://localhost:19530"``)
  - Zilliz Cloud (e.g., ``"https://in01-xxx.cloud.zilliz.com"`` + token)
  """

  def __init__(
      self,
      *,
      embedding_fn: Callable[[list[str]], list[list[float]]],
      collection_name: str = "adk_memory",
      uri: str = "http://localhost:19530",
      token: Optional[str] = None,
      db_name: str = "default",
      dimension: int = 768,
      metric_type: str = "COSINE",
      top_k: int = 10,
  ):
    """Initializes the MilvusMemoryService.

    Args:
      embedding_fn: A function that takes a list of texts and returns a
        list of embedding vectors.
      collection_name: The Milvus collection name for storing memories.
      uri: The Milvus server URI or local file path.
      token: Optional authentication token (e.g., for Zilliz Cloud).
      db_name: The Milvus database name.
      dimension: The dimension of the embedding vectors.
      metric_type: The distance metric (COSINE, L2, or IP).
      top_k: Default number of results for memory search.

    Raises:
      ImportError: If pymilvus is not installed.
    """
    if MilvusClient is None:
      raise ImportError(
          "pymilvus package not found. "
          'Please install with: pip install "google-adk[milvus]"'
      )

    self._embedding_fn = embedding_fn
    self._collection_name = collection_name
    self._dimension = dimension
    self._metric_type = metric_type
    self._top_k = top_k
    self._collection_ready = False

    self._client = MilvusClient(
        uri=uri,
        token=token,
        db_name=db_name,
    )

  def _ensure_collection(self) -> None:
    """Creates the collection and index if they do not exist (idempotent)."""
    if self._collection_ready:
      return

    if self._client.has_collection(self._collection_name):
      logger.info(
          "Memory collection '%s' already exists.",
          self._collection_name,
      )
      self._collection_ready = True
      return

    schema = self._client.create_schema(
        auto_id=True, enable_dynamic_field=True
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="app_name",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="user_id",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="session_id",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="author",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name="timestamp",
        datatype=DataType.DOUBLE,
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=self._dimension,
    )

    index_params = self._client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type=self._metric_type,
    )

    self._client.create_collection(
        collection_name=self._collection_name,
        schema=schema,
        index_params=index_params,
    )

    logger.info(
        "Created memory collection '%s' (dim=%d, metric=%s).",
        self._collection_name,
        self._dimension,
        self._metric_type,
    )
    self._collection_ready = True

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Adds session events to Milvus as vector-embedded memories.

    Filters out events with no user-readable content (e.g., function
    calls). Deduplicates by checking existing events for the same
    session before inserting.

    Args:
      session: The session whose events will be stored.
    """
    self._ensure_collection()

    # Collect events with meaningful text content.
    events_to_store = []
    for event in session.events:
      if _should_skip_event(event.content):
        continue
      text = _extract_event_text(event.content)
      if not text.strip():
        continue
      events_to_store.append((event, text))

    if not events_to_store:
      logger.info("No events to add to memory for session %s.", session.id)
      return

    # Deduplicate: find which timestamps already exist for this session.
    existing_timestamps = set()
    try:
      existing = self._client.query(
          collection_name=self._collection_name,
          filter=(
              f'app_name == "{session.app_name}"'
              f' and user_id == "{session.user_id}"'
              f' and session_id == "{session.id}"'
          ),
          output_fields=["timestamp"],
      )
      existing_timestamps = {row["timestamp"] for row in existing}
    except Exception:
      logger.debug(
          "Could not query existing events for dedup, inserting all.",
          exc_info=True,
      )

    new_events = [
        (event, text)
        for event, text in events_to_store
        if event.timestamp not in existing_timestamps
    ]

    if not new_events:
      logger.info(
          "All events for session %s already in memory.", session.id
      )
      return

    # Embed and insert.
    texts = [text for _, text in new_events]
    embeddings = self._embedding_fn(texts)

    data = []
    for (event, text), embedding in zip(new_events, embeddings):
      data.append({
          "app_name": session.app_name,
          "user_id": session.user_id,
          "session_id": session.id,
          "author": event.author or "",
          "content": text,
          "timestamp": event.timestamp,
          "embedding": embedding,
      })

    self._client.insert(
        collection_name=self._collection_name,
        data=data,
    )

    logger.info(
        "Added %d events from session %s to memory collection '%s'.",
        len(data),
        session.id,
        self._collection_name,
    )

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Searches memories by semantic similarity, scoped to the user.

    Args:
      app_name: The application name.
      user_id: The user ID.
      query: The natural-language search query.

    Returns:
      A SearchMemoryResponse with matching MemoryEntry objects.
    """
    self._ensure_collection()

    query_embedding = self._embedding_fn([query])[0]

    filter_expr = (
        f'app_name == "{app_name}" and user_id == "{user_id}"'
    )

    results = self._client.search(
        collection_name=self._collection_name,
        data=[query_embedding],
        limit=self._top_k,
        output_fields=["content", "author", "timestamp"],
        search_params={"metric_type": self._metric_type},
        filter=filter_expr,
    )

    if not results or not results[0]:
      return SearchMemoryResponse()

    memories = []
    for hit in results[0]:
      entity = hit["entity"]
      author = entity.get("author", "")
      content_text = entity.get("content", "")
      timestamp = entity.get("timestamp")

      role = "user" if author == "user" else "model"
      memory_entry = MemoryEntry(
          content=types.Content(
              parts=[types.Part(text=content_text)],
              role=role,
          ),
          author=author,
          timestamp=(
              _utils.format_timestamp(timestamp) if timestamp else None
          ),
      )
      memories.append(memory_entry)

    return SearchMemoryResponse(memories=memories)

  def close(self) -> None:
    """Closes the Milvus client connection."""
    self._client.close()
