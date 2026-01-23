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

"""ChromaDB-based memory service with semantic search capabilities."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .embeddings.base_embedding_provider import BaseEmbeddingProvider
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)


def _user_key(app_name: str, user_id: str) -> str:
  """Generate a unique key for a user within an app."""
  return f"{app_name}/{user_id}"


def _event_id(session_id: str, event_id: str) -> str:
  """Generate a unique document ID for an event."""
  return hashlib.sha256(f"{session_id}/{event_id}".encode()).hexdigest()[:32]


class ChromaMemoryService(BaseMemoryService):
  """A memory service that uses ChromaDB for semantic search.

  This service stores session events as documents in a ChromaDB collection
  and uses vector embeddings for semantic similarity search.

  Example:
      >>> from google.adk.memory.embeddings import OllamaEmbeddingProvider
      >>> embedding_provider = OllamaEmbeddingProvider(model="nomic-embed-text")
      >>> memory = ChromaMemoryService(
      ...     embedding_provider=embedding_provider,
      ...     persist_directory="./memory_db"
      ... )
  """

  def __init__(
      self,
      embedding_provider: BaseEmbeddingProvider,
      collection_name: str = "adk_memory",
      persist_directory: Optional[str] = None,
  ):
    """Initialize the ChromaMemoryService.

    Args:
        embedding_provider: The embedding provider to use for generating
            vector representations of text.
        collection_name: The name of the ChromaDB collection to use.
        persist_directory: Optional directory path for persisting the
            ChromaDB data. If None, data is stored in memory only.
    """
    try:
      import chromadb
    except ImportError as exc:
      raise ImportError(
          "chromadb is required for ChromaMemoryService. "
          "Install it with: pip install chromadb"
      ) from exc

    self._embedding_provider = embedding_provider
    self._collection_name = collection_name

    if persist_directory:
      self._client = chromadb.PersistentClient(path=persist_directory)
    else:
      self._client = chromadb.Client()

    self._collection = self._client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

  @override
  async def add_session_to_memory(self, session: "Session"):
    """Add a session's events to the ChromaDB collection.

    Each event with text content is stored as a separate document with
    its embedding, along with metadata for filtering.

    Args:
        session: The session to add to memory.
    """
    user_key = _user_key(session.app_name, session.user_id)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for event in session.events:
      if not event.content or not event.content.parts:
        continue

      text_parts = [part.text for part in event.content.parts if part.text]
      if not text_parts:
        continue

      document_text = " ".join(text_parts)
      documents.append(document_text)
      metadatas.append({
          "user_key": user_key,
          "app_name": session.app_name,
          "user_id": session.user_id,
          "session_id": session.id,
          "event_id": event.id,
          "author": event.author or "",
          "timestamp": event.timestamp or 0,
      })
      ids.append(_event_id(session.id, event.id))

    if not documents:
      return

    # Generate embeddings
    embeddings = await self._embedding_provider.embed(documents)

    # Upsert to ChromaDB (update if exists, insert otherwise)
    self._collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.debug(
        "Added %d events from session %s to ChromaDB",
        len(documents),
        session.id,
    )

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Search for memories semantically similar to the query.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing the matching memories.
    """

    user_key = _user_key(app_name, user_id)

    # Generate embedding for query
    query_embeddings = await self._embedding_provider.embed([query])
    if not query_embeddings:
      return SearchMemoryResponse()

    # Search ChromaDB with user filtering
    results = self._collection.query(
        query_embeddings=query_embeddings,
        n_results=10,
        where={"user_key": user_key},
        include=["documents", "metadatas"],
    )

    memories: list[MemoryEntry] = []

    if results["documents"] and results["metadatas"]:
      for doc, metadata in zip(
          results["documents"][0], results["metadatas"][0]
      ):
        content = types.Content(parts=[types.Part(text=doc)])
        memories.append(
            MemoryEntry(
                content=content,
                author=metadata.get("author", ""),
                timestamp=_utils.format_timestamp(metadata.get("timestamp", 0)),
            )
        )

    return SearchMemoryResponse(memories=memories)
