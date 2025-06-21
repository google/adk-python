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

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session


class LangchainVectorStoreMemoryService(BaseMemoryService):
  """A memory service that uses a Langchain vector store for memory.

  This service allows adding sessions to a Langchain vector store and searching
  for relevant memories based on user queries.
  """

  def __init__(self, vector_store: VectorStore):
    """Initializes the LangchainVectorStoreMemoryService.

    Args:
        vector_store: The Langchain vector store to use for memory.
    """
    self._vector_store = vector_store

  async def add_session_to_memory(
      self,
      session: Session,
  ):
    """Adds a session to the Langchain vector store.

    Args:
        session: The session to add.
    """
    # TODO(b/338460078): Implement conversation summarization.
    documents = [
        Document(
            page_content=turn.exchange.user_prompt,
            metadata={
                "app_name": session.app_name,
                "session_id": session.session_id,
                "user_id": session.user_id,
                "turn_id": turn.turn_id,
            },
        )
        for turn in session.turns
    ]
    await self._vector_store.aadd_documents(documents=documents)

  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Searches for sessions in the Langchain vector store that match the query.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing the matching memories.
    """
    # TODO(b/338460135): Add filtering by app_name and user_id.
    docs = await self._vector_store.asimilarity_search(query=query)
    memories = [
        MemoryEntry(
            app_name=doc.metadata["app_name"],
            session_id=doc.metadata["session_id"],
            user_id=doc.metadata["user_id"],
            turn_id=doc.metadata["turn_id"],
            content=doc.page_content,
        )
        for doc in docs
    ]
    return SearchMemoryResponse(memories=memories)
