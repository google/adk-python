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

import asyncio
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from langchain_core.documents import Document

from src.google.adk.memory import LangchainVectorStoreMemoryService
from src.google.adk.memory.memory_entry import MemoryEntry
from src.google.adk.memory.base_memory_service import SearchMemoryResponse
from src.google.adk.sessions.session import Session
from src.google.adk.sessions.turn import Turn
from src.google.adk.shared.exchange import Exchange


class TestLangchainVectorStoreMemoryService(unittest.TestCase):

  def setUp(self):
    self.mock_vector_store = MagicMock()
    self.mock_vector_store.aadd_documents = AsyncMock()
    self.mock_vector_store.asimilarity_search = AsyncMock()
    self.memory_service = LangchainVectorStoreMemoryService(
        vector_store=self.mock_vector_store
    )

  def test_add_session_to_memory_successful(self):
    session = Session(
        app_name="test_app", session_id="test_session", user_id="test_user"
    )
    session.add_turn(
        Turn(
            turn_id="turn1",
            exchange=Exchange(user_prompt="Hello", llm_response="Hi there!"),
        )
    )
    session.add_turn(
        Turn(
            turn_id="turn2",
            exchange=Exchange(user_prompt="How are you?", llm_response="I'm good!"),
        )
    )

    asyncio.run(self.memory_service.add_session_to_memory(session))

    self.mock_vector_store.aadd_documents.assert_called_once()
    call_args = self.mock_vector_store.aadd_documents.call_args
    added_documents = call_args.kwargs["documents"]
    self.assertEqual(len(added_documents), 2)

    doc1 = added_documents[0]
    self.assertIsInstance(doc1, Document)
    self.assertEqual(doc1.page_content, "Hello")
    self.assertEqual(
        doc1.metadata,
        {
            "app_name": "test_app",
            "session_id": "test_session",
            "user_id": "test_user",
            "turn_id": "turn1",
        },
    )

    doc2 = added_documents[1]
    self.assertIsInstance(doc2, Document)
    self.assertEqual(doc2.page_content, "How are you?")
    self.assertEqual(
        doc2.metadata,
        {
            "app_name": "test_app",
            "session_id": "test_session",
            "user_id": "test_user",
            "turn_id": "turn2",
        },
    )

  def test_add_session_to_memory_no_content(self):
    session = Session(
        app_name="test_app", session_id="test_session", user_id="test_user"
    )
    # Add a turn with an exchange that has an empty user_prompt
    session.add_turn(
        Turn(
            turn_id="turn1",
            exchange=Exchange(user_prompt="", llm_response="No prompt provided."),
        )
    )
    # Add a turn with an exchange that has a None user_prompt
    session.add_turn(
        Turn(
            turn_id="turn2",
            exchange=Exchange(user_prompt=None, llm_response="Still no prompt."),
        )
    )

    asyncio.run(self.memory_service.add_session_to_memory(session))

    self.mock_vector_store.aadd_documents.assert_called_once()
    call_args = self.mock_vector_store.aadd_documents.call_args
    added_documents = call_args.kwargs["documents"]
    # Even with empty/None prompts, documents are created with that content
    self.assertEqual(len(added_documents), 2)
    self.assertEqual(added_documents[0].page_content, "")
    self.assertIsNone(added_documents[1].page_content)


  def test_search_memory_found_results(self):
    mock_docs = [
        Document(
            page_content="Test content 1",
            metadata={
                "app_name": "test_app",
                "session_id": "s1",
                "user_id": "user1",
                "turn_id": "t1",
            },
        ),
        Document(
            page_content="Test content 2",
            metadata={
                "app_name": "test_app",
                "session_id": "s2",
                "user_id": "user1",
                "turn_id": "t2",
            },
        ),
    ]
    self.mock_vector_store.asimilarity_search.return_value = mock_docs

    response = asyncio.run(
        self.memory_service.search_memory(
            app_name="test_app", user_id="user1", query="test query"
        )
    )

    self.mock_vector_store.asimilarity_search.assert_called_once_with(
        query="test query"
    )
    self.assertIsInstance(response, SearchMemoryResponse)
    self.assertEqual(len(response.memories), 2)

    mem1 = response.memories[0]
    self.assertIsInstance(mem1, MemoryEntry)
    self.assertEqual(mem1.content, "Test content 1")
    self.assertEqual(mem1.app_name, "test_app")
    self.assertEqual(mem1.session_id, "s1")
    self.assertEqual(mem1.user_id, "user1")
    self.assertEqual(mem1.turn_id, "t1")

    mem2 = response.memories[1]
    self.assertIsInstance(mem2, MemoryEntry)
    self.assertEqual(mem2.content, "Test content 2")
    self.assertEqual(mem2.app_name, "test_app")
    self.assertEqual(mem2.session_id, "s2")
    self.assertEqual(mem2.user_id, "user1")
    self.assertEqual(mem2.turn_id, "t2")

  def test_search_memory_no_results(self):
    self.mock_vector_store.asimilarity_search.return_value = []

    response = asyncio.run(
        self.memory_service.search_memory(
            app_name="test_app", user_id="user1", query="test query"
        )
    )

    self.mock_vector_store.asimilarity_search.assert_called_once_with(
        query="test query"
    )
    self.assertIsInstance(response, SearchMemoryResponse)
    self.assertEqual(len(response.memories), 0)

  # The LangchainVectorStoreMemoryService currently does not implement
  # filtering by app_name and user_id before calling the vector store.
  # It relies on the vector store itself or the context (e.g. a per-user
  # vector store) to handle this.
  # This test verifies that these identifiers are passed in the metadata
  # when adding documents, which is a prerequisite for such filtering.
  # A more comprehensive test for isolation would require mocking different
  # vector store behaviors or having multiple vector store instances.
  def test_search_memory_metadata_for_isolation(self):
    # This test is largely covered by test_add_session_to_memory_successful
    # which checks the metadata. We can add a specific search scenario.

    # 1. Add data for user1
    session1 = Session(app_name="app1", session_id="s1", user_id="user1")
    session1.add_turn(Turn(turn_id="t1", exchange=Exchange(user_prompt="user1 data", llm_response="...")))
    asyncio.run(self.memory_service.add_session_to_memory(session1))

    # 2. Add data for user2
    session2 = Session(app_name="app1", session_id="s2", user_id="user2")
    session2.add_turn(Turn(turn_id="t2", exchange=Exchange(user_prompt="user2 data", llm_response="...")))
    asyncio.run(self.memory_service.add_session_to_memory(session2))

    # 3. Configure search to return both if no filtering happens at vector store
    mock_doc1 = Document(page_content="user1 data", metadata={"app_name": "app1", "session_id": "s1", "user_id": "user1", "turn_id": "t1"})
    mock_doc2 = Document(page_content="user2 data", metadata={"app_name": "app1", "session_id": "s2", "user_id": "user2", "turn_id": "t2"})
    self.mock_vector_store.asimilarity_search.return_value = [mock_doc1, mock_doc2]

    # 4. Search for user1's data
    # The current implementation doesn't filter by user_id or app_name in search_memory,
    # so we expect both results back from the mock.
    # The TODO(b/338460135) in the service points this out.
    response = asyncio.run(
        self.memory_service.search_memory(
            app_name="app1", user_id="user1", query="data"
        )
    )
    self.mock_vector_store.asimilarity_search.assert_called_with(query="data")
    self.assertEqual(len(response.memories), 2) # Expecting both due to no filtering

    # If filtering were implemented in search_memory, this would be different.
    # For now, this test confirms the as-is behavior and relies on the
    # add_session_to_memory tests for metadata correctness.


if __name__ == "__main__":
  unittest.main()
