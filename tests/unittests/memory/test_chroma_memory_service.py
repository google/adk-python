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

"""Tests for ChromaMemoryService."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.memory.chroma_memory_service import ChromaMemoryService
from google.adk.memory.embeddings.base_embedding_provider import BaseEmbeddingProvider
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_OTHER_USER_ID = "another-user"


class MockEmbeddingProvider(BaseEmbeddingProvider):
  """A mock embedding provider for testing."""

  def __init__(self, dimension: int = 384):
    self._dimension = dimension

  @property
  def dimension(self) -> int:
    return self._dimension

  async def embed(self, texts: list[str]) -> list[list[float]]:
    """Return deterministic mock embeddings based on text content."""
    embeddings = []
    for text in texts:
      # Create a simple deterministic embedding based on hash
      base_value = hash(text) % 1000 / 1000.0
      embedding = [base_value + i * 0.001 for i in range(self._dimension)]
      embeddings.append(embedding)
    return embeddings


MOCK_SESSION_1 = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id="session-1",
    last_update_time=1000,
    events=[
        Event(
            id="event-1a",
            invocation_id="inv-1",
            author="user",
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text="The ADK is a great toolkit.")]
            ),
        ),
        # Event with no content, should be ignored by the service
        Event(
            id="event-1b",
            invocation_id="inv-2",
            author="user",
            timestamp=12346,
        ),
        Event(
            id="event-1c",
            invocation_id="inv-3",
            author="model",
            timestamp=12347,
            content=types.Content(
                parts=[
                    types.Part(
                        text="I agree. The Agent Development Kit (ADK) rocks!"
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_2 = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id="session-2",
    last_update_time=2000,
    events=[
        Event(
            id="event-2a",
            invocation_id="inv-4",
            author="user",
            timestamp=54321,
            content=types.Content(
                parts=[types.Part(text="I like to code in Python.")]
            ),
        ),
    ],
)

MOCK_SESSION_DIFFERENT_USER = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_OTHER_USER_ID,
    id="session-3",
    last_update_time=3000,
    events=[
        Event(
            id="event-3a",
            invocation_id="inv-5",
            author="user",
            timestamp=60000,
            content=types.Content(parts=[types.Part(text="This is a secret.")]),
        ),
    ],
)

MOCK_SESSION_WITH_NO_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id="session-4",
    last_update_time=4000,
)


@pytest.fixture
def embedding_provider():
  """Create a mock embedding provider."""
  return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def memory_service(embedding_provider, request):
  """Create a ChromaMemoryService with in-memory storage and unique collection."""
  # Use test function name to create unique collection per test
  collection_name = f"test_{request.node.name}"
  return ChromaMemoryService(
      embedding_provider=embedding_provider,
      collection_name=collection_name,
  )


@pytest.mark.asyncio
async def test_add_session_to_memory(memory_service):
  """Tests that a session with events is correctly added to memory."""
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  # Verify documents were added to the collection
  count = memory_service._collection.count()
  # Should have 2 events (one has no content and is filtered)
  assert count == 2


@pytest.mark.asyncio
async def test_add_session_with_no_events_to_memory(memory_service):
  """Tests that adding a session with no events does not cause an error."""
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_NO_EVENTS)

  # Verify no documents were added
  count = memory_service._collection.count()
  assert count == 0


@pytest.mark.asyncio
async def test_search_memory_returns_results(memory_service):
  """Tests that search returns relevant results."""
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_2)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK toolkit"
  )

  # Should return results (exact matching depends on embedding similarity)
  assert len(result.memories) > 0


@pytest.mark.asyncio
async def test_search_memory_no_match(memory_service):
  """Tests search with no matching user returns empty results."""
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id="nonexistent-user", query="ADK"
  )

  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_is_scoped_by_user(memory_service):
  """Tests that search results are correctly scoped to the user_id."""
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_DIFFERENT_USER)

  # Verify that searching as MOCK_OTHER_USER_ID returns the secret
  result_other_user = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_OTHER_USER_ID, query="secret"
  )
  assert len(result_other_user.memories) == 1
  assert (
      result_other_user.memories[0].content.parts[0].text == "This is a secret."
  )

  # Verify that searching as MOCK_USER_ID does NOT return the secret
  # (it should return MOCK_USER_ID's data, not MOCK_OTHER_USER_ID's)
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="secret"
  )
  # Results should only contain MOCK_USER_ID's content, not the secret
  for memory in result.memories:
    assert "secret" not in memory.content.parts[0].text.lower()


@pytest.mark.asyncio
async def test_upsert_updates_existing_documents(memory_service):
  """Tests that adding the same session twice updates existing documents."""
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  initial_count = memory_service._collection.count()

  # Add the same session again
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  final_count = memory_service._collection.count()

  # Count should remain the same (upsert, not duplicate)
  assert initial_count == final_count
