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

from unittest import mock

from google.adk.events.event import Event
from google.adk.memory.milvus_memory_service import MilvusMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

DIMENSION = 4
APP_NAME = "test-app"
USER_ID = "test-user"


def _mock_embedding_fn(texts: list[str]) -> list[list[float]]:
  """A mock embedding function returning fixed-dimension vectors."""
  return [[0.1] * DIMENSION for _ in texts]


def _make_session(
    session_id: str = "session-1",
    app_name: str = APP_NAME,
    user_id: str = USER_ID,
    events: list[Event] | None = None,
) -> Session:
  if events is None:
    events = [
        Event(
            id="event-1",
            invocation_id="inv-1",
            author="user",
            timestamp=1000.0,
            content=types.Content(
                parts=[types.Part(text="Hello, I like Python.")]
            ),
        ),
        Event(
            id="event-2",
            invocation_id="inv-2",
            author="model",
            timestamp=1001.0,
            content=types.Content(
                parts=[types.Part(text="Python is a great language!")]
            ),
        ),
    ]
  return Session(
      app_name=app_name,
      user_id=user_id,
      id=session_id,
      last_update_time=2000,
      events=events,
  )


@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
def test_init_success(mock_milvus_client_cls):
  """Test successful initialization."""
  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      uri="http://localhost:19530",
      collection_name="test_memory",
      dimension=DIMENSION,
  )
  mock_milvus_client_cls.assert_called_once_with(
      uri="http://localhost:19530",
      token=None,
      db_name="default",
  )
  assert service._embedding_fn is _mock_embedding_fn
  assert service._collection_name == "test_memory"


@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
def test_ensure_collection_creates_when_missing(mock_milvus_client_cls):
  """Test that _ensure_collection creates collection when it doesn't exist."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = False

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  service._ensure_collection()

  mock_client.has_collection.assert_called_once_with("adk_memory")
  mock_client.create_collection.assert_called_once()


@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
def test_ensure_collection_skips_existing(mock_milvus_client_cls):
  """Test that _ensure_collection skips if collection exists."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  service._ensure_collection()

  mock_client.has_collection.assert_called_once_with("adk_memory")
  mock_client.create_collection.assert_not_called()


@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
def test_ensure_collection_idempotent(mock_milvus_client_cls):
  """Test that _ensure_collection only checks once."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = False

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  service._ensure_collection()
  service._ensure_collection()

  # has_collection should only be called once due to _collection_ready flag.
  mock_client.has_collection.assert_called_once()


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_add_session_to_memory(mock_milvus_client_cls):
  """Test adding a session with events to memory."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.query.return_value = []  # No existing events.

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  session = _make_session()

  await service.add_session_to_memory(session)

  mock_client.insert.assert_called_once()
  call_args = mock_client.insert.call_args
  data = call_args.kwargs["data"]
  assert len(data) == 2
  assert data[0]["app_name"] == APP_NAME
  assert data[0]["user_id"] == USER_ID
  assert data[0]["session_id"] == "session-1"
  assert data[0]["author"] == "user"
  assert data[0]["content"] == "Hello, I like Python."
  assert data[0]["timestamp"] == 1000.0
  assert len(data[0]["embedding"]) == DIMENSION


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_add_session_skips_empty_events(mock_milvus_client_cls):
  """Test that events without text content are skipped."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.query.return_value = []

  session = _make_session(events=[
      Event(
          id="event-empty",
          invocation_id="inv-1",
          author="user",
          timestamp=1000.0,
          # No content at all.
      ),
      Event(
          id="event-func",
          invocation_id="inv-2",
          author="model",
          timestamp=1001.0,
          content=types.Content(
              parts=[types.Part(function_call=types.FunctionCall(
                  name="test_fn", args={}
              ))]
          ),
      ),
      Event(
          id="event-text",
          invocation_id="inv-3",
          author="user",
          timestamp=1002.0,
          content=types.Content(
              parts=[types.Part(text="Real message")]
          ),
      ),
  ])

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )

  await service.add_session_to_memory(session)

  call_args = mock_client.insert.call_args
  data = call_args.kwargs["data"]
  assert len(data) == 1
  assert data[0]["content"] == "Real message"


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_add_session_deduplication(mock_milvus_client_cls):
  """Test that events already in Milvus are not re-inserted."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  # Simulate one event already existing.
  mock_client.query.return_value = [{"timestamp": 1000.0}]

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  session = _make_session()

  await service.add_session_to_memory(session)

  call_args = mock_client.insert.call_args
  data = call_args.kwargs["data"]
  # Only the second event (timestamp=1001.0) should be inserted.
  assert len(data) == 1
  assert data[0]["timestamp"] == 1001.0


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_add_session_all_duplicated(mock_milvus_client_cls):
  """Test that no insert happens when all events already exist."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.query.return_value = [
      {"timestamp": 1000.0},
      {"timestamp": 1001.0},
  ]

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  session = _make_session()

  await service.add_session_to_memory(session)

  mock_client.insert.assert_not_called()


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_search_memory(mock_milvus_client_cls):
  """Test similarity search returns MemoryEntry objects."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.search.return_value = [[
      {
          "entity": {
              "content": "I like Python.",
              "author": "user",
              "timestamp": 1000.0,
          },
          "distance": 0.95,
      },
      {
          "entity": {
              "content": "Python is great!",
              "author": "model",
              "timestamp": 1001.0,
          },
          "distance": 0.90,
      },
  ]]

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )

  response = await service.search_memory(
      app_name=APP_NAME,
      user_id=USER_ID,
      query="Python",
  )

  assert len(response.memories) == 2
  assert response.memories[0].author == "user"
  assert response.memories[0].content.parts[0].text == "I like Python."
  assert response.memories[0].content.role == "user"
  assert response.memories[1].author == "model"
  assert response.memories[1].content.role == "model"

  # Verify search was called with correct filter.
  call_args = mock_client.search.call_args
  assert f'app_name == "{APP_NAME}"' in call_args.kwargs["filter"]
  assert f'user_id == "{USER_ID}"' in call_args.kwargs["filter"]


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_search_memory_empty_results(mock_milvus_client_cls):
  """Test search returning no results."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.search.return_value = [[]]

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )

  response = await service.search_memory(
      app_name=APP_NAME,
      user_id=USER_ID,
      query="something",
  )

  assert len(response.memories) == 0


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_search_memory_user_scoping(mock_milvus_client_cls):
  """Test that search scopes by app_name and user_id."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True
  mock_client.search.return_value = [[]]

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )

  await service.search_memory(
      app_name="app-A",
      user_id="user-B",
      query="test",
  )

  call_args = mock_client.search.call_args
  filter_expr = call_args.kwargs["filter"]
  assert 'app_name == "app-A"' in filter_expr
  assert 'user_id == "user-B"' in filter_expr


@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
def test_close(mock_milvus_client_cls):
  """Test closing the memory service."""
  mock_client = mock_milvus_client_cls.return_value

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  service.close()

  mock_client.close.assert_called_once()


@pytest.mark.asyncio
@mock.patch("google.adk.memory.milvus_memory_service.MilvusClient")
async def test_add_session_no_events(mock_milvus_client_cls):
  """Test adding a session with no events does nothing."""
  mock_client = mock_milvus_client_cls.return_value
  mock_client.has_collection.return_value = True

  service = MilvusMemoryService(
      embedding_fn=_mock_embedding_fn,
      dimension=DIMENSION,
  )
  session = _make_session(events=[])

  await service.add_session_to_memory(session)

  mock_client.insert.assert_not_called()
