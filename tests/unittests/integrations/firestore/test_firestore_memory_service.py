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
from google.adk.integrations.firestore.firestore_memory_service import FirestoreMemoryService
from google.cloud.firestore_v1.base_query import FieldFilter
from google.genai import types
import pytest


@pytest.fixture
def mock_firestore_client():
  client = mock.MagicMock()
  collection_ref = mock.MagicMock()
  client.collection.return_value = collection_ref

  collection_ref.where.return_value = collection_ref

  doc_snapshot = mock.MagicMock()
  doc_snapshot.to_dict.return_value = {}

  collection_ref.get = mock.AsyncMock(return_value=[doc_snapshot])

  return client


def test_extract_keywords(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  text = "The quick brown fox jumps over the lazy dog."
  keywords = service._extract_keywords(text)

  assert "the" not in keywords
  assert "over" not in keywords
  assert "quick" in keywords
  assert "brown" in keywords
  assert "fox" in keywords
  assert "jumps" in keywords
  assert "lazy" in keywords
  assert "dog" in keywords


@pytest.mark.asyncio
async def test_search_memory_empty_query(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  response = await service.search_memory(
      app_name="test_app", user_id="test_user", query=""
  )
  assert not response.memories
  mock_firestore_client.collection.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_with_results(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  query = "quick fox"

  doc_snapshot = mock_firestore_client.collection.return_value.where.return_value.where.return_value.where.return_value.get.return_value[
      0
  ]

  content = types.Content(parts=[types.Part.from_text(text="quick fox jumps")])

  doc_snapshot.to_dict.return_value = {
      "appName": app_name,
      "userId": user_id,
      "author": "user",
      "content": content.model_dump(exclude_none=True, mode="json"),
      "timestamp": 1234567890.0,
  }

  response = await service.search_memory(
      app_name=app_name, user_id=user_id, query=query
  )

  assert response.memories
  assert len(response.memories) == 1
  assert response.memories[0].author == "user"

  mock_firestore_client.collection.assert_called_with("memories")
  collection_ref = mock_firestore_client.collection.return_value

  assert collection_ref.where.call_count == 6
  calls = collection_ref.where.call_args_list

  app_name_calls = 0
  user_id_calls = 0
  keyword_calls = 0

  for call in calls:
    kwargs = call.kwargs
    filt = kwargs.get("filter")
    if filt:
      if (
          filt.field_path == "appName"
          and filt.op_string == "=="
          and filt.value == app_name
      ):
        app_name_calls += 1
      elif (
          filt.field_path == "userId"
          and filt.op_string == "=="
          and filt.value == user_id
      ):
        user_id_calls += 1
      elif filt.field_path == "keywords" and filt.op_string == "array_contains":

        if filt.value in ["quick", "fox"]:
          keyword_calls += 1

  assert app_name_calls == 2
  assert user_id_calls == 2
  assert keyword_calls == 2


@pytest.mark.asyncio
async def test_search_memory_deduplication(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  query = "quick fox"

  content = types.Content(parts=[types.Part.from_text(text="quick fox jumps")])

  doc_snapshot1 = mock.MagicMock()
  doc_snapshot1.to_dict.return_value = {
      "appName": app_name,
      "userId": user_id,
      "author": "user",
      "content": content.model_dump(exclude_none=True, mode="json"),
      "timestamp": 1234567890.0,
  }

  doc_snapshot2 = mock.MagicMock()
  doc_snapshot2.to_dict.return_value = {
      "appName": app_name,
      "userId": user_id,
      "author": "user",
      "content": content.model_dump(exclude_none=True, mode="json"),
      "timestamp": 1234567890.0,
  }

  get_mock = mock.AsyncMock(side_effect=[[doc_snapshot1], [doc_snapshot2]])

  mock_firestore_client.collection.return_value.where.return_value.where.return_value.where.return_value.get = (
      get_mock
  )

  response = await service.search_memory(
      app_name=app_name, user_id=user_id, query=query
  )

  assert response.memories
  assert len(response.memories) == 1
  assert response.memories[0].author == "user"


@pytest.mark.asyncio
async def test_search_memory_parsing_error(mock_firestore_client, caplog):
  service = FirestoreMemoryService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  query = "quick"

  doc_snapshot = mock_firestore_client.collection.return_value.where.return_value.where.return_value.where.return_value.get.return_value[
      0
  ]
  doc_snapshot.to_dict.return_value = {"content": "invalid_data"}

  response = await service.search_memory(
      app_name=app_name, user_id=user_id, query=query
  )

  assert not response.memories
  assert "Failed to parse memory entry" in caplog.text


@pytest.mark.asyncio
async def test_search_memory_only_stop_words(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  response = await service.search_memory(
      app_name="test_app", user_id="test_user", query="the and or"
  )
  assert not response.memories
  mock_firestore_client.collection.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_partial_failures(mock_firestore_client, caplog):
  service = FirestoreMemoryService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  query = "fox quick"

  coll_ref = (
      mock_firestore_client.collection.return_value.where.return_value.where.return_value.where.return_value
  )

  doc_snapshot = mock.MagicMock()
  doc_snapshot.to_dict.return_value = {
      "content": {"parts": [{"text": "quick response"}]},
      "author": "user",
      "timestamp": 1234567890.0,
  }

  call_count = 0

  async def mock_get():
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise ValueError("Mock generic network failure standalone")
    return [doc_snapshot]

  coll_ref.get = mock.AsyncMock(side_effect=mock_get)

  response = await service.search_memory(
      app_name=app_name, user_id=user_id, query=query
  )

  assert len(response.memories) == 1
  assert response.memories[0].author == "user"
  assert "Memory keyword search partial failure" in caplog.text


def test_init_default_client():
  with mock.patch("google.cloud.firestore.AsyncClient") as mock_client_class:
    mock_instance = mock.MagicMock()
    mock_client_class.return_value = mock_instance

    service = FirestoreMemoryService()

    mock_client_class.assert_called_once()
    assert service.client == mock_instance


@pytest.mark.asyncio
async def test_add_session_to_memory(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)

  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  content = types.Content(parts=[types.Part.from_text(text="quick brown fox")])
  event = Event(
      invocation_id="test_inv",
      author="user",
      content=content,
      timestamp=1234567890.0,
  )
  session.events.append(event)

  batch = mock.MagicMock()
  mock_firestore_client.batch.return_value = batch
  batch.commit = mock.AsyncMock()

  doc_ref = mock.MagicMock()
  mock_firestore_client.collection.return_value.document.return_value = doc_ref

  await service.add_session_to_memory(session)

  mock_firestore_client.batch.assert_called_once()
  mock_firestore_client.collection.assert_called_with("memories")
  batch.set.assert_called_once()
  batch.commit.assert_called_once()

  args, kwargs = batch.set.call_args
  assert args[0] == doc_ref
  data = args[1]
  assert data["appName"] == "test_app"
  assert data["userId"] == "test_user"
  assert "quick" in data["keywords"]
  assert data["author"] == "user"
  assert data["timestamp"] == 1234567890.0


@pytest.mark.asyncio
async def test_add_session_to_memory_no_events(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)

  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  batch = mock.MagicMock()
  mock_firestore_client.batch.return_value = batch

  await service.add_session_to_memory(session)

  mock_firestore_client.batch.assert_called_once()
  batch.set.assert_not_called()
  batch.commit.assert_not_called()


@pytest.mark.asyncio
async def test_add_session_to_memory_no_keywords(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)

  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  content = types.Content(parts=[types.Part.from_text(text="the and or")])
  event = Event(invocation_id="test_inv", author="user", content=content)
  session.events.append(event)

  batch = mock.MagicMock()
  mock_firestore_client.batch.return_value = batch

  await service.add_session_to_memory(session)

  mock_firestore_client.batch.assert_called_once()
  batch.set.assert_not_called()
  batch.commit.assert_not_called()


@pytest.mark.asyncio
async def test_add_session_to_memory_commit_error(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)

  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  content = types.Content(parts=[types.Part.from_text(text="quick brown fox")])
  event = Event(invocation_id="test_inv", author="user", content=content)
  session.events.append(event)

  batch = mock.MagicMock()
  mock_firestore_client.batch.return_value = batch
  batch.commit = mock.AsyncMock(
      side_effect=Exception("Firestore commit failed")
  )

  with pytest.raises(Exception, match="Firestore commit failed"):
    await service.add_session_to_memory(session)


@pytest.mark.asyncio
async def test_add_session_to_memory_exceeds_batch_limit(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)

  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  for i in range(501):
    content = types.Content(
        parts=[types.Part.from_text(text=f"event keyword {i}")]
    )
    event = Event(
        invocation_id=f"test_inv_{i}",
        author="user",
        content=content,
        timestamp=1234567890.0 + i,
    )
    session.events.append(event)

  batch1 = mock.MagicMock()
  batch2 = mock.MagicMock()
  batch1.commit = mock.AsyncMock()
  batch2.commit = mock.AsyncMock()
  mock_firestore_client.batch.side_effect = [batch1, batch2]

  await service.add_session_to_memory(session)

  assert mock_firestore_client.batch.call_count == 2
  assert batch1.set.call_count == 500
  batch1.commit.assert_called_once()
  assert batch2.set.call_count == 1
  batch2.commit.assert_called_once()
