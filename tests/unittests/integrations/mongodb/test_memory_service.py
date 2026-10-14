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

"""Tests for MongoDbMemoryService, backed by mongomock."""

from __future__ import annotations

import asyncio
from unittest import mock

from google.adk.events.event import Event
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import mongomock
import pytest


@pytest.fixture(autouse=True)
def _mongomock_bulk_write_compat(monkeypatch):
  """Drops the `sort` key pymongo 4.9+ adds to bulk write operations.

  mongomock's BulkOperationBuilder does not accept `sort` (added to
  ReplaceOne/UpdateOne in pymongo 4.9), so bulk_write against mongomock
  raises TypeError without this shim. Real MongoDB accepts the key; this
  only patches the test double.
  """
  from mongomock.collection import BulkOperationBuilder

  real_add_replace = BulkOperationBuilder.add_replace
  real_add_update = BulkOperationBuilder.add_update

  def add_replace(self, selector, doc, upsert, sort=None, **kwargs):
    return real_add_replace(self, selector, doc, upsert, **kwargs)

  def add_update(
      self, selector, doc, multi=False, upsert=False, sort=None, **kwargs
  ):
    return real_add_update(self, selector, doc, multi, upsert, **kwargs)

  monkeypatch.setattr(BulkOperationBuilder, "add_replace", add_replace)
  monkeypatch.setattr(BulkOperationBuilder, "add_update", add_update)


APP = "test_app"
USER = "test_user"


@pytest.fixture
def service():
  return MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(), database_name="test_db"
  )


@pytest.mark.asyncio
async def test_ensure_indexes_creates_recall_index():
  """ensure_indexes=True creates the app/user/keywords index on memories."""
  client = mongomock.MongoClient()
  service = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", ensure_indexes=True
  )
  await service.add_session_to_memory(_session_with_texts("hello world"))

  indexes = client["test_db"]["memories"].index_information()
  assert "app_user_keywords" in indexes


@pytest.mark.asyncio
async def test_index_not_created_by_default(service):
  """Without ensure_indexes the implicit _id index is the only one present."""
  await service.add_session_to_memory(_session_with_texts("hello world"))

  indexes = service._memories().index_information()
  assert "app_user_keywords" not in indexes


def _session_with_texts(*texts: str, session_id: str = "s1") -> Session:
  events = [
      Event(
          invocation_id="inv",
          author="user" if i % 2 == 0 else "agent",
          content=types.Content(
              role="user" if i % 2 == 0 else "model",
              parts=[types.Part(text=text)],
          ),
          timestamp=float(i + 1),
      )
      for i, text in enumerate(texts)
  ]
  return Session(id=session_id, app_name=APP, user_id=USER, events=events)


@pytest.mark.asyncio
async def test_add_session_then_search_by_keyword(service):
  await service.add_session_to_memory(
      _session_with_texts(
          "I love hiking in the mountains",
          "My favorite trail is the Pacific Crest Trail",
      )
  )

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1
  assert "hiking" in response.memories[0].content.parts[0].text
  assert response.memories[0].author == "user"

  # A query can match several memories.
  response = await service.search_memory(
      app_name=APP, user_id=USER, query="trail mountains"
  )
  assert len(response.memories) == 2


@pytest.mark.asyncio
async def test_search_memory_scopes_by_app_and_user(service):
  await service.add_session_to_memory(_session_with_texts("remember hiking"))

  assert (
      await service.search_memory(
          app_name="other_app", user_id=USER, query="hiking"
      )
  ).memories == []
  assert (
      await service.search_memory(
          app_name=APP, user_id="other_user", query="hiking"
      )
  ).memories == []


@pytest.mark.asyncio
async def test_reingesting_session_does_not_duplicate(service):
  session = _session_with_texts("I love hiking")
  await service.add_session_to_memory(session)
  await service.add_session_to_memory(session)

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1


@pytest.mark.asyncio
async def test_search_memory_ignores_stop_words(service):
  await service.add_session_to_memory(_session_with_texts("the cat sat"))

  # "the" is a stop word, so a stop-words-only query matches nothing even
  # though the ingested text contains "the".
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="the")
  ).memories == []
  response = await service.search_memory(
      app_name=APP, user_id=USER, query="cat"
  )
  assert len(response.memories) == 1


@pytest.mark.asyncio
async def test_search_memory_empty_or_stop_word_query(service):
  await service.add_session_to_memory(_session_with_texts("remember hiking"))
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="")
  ).memories == []
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="?!")
  ).memories == []


@pytest.mark.asyncio
async def test_add_session_skips_events_without_text(service):
  session = Session(id="s1", app_name=APP, user_id=USER)
  session.events.append(Event(invocation_id="inv", author="agent"))
  await service.add_session_to_memory(session)
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="anything")
  ).memories == []


def test_constructor_validates_client_args():
  with pytest.raises(ValueError):
    MongoDbMemoryService(database_name="db")
  with pytest.raises(ValueError):
    MongoDbMemoryService(
        database_name="db",
        mongo_client=mongomock.MongoClient(),
        connection_string="mongodb://localhost:27017",
    )


@pytest.mark.asyncio
async def test_ingest_uses_one_bulk_write(service):
  """Ingestion issues one bulk_write, not one round-trip per event."""
  bulk_calls = []
  replace_calls = []
  collection = service._memories()
  real_bulk = collection.bulk_write

  def _counting_bulk(ops, **kwargs):
    bulk_calls.append(ops)
    return real_bulk(ops, **kwargs)

  def _forbidden_replace(*args, **kwargs):
    replace_calls.append(args)

  with (
      mock.patch.object(collection, "bulk_write", _counting_bulk),
      mock.patch.object(collection, "replace_one", _forbidden_replace),
  ):
    await service.add_session_to_memory(
        _session_with_texts("first memory", "second memory", "third memory")
    )

  assert len(bulk_calls) == 1
  assert len(bulk_calls[0]) == 3  # one ReplaceOne per event
  assert replace_calls == []

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="memory"
  )
  assert len(response.memories) == 3


@pytest.mark.asyncio
async def test_index_creation_runs_once_under_concurrency():
  """Concurrent first calls create the recall index exactly once."""
  client = mongomock.MongoClient()
  service = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", ensure_indexes=True
  )
  create_calls = []
  memories = client["test_db"]["memories"]
  real_create = memories.create_index

  def _counting_create(*args, **kwargs):
    create_calls.append(kwargs.get("name"))
    return real_create(*args, **kwargs)

  with mock.patch.object(memories, "create_index", _counting_create):
    await asyncio.gather(*(
        service.add_session_to_memory(
            _session_with_texts(f"memory {i}", session_id=f"s{i}")
        )
        for i in range(10)
    ))

  assert create_calls == ["app_user_keywords"]
