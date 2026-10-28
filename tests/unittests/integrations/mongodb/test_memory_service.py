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
from datetime import timedelta
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
async def test_index_created_by_default(service):
  """Indexes are on by default: no opt-in is needed for the recall index."""
  await service.add_session_to_memory(_session_with_texts("hello world"))

  indexes = service._memories().index_information()
  assert "app_user_keywords" in indexes


@pytest.mark.asyncio
async def test_ensure_indexes_false_skips_index_creation():
  """Opting out leaves the implicit _id index as the only one."""
  service = MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(),
      database_name="test_db",
      ensure_indexes=False,
  )
  await service.add_session_to_memory(_session_with_texts("hello world"))

  indexes = service._memories().index_information()
  assert "app_user_keywords" not in indexes


@pytest.mark.asyncio
async def test_index_privilege_error_falls_back_to_unindexed_reads(
    caplog, monkeypatch
):
  """A denied createIndex warns once and never breaks reads or writes."""
  from pymongo.errors import OperationFailure

  client = mongomock.MongoClient()
  service = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  collection = client["test_db"]["memories"]

  def _denied(*args, **kwargs):
    raise OperationFailure("not authorized", code=13)

  monkeypatch.setattr(collection, "create_index", _denied)

  with caplog.at_level("WARNING"):
    await service.add_session_to_memory(_session_with_texts("hello world"))
    response = await service.search_memory(
        app_name=APP, user_id=USER, query="hello"
    )

  assert response.memories
  warnings = [r for r in caplog.records if "createIndex" in r.message]
  assert len(warnings) == 1  # warned once, not per operation


@pytest.mark.asyncio
async def test_index_creation_errors_other_than_privilege_propagate(
    monkeypatch,
):
  """Unexpected createIndex failures are not swallowed."""
  client = mongomock.MongoClient()
  service = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  collection = client["test_db"]["memories"]

  def _boom(*args, **kwargs):
    raise RuntimeError("connection dropped")

  monkeypatch.setattr(collection, "create_index", _boom)

  with pytest.raises(RuntimeError, match="connection dropped"):
    await service.add_session_to_memory(_session_with_texts("hello world"))


@pytest.mark.asyncio
async def test_search_memory_respects_max_results():
  """A common keyword returns at most max_results memories."""
  service = MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(),
      database_name="test_db",
      max_results=2,
  )
  await service.add_session_to_memory(
      _session_with_texts(
          "coffee one", "coffee two", "coffee three", "coffee four"
      )
  )

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="coffee"
  )

  assert len(response.memories) == 2


@pytest.mark.asyncio
async def test_search_memory_ranks_by_keyword_overlap_then_recency():
  """Higher keyword overlap ranks first; recency breaks ties."""
  service = MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(), database_name="test_db"
  )
  await service.add_session_to_memory(
      _session_with_texts(
          "espresso machine",  # overlap 2, oldest
          "espresso machine with milk frother",  # overlap 3
          "my espresso machine",  # overlap 2, newest
      )
  )

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="espresso machine frother"
  )

  texts = [
      " ".join(part.text for part in m.content.parts if part.text)
      for m in response.memories
  ]
  assert texts == [
      "espresso machine with milk frother",
      "my espresso machine",
      "espresso machine",
  ]


def test_max_results_must_be_positive():
  """max_results bounds the search; zero or negative is a config error."""
  with pytest.raises(ValueError, match="max_results"):
    MongoDbMemoryService(
        mongo_client=mongomock.MongoClient(),
        database_name="test_db",
        max_results=0,
    )


def test_memory_ttl_days_must_be_positive_or_none():
  """memory_ttl_days of zero or negative is a config error; None is valid."""
  with pytest.raises(ValueError, match="memory_ttl_days"):
    MongoDbMemoryService(
        mongo_client=mongomock.MongoClient(),
        database_name="test_db",
        memory_ttl_days=0,
    )
  # None disables expiry and is accepted.
  MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(),
      database_name="test_db",
      memory_ttl_days=None,
  )


@pytest.mark.asyncio
async def test_ttl_index_created_by_default():
  """A TTL index on expire_at is created by default (7-day retention)."""
  client = mongomock.MongoClient()
  service = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  await service.add_session_to_memory(_session_with_texts("hello world"))

  indexes = client["test_db"]["memories"].index_information()
  assert "memory_ttl" in indexes
  assert indexes["memory_ttl"]["key"] == [("expire_at", 1)]
  assert indexes["memory_ttl"]["expireAfterSeconds"] == 0


@pytest.mark.asyncio
async def test_ingest_writes_expire_at_default_seven_days():
  """Each memory document stores an expire_at ~7 days in the future."""
  from datetime import datetime
  from datetime import timezone

  client = mongomock.MongoClient()
  service = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  before = datetime.now(timezone.utc)
  await service.add_session_to_memory(
      _session_with_texts("first memory", "second memory")
  )
  after = datetime.now(timezone.utc)

  docs = list(client["test_db"]["memories"].find({}))
  assert len(docs) == 2
  for doc in docs:
    expire_at = doc["expire_at"]
    # mongomock returns naive datetimes; treat as UTC for the comparison.
    if expire_at.tzinfo is None:
      expire_at = expire_at.replace(tzinfo=timezone.utc)
    # mongomock truncates microseconds, so allow a small slack below 7 days.
    assert expire_at - before >= timedelta(days=7) - timedelta(seconds=1)
    assert expire_at - after <= timedelta(days=7)


@pytest.mark.asyncio
async def test_ttl_days_configurable():
  """A custom memory_ttl_days shifts the stored expire_at accordingly."""
  from datetime import datetime
  from datetime import timezone

  client = mongomock.MongoClient()
  service = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", memory_ttl_days=30
  )
  before = datetime.now(timezone.utc)
  await service.add_session_to_memory(_session_with_texts("hello world"))

  doc = client["test_db"]["memories"].find_one({})
  expire_at = doc["expire_at"]
  if expire_at.tzinfo is None:
    expire_at = expire_at.replace(tzinfo=timezone.utc)
  # 30-day retention; mongomock truncates microseconds, so allow slack.
  assert expire_at - before >= timedelta(days=30) - timedelta(seconds=1)


@pytest.mark.asyncio
async def test_ttl_disabled_omits_expire_at_and_index():
  """memory_ttl_days=None stores no expire_at and creates no TTL index."""
  client = mongomock.MongoClient()
  service = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", memory_ttl_days=None
  )
  await service.add_session_to_memory(_session_with_texts("hello world"))

  doc = client["test_db"]["memories"].find_one({})
  assert "expire_at" not in doc
  indexes = client["test_db"]["memories"].index_information()
  assert "memory_ttl" not in indexes
  # The recall index is still created regardless of the TTL setting.
  assert "app_user_keywords" in indexes


def test_retrieval_mode_must_be_known():
  """An unknown retrieval mode is a config error."""
  with pytest.raises(ValueError, match="retrieval_mode"):
    MongoDbMemoryService(
        mongo_client=mongomock.MongoClient(),
        database_name="test_db",
        retrieval_mode="semantic",
    )


def _memory_doc(text: str, doc_id: str = "m1") -> dict:
  return {
      "_id": doc_id,
      "app_name": APP,
      "user_id": USER,
      "author": "user",
      "text": text,
      "content": (
          types.Content(role="user", parts=[types.Part(text=text)]).model_dump(
              exclude_none=True, mode="json"
          )
      ),
      "timestamp": 1.0,
  }


def _vector_service(**kwargs) -> MongoDbMemoryService:
  defaults = dict(
      database_name="test_db",
      ensure_indexes=False,
      retrieval_mode="vector",
      genai_client=mock.MagicMock(),
  )
  defaults.update(kwargs)
  return MongoDbMemoryService(**defaults)


@pytest.mark.asyncio
async def test_vector_mode_ingest_embeds_in_one_batch():
  """Vector mode embeds all event texts in one call and stores the vectors."""
  client = mongomock.MongoClient()
  genai_client = mock.MagicMock()
  embedding = mock.MagicMock()
  embedding.values = [0.1, 0.2]
  genai_client.aio.models.embed_content = mock.AsyncMock(
      return_value=mock.MagicMock(embeddings=[embedding, embedding])
  )
  service = _vector_service(mongo_client=client, genai_client=genai_client)

  await service.add_session_to_memory(
      _session_with_texts("first memory", "second memory")
  )

  embed_call = genai_client.aio.models.embed_content.await_args
  assert embed_call.kwargs["contents"] == ["first memory", "second memory"]
  assert embed_call.kwargs["model"] == "text-embedding-005"
  docs = list(client["test_db"]["memories"].find({}))
  assert len(docs) == 2
  assert all(doc["embedding"] == [0.1, 0.2] for doc in docs)
  assert all("text" in doc and "keywords" not in doc for doc in docs)


@pytest.mark.asyncio
async def test_vector_mode_recall_runs_vector_search(monkeypatch):
  """Vector mode recalls via $vectorSearch with an embedded query."""
  embed_query = mock.AsyncMock(return_value=[0.1, 0.2])
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._search_tool._embed_query", embed_query
  )
  client = mock.MagicMock()
  collection = client["test_db"]["memories"]
  collection.aggregate.return_value = iter([_memory_doc("nearest memory")])
  service = _vector_service(mongo_client=client)

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="coffee machine"
  )

  assert embed_query.await_args.kwargs["query"] == "coffee machine"
  stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
  assert stage["queryVector"] == [0.1, 0.2]
  assert stage["path"] == "embedding"
  assert stage["filter"] == {"app_name": APP, "user_id": USER}
  assert stage["limit"] == service._max_results
  texts = [
      " ".join(p.text for p in m.content.parts if p.text)
      for m in response.memories
  ]
  assert texts == ["nearest memory"]


@pytest.mark.asyncio
async def test_auto_mode_ingest_stores_plain_text_without_embeddings():
  """Auto mode stores plain text; Atlas embeds it, not the genai client."""
  client = mongomock.MongoClient()
  genai_client = mock.MagicMock()
  service = MongoDbMemoryService(
      mongo_client=client,
      database_name="test_db",
      ensure_indexes=False,
      retrieval_mode="auto",
      genai_client=genai_client,
  )

  await service.add_session_to_memory(_session_with_texts("just text"))

  doc = client["test_db"]["memories"].find_one({})
  assert doc["text"] == "just text"
  assert "embedding" not in doc and "keywords" not in doc
  genai_client.aio.models.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_auto_mode_recall_sends_query_text(monkeypatch):
  """Auto mode sends query.text to the autoEmbed index; no embed call."""
  embed_query = mock.AsyncMock(return_value=[0.1])
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._search_tool._embed_query", embed_query
  )
  client = mock.MagicMock()
  collection = client["test_db"]["memories"]
  collection.aggregate.return_value = iter([_memory_doc("atlas recall")])
  service = MongoDbMemoryService(
      mongo_client=client,
      database_name="test_db",
      ensure_indexes=False,
      retrieval_mode="auto",
  )

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="coffee machine"
  )

  embed_query.assert_not_awaited()
  stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
  assert stage["query"] == {"text": "coffee machine"}
  assert stage["path"] == "text"
  assert stage["filter"] == {"app_name": APP, "user_id": USER}
  assert len(response.memories) == 1


def test_vector_mode_creates_vector_index():
  """ensure_indexes in vector mode creates a vectorSearch index."""
  client = mock.MagicMock()
  service = _vector_service(mongo_client=client, ensure_indexes=True)

  service._ensure_indexes_once()

  # The collection is created first: createSearchIndexes rejects a
  # collection that does not exist (fresh deployment's first ingest).
  client["test_db"].create_collection.assert_called_once_with("memories")
  create = client["test_db"]["memories"].create_search_index
  assert create.call_count == 1
  model = create.call_args.kwargs["model"]
  fields = model.document["definition"]["fields"]
  assert {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine",
  } in fields
  assert {"type": "filter", "path": "app_name"} in fields
  assert model.document["name"] == "memory_vector_index"


def test_auto_mode_creates_autoembed_index():
  """ensure_indexes in auto mode creates an autoEmbed index on `text`."""
  client = mock.MagicMock()
  service = MongoDbMemoryService(
      mongo_client=client,
      database_name="test_db",
      ensure_indexes=True,
      retrieval_mode="auto",
  )

  service._ensure_indexes_once()

  create = client["test_db"]["memories"].create_search_index
  model = create.call_args.kwargs["model"]
  fields = model.document["definition"]["fields"]
  assert {
      "type": "autoEmbed",
      "modality": "text",
      "path": "text",
      "model": "voyage-4",
  } in fields


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

  # Each recommended index (recall + TTL) is created exactly once.
  assert sorted(create_calls) == ["app_user_keywords", "memory_ttl"]
