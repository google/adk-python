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

"""Tests for MongoDbMemoryService."""

from __future__ import annotations

import copy
from typing import Any
from unittest import mock

from google.adk.events.event import Event
from google.adk.integrations.mongodb import _client
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.sessions.session import Session
from google.genai import types
from pydantic import ValidationError
import pytest

APP = "test_app"
USER = "test_user"


def _doc_key(doc_id: Any) -> Any:
  if isinstance(doc_id, dict):
    return tuple(doc_id.items())
  return doc_id


class _FakeCollection:
  """In-memory stand-in for a PyMongo Collection."""

  def __init__(self) -> None:
    self._docs: dict[Any, dict[str, Any]] = {}
    self._indexes: dict[str, dict[str, Any]] = {
        "_id_": {"key": [("_id", 1)]},
    }
    self.create_index_calls: int = 0
    self.create_index_error: Exception | None = None
    self.bulk_write_calls: list[tuple[list[Any], bool]] = []
    self.last_pipeline: list[dict[str, Any]] | None = None

  def create_index(
      self, keys: list[tuple[str, int]], *, name: str | None = None
  ) -> str:
    self.create_index_calls += 1
    if self.create_index_error is not None:
      raise self.create_index_error
    index_name = name or "_".join(f"{k}_{v}" for k, v in keys)
    self._indexes[index_name] = {"key": list(keys)}
    return index_name

  def index_information(self) -> dict[str, dict[str, Any]]:
    return copy.deepcopy(self._indexes)

  def replace_one(
      self,
      filter_doc: dict[str, Any],
      replacement: dict[str, Any],
      *,
      upsert: bool = False,
  ) -> None:
    doc_id = filter_doc.get("_id")
    key = _doc_key(doc_id)
    if key in self._docs or upsert:
      stored = copy.deepcopy(replacement)
      stored["_id"] = copy.deepcopy(doc_id)
      self._docs[key] = stored

  def bulk_write(
      self,
      requests: list[Any],
      *,
      ordered: bool = True,
  ) -> None:
    if not requests:
      raise ValueError("requests must not be empty")
    self.bulk_write_calls.append((list(requests), ordered))
    for req in requests:
      self.replace_one(req._filter, req._doc, upsert=req._upsert)

  def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    self.last_pipeline = copy.deepcopy(pipeline)
    docs = [copy.deepcopy(doc) for doc in self._docs.values()]
    for stage in pipeline:
      if "$match" in stage:
        docs = [doc for doc in docs if _matches(doc, stage["$match"])]
      elif "$addFields" in stage:
        for doc in docs:
          for field, expr in stage["$addFields"].items():
            doc[field] = _eval_expr(doc, expr)
      elif "$sort" in stage:
        for sort_key, direction in reversed(list(stage["$sort"].items())):
          docs.sort(
              key=lambda d, k=sort_key: (
                  float(d[k]) if isinstance(d.get(k), int | float) else 0.0
              ),
              reverse=(direction < 0),
          )
      elif "$limit" in stage:
        docs = docs[: stage["$limit"]]
      else:
        raise NotImplementedError(f"Unsupported stage: {stage}")
    return docs


def _eval_expr(doc: dict[str, Any], expr: Any) -> Any:
  if isinstance(expr, str) and expr.startswith("$"):
    return doc.get(expr[1:])
  if isinstance(expr, dict):
    if "$size" in expr:
      return len(_eval_expr(doc, expr["$size"]))
    if "$setIntersection" in expr:
      operands = [_eval_expr(doc, op) for op in expr["$setIntersection"]]
      if not operands:
        return []
      result = set(operands[0] or [])
      for op in operands[1:]:
        result &= set(op or [])
      return list(result)
  return expr


def _matches(doc: dict[str, Any], filter_doc: dict[str, Any]) -> bool:
  for key, condition in filter_doc.items():
    val = doc.get(key)
    if isinstance(condition, dict) and "$in" in condition:
      candidates = condition["$in"]
      if isinstance(val, list):
        if not any(item in candidates for item in val):
          return False
      elif val not in candidates:
        return False
    elif val != condition:
      return False
  return True


class _FakeDatabase:
  """In-memory stand-in for a PyMongo Database."""

  def __init__(self) -> None:
    self._collections: dict[str, _FakeCollection] = {}

  def __getitem__(self, name: str) -> _FakeCollection:
    if name not in self._collections:
      self._collections[name] = _FakeCollection()
    return self._collections[name]


class _FakeMongoClient:
  """In-memory stand-in for pymongo.MongoClient."""

  def __init__(self) -> None:
    self._dbs: dict[str, _FakeDatabase] = {}
    self.closed: bool = False

  def __getitem__(self, name: str) -> _FakeDatabase:
    if name not in self._dbs:
      self._dbs[name] = _FakeDatabase()
    return self._dbs[name]

  def close(self) -> None:
    self.closed = True


@pytest.fixture
def service() -> MongoDbMemoryService:
  client: Any = _FakeMongoClient()
  return MongoDbMemoryService(mongo_client=client, database_name="test_db")


def _session_with_texts(*texts: str, session_id: str = "s1") -> Session:
  events = [
      Event(
          id=f"e{i}",
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


class _OperationFailure(Exception):
  """Stand-in for pymongo.errors.OperationFailure."""

  def __init__(self, message: str, code: int | None = None) -> None:
    super().__init__(message)
    self.code = code


async def test_recall_index_created_once_by_default() -> None:
  """The app/user/keywords index is created once without opting in."""
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  await svc.add_session_to_memory(_session_with_texts("hello world"))
  await svc.search_memory(app_name=APP, user_id=USER, query="hello")

  collection = client["test_db"]["memories"]
  indexes = collection.index_information()
  assert "app_user_keywords" in indexes
  assert collection.create_index_calls == 1


async def test_search_without_keywords_skips_index_creation() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  await svc.search_memory(app_name=APP, user_id=USER, query="the")

  assert client["test_db"]["memories"].create_index_calls == 0


async def test_ensure_indexes_false_skips_index_creation() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", ensure_indexes=False
  )
  await svc.add_session_to_memory(_session_with_texts("hello world"))
  await svc.search_memory(app_name=APP, user_id=USER, query="hello")

  collection = client["test_db"]["memories"]
  assert "app_user_keywords" not in collection.index_information()
  assert collection.create_index_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        _OperationFailure("denied", code=13),
        _OperationFailure("user is not authorized on my_app to execute"),
        _OperationFailure(
            "user is not allowed to do action [createIndex] on"
            " [my_app.memories]",
            code=8000,
        ),
    ],
)
async def test_index_privilege_error_warns_once_and_falls_back(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  collection = client["test_db"]["memories"]
  collection.create_index_error = error

  with caplog.at_level("WARNING"):
    await svc.add_session_to_memory(_session_with_texts("hello world"))
    response = await svc.search_memory(
        app_name=APP, user_id=USER, query="hello"
    )

  assert len(response.memories) == 1
  assert collection.create_index_calls == 1
  warnings = [r for r in caplog.records if "createIndex" in r.getMessage()]
  assert len(warnings) == 1


async def test_existing_index_under_other_name_is_accepted(
    caplog: pytest.LogCaptureFixture,
) -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  collection = client["test_db"]["memories"]
  collection.create_index_error = _OperationFailure(
      "Index already exists with a different name: my_index", code=85
  )

  with caplog.at_level("WARNING"):
    await svc.add_session_to_memory(_session_with_texts("hello world"))
    response = await svc.search_memory(
        app_name=APP, user_id=USER, query="hello"
    )

  assert len(response.memories) == 1
  assert collection.create_index_calls == 1
  assert not [r for r in caplog.records if "createIndex" in r.getMessage()]


async def test_index_creation_errors_other_than_privilege_propagate() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  collection = client["test_db"]["memories"]
  collection.create_index_error = _OperationFailure("connection dropped", 6)

  with pytest.raises(_OperationFailure, match="connection dropped"):
    await svc.add_session_to_memory(_session_with_texts("hello world"))

  # The failure is not cached: the next operation retries index creation.
  collection.create_index_error = None
  await svc.add_session_to_memory(_session_with_texts("hello world"))
  assert collection.create_index_calls == 2
  assert "app_user_keywords" in collection.index_information()


async def test_add_session_then_search_by_keyword(
    service: MongoDbMemoryService,
) -> None:
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
  assert response.memories[0].content.parts is not None
  assert response.memories[0].content.parts[0].text is not None
  assert "hiking" in response.memories[0].content.parts[0].text
  assert response.memories[0].author == "user"
  assert response.memories[0].id == "e0"

  # A query can match several memories.
  response = await service.search_memory(
      app_name=APP, user_id=USER, query="trail mountains"
  )
  assert len(response.memories) == 2


async def test_search_memory_scopes_by_app_and_user(
    service: MongoDbMemoryService,
) -> None:
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


async def test_slash_in_keys_does_not_collide(
    service: MongoDbMemoryService,
) -> None:
  s1 = Session(
      id="d/e",
      app_name="a/b",
      user_id="c",
      events=[
          Event(
              id="f",
              invocation_id="inv1",
              author="user",
              content=types.Content(
                  role="user", parts=[types.Part(text="hiking alpha")]
              ),
              timestamp=1.0,
          )
      ],
  )
  s2 = Session(
      id="d",
      app_name="a",
      user_id="b/c",
      events=[
          Event(
              id="e/f",
              invocation_id="inv2",
              author="user",
              content=types.Content(
                  role="user", parts=[types.Part(text="hiking beta")]
              ),
              timestamp=2.0,
          )
      ],
  )
  await service.add_session_to_memory(s1)
  await service.add_session_to_memory(s2)

  r1 = await service.search_memory(app_name="a/b", user_id="c", query="hiking")
  assert len(r1.memories) == 1
  assert r1.memories[0].content.parts is not None
  assert r1.memories[0].content.parts[0].text == "hiking alpha"
  assert r1.memories[0].id == "f"

  r2 = await service.search_memory(app_name="a", user_id="b/c", query="hiking")
  assert len(r2.memories) == 1
  assert r2.memories[0].content.parts is not None
  assert r2.memories[0].content.parts[0].text == "hiking beta"
  assert r2.memories[0].id == "e/f"


async def test_search_memory_ranks_by_keyword_overlap_and_caps_results() -> (
    None
):
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(
      mongo_client=client, database_name="test_db", max_results=2
  )
  await svc.add_session_to_memory(
      _session_with_texts(
          "alpha only",
          "alpha beta gamma all three",
          "alpha beta two words",
      )
  )

  response = await svc.search_memory(
      app_name=APP, user_id=USER, query="alpha beta gamma"
  )
  assert [
      m.content.parts[0].text
      for m in response.memories
      if m.content.parts and m.content.parts[0].text
  ] == [
      "alpha beta gamma all three",
      "alpha beta two words",
  ]
  pipeline = client["test_db"]["memories"].last_pipeline
  assert pipeline is not None
  assert {"$sort": {"_score": -1, "timestamp": -1}} in pipeline
  assert {"$limit": 2} in pipeline

  # A one-word query ties on _score=1 across all three events, so the
  # timestamp tie-breaker returns the two most recent (e2 at 3.0, e1 at 2.0).
  tie_response = await svc.search_memory(
      app_name=APP, user_id=USER, query="alpha"
  )
  assert [m.id for m in tie_response.memories] == ["e2", "e1"]


async def test_add_events_to_memory_batches_and_scopes_session() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  session = _session_with_texts(
      "hiking in the Alps", "camping by the lake", session_id="s1"
  )

  await svc.add_events_to_memory(
      app_name=APP,
      user_id=USER,
      events=session.events,
      session_id=session.id,
  )

  collection = client["test_db"]["memories"]
  assert len(collection.bulk_write_calls) == 1
  requests, ordered = collection.bulk_write_calls[0]
  assert len(requests) == 2
  assert ordered is False

  # Incremental event ingestion into the same session adds only the delta
  # while re-adding the full session remains idempotent.
  turn2_event = Event(
      id="e2",
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user", parts=[types.Part(text="kayaking down the river")]
      ),
      timestamp=3.0,
  )
  await svc.add_events_to_memory(
      app_name=APP,
      user_id=USER,
      events=[turn2_event],
      session_id="s1",
  )
  session.events.append(turn2_event)
  await svc.add_session_to_memory(session)
  assert len(collection.bulk_write_calls) == 3

  response = await svc.search_memory(
      app_name=APP, user_id=USER, query="hiking camping kayaking"
  )
  assert [m.id for m in response.memories] == ["e2", "e1", "e0"]

  # Without session_id, events are scoped under __unknown_session_id__.
  await svc.add_events_to_memory(
      app_name=APP,
      user_id=USER,
      events=[
          Event(
              id="anon1",
              invocation_id="inv3",
              author="user",
              content=types.Content(
                  role="user", parts=[types.Part(text="bouldering indoors")]
              ),
              timestamp=4.0,
          )
      ],
  )
  assert (
      _doc_key({
          "app_name": APP,
          "user_id": USER,
          "session_id": "__unknown_session_id__",
          "event_id": "anon1",
      })
      in collection._docs
  )

  # Empty or stop-word-only event lists skip bulk_write completely.
  calls_before = len(collection.bulk_write_calls)
  await svc.add_events_to_memory(app_name=APP, user_id=USER, events=[])
  assert len(collection.bulk_write_calls) == calls_before


async def test_unicode_and_nfc_keyword_extraction_and_search(
    service: MongoDbMemoryService,
) -> None:
  await service.add_session_to_memory(
      _session_with_texts(
          "Visit cafe\u0301 in Paris",
          "日本語のイベント",
          "Привет мир",
      )
  )

  # Precomposed NFC query matches decomposed NFD stored event, while a
  # truncated ASCII stem does not.
  cafe_response = await service.search_memory(
      app_name=APP, user_id=USER, query="café"
  )
  assert len(cafe_response.memories) == 1
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="c" + "af")
  ).memories == []

  # CJK and Cyrillic events are indexed and searchable.
  cjk_response = await service.search_memory(
      app_name=APP, user_id=USER, query="日本語のイベント"
  )
  assert len(cjk_response.memories) == 1
  cyrillic_response = await service.search_memory(
      app_name=APP, user_id=USER, query="привет"
  )
  assert len(cyrillic_response.memories) == 1


async def test_latin_word_inside_non_latin_text_is_searchable(
    service: MongoDbMemoryService,
) -> None:
  await service.add_session_to_memory(_session_with_texts("使用MongoDB存储"))

  for query in ("MongoDB", "存储", "使用MongoDB存储"):
    response = await service.search_memory(
        app_name=APP, user_id=USER, query=query
    )
    assert len(response.memories) == 1, query
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="mongo")
  ).memories == []


async def test_lowercasing_does_not_store_bare_combining_marks() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  await svc.add_session_to_memory(_session_with_texts("İstanbul日本"))

  (doc,) = client["test_db"]["memories"]._docs.values()
  assert "\u0307" not in doc["keywords"]
  assert "i\u0307stanbul" in doc["keywords"]
  assert "日本" in doc["keywords"]
  response = await svc.search_memory(
      app_name=APP, user_id=USER, query="İstanbul"
  )
  assert len(response.memories) == 1


async def test_reingesting_session_does_not_duplicate(
    service: MongoDbMemoryService,
) -> None:
  session = _session_with_texts("I love hiking")
  await service.add_session_to_memory(session)
  await service.add_session_to_memory(session)

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1


async def test_search_memory_deduplicates_identical_events(
    service: MongoDbMemoryService,
) -> None:
  events = [
      Event(
          id="e1",
          invocation_id="inv1",
          author="user",
          content=types.Content(
              role="user", parts=[types.Part(text="I love hiking")]
          ),
          timestamp=10.0,
      ),
      Event(
          id="e2",
          invocation_id="inv2",
          author="user",
          content=types.Content(
              role="user", parts=[types.Part(text="I love hiking")]
          ),
          timestamp=10.0,
      ),
  ]
  session = Session(id="s1", app_name=APP, user_id=USER, events=events)
  await service.add_session_to_memory(session)

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1


async def test_search_memory_ignores_stop_words(
    service: MongoDbMemoryService,
) -> None:
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


async def test_custom_stop_words_and_collection() -> None:
  client: Any = _FakeMongoClient()
  svc = MongoDbMemoryService(
      mongo_client=client,
      database_name="test_db",
      memories_collection="custom_memories",
      stop_words={"CustomStop"},
  )
  await svc.add_session_to_memory(_session_with_texts("customstop keepword"))

  assert (
      await svc.search_memory(app_name=APP, user_id=USER, query="customstop")
  ).memories == []
  response = await svc.search_memory(
      app_name=APP, user_id=USER, query="keepword"
  )
  assert len(response.memories) == 1
  assert (
      _doc_key({
          "app_name": APP,
          "user_id": USER,
          "session_id": "s1",
          "event_id": "e0",
      })
      in client["test_db"]["custom_memories"]._docs
  )


async def test_default_stop_words_not_aliased_across_instances() -> None:
  client: Any = _FakeMongoClient()
  svc1 = MongoDbMemoryService(mongo_client=client, database_name="test_db")
  svc2 = MongoDbMemoryService(mongo_client=client, database_name="test_db")

  svc1._stop_words.add("hiking")
  assert "hiking" not in svc2._stop_words

  await svc2.add_session_to_memory(_session_with_texts("hiking trail"))
  response = await svc2.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1


async def test_search_memory_empty_or_stop_word_query(
    service: MongoDbMemoryService,
) -> None:
  await service.add_session_to_memory(_session_with_texts("remember hiking"))
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="")
  ).memories == []
  assert (
      await service.search_memory(app_name=APP, user_id=USER, query="?!")
  ).memories == []


async def test_add_session_skips_events_without_text_or_thought_parts(
    service: MongoDbMemoryService,
) -> None:
  session = Session(id="s1", app_name=APP, user_id=USER)
  session.events.append(Event(id="e1", invocation_id="inv", author="agent"))
  session.events.append(
      Event(
          id="e2",
          invocation_id="inv",
          author="agent",
          content=types.Content(
              role="model",
              parts=[types.Part(text="secretthought", thought=True)],
          ),
      )
  )
  session.events.append(
      Event(
          id="e3",
          invocation_id="inv",
          author="agent",
          content=types.Content(
              role="model",
              parts=[
                  types.Part(text="hiddenreasoning", thought=True),
                  types.Part(text="visibleanswer"),
              ],
          ),
      )
  )
  await service.add_session_to_memory(session)
  assert (
      await service.search_memory(
          app_name=APP, user_id=USER, query="secretthought"
      )
  ).memories == []
  assert (
      await service.search_memory(
          app_name=APP, user_id=USER, query="hiddenreasoning"
      )
  ).memories == []
  response = await service.search_memory(
      app_name=APP, user_id=USER, query="visibleanswer"
  )
  assert len(response.memories) == 1
  assert response.memories[0].content.parts == [
      types.Part(text="visibleanswer")
  ]


async def test_search_memory_skips_corrupt_documents(
    service: MongoDbMemoryService,
) -> None:
  await service.add_session_to_memory(_session_with_texts("remember hiking"))
  service._memories().replace_one(
      {"_id": "corrupt"},
      {
          "app_name": APP,
          "user_id": USER,
          "keywords": ["hiking"],
          "content": "not-a-valid-content-dict",
      },
      upsert=True,
  )
  service._memories().replace_one(
      {"_id": "missing_content"},
      {
          "app_name": APP,
          "user_id": USER,
          "keywords": ["hiking"],
      },
      upsert=True,
  )

  response = await service.search_memory(
      app_name=APP, user_id=USER, query="hiking"
  )
  assert len(response.memories) == 1

  # Non-content errors (timestamp formatting or MemoryEntry validation)
  # propagate instead of being swallowed as corrupt content.
  service._memories().replace_one(
      {"_id": "bad_timestamp"},
      {
          "app_name": APP,
          "user_id": USER,
          "keywords": ["camping"],
          "content": {"role": "user", "parts": [{"text": "camping"}]},
          "timestamp": "not-a-float",
      },
      upsert=True,
  )
  with pytest.raises(TypeError):
    await service.search_memory(app_name=APP, user_id=USER, query="camping")

  service._memories().replace_one(
      {"_id": "bad_author"},
      {
          "app_name": APP,
          "user_id": USER,
          "keywords": ["kayaking"],
          "content": {"role": "user", "parts": [{"text": "kayaking"}]},
          "author": 123,
      },
      upsert=True,
  )
  with pytest.raises(ValidationError):
    await service.search_memory(app_name=APP, user_id=USER, query="kayaking")


async def test_close_only_closes_owned_client() -> None:
  caller_client: Any = _FakeMongoClient()
  svc_unowned = MongoDbMemoryService(
      mongo_client=caller_client, database_name="test_db"
  )
  await svc_unowned.close()
  assert not caller_client.closed

  owned_client = _FakeMongoClient()
  with mock.patch.object(
      _client, "get_mongo_client", return_value=owned_client
  ):
    svc_owned = MongoDbMemoryService(
        connection_string="mongodb://localhost:27017", database_name="test_db"
    )
  await svc_owned.close()
  assert owned_client.closed


def test_constructor_validates_client_args() -> None:
  client: Any = _FakeMongoClient()
  with pytest.raises(ValueError, match="Either `connection_string`"):
    MongoDbMemoryService(database_name="db")
  with pytest.raises(ValueError, match="Only one of"):
    MongoDbMemoryService(
        database_name="db",
        mongo_client=client,
        connection_string="mongodb://localhost:27017",
    )
  with pytest.raises(ValueError, match="max_results must be positive"):
    MongoDbMemoryService(database_name="db", mongo_client=client, max_results=0)


def test_invalid_max_results_does_not_create_client() -> None:
  with mock.patch.object(_client, "get_mongo_client") as get_mongo_client:
    with pytest.raises(ValueError, match="max_results must be positive"):
      MongoDbMemoryService(
          connection_string="mongodb://localhost:27017",
          database_name="db",
          max_results=0,
      )
  get_mongo_client.assert_not_called()
