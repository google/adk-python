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

"""Unit tests for PgVectorMemoryService.

The tests run against an in-process fake connection pool and a deterministic
bag-of-words embedder, so they exercise the service's ingestion, ranking, and
serialization logic without a live PostgreSQL database, the psycopg driver, or
any network calls.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from google.adk.events.event import Event
from google.adk.integrations.pgvector import _pgvector_memory_service
from google.adk.integrations.pgvector import PgVectorMemoryService
from google.adk.integrations.pgvector import PgVectorMemoryServiceConfig
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types
import pytest

_DIM = 64


@pytest.fixture(autouse=True)
def _no_vector_registration(monkeypatch):
  """Skips pgvector's connection type registration in unit tests.

  The fake connection is not a real psycopg connection, so registering the
  pgvector type (which queries a live database) is neither possible nor
  meaningful here; that path is covered by the end-to-end tests. Forcing it off
  also keeps the suite hermetic whether or not pgvector is installed, matching
  CI where the optional driver is absent.
  """
  monkeypatch.setattr(
      _pgvector_memory_service, "register_vector_async", None, raising=False
  )


def _bag_of_words_embedder(dim: int = _DIM):
  """Returns a deterministic embedder that hashes tokens into buckets.

  Texts that share words get overlapping vectors, so cosine distance ranks a
  query nearest the events that share its words - enough to test ranking
  without a real embedding model.
  """

  async def embed(texts):
    vectors = []
    for text in texts:
      vector = [0.0] * dim
      for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vector[bucket] += 1.0
      vectors.append(vector)
    return vectors

  return embed


def _cosine_distance(a, b) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(y * y for y in b))
  if norm_a == 0 or norm_b == 0:
    return 1.0
  return 1.0 - dot / (norm_a * norm_b)


class _FakeCursor:

  def __init__(self, rows):
    self._rows = rows

  async def fetchall(self):
    return self._rows


class _FakeConnection:
  """Interprets the small set of statements PgVectorMemoryService issues."""

  def __init__(self, store: dict[str, tuple[Any, ...]]):
    self._store = store

  async def __aenter__(self):
    return self

  async def __aexit__(self, *exc):
    return False

  async def execute(self, sql: str, params: tuple[Any, ...] | None = None):
    keyword = sql.strip().split(None, 1)[0].upper()
    if keyword in ("CREATE",):
      return _FakeCursor([])
    if keyword == "INSERT":
      # params: id, app, user, session, author, timestamp, text, content,
      # custom_metadata, embedding
      self._store[params[0]] = params
      return _FakeCursor([])
    if keyword == "SELECT":
      # params: embedding, app_name, user_id, top_k
      query_vec, app_name, user_id, top_k = params
      scored = []
      for row in self._store.values():
        if row[1] != app_name or row[2] != user_id:
          continue
        distance = _cosine_distance(query_vec, row[9])
        # author, timestamp, content, custom_metadata, id, distance
        scored.append((row[4], row[5], row[7], row[8], row[0], distance))
      scored.sort(key=lambda r: r[5])
      return _FakeCursor(scored[:top_k])
    return _FakeCursor([])


class _FakePgVectorPool:

  def __init__(self):
    self.store: dict[str, tuple[Any, ...]] = {}
    self.open_count = 0
    self.closed = False

  async def open(self):
    self.open_count += 1

  async def close(self):
    self.closed = True

  def connection(self):
    return _FakeConnection(self.store)


def _event(author: str, text: str, timestamp: float = 12345.0) -> Event:
  return Event(
      author=author,
      timestamp=timestamp,
      content=types.Content(parts=[types.Part(text=text)]),
  )


def _session(app_name: str, user_id: str, session_id: str, events) -> Session:
  return Session(
      app_name=app_name,
      user_id=user_id,
      id=session_id,
      events=events,
  )


def _make_service(pool, **config_kwargs):
  config = PgVectorMemoryServiceConfig(
      dsn="postgresql://ignored",
      embedding_dimension=_DIM,
      **config_kwargs,
  )
  return PgVectorMemoryService(
      config,
      connection_pool=pool,
      embedder=_bag_of_words_embedder(),
  )


@pytest.mark.asyncio
async def test_add_session_and_search_returns_semantically_closest():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  session = _session(
      "app1",
      "user1",
      "s1",
      [
          _event("user", "How do I dispute a billing charge on my invoice?"),
          _event("model", "The weather forecast for tomorrow is sunny."),
          _event("user", "Who won the basketball game last night?"),
      ],
  )

  await service.add_session_to_memory(session)
  response = await service.search_memory(
      app_name="app1", user_id="user1", query="billing invoice dispute"
  )

  assert response.memories
  top = response.memories[0]
  assert "billing" in top.content.parts[0].text
  assert top.author == "user"


@pytest.mark.asyncio
async def test_search_ignores_other_users_and_apps():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  await service.add_session_to_memory(
      _session("app1", "user1", "s1", [_event("user", "billing invoice")])
  )
  await service.add_session_to_memory(
      _session("app1", "user2", "s2", [_event("user", "billing invoice")])
  )

  response = await service.search_memory(
      app_name="app1", user_id="user2", query="billing"
  )

  assert len(response.memories) == 1


@pytest.mark.asyncio
async def test_empty_query_returns_no_memories():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  await service.add_session_to_memory(
      _session("app1", "user1", "s1", [_event("user", "billing invoice")])
  )

  response = await service.search_memory(
      app_name="app1", user_id="user1", query="   "
  )

  assert response.memories == []


@pytest.mark.asyncio
async def test_events_without_text_are_skipped():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  session = _session(
      "app1",
      "user1",
      "s1",
      [
          _event("user", "billing invoice"),
          Event(author="user", timestamp=1.0),  # no content
      ],
  )

  await service.add_session_to_memory(session)

  assert len(pool.store) == 1


@pytest.mark.asyncio
async def test_reingesting_a_session_is_idempotent():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  session = _session(
      "app1",
      "user1",
      "s1",
      [_event("user", "billing invoice"), _event("model", "sunny weather")],
  )

  await service.add_session_to_memory(session)
  await service.add_session_to_memory(session)

  # Two text events, ingested twice, must not create duplicate rows.
  assert len(pool.store) == 2


@pytest.mark.asyncio
async def test_add_events_to_memory_persists_delta():
  pool = _FakePgVectorPool()
  service = _make_service(pool)

  await service.add_events_to_memory(
      app_name="app1",
      user_id="user1",
      events=[_event("user", "billing invoice dispute")],
      session_id="s1",
  )

  response = await service.search_memory(
      app_name="app1", user_id="user1", query="billing"
  )
  assert len(response.memories) == 1


@pytest.mark.asyncio
async def test_add_memory_writes_explicit_entries():
  pool = _FakePgVectorPool()
  service = _make_service(pool)

  await service.add_memory(
      app_name="app1",
      user_id="user1",
      memories=[
          MemoryEntry(
              content=types.Content(
                  parts=[types.Part(text="the user prefers dark mode")]
              ),
              author="user",
              custom_metadata={"source": "profile"},
          )
      ],
  )

  response = await service.search_memory(
      app_name="app1", user_id="user1", query="dark mode preference"
  )
  assert len(response.memories) == 1
  assert response.memories[0].custom_metadata["source"] == "profile"


@pytest.mark.asyncio
async def test_distance_threshold_drops_unrelated_memories():
  pool = _FakePgVectorPool()
  service = _make_service(pool, distance_threshold=0.5)
  await service.add_session_to_memory(
      _session(
          "app1",
          "user1",
          "s1",
          [
              _event("user", "billing invoice dispute charge"),
              _event("model", "basketball score tonight"),
          ],
      )
  )

  response = await service.search_memory(
      app_name="app1", user_id="user1", query="billing invoice dispute charge"
  )

  # Only the near-identical billing memory is within the distance ceiling.
  assert len(response.memories) == 1
  assert "billing" in response.memories[0].content.parts[0].text


@pytest.mark.asyncio
async def test_content_survives_serialization_roundtrip():
  pool = _FakePgVectorPool()
  service = _make_service(pool)
  await service.add_session_to_memory(
      _session(
          "app1",
          "user1",
          "s1",
          [_event("model", "billing details: multi part reply")],
      )
  )

  response = await service.search_memory(
      app_name="app1", user_id="user1", query="billing details"
  )

  memory = response.memories[0]
  assert isinstance(memory.content, types.Content)
  assert memory.content.parts[0].text == "billing details: multi part reply"
  assert memory.timestamp is not None


def test_missing_driver_raises_helpful_error(monkeypatch):
  monkeypatch.setattr(_pgvector_memory_service, "AsyncConnectionPool", None)
  service = PgVectorMemoryService(
      PgVectorMemoryServiceConfig(dsn="postgresql://x"),
      embedder=_bag_of_words_embedder(),
  )

  with pytest.raises(ImportError, match="google-adk\\[pgvector\\]"):
    service._get_pool()


def test_dsn_required_when_no_pool(monkeypatch):
  # Pretend the driver is installed so the missing dsn is what fails.
  monkeypatch.setattr(_pgvector_memory_service, "AsyncConnectionPool", object)
  service = PgVectorMemoryService(
      PgVectorMemoryServiceConfig(dsn=None),
      embedder=_bag_of_words_embedder(),
  )

  with pytest.raises(ValueError, match="dsn is required"):
    service._get_pool()


@pytest.mark.asyncio
async def test_owned_pool_is_opened_once_and_closed(monkeypatch):
  fake = _FakePgVectorPool()
  monkeypatch.setattr(
      _pgvector_memory_service,
      "AsyncConnectionPool",
      lambda dsn, open: fake,
  )
  # No pool injected: the service creates and owns it, so it must open it.
  service = PgVectorMemoryService(
      PgVectorMemoryServiceConfig(
          dsn="postgresql://x", embedding_dimension=_DIM
      ),
      embedder=_bag_of_words_embedder(),
  )

  await service.add_session_to_memory(
      _session("app1", "user1", "s1", [_event("user", "billing invoice")])
  )
  await service.search_memory(app_name="app1", user_id="user1", query="billing")

  assert fake.open_count == 1  # opened once, not per operation
  await service.close()
  assert fake.closed is True


@pytest.mark.asyncio
async def test_injected_pool_is_not_opened_or_closed():
  # A caller-supplied pool is managed by the caller, so the service must not
  # open or close it.
  pool = _FakePgVectorPool()
  service = _make_service(pool)

  await service.add_session_to_memory(
      _session("app1", "user1", "s1", [_event("user", "billing invoice")])
  )
  await service.close()

  assert pool.open_count == 0
  assert pool.closed is False
