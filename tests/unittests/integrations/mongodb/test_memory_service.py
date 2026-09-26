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

from google.adk.events.event import Event
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import mongomock
import pytest

APP = "test_app"
USER = "test_user"


@pytest.fixture
def service():
  return MongoDbMemoryService(
      mongo_client=mongomock.MongoClient(), database_name="test_db"
  )


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
