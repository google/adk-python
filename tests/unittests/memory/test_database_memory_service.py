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

from google.adk.events.event import Event
from google.adk.memory.database_memory_service import DatabaseMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_OTHER_USER_ID = "another-user"

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
async def memory_service():
  service = DatabaseMemoryService("sqlite+aiosqlite:///:memory:")
  async with service:
    yield service


def test_extract_keywords():
  service = DatabaseMemoryService("sqlite+aiosqlite:///:memory:")
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
async def test_add_session_to_memory(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK toolkit"
  )
  assert result.memories
  assert len(result.memories) >= 1


@pytest.mark.asyncio
async def test_add_session_with_no_events(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_NO_EVENTS)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="anything"
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_add_session_to_memory_filters_no_content_events(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
  )
  assert len(result.memories) == 2
  authors = {m.author for m in result.memories}
  assert "user" in authors
  assert "model" in authors


@pytest.mark.asyncio
async def test_add_session_to_memory_skips_stop_words_only(memory_service):
  session = Session(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      id="session-stop",
      events=[
          Event(
              id="e-stop",
              invocation_id="inv-stop",
              author="user",
              content=types.Content(parts=[types.Part(text="the and or")]),
          ),
      ],
  )
  await memory_service.add_session_to_memory(session)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="random"
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_empty_query(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query=""
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_only_stop_words(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="the and or"
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_simple_match(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_2)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="Python"
  )
  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == "I like to code in Python."
  assert result.memories[0].author == "user"


@pytest.mark.asyncio
async def test_search_memory_case_insensitive(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="development"
  )
  assert len(result.memories) == 1
  assert (
      result.memories[0].content.parts[0].text
      == "I agree. The Agent Development Kit (ADK) rocks!"
  )


@pytest.mark.asyncio
async def test_search_memory_multiple_matches(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
  )
  assert len(result.memories) == 2
  texts = {m.content.parts[0].text for m in result.memories}
  assert "The ADK is a great toolkit." in texts
  assert "I agree. The Agent Development Kit (ADK) rocks!" in texts


@pytest.mark.asyncio
async def test_search_memory_no_match(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="nonexistent"
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_scoped_by_user(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_DIFFERENT_USER)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="secret"
  )
  assert not result.memories

  result_other = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_OTHER_USER_ID, query="secret"
  )
  assert len(result_other.memories) == 1
  assert result_other.memories[0].content.parts[0].text == "This is a secret."


@pytest.mark.asyncio
async def test_search_memory_deduplication(memory_service):
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
  )
  assert len(result.memories) == 2


@pytest.mark.asyncio
async def test_add_events_to_memory(memory_service):
  events = [MOCK_SESSION_1.events[0]]
  await memory_service.add_events_to_memory(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      session_id="session-1",
      events=events,
  )

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK toolkit"
  )
  assert len(result.memories) == 1
  assert result.memories[0].author == "user"


@pytest.mark.asyncio
async def test_add_events_to_memory_without_session_id(memory_service):
  events = [MOCK_SESSION_2.events[0]]
  await memory_service.add_events_to_memory(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      events=events,
  )

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="Python"
  )
  assert len(result.memories) == 1


@pytest.mark.asyncio
async def test_add_events_to_memory_skips_empty(memory_service):
  events = [
      Event(
          id="e-empty",
          invocation_id="inv-empty",
          author="user",
          timestamp=12345,
      ),
  ]
  await memory_service.add_events_to_memory(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      events=events,
  )

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="anything"
  )
  assert not result.memories


@pytest.mark.asyncio
async def test_context_manager():
  service = DatabaseMemoryService("sqlite+aiosqlite:///:memory:")
  async with service:
    await service.add_session_to_memory(MOCK_SESSION_1)
    result = await service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
    )
    assert result.memories


@pytest.mark.asyncio
async def test_file_based_sqlite(tmp_path):
  db_path = tmp_path / "test_memory.db"
  async with DatabaseMemoryService(f"sqlite+aiosqlite:///{db_path}") as service:
    await service.add_session_to_memory(MOCK_SESSION_1)
    result = await service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
    )
    assert len(result.memories) == 2

  async with DatabaseMemoryService(
      f"sqlite+aiosqlite:///{db_path}"
  ) as service2:
    result = await service2.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="ADK"
    )
    assert len(result.memories) == 2


def test_invalid_db_url():
  with pytest.raises(ValueError, match="Invalid database URL"):
    DatabaseMemoryService("not-a-valid-url://")
