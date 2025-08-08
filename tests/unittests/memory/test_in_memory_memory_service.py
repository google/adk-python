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

from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'
MOCK_OTHER_USER_ID = 'another-user'

MOCK_SESSION_1 = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id='session-1',
    last_update_time=1000,
    events=[
        Event(
            id='event-1a',
            invocation_id='inv-1',
            author='user',
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text='The ADK is a great toolkit.')]
            ),
        ),
        # Event with no content, should be ignored by the service
        Event(
            id='event-1b',
            invocation_id='inv-2',
            author='user',
            timestamp=12346,
        ),
        Event(
            id='event-1c',
            invocation_id='inv-3',
            author='model',
            timestamp=12347,
            content=types.Content(
                parts=[
                    types.Part(
                        text='I agree. The Agent Development Kit (ADK) rocks!'
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_2 = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id='session-2',
    last_update_time=2000,
    events=[
        Event(
            id='event-2a',
            invocation_id='inv-4',
            author='user',
            timestamp=54321,
            content=types.Content(
                parts=[types.Part(text='I like to code in Python.')]
            ),
        ),
    ],
)

MOCK_SESSION_DIFFERENT_USER = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_OTHER_USER_ID,
    id='session-3',
    last_update_time=3000,
    events=[
        Event(
            id='event-3a',
            invocation_id='inv-5',
            author='user',
            timestamp=60000,
            content=types.Content(parts=[types.Part(text='This is a secret.')]),
        ),
    ],
)

MOCK_SESSION_WITH_NO_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id='session-4',
    last_update_time=4000,
)


@pytest.mark.asyncio
async def test_add_session_to_memory():
  """Tests that a session with events is correctly added to memory."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  user_key = f'{MOCK_APP_NAME}/{MOCK_USER_ID}'
  assert user_key in memory_service._session_events
  session_memory = memory_service._session_events[user_key]
  assert MOCK_SESSION_1.id in session_memory
  # Check that the event with no content was filtered out
  assert len(session_memory[MOCK_SESSION_1.id]) == 2
  assert session_memory[MOCK_SESSION_1.id][0].id == 'event-1a'
  assert session_memory[MOCK_SESSION_1.id][1].id == 'event-1c'


@pytest.mark.asyncio
async def test_add_session_with_no_events_to_memory():
  """Tests that adding a session with no events does not cause an error."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_NO_EVENTS)

  user_key = f'{MOCK_APP_NAME}/{MOCK_USER_ID}'
  assert user_key in memory_service._session_events
  session_memory = memory_service._session_events[user_key]
  assert MOCK_SESSION_WITH_NO_EVENTS.id in session_memory
  assert not session_memory[MOCK_SESSION_WITH_NO_EVENTS.id]


@pytest.mark.asyncio
async def test_search_memory_simple_match():
  """Tests a simple keyword search that should find a match."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_2)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='Python'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'I like to code in Python.'
  assert result.memories[0].author == 'user'


@pytest.mark.asyncio
async def test_search_memory_case_insensitive_match():
  """Tests that search is case-insensitive."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='development'
  )

  assert len(result.memories) == 1
  assert (
      result.memories[0].content.parts[0].text
      == 'I agree. The Agent Development Kit (ADK) rocks!'
  )


@pytest.mark.asyncio
async def test_search_memory_multiple_matches():
  """Tests that a query can match multiple events."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='How about ADK?'
  )

  assert len(result.memories) == 2
  texts = {memory.content.parts[0].text for memory in result.memories}
  assert 'The ADK is a great toolkit.' in texts
  assert 'I agree. The Agent Development Kit (ADK) rocks!' in texts


@pytest.mark.asyncio
async def test_search_memory_no_match():
  """Tests a search query that should not match any memories."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='nonexistent'
  )

  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_is_scoped_by_user():
  """Tests that search results are correctly scoped to the user_id."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_DIFFERENT_USER)

  # Search for "secret", which only exists for MOCK_OTHER_USER_ID,
  # but search as MOCK_USER_ID.
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='secret'
  )

  # No results should be returned for MOCK_USER_ID
  assert not result.memories

  # The result should be found when searching as the correct user
  result_other_user = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_OTHER_USER_ID, query='secret'
  )
  assert len(result_other_user.memories) == 1
  assert (
      result_other_user.memories[0].content.parts[0].text == 'This is a secret.'
  )
