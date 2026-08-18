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

import asyncio
import threading

from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.platform import thread as platform_thread
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

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  assert user_key in memory_service._session_events
  session_memory = memory_service._session_events[user_key]
  assert MOCK_SESSION_1.id in session_memory
  # Check that the event with no content was filtered out
  assert len(session_memory[MOCK_SESSION_1.id]) == 2
  assert session_memory[MOCK_SESSION_1.id][0].id == 'event-1a'
  assert session_memory[MOCK_SESSION_1.id][1].id == 'event-1c'


@pytest.mark.asyncio
async def test_add_events_to_memory_with_explicit_events():
  """Tests that add_events_to_memory can ingest an explicit event list."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      session_id=MOCK_SESSION_1.id,
      events=[MOCK_SESSION_1.events[0]],
  )

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  session_memory = memory_service._session_events[user_key]
  assert len(session_memory[MOCK_SESSION_1.id]) == 1
  assert session_memory[MOCK_SESSION_1.id][0].id == 'event-1a'


@pytest.mark.asyncio
async def test_add_events_to_memory_without_session_id_uses_default_bucket():
  """Tests add_events_to_memory when no session_id is provided."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      events=[MOCK_SESSION_1.events[0]],
  )

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  session_memory = memory_service._session_events[user_key]
  assert len(session_memory) == 1
  unknown_session_events = next(iter(session_memory.values()))
  assert len(unknown_session_events) == 1
  assert unknown_session_events[0].id == 'event-1a'


@pytest.mark.asyncio
async def test_add_events_to_memory_alias_is_supported():
  """Tests that add_events_to_memory remains a compatibility alias."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      session_id=MOCK_SESSION_1.id,
      events=[MOCK_SESSION_1.events[0]],
  )

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  session_memory = memory_service._session_events[user_key]
  assert [event.id for event in session_memory[MOCK_SESSION_1.id]] == [
      'event-1a'
  ]


@pytest.mark.asyncio
async def test_add_events_to_memory_appends_without_replacing():
  """Tests that add_events_to_memory appends events rather than replacing."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  new_event = Event(
      id='event-1d',
      invocation_id='inv-6',
      author='user',
      timestamp=12348,
      content=types.Content(parts=[types.Part(text='A new fact.')]),
  )
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      session_id=MOCK_SESSION_1.id,
      events=[new_event],
  )

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  session_memory = memory_service._session_events[user_key]
  assert [event.id for event in session_memory[MOCK_SESSION_1.id]] == [
      'event-1a',
      'event-1c',
      'event-1d',
  ]


@pytest.mark.asyncio
async def test_add_events_to_memory_deduplicates_event_ids():
  """Tests that duplicate event IDs are not appended multiple times."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  duplicate_event = Event(
      id='event-1a',
      invocation_id='inv-7',
      author='user',
      timestamp=12349,
      content=types.Content(parts=[types.Part(text='Updated duplicate text.')]),
  )
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      session_id=MOCK_SESSION_1.id,
      events=[duplicate_event],
  )

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
  session_memory = memory_service._session_events[user_key]
  assert [event.id for event in session_memory[MOCK_SESSION_1.id]] == [
      'event-1a',
      'event-1c',
  ]


@pytest.mark.asyncio
async def test_add_session_with_no_events_to_memory():
  """Tests that adding a session with no events does not cause an error."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_NO_EVENTS)

  user_key = (MOCK_APP_NAME, MOCK_USER_ID)
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


@pytest.mark.asyncio
async def test_search_memory_does_not_collide_on_slash_in_identifiers():
  """Tests that a slash in app_name cannot alias another app/user pair."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(
      Session(
          app_name='app/other-user',
          user_id='user',
          id='session-slashed-app',
          last_update_time=1000,
          events=[
              Event(
                  id='event-slashed-app',
                  invocation_id='inv-slashed-app',
                  author='user',
                  timestamp=12345,
                  content=types.Content(
                      parts=[types.Part(text='This is a secret.')]
                  ),
              ),
          ],
      )
  )

  result = await memory_service.search_memory(
      app_name='app', user_id='other-user/user', query='secret'
  )

  assert not result.memories


# --- Non-Latin language tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'event_text,query,expected_count',
    [
        # Japanese (no space delimiters — substring fallback)
        ('私の名前は太郎です', '太郎', 1),
        ('私の名前は太郎です', '天気', 0),
        # Chinese (no space delimiters — substring fallback)
        ('我喜欢机器学习', '机器学习', 1),
        ('我喜欢机器学习', '天气预报', 0),
        # Korean (space-delimited — token match)
        ('제 이름은 민수입니다', '민수입니다', 1),
        # Cyrillic (space-delimited — token match)
        ('Меня зовут Алексей', 'Алексей', 1),
        # Mixed: non-Latin substring + Latin token in same event
        ('太郎 works at ABC Corp', '太郎', 1),
        ('太郎 works at ABC Corp', 'ABC', 1),
        # Latin partial-word must NOT match (regression guard)
        ('I like to code in Python.', 'thon', 0),
    ],
)
async def test_search_memory_non_latin(event_text, query, expected_count):
  """Tests search_memory with non-Latin scripts and mixed content."""
  session = Session(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      id='session-i18n',
      last_update_time=7000,
      events=[
          Event(
              id='event-i18n',
              invocation_id='inv-i18n',
              author='user',
              timestamp=90000,
              content=types.Content(parts=[types.Part(text=event_text)]),
          ),
      ],
  )
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(session)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query=query
  )
  assert len(result.memories) == expected_count


def _text_event(tag: str, text: str) -> Event:
  return Event(
      id=f'event-{tag}',
      invocation_id=f'inv-{tag}',
      author='user',
      timestamp=1.0,
      content=types.Content(parts=[types.Part(text=text)]),
  )


@pytest.mark.asyncio
async def test_search_memory_ranks_by_number_of_matching_words():
  """Tests that the events matching the most query words come first."""
  memory_service = InMemoryMemoryService()
  await memory_service.add_session_to_memory(
      Session(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          id='session-ranked',
          last_update_time=1000,
          events=[
              _text_event('ranked-a', 'The deploy is ready.'),
              _text_event('ranked-b', 'Ready.'),
              _text_event('ranked-c', 'The deploy status is ready.'),
          ],
      )
  )

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='deploy status ready'
  )

  assert [memory.content.parts[0].text for memory in result.memories] == [
      'The deploy status is ready.',
      'The deploy is ready.',
      'Ready.',
  ]


@pytest.mark.asyncio
async def test_search_memory_returns_at_most_ten_memories():
  """Tests that a word shared with the whole store cannot return the store."""
  memory_service = InMemoryMemoryService()
  events = [_text_event(f'note-{i}', f'note {i} about work') for i in range(20)]
  events.append(_text_event('backlog', 'the backlog note about work'))
  await memory_service.add_session_to_memory(
      Session(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          id='session-many',
          last_update_time=1000,
          events=events,
      )
  )

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='work backlog note'
  )

  texts = [memory.content.parts[0].text for memory in result.memories]
  # The best match is stored last but ranks first, and the rest tie, so they
  # keep the order they were added in.
  assert texts == ['the backlog note about work'] + [
      f'note {i} about work' for i in range(9)
  ]


def _make_event(tag: str) -> Event:
  return Event(
      id=f'event-{tag}',
      invocation_id=f'inv-{tag}',
      author='user',
      timestamp=1.0,
      content=types.Content(parts=[types.Part(text=f'fact about {tag}')]),
  )


def _make_session(tag: str) -> Session:
  return Session(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      id=f'session-{tag}',
      last_update_time=1,
      events=[_make_event(tag)],
  )


def test_search_memory_is_thread_safe_against_concurrent_writes():
  """Searching while other threads add memory must not crash.

  InMemoryMemoryService documents itself as thread-safe. search_memory must
  therefore iterate a stable snapshot taken under the lock; iterating a live
  reference to the shared store while a concurrent writer mutates it raises
  "RuntimeError: dictionary changed size during iteration".
  """
  memory_service = InMemoryMemoryService()
  seed_loop = asyncio.new_event_loop()
  try:
    for i in range(50):
      seed_loop.run_until_complete(
          memory_service.add_session_to_memory(_make_session(f'seed-{i}'))
      )
  finally:
    seed_loop.close()

  errors = []
  stop = threading.Event()
  barrier = threading.Barrier(3)

  def writer():
    loop = asyncio.new_event_loop()
    barrier.wait()
    try:
      for i in range(500):
        if stop.is_set():
          return
        loop.run_until_complete(
            memory_service.add_session_to_memory(_make_session(f'writer-{i}'))
        )
    except Exception as e:  # pylint: disable=broad-except
      errors.append(e)
      stop.set()
    finally:
      loop.close()

  def reader():
    loop = asyncio.new_event_loop()
    barrier.wait()
    try:
      for _ in range(500):
        if stop.is_set():
          return
        loop.run_until_complete(
            memory_service.search_memory(
                app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='fact'
            )
        )
    except Exception as e:  # pylint: disable=broad-except
      errors.append(e)
      stop.set()
    finally:
      loop.close()

  threads = [
      platform_thread.create_thread(writer),
      platform_thread.create_thread(reader),
      platform_thread.create_thread(reader),
  ]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  assert (
      not errors
  ), f'search_memory raced with concurrent writes: {errors[0]!r}'
