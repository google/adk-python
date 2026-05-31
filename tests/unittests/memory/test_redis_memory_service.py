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
from google.adk.memory.redis_memory_service import RedisMemoryService
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


class FakeAsyncRedis:
  def __init__(self):
    self.sets: dict[str, set[str]] = {}
    self.lists: dict[str, list[str]] = {}
    self.hashes: dict[str, dict[str, str]] = {}
    self.closed = False

  async def sadd(self, key: str, *values: str) -> int:
    values_set = self.sets.setdefault(key, set())
    old_len = len(values_set)
    values_set.update(values)
    return len(values_set) - old_len

  async def smembers(self, key: str) -> set[str]:
    return set(self.sets.get(key, set()))

  async def delete(self, *keys: str) -> int:
    deleted = 0
    for key in keys:
      for store in (self.sets, self.lists, self.hashes):
        if key in store:
          del store[key]
          deleted += 1
    return deleted

  async def hsetnx(self, key: str, field: str, value: str) -> int:
    values = self.hashes.setdefault(key, {})
    if field in values:
      return 0
    values[field] = value
    return 1

  async def rpush(self, key: str, value: str) -> int:
    values = self.lists.setdefault(key, [])
    values.append(value)
    return len(values)

  async def lrange(self, key: str, start: int, end: int) -> list[str]:
    values = self.lists.get(key, [])
    if end == -1:
      return values[start:]
    return values[start : end + 1]

  async def hget(self, key: str, field: str) -> str | None:
    return self.hashes.get(key, {}).get(field)

  async def aclose(self) -> None:
    self.closed = True


def redis_memory_service() -> RedisMemoryService:
  return RedisMemoryService(client=FakeAsyncRedis())


@pytest.mark.asyncio
async def test_add_session_to_memory():
  memory_service = redis_memory_service()

  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='ADK'
  )

  assert len(result.memories) == 2
  assert {memory.id for memory in result.memories} == {'event-1a', 'event-1c'}


@pytest.mark.asyncio
async def test_add_events_to_memory_with_explicit_events():
  memory_service = redis_memory_service()

  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      session_id=MOCK_SESSION_1.id,
      events=[MOCK_SESSION_1.events[0]],
  )
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='toolkit'
  )

  assert len(result.memories) == 1
  assert result.memories[0].id == 'event-1a'


@pytest.mark.asyncio
async def test_add_events_to_memory_without_session_id_uses_default_bucket():
  memory_service = redis_memory_service()

  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION_1.app_name,
      user_id=MOCK_SESSION_1.user_id,
      events=[MOCK_SESSION_1.events[0]],
  )
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='toolkit'
  )

  assert len(result.memories) == 1
  assert result.memories[0].custom_metadata['session_id']


@pytest.mark.asyncio
async def test_add_events_to_memory_appends_without_replacing():
  memory_service = redis_memory_service()
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
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='fact'
  )

  assert len(result.memories) == 1
  assert result.memories[0].id == 'event-1d'


@pytest.mark.asyncio
async def test_add_events_to_memory_deduplicates_event_ids():
  memory_service = redis_memory_service()
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
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='duplicate'
  )

  assert not result.memories


@pytest.mark.asyncio
async def test_add_session_replaces_existing_session_memory():
  memory_service = redis_memory_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  replacement_session = Session(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      id=MOCK_SESSION_1.id,
      last_update_time=5000,
      events=[
          Event(
              id='replacement',
              invocation_id='inv-8',
              author='user',
              timestamp=12350,
              content=types.Content(parts=[types.Part(text='Replacement')]),
          )
      ],
  )

  await memory_service.add_session_to_memory(replacement_session)
  old_result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='ADK'
  )
  new_result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='Replacement'
  )

  assert not old_result.memories
  assert len(new_result.memories) == 1
  assert new_result.memories[0].id == 'replacement'


@pytest.mark.asyncio
async def test_add_session_with_no_events_to_memory():
  memory_service = redis_memory_service()

  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_NO_EVENTS)
  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='anything'
  )

  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_simple_match():
  memory_service = redis_memory_service()
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
  memory_service = redis_memory_service()
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
  memory_service = redis_memory_service()
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
  memory_service = redis_memory_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='nonexistent'
  )

  assert not result.memories


@pytest.mark.asyncio
async def test_search_memory_is_scoped_by_user():
  memory_service = redis_memory_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_1)
  await memory_service.add_session_to_memory(MOCK_SESSION_DIFFERENT_USER)

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='secret'
  )
  result_other_user = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_OTHER_USER_ID, query='secret'
  )

  assert not result.memories
  assert len(result_other_user.memories) == 1
  assert (
      result_other_user.memories[0].content.parts[0].text == 'This is a secret.'
  )


@pytest.mark.asyncio
async def test_close_closes_client():
  client = FakeAsyncRedis()
  memory_service = RedisMemoryService(client=client)

  await memory_service.close()

  assert client.closed
