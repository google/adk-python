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

from datetime import datetime
from datetime import timezone
import enum

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


class SessionServiceType(enum.Enum):
  IN_MEMORY = 'IN_MEMORY'
  DATABASE = 'DATABASE'


def get_session_service(
    service_type: SessionServiceType = SessionServiceType.IN_MEMORY,
):
  """Creates a session service for testing."""
  if service_type == SessionServiceType.DATABASE:
    return DatabaseSessionService('sqlite:///:memory:')
  return InMemorySessionService()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_get_empty_session(service_type):
  session_service = get_session_service(service_type)
  assert not await session_service.get_session(
      app_name='my_app', user_id='test_user', session_id='123'
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_create_get_session(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'test_user'
  state = {'key': 'value'}

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, state=state
  )
  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.id
  assert session.state == state
  assert (
      session.last_update_time
      <= datetime.now().astimezone(timezone.utc).timestamp()
  )

  got_session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  assert got_session == session
  assert (
      got_session.last_update_time
      <= datetime.now().astimezone(timezone.utc).timestamp()
  )

  session_id = session.id
  await session_service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert (
      await session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session.id
      )
      is None
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_create_and_list_sessions(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'test_user'

  session_ids = ['session' + str(i) for i in range(5)]
  for session_id in session_ids:
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={'key': 'value' + session_id},
    )

  list_sessions_response = await session_service.list_sessions(
      app_name=app_name, user_id=user_id
  )
  sessions = list_sessions_response.sessions
  assert len(sessions) == len(session_ids)
  assert {s.id for s in sessions} == set(session_ids)
  for session in sessions:
    assert session.state == {'key': 'value' + session.id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_list_sessions_all_users(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id_1 = 'user1'
  user_id_2 = 'user2'

  await session_service.create_session(
      app_name=app_name,
      user_id=user_id_1,
      session_id='session1a',
      state={'key': 'value1a'},
  )
  await session_service.create_session(
      app_name=app_name,
      user_id=user_id_1,
      session_id='session1b',
      state={'key': 'value1b'},
  )
  await session_service.create_session(
      app_name=app_name,
      user_id=user_id_2,
      session_id='session2a',
      state={'key': 'value2a'},
  )

  # List sessions for user1 - should contain merged state
  list_sessions_response_1 = await session_service.list_sessions(
      app_name=app_name, user_id=user_id_1
  )
  sessions_1 = list_sessions_response_1.sessions
  assert len(sessions_1) == 2
  sessions_1_map = {s.id: s for s in sessions_1}
  assert sessions_1_map['session1a'].state == {'key': 'value1a'}
  assert sessions_1_map['session1b'].state == {'key': 'value1b'}

  # List sessions for user2 - should contain merged state
  list_sessions_response_2 = await session_service.list_sessions(
      app_name=app_name, user_id=user_id_2
  )
  sessions_2 = list_sessions_response_2.sessions
  assert len(sessions_2) == 1
  assert sessions_2[0].id == 'session2a'
  assert sessions_2[0].state == {'key': 'value2a'}

  # List sessions for all users - should contain merged state
  list_sessions_response_all = await session_service.list_sessions(
      app_name=app_name, user_id=None
  )
  sessions_all = list_sessions_response_all.sessions
  assert len(sessions_all) == 3
  sessions_all_map = {s.id: s for s in sessions_all}
  assert sessions_all_map['session1a'].state == {'key': 'value1a'}
  assert sessions_all_map['session1b'].state == {'key': 'value1b'}
  assert sessions_all_map['session2a'].state == {'key': 'value2a'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_app_state_is_shared_by_all_users_of_app(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  # User 1 creates a session, establishing app:k1
  session1 = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1', state={'app:k1': 'v1'}
  )
  # User 1 appends an event to session1, establishing app:k2
  event = Event(
      invocation_id='inv1',
      author='user',
      actions=EventActions(state_delta={'app:k2': 'v2'}),
  )
  await session_service.append_event(session=session1, event=event)

  # User 2 creates a new session session2, it should see app:k1 and app:k2
  session2 = await session_service.create_session(
      app_name=app_name, user_id='u2', session_id='s2'
  )
  assert session2.state == {'app:k1': 'v1', 'app:k2': 'v2'}

  # If we get session session1 again, it should also see both
  session1_got = await session_service.get_session(
      app_name=app_name, user_id='u1', session_id='s1'
  )
  assert session1_got.state.get('app:k1') == 'v1'
  assert session1_got.state.get('app:k2') == 'v2'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_user_state_is_shared_only_by_user_sessions(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  # User 1 creates a session, establishing user:k1 for user 1
  session1 = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1', state={'user:k1': 'v1'}
  )
  # User 1 appends an event to session1, establishing user:k2 for user 1
  event = Event(
      invocation_id='inv1',
      author='user',
      actions=EventActions(state_delta={'user:k2': 'v2'}),
  )
  await session_service.append_event(session=session1, event=event)

  # Another session for User 1 should see user:k1 and user:k2
  session1b = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1b'
  )
  assert session1b.state == {'user:k1': 'v1', 'user:k2': 'v2'}

  # A session for User 2 should NOT see user:k1 or user:k2
  session2 = await session_service.create_session(
      app_name=app_name, user_id='u2', session_id='s2'
  )
  assert session2.state == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_session_state_is_not_shared(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  # User 1 creates a session session1, establishing sk1 only for session1
  session1 = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1', state={'sk1': 'v1'}
  )
  # User 1 appends an event to session1, establishing sk2 only for session1
  event = Event(
      invocation_id='inv1',
      author='user',
      actions=EventActions(state_delta={'sk2': 'v2'}),
  )
  await session_service.append_event(session=session1, event=event)

  # Getting session1 should show sk1 and sk2
  session1_got = await session_service.get_session(
      app_name=app_name, user_id='u1', session_id='s1'
  )
  assert session1_got.state.get('sk1') == 'v1'
  assert session1_got.state.get('sk2') == 'v2'

  # Creating another session session1b for User 1 should NOT see sk1 or sk2
  session1b = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1b'
  )
  assert session1b.state == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_temp_state_is_not_persisted_in_state_or_events(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'u1'
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, session_id='s1'
  )
  event = Event(
      invocation_id='inv1',
      author='user',
      actions=EventActions(state_delta={'temp:k1': 'v1', 'sk': 'v2'}),
  )
  await session_service.append_event(session=session, event=event)

  # Refetch session and check state and event
  session_got = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id='s1'
  )
  # Check session state does not contain temp keys
  assert session_got.state.get('sk') == 'v2'
  assert 'temp:k1' not in session_got.state
  # Check event as stored in session does not contain temp keys in state_delta
  assert 'temp:k1' not in session_got.events[0].actions.state_delta
  assert session_got.events[0].actions.state_delta.get('sk') == 'v2'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_get_session_respects_user_id(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  # u1 creates session 's1' and adds an event
  session1 = await session_service.create_session(
      app_name=app_name, user_id='u1', session_id='s1'
  )
  event = Event(invocation_id='inv1', author='user')
  await session_service.append_event(session1, event)
  # u2 creates a session with the same session_id 's1'
  await session_service.create_session(
      app_name=app_name, user_id='u2', session_id='s1'
  )
  # Check that getting s1 for u2 returns u2's session (with no events)
  # not u1's session.
  session2_got = await session_service.get_session(
      app_name=app_name, user_id='u2', session_id='s1'
  )
  assert session2_got.user_id == 'u2'
  assert len(session2_got.events) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_create_session_with_existing_id_raises_error(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'test_user'
  session_id = 'existing_session'

  # Create the first session
  await session_service.create_session(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
  )

  # Attempt to create a session with the same ID
  with pytest.raises(AlreadyExistsError):
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_append_event_bytes(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'user'

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  test_content = types.Content(
      role='user',
      parts=[
          types.Part.from_bytes(data=b'test_image_data', mime_type='image/png'),
      ],
  )
  test_grounding_metadata = types.GroundingMetadata(
      search_entry_point=types.SearchEntryPoint(sdk_blob=b'test_sdk_blob')
  )
  event = Event(
      invocation_id='invocation',
      author='user',
      content=test_content,
      grounding_metadata=test_grounding_metadata,
  )
  await session_service.append_event(session=session, event=event)

  assert session.events[0].content == test_content

  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  events = session.events
  assert len(events) == 1
  assert events[0].content == test_content
  assert events[0].grounding_metadata == test_grounding_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_append_event_complete(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'user'

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  event = Event(
      invocation_id='invocation',
      author='user',
      content=types.Content(role='user', parts=[types.Part(text='test_text')]),
      turn_complete=True,
      partial=False,
      actions=EventActions(
          artifact_delta={
              'file': 0,
          },
          transfer_to_agent='agent',
          escalate=True,
      ),
      long_running_tool_ids={'tool1'},
      error_code='error_code',
      error_message='error_message',
      interrupted=True,
      grounding_metadata=types.GroundingMetadata(
          web_search_queries=['query1'],
      ),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=1, candidates_token_count=1, total_token_count=2
      ),
      citation_metadata=types.CitationMetadata(),
      custom_metadata={'custom_key': 'custom_value'},
  )
  await session_service.append_event(session=session, event=event)

  assert (
      await session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session.id
      )
      == session
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_get_session_with_config(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'user'

  num_test_events = 5
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  for i in range(1, num_test_events + 1):
    event = Event(author='user', timestamp=i)
    await session_service.append_event(session, event)

  # No config, expect all events to be returned.
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  events = session.events
  assert len(events) == num_test_events

  # Only expect the most recent 3 events.
  num_recent_events = 3
  config = GetSessionConfig(num_recent_events=num_recent_events)
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id, config=config
  )
  events = session.events
  assert len(events) == num_recent_events
  assert events[0].timestamp == num_test_events - num_recent_events + 1

  # Only expect events after timestamp 4.0 (inclusive), i.e., 2 events.
  after_timestamp = 4.0
  config = GetSessionConfig(after_timestamp=after_timestamp)
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id, config=config
  )
  events = session.events
  assert len(events) == num_test_events - after_timestamp + 1
  assert events[0].timestamp == after_timestamp

  # Expect no events if none are > after_timestamp.
  way_after_timestamp = num_test_events * 10
  config = GetSessionConfig(after_timestamp=way_after_timestamp)
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id, config=config
  )
  assert not session.events

  # Both filters applied, i.e., of 3 most recent events, only 2 are after
  # timestamp 4.0, so expect 2 events.
  config = GetSessionConfig(
      after_timestamp=after_timestamp, num_recent_events=num_recent_events
  )
  session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id, config=config
  )
  events = session.events
  assert len(events) == num_test_events - after_timestamp + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
async def test_partial_events_are_not_persisted(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'user'
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  event = Event(author='user', partial=True)
  await session_service.append_event(session, event)

  # Check in-memory session
  assert len(session.events) == 0
  # Check persisted session
  session_got = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  assert len(session_got.events) == 0
