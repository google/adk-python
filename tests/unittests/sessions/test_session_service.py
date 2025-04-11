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

import enum
import pytest

from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.genai import types


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


@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
def test_get_empty_session(service_type):
  session_service = get_session_service(service_type)
  assert not session_service.get_session(
      app_name='my_app', user_id='test_user', session_id='123'
  )


@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
def test_create_get_session(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'test_user'
  state = {'key': 'value'}

  session = session_service.create_session(
      app_name=app_name, user_id=user_id, state=state
  )
  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.id
  assert session.state == state
  assert (
      session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session.id
      )
      == session
  )

  session_id = session.id
  session_service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert (
      not session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session.id
      )
      == session
  )


@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
def test_create_and_list_sessions(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'test_user'

  session_ids = ['session' + str(i) for i in range(5)]
  for session_id in session_ids:
    session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  sessions = session_service.list_sessions(
      app_name=app_name, user_id=user_id
  ).sessions
  for i in range(len(sessions)):
    assert sessions[i].id == session_ids[i]


@pytest.mark.parametrize(
    'service_type', [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
def test_session_state(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id_1 = 'user1'
  user_id_2 = 'user2'
  session_id_11 = 'session11'
  session_id_12 = 'session12'
  session_id_2 = 'session2'
  state_11 = {'key11': 'value11'}
  state_12 = {'key12': 'value12'}

  session_11 = session_service.create_session(
      app_name=app_name,
      user_id=user_id_1,
      state=state_11,
      session_id=session_id_11,
  )
  session_service.create_session(
      app_name=app_name,
      user_id=user_id_1,
      state=state_12,
      session_id=session_id_12,
  )
  session_service.create_session(
      app_name=app_name, user_id=user_id_2, session_id=session_id_2
  )

  assert session_11.state.get('key11') == 'value11'

  event = Event(
      invocation_id='invocation',
      author='user',
      content=types.Content(role='user', parts=[types.Part(text='text')]),
      actions=EventActions(
          state_delta={
              'app:key': 'value',
              'user:key1': 'value1',
              'temp:key': 'temp',
              'key11': 'value11_new',
          }
      ),
  )
  session_service.append_event(session=session_11, event=event)

  # User and app state is stored, temp state is filtered.
  assert session_11.state.get('app:key') == 'value'
  assert session_11.state.get('key11') == 'value11_new'
  assert session_11.state.get('user:key1') == 'value1'
  assert not session_11.state.get('temp:key')

  session_12 = session_service.get_session(
      app_name=app_name, user_id=user_id_1, session_id=session_id_12
  )
  # After getting a new instance, the session_12 got the user and app state,
  # even append_event is not applied to it, temp state has no effect
  assert session_12.state.get('key12') == 'value12'
  assert not session_12.state.get('temp:key')

  # The user1's state is not visible to user2, app state is visible
  session_2 = session_service.get_session(
      app_name=app_name, user_id=user_id_2, session_id=session_id_2
  )
  assert session_2.state.get('app:key') == 'value'
  assert not session_2.state.get('user:key1')

  assert not session_2.state.get('user:key1')

  # The change to session_11 is persisted
  session_11 = session_service.get_session(
      app_name=app_name, user_id=user_id_1, session_id=session_id_11
  )
  assert session_11.state.get('key11') == 'value11_new'
  assert session_11.state.get('user:key1') == 'value1'
  assert not session_11.state.get('temp:key')


@pytest.mark.parametrize(
    "service_type", [SessionServiceType.IN_MEMORY, SessionServiceType.DATABASE]
)
def test_create_new_session_will_merge_states(service_type):
  session_service = get_session_service(service_type)
  app_name = 'my_app'
  user_id = 'user'
  session_id_1 = 'session1'
  session_id_2 = 'session2'
  state_1 = {'key1': 'value1'}

  session_1 = session_service.create_session(
      app_name=app_name, user_id=user_id, state=state_1, session_id=session_id_1
  )

  event = Event(
      invocation_id='invocation',
      author='user',
      content=types.Content(role='user', parts=[types.Part(text='text')]),
      actions=EventActions(
          state_delta={
              'app:key': 'value',
              'user:key1': 'value1',
              'temp:key': 'temp',
          }
      ),
  )
  session_service.append_event(session=session_1, event=event)

  # User and app state is stored, temp state is filtered.
  assert session_1.state.get('app:key') == 'value'
  assert session_1.state.get('key1') == 'value1'
  assert session_1.state.get('user:key1') == 'value1'
  assert not session_1.state.get('temp:key')

  session_2 = session_service.create_session(
      app_name=app_name, user_id=user_id, state={}, session_id=session_id_2
  )
  # Session 2 has the persisted states
  assert session_2.state.get('app:key') == 'value'
  assert session_2.state.get('user:key1') == 'value1'
  assert not session_2.state.get('key1')
  assert not session_2.state.get('temp:key')
