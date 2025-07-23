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

import re
import this
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from unittest import mock

from dateutil.parser import isoparse
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.genai import types
import pytest

MOCK_SESSION_JSON_1 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/1'
    ),
    'createTime': '2024-12-12T12:12:12.123456Z',
    'updateTime': '2024-12-12T12:12:12.123456Z',
    'sessionState': {
        'key': {'value': 'test_value'},
    },
    'userId': 'user',
}
MOCK_SESSION_JSON_2 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/2'
    ),
    'updateTime': '2024-12-13T12:12:12.123456Z',
    'userId': 'user',
}
MOCK_SESSION_JSON_3 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/3'
    ),
    'updateTime': '2024-12-14T12:12:12.123456Z',
    'userId': 'user2',
}
MOCK_EVENT_JSON = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/1/events/123'
        ),
        'invocationId': '123',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:12.123456Z',
        'content': {
            'parts': [
                {'text': 'test_content'},
            ],
        },
        'actions': {
            'stateDelta': {
                'key': {'value': 'test_value'},
            },
            'transferAgent': 'agent',
        },
        'eventMetadata': {
            'partial': False,
            'turnComplete': True,
            'interrupted': False,
            'branch': '',
            'longRunningToolIds': ['tool1'],
        },
    },
]
MOCK_EVENT_JSON_2 = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/2/events/123'
        ),
        'invocationId': '222',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:12.123456Z',
    },
]
MOCK_EVENT_JSON_3 = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/2/events/456'
        ),
        'invocationId': '333',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:12.123456Z',
    },
]

MOCK_SESSION = Session(
    app_name='123',
    user_id='user',
    id='1',
    state=MOCK_SESSION_JSON_1['sessionState'],
    last_update_time=isoparse(MOCK_SESSION_JSON_1['updateTime']).timestamp(),
    events=[
        Event(
            id='123',
            invocation_id='123',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON[0]['timestamp']).timestamp(),
            content=types.Content(parts=[types.Part(text='test_content')]),
            actions=EventActions(
                transfer_to_agent='agent',
                state_delta={'key': {'value': 'test_value'}},
            ),
            partial=False,
            turn_complete=True,
            interrupted=False,
            branch='',
            long_running_tool_ids={'tool1'},
        ),
    ],
)

MOCK_SESSION_2 = Session(
    app_name='123',
    user_id='user',
    id='2',
    last_update_time=isoparse(MOCK_SESSION_JSON_2['updateTime']).timestamp(),
    events=[
        Event(
            id='123',
            invocation_id='222',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON_2[0]['timestamp']).timestamp(),
        ),
        Event(
            id='456',
            invocation_id='333',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON_3[0]['timestamp']).timestamp(),
        ),
    ],
)


SESSION_REGEX = r'^reasoningEngines/([^/]+)/sessions/([^/]+)$'
SESSIONS_REGEX = (  # %22 represents double-quotes in a URL-encoded string
    r'^reasoningEngines/([^/]+)/sessions\?filter=user_id=%22([^%]+)%22.*$'
)
EVENTS_REGEX = (
    r'^reasoningEngines/([^/]+)/sessions/([^/]+)/events(?:\?pageToken=([^/]+))?'
)
LRO_REGEX = r'^operations/([^/]+)$'


class MockApiClient:
  """Mocks the API Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    this.session_dict: dict[str, Any] = {}
    this.event_dict: dict[str, Tuple[List[Any], Optional[str]]] = {}

  async def async_request(
      self, http_method: str, path: str, request_dict: dict[str, Any]
  ):
    """Mocks the API Client request method"""
    if http_method == 'GET':
      if re.match(SESSION_REGEX, path):
        match = re.match(SESSION_REGEX, path)
        if match:
          session_id = match.group(2)
          if session_id in self.session_dict:
            return self.session_dict[session_id]
          else:
            raise ValueError(f'Session not found: {session_id}')
      elif re.match(SESSIONS_REGEX, path):
        match = re.match(SESSIONS_REGEX, path)
        return {
            'sessions': [
                session
                for session in self.session_dict.values()
                if session['userId'] == match.group(2)
            ],
        }
      elif re.match(EVENTS_REGEX, path):
        match = re.match(EVENTS_REGEX, path)
        if match:
          session_id = match.group(2)
          if match.group(3):
            page_token = match.group(3)
            if page_token == 'my_token':
              response = {'sessionEvents': MOCK_EVENT_JSON_3}
              response['nextPageToken'] = 'my_token2'
              return response
            else:
              return {}
          events_tuple = self.event_dict.get(session_id, ([], None))
          response = {'sessionEvents': events_tuple[0]}
          if events_tuple[1]:
            response['nextPageToken'] = events_tuple[1]
          return response
      elif re.match(LRO_REGEX, path):
        # Mock long-running operation as completed
        return {
            'name': path,
            'done': True,
            'response': self.session_dict['4'],  # Return the created session
        }
      else:
        raise ValueError(f'Unsupported path: {path}')
    elif http_method == 'POST':
      new_session_id = '4'
      self.session_dict[new_session_id] = {
          'name': (
              'projects/test-project/locations/test-location/'
              'reasoningEngines/123/sessions/'
              + new_session_id
          ),
          'userId': request_dict['user_id'],
          'sessionState': request_dict.get('session_state', {}),
          'updateTime': '2024-12-12T12:12:12.123456Z',
      }
      return {
          'name': (
              'projects/test_project/locations/test_location/'
              'reasoningEngines/123/sessions/'
              + new_session_id
              + '/operations/111'
          ),
          'done': False,
      }
    elif http_method == 'DELETE':
      match = re.match(SESSION_REGEX, path)
      if match:
        self.session_dict.pop(match.group(2))
    else:
      raise ValueError(f'Unsupported http method: {http_method}')


def mock_vertex_ai_session_service(agent_engine_id: Optional[str] = None):
  """Creates a mock Vertex AI Session service for testing."""
  if agent_engine_id:
    return VertexAiSessionService(
        project='test-project',
        location='test-location',
        agent_engine_id=agent_engine_id,
    )
  return VertexAiSessionService(
      project='test-project', location='test-location'
  )


@pytest.fixture
def mock_get_api_client():
  api_client = MockApiClient()
  api_client.session_dict = {
      '1': MOCK_SESSION_JSON_1,
      '2': MOCK_SESSION_JSON_2,
      '3': MOCK_SESSION_JSON_3,
  }
  api_client.event_dict = {
      '1': (MOCK_EVENT_JSON, None),
      '2': (MOCK_EVENT_JSON_2, 'my_token'),
  }
  with mock.patch(
      'google.adk.sessions.vertex_ai_session_service.VertexAiSessionService._get_api_client',
      return_value=api_client,
  ):
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
@pytest.mark.parametrize('agent_engine_id', [None, '123'])
async def test_get_empty_session(agent_engine_id):
  if agent_engine_id:
    session_service = mock_vertex_ai_session_service(agent_engine_id)
  else:
    session_service = mock_vertex_ai_session_service()
  with pytest.raises(ValueError) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user', session_id='0'
    )
  assert str(excinfo.value) == 'Session not found: 0'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
@pytest.mark.parametrize('agent_engine_id', [None, '123'])
async def test_get_another_user_session(agent_engine_id):
  if agent_engine_id:
    session_service = mock_vertex_ai_session_service(agent_engine_id)
  else:
    session_service = mock_vertex_ai_session_service()
  with pytest.raises(ValueError) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user2', session_id='1'
    )
  assert str(excinfo.value) == 'Session not found: 1'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_get_and_delete_session():
  session_service = mock_vertex_ai_session_service()

  assert (
      await session_service.get_session(
          app_name='123', user_id='user', session_id='1'
      )
      == MOCK_SESSION
  )

  await session_service.delete_session(
      app_name='123', user_id='user', session_id='1'
  )
  with pytest.raises(ValueError) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user', session_id='1'
    )
  assert str(excinfo.value) == 'Session not found: 1'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_get_session_with_page_token():
  session_service = mock_vertex_ai_session_service()

  assert (
      await session_service.get_session(
          app_name='123', user_id='user', session_id='2'
      )
      == MOCK_SESSION_2
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_list_sessions():
  session_service = mock_vertex_ai_session_service()
  sessions = await session_service.list_sessions(app_name='123', user_id='user')
  assert len(sessions.sessions) == 2
  assert sessions.sessions[0].id == '1'
  assert sessions.sessions[1].id == '2'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_create_session():
  session_service = mock_vertex_ai_session_service()

  state = {'key': 'value'}
  session = await session_service.create_session(
      app_name='123', user_id='user', state=state
  )
  assert session.state == state
  assert session.app_name == '123'
  assert session.user_id == 'user'
  assert session.last_update_time is not None

  session_id = session.id
  assert session == await session_service.get_session(
      app_name='123', user_id='user', session_id=session_id
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_create_session_with_custom_session_id():
  session_service = mock_vertex_ai_session_service()

  with pytest.raises(ValueError) as excinfo:
    await session_service.create_session(
        app_name='123', user_id='user', session_id='1'
    )
  assert str(excinfo.value) == (
      'User-provided Session id is not supported for VertexAISessionService.'
  )
