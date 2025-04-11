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
import uuid
from dateutil.parser import isoparse
from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.sessions import Session
from google.adk.sessions import VertexAiSessionService
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
            'reasoningEngines/test_engine/sessions/1/events/123'
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


SESSION_REGEX = r'^reasoningEngines/([^/]+)/sessions/([^/]+)$'
SESSIONS_REGEX = r'^reasoningEngines/([^/]+)/sessions$'
EVENTS_REGEX = r'^reasoningEngines/([^/]+)/sessions/([^/]+)/events$'
LRO_REGEX = r'^operations/([^/]+)$'


class MockApiClient:
  """Mocks the API Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    this.session_dict: dict[str, Any] = {}
    this.event_dict: dict[str, list[Any]] = {}

  def request(self, http_method: str, path: str, request_dict: dict[str, Any]):
    """Mocks the API Client request method."""
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
        return {
            'sessions': self.session_dict.values(),
        }
      elif re.match(EVENTS_REGEX, path):
        match = re.match(EVENTS_REGEX, path)
        if match:
          return {'sessionEvents': self.event_dict[match.group(2)]}
      elif re.match(LRO_REGEX, path):
        return {
            'name': (
                'projects/test-project/locations/test-location/'
                'reasoningEngines/123/sessions/123'
            ),
            'done': True,
        }
      else:
        raise ValueError(f'Unsupported path: {path}')
    elif http_method == 'POST':
      id = str(uuid.uuid4())
      self.session_dict[id] = {
          'name': (
              'projects/test-project/locations/test-location/'
              'reasoningEngines/123/sessions/'
              + id
          ),
          'userId': request_dict['user_id'],
          'sessionState': request_dict.get('sessionState', {}),
          'updateTime': '2024-12-12T12:12:12.123456Z',
      }
      return {
          'name': (
              'projects/test_project/locations/test_location/'
              'reasoningEngines/test_engine/sessions/123'
          ),
          'done': False,
      }
    elif http_method == 'DELETE':
      match = re.match(SESSION_REGEX, path)
      if match:
        self.session_dict.pop(match.group(2))
    else:
      raise ValueError(f'Unsupported http method: {http_method}')


def mock_vertex_ai_session_service():
  """Creates a mock Vertex AI Session service for testing."""
  service = VertexAiSessionService(
      project='test-project', location='test-location'
  )
  service.api_client = MockApiClient()
  service.api_client.session_dict = {
      '1': MOCK_SESSION_JSON_1,
      '2': MOCK_SESSION_JSON_2,
      '3': MOCK_SESSION_JSON_3,
  }
  service.api_client.event_dict = {
      '1': MOCK_EVENT_JSON,
  }
  return service


def test_get_empty_session():
  session_service = mock_vertex_ai_session_service()
  with pytest.raises(ValueError) as excinfo:
    assert session_service.get_session(
        app_name='123', user_id='user', session_id='0'
    )
    assert str(excinfo.value) == 'Session not found: 0'


def test_get_and_delete_session():
  session_service = mock_vertex_ai_session_service()

  assert (
      session_service.get_session(
          app_name='123', user_id='user', session_id='1'
      )
      == MOCK_SESSION
  )

  session_service.delete_session(app_name='123', user_id='user', session_id='1')
  with pytest.raises(ValueError) as excinfo:
    assert session_service.get_session(
        app_name='123', user_id='user', session_id='1'
    )
    assert str(excinfo.value) == 'Session not found: 1'

  def test_list_sessions():
    session_service = mock_vertex_ai_session_service()
    sessions = session_service.list_sessions(app_name='123', user_id='user')
    assert len(sessions.sessions) == 2
    assert sessions.sessions[0].id == '1'
    assert sessions.sessions[1].id == '2'

  def test_create_session():
    session_service = mock_vertex_ai_session_service()
    session = session_service.create_session(
        app_name='123', user_id='user', state={'key': 'value'}
    )
    assert session.state == {'key': 'value'}
    assert session.app_name == '123'
    assert session.user_id == 'user'
    assert session.last_update_time is not None

    session_id = session.id
    assert session == session_service.get_session(
        app_name='123', user_id='user', session_id=session_id
    )
