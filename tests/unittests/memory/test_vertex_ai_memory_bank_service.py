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
from typing import Any
from unittest import mock

from google.adk.events.event import Event
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id='333',
    last_update_time=22333,
    events=[
        Event(
            id='444',
            invocation_id='123',
            author='user',
            timestamp=12345,
            content=types.Content(parts=[types.Part(text='test_content')]),
        ),
        # Empty event, should be ignored
        Event(
            id='555',
            invocation_id='456',
            author='user',
            timestamp=12345,
        ),
        # Function call event, should be ignored
        Event(
            id='666',
            invocation_id='456',
            author='agent',
            timestamp=23456,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name='test_function')
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id='444',
    last_update_time=22333,
)


RETRIEVE_MEMORIES_REGEX = r'^reasoningEngines/([^/]+)/memories:retrieve$'
GENERATE_MEMORIES_REGEX = r'^reasoningEngines/([^/]+)/memories:generate$'


class MockApiClient:
  """Mocks the API Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    self.async_request = mock.AsyncMock()
    self.async_request.side_effect = self._mock_async_request

  async def _mock_async_request(
      self, http_method: str, path: str, request_dict: dict[str, Any]
  ):
    """Mocks the API Client request method."""
    if http_method == 'POST':
      if re.match(GENERATE_MEMORIES_REGEX, path):
        return {}
      elif re.match(RETRIEVE_MEMORIES_REGEX, path):
        if (
            request_dict.get('scope', None)
            and request_dict['scope'].get('app_name', None) == MOCK_APP_NAME
        ):
          return {
              'retrievedMemories': [
                  {
                      'memory': {
                          'fact': 'test_content',
                      },
                      'updateTime': '2024-12-12T12:12:12.123456Z',
                  },
              ],
          }
        else:
          return {'retrievedMemories': []}
      else:
        raise ValueError(f'Unsupported path: {path}')
    else:
      raise ValueError(f'Unsupported http method: {http_method}')


def mock_vertex_ai_memory_bank_service():
  """Creates a mock Vertex AI Memory Bank service for testing."""
  return VertexAiMemoryBankService(
      project='test-project',
      location='test-location',
      agent_engine_id='123',
  )


@pytest.fixture
def mock_get_api_client():
  api_client = MockApiClient()
  with mock.patch(
      'google.adk.memory.vertex_ai_memory_bank_service.VertexAiMemoryBankService._get_api_client',
      return_value=api_client,
  ):
    yield api_client


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_add_session_to_memory(mock_get_api_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION)

  mock_get_api_client.async_request.assert_awaited_once_with(
      http_method='POST',
      path='reasoningEngines/123/memories:generate',
      request_dict={
          'direct_contents_source': {
              'events': [
                  {
                      'content': {
                          'parts': [
                              {'text': 'test_content'},
                          ],
                      },
                  },
              ],
          },
          'scope': {'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      },
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_add_empty_session_to_memory(mock_get_api_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

  mock_get_api_client.async_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_search_memory(mock_get_api_client):
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  mock_get_api_client.async_request.assert_awaited_once_with(
      http_method='POST',
      path='reasoningEngines/123/memories:retrieve',
      request_dict={
          'scope': {'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
          'similarity_search_params': {'search_query': 'query'},
      },
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'test_content'
