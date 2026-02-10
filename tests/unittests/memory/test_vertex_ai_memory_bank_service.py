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

from datetime import datetime
from typing import Any
from typing import Iterable
from typing import Optional
from unittest import mock

from google.adk.events.event import Event
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'


class _AsyncListIterator:
  """Minimal async iterator wrapper for list-like results."""

  def __init__(self, items: Iterable[Any]):
    self._items = list(items)
    self._index = 0

  def __aiter__(self) -> '_AsyncListIterator':
    return self

  async def __anext__(self) -> Any:
    if self._index >= len(self._items):
      raise StopAsyncIteration
    item = self._items[self._index]
    self._index += 1
    return item


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


def mock_vertex_ai_memory_bank_service(
    project: Optional[str] = 'test-project',
    location: Optional[str] = 'test-location',
    agent_engine_id: Optional[str] = '123',
    express_mode_api_key: Optional[str] = None,
):
  """Creates a mock Vertex AI Memory Bank service for testing."""
  return VertexAiMemoryBankService(
      project=project,
      location=location,
      agent_engine_id=agent_engine_id,
      express_mode_api_key=express_mode_api_key,
  )


@pytest.fixture
def mock_vertexai_client():
  with mock.patch('vertexai.Client') as mock_client_constructor:
    mock_async_client = mock.MagicMock()
    mock_async_client.agent_engines.memories.generate = mock.AsyncMock()
    mock_async_client.agent_engines.memories.retrieve = mock.AsyncMock()

    mock_client = mock.MagicMock()
    mock_client.aio = mock_async_client

    mock_client_constructor.return_value = mock_client
    yield mock_async_client


@pytest.mark.asyncio
async def test_initialize_with_project_location_and_api_key_error():
  with pytest.raises(ValueError) as excinfo:
    mock_vertex_ai_memory_bank_service(
        project='test-project',
        location='test-location',
        express_mode_api_key='test-api-key',
    )
  assert (
      'Cannot specify project or location and express_mode_api_key. Either use'
      ' project and location, or just the express_mode_api_key.'
      in str(excinfo.value)
  )


@pytest.mark.asyncio
async def test_add_session_to_memory(mock_vertexai_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION)

  mock_vertexai_client.agent_engines.memories.generate.assert_awaited_once_with(
      name='reasoningEngines/123',
      direct_contents_source={
          'events': [
              {
                  'content': {
                      'parts': [{'text': 'test_content'}],
                  }
              }
          ]
      },
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      config={'wait_for_completion': False},
  )


@pytest.mark.asyncio
async def test_add_empty_session_to_memory(mock_vertexai_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

  mock_vertexai_client.agent_engines.memories.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_memory(mock_vertexai_client):
  retrieved_memory = mock.MagicMock()
  retrieved_memory.memory.fact = 'test_content'
  retrieved_memory.memory.update_time = datetime(
      2024, 12, 12, 12, 12, 12, 123456
  )

  mock_vertexai_client.agent_engines.memories.retrieve.return_value = (
      _AsyncListIterator([retrieved_memory])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  mock_vertexai_client.agent_engines.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'test_content'


@pytest.mark.asyncio
async def test_search_memory_empty_results(mock_vertexai_client):
  mock_vertexai_client.agent_engines.memories.retrieve.return_value = (
      _AsyncListIterator([])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  mock_vertexai_client.agent_engines.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )

  assert len(result.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_uses_async_client_path():
  sync_client = mock.MagicMock()
  sync_client.agent_engines.memories.retrieve.side_effect = AssertionError(
      'sync retrieve should not be called'
  )

  async_client = mock.MagicMock()
  async_client.agent_engines.memories.retrieve = mock.AsyncMock(
      return_value=_AsyncListIterator([])
  )

  with mock.patch('vertexai.Client') as mock_client_constructor:
    mock_client_constructor.return_value = mock.MagicMock(
        aio=async_client,
        agent_engines=sync_client.agent_engines,
    )
    memory_service = mock_vertex_ai_memory_bank_service()
    await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='query',
    )

  async_client.agent_engines.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )
  sync_client.agent_engines.memories.retrieve.assert_not_called()
