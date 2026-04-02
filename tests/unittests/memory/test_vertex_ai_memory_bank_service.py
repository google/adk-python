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
from google.adk.memory import vertex_ai_memory_bank_service as memory_service_module
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.session import Session
from google.genai import types
import pytest
from vertexai._genai.types import common as vertex_common_types

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'


def _supports_generate_memories_metadata() -> bool:
  return (
      'metadata'
      in vertex_common_types.GenerateAgentEngineMemoriesConfig.model_fields
  )


def _supports_create_memory_metadata() -> bool:
  return 'metadata' in vertex_common_types.AgentEngineMemoryConfig.model_fields


def _supports_create_memory_revision_labels() -> bool:
  return (
      'revision_labels'
      in vertex_common_types.AgentEngineMemoryConfig.model_fields
  )


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


def test_build_generate_memories_config_uses_runtime_config_keys():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_generate_memories_config_keys',
          return_value=frozenset({'wait_for_completion', 'new_generate_key'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_generate_memories_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_generate_memories_config(
        {'new_generate_key': 'value'}
    )

  assert config == {
      'wait_for_completion': False,
      'new_generate_key': 'value',
  }


def test_build_create_memory_config_uses_runtime_config_keys():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_create_memory_config_keys',
          return_value=frozenset({'wait_for_completion', 'new_create_key'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_create_memory_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_create_memory_config(
        {'new_create_key': 'value'}
    )

  assert config == {
      'wait_for_completion': False,
      'new_create_key': 'value',
  }


def test_build_create_memory_config_merges_revision_labels_when_supported():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_create_memory_config_keys',
          return_value=frozenset({'wait_for_completion', 'revision_labels'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_create_memory_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_create_memory_config(
        {'revision_labels': {'source': 'global'}},
        memory_revision_labels={'author': 'agent'},
    )

  assert config == {
      'wait_for_completion': False,
      'revision_labels': {
          'source': 'global',
          'author': 'agent',
      },
  }


@pytest.fixture
def mock_vertexai_client():
  with mock.patch('vertexai.Client') as mock_client_constructor:
    mock_async_client = mock.MagicMock()
    mock_async_client.agent_engines.memories.generate = mock.AsyncMock()
    mock_async_client.agent_engines.memories.create = mock.AsyncMock()
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


def test_initialize_without_agent_engine_id_error():
  with pytest.raises(
      ValueError,
      match='agent_engine_id is required for VertexAiMemoryBankService',
  ):
    mock_vertex_ai_memory_bank_service(agent_engine_id=None)


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
async def test_add_events_to_memory_with_explicit_events_and_metadata(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={'ttl': '6000s', 'source': 'agent'},
  )

  expected_config = {
      'wait_for_completion': False,
      'revision_ttl': '6000s',
  }
  if _supports_generate_memories_metadata():
    expected_config['metadata'] = {'source': {'string_value': 'agent'}}

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
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
      config=expected_config,
  )
  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_events_to_memory_without_session_id(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      events=[MOCK_SESSION.events[0]],
  )

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
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
  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_events_to_memory_merges_metadata_field_and_unknown_keys(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={
          'metadata': {'origin': 'unit-test'},
          'source': 'agent',
      },
  )

  expected_config = {'wait_for_completion': False}
  if _supports_generate_memories_metadata():
    expected_config['metadata'] = {
        'origin': {'string_value': 'unit-test'},
        'source': {'string_value': 'agent'},
    }

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
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
      config=expected_config,
  )
  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_events_to_memory_none_wait_for_completion_keeps_default(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={'wait_for_completion': None},
  )

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
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
  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_events_to_memory_ttl_used_when_revision_ttl_is_none(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={
          'ttl': '6000s',
          'revision_ttl': None,
      },
  )

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
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
      config={
          'wait_for_completion': False,
          'revision_ttl': '6000s',
      },
  )
  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_events_to_memory_with_filtered_events_skips_rpc(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[1], MOCK_SESSION.events[2]],
  )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_calls_create(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      memories=[
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact one')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact two')])
          ),
      ],
      custom_metadata={
          'enable_consolidation': False,
          'ttl': '6000s',
          'source': 'agent',
      },
  )

  expected_config = {
      'wait_for_completion': False,
      'ttl': '6000s',
  }
  if _supports_create_memory_metadata():
    expected_config['metadata'] = {'source': {'string_value': 'agent'}}

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_has_awaits([
      mock.call(
          name='reasoningEngines/123',
          fact='fact one',
          scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
          config=expected_config,
      ),
      mock.call(
          name='reasoningEngines/123',
          fact='fact two',
          scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
          config=expected_config,
      ),
  ])
  assert mock_vertexai_client.agent_engines.memories.create.await_count == 2

  create_config = (
      mock_vertexai_client.agent_engines.memories.create.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.AgentEngineMemoryConfig(**create_config)


@pytest.mark.asyncio
async def test_add_memory_enable_consolidation_calls_generate_direct_source(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      memories=[
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact one')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact two')])
          ),
      ],
      custom_metadata={
          'enable_consolidation': True,
          'source': 'agent',
      },
  )

  expected_config = {'wait_for_completion': False}
  if _supports_generate_memories_metadata():
    expected_config['metadata'] = {'source': {'string_value': 'agent'}}

  mock_vertexai_client.agent_engines.memories.generate.assert_called_once_with(
      name='reasoningEngines/123',
      direct_memories_source={
          'direct_memories': [
              {'fact': 'fact one'},
              {'fact': 'fact two'},
          ]
      },
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      config=expected_config,
  )
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()

  generate_config = (
      mock_vertexai_client.agent_engines.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.GenerateAgentEngineMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_memory_enable_consolidation_batches_generate_calls(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      memories=[
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact one')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact two')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact three')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact four')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact five')])
          ),
          MemoryEntry(
              content=types.Content(parts=[types.Part(text='fact six')])
          ),
      ],
      custom_metadata={
          'enable_consolidation': True,
      },
  )

  mock_vertexai_client.agent_engines.memories.generate.assert_has_awaits([
      mock.call(
          name='reasoningEngines/123',
          direct_memories_source={
              'direct_memories': [
                  {'fact': 'fact one'},
                  {'fact': 'fact two'},
                  {'fact': 'fact three'},
                  {'fact': 'fact four'},
                  {'fact': 'fact five'},
              ]
          },
          scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
          config={'wait_for_completion': False},
      ),
      mock.call(
          name='reasoningEngines/123',
          direct_memories_source={
              'direct_memories': [
                  {'fact': 'fact six'},
              ]
          },
          scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
          config={'wait_for_completion': False},
      ),
  ])
  assert mock_vertexai_client.agent_engines.memories.generate.await_count == 2
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_invalid_enable_consolidation_type_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      TypeError,
      match=r'custom_metadata\["enable_consolidation"\] must be a bool',
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                content=types.Content(parts=[types.Part(text='fact one')])
            )
        ],
        custom_metadata={'enable_consolidation': 'yes'},
    )
  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_calls_create_with_memory_entry_metadata(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      memories=[
          MemoryEntry(
              author='agent',
              timestamp='2026-02-13T14:46:21Z',
              content=types.Content(parts=[types.Part(text='fact one')]),
              custom_metadata={'source': 'entry'},
          )
      ],
      custom_metadata={'ttl': '6000s', 'source': 'global'},
  )

  expected_config = {
      'wait_for_completion': False,
      'ttl': '6000s',
  }
  if _supports_create_memory_metadata():
    expected_config['metadata'] = {
        'source': {'string_value': 'entry'},
    }
  if _supports_create_memory_revision_labels():
    expected_config['revision_labels'] = {
        'author': 'agent',
        'timestamp': '2026-02-13T14:46:21Z',
    }

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_awaited_once_with(
      name='reasoningEngines/123',
      fact='fact one',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      config=expected_config,
  )
  create_config = (
      mock_vertexai_client.agent_engines.memories.create.call_args.kwargs[
          'config'
      ]
  )
  vertex_common_types.AgentEngineMemoryConfig(**create_config)


@pytest.mark.asyncio
async def test_add_memory_calls_create_with_multimodal_content(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      ValueError,
      match=(
          r'memories\[0\] must include text only; inline_data and file_data '
          r'are not supported'
      ),
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[
                        types.Part(text='caption'),
                        types.Part(
                            file_data=types.FileData(
                                mime_type='image/png',
                                file_uri='gs://bucket/image.png',
                            )
                        ),
                    ]
                )
            )
        ],
    )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_missing_text_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      ValueError,
      match=r'memories\[0\] must include text',
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(name='tool')
                        )
                    ]
                )
            )
        ],
    )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_whitespace_only_text_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      ValueError,
      match=r'memories\[0\] must include non-whitespace text',
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(content=types.Content(parts=[types.Part(text='   ')]))
        ],
    )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_whitespace_and_non_text_parts_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      ValueError,
      match=(
          r'memories\[0\] must include text only; inline_data and file_data '
          r'are not supported'
      ),
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                content=types.Content(
                    parts=[
                        types.Part(text='  '),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type='image/png',
                                data=b'abc',
                            )
                        ),
                    ]
                )
            )
        ],
    )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_missing_memories_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      ValueError, match=r'memories must contain at least one entry'
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[],
    )
  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_invalid_memory_type_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      TypeError,
      match=r'memories\[0\] must be a MemoryEntry',
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[123],
    )
  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_content_type_raises(
    mock_vertexai_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with pytest.raises(
      TypeError,
      match=r'memories\[0\] must be a MemoryEntry',
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[types.Content(parts=[types.Part(text='fact one')])],
    )

  mock_vertexai_client.agent_engines.memories.generate.assert_not_called()
  mock_vertexai_client.agent_engines.memories.create.assert_not_called()


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
