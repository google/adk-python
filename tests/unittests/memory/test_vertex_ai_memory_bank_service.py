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
import datetime
import logging
from types import SimpleNamespace
from typing import Any
from typing import Iterable
from typing import Optional
from unittest import mock

from agentplatform import types as agentplatform_types
from google.adk.events.event import Event
from google.adk.memory import vertex_ai_memory_bank_service as memory_service_module
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.session import Session
from google.auth.credentials import Credentials
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'


def _supports_generate_memories_metadata() -> bool:
  return 'metadata' in agentplatform_types.GenerateMemoriesConfig.model_fields


def _supports_create_memory_metadata() -> bool:
  return 'metadata' in agentplatform_types.MemoryConfig.model_fields


def _supports_create_memory_revision_labels() -> bool:
  return 'revision_labels' in agentplatform_types.MemoryConfig.model_fields


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
        # Function call event, no longer filtered out
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
    credentials: Optional[Credentials] = None,
):
  """Creates a mock Vertex AI Memory Bank service for testing."""
  return VertexAiMemoryBankService(
      project=project,
      location=location,
      agent_engine_id=agent_engine_id,
      express_mode_api_key=express_mode_api_key,
      credentials=credentials,
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


def test_build_create_memory_config_sets_memory_id_when_supported():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_create_memory_config_keys',
          return_value=frozenset({'wait_for_completion', 'memory_id'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_create_memory_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_create_memory_config(
        None, memory_id='mem-123'
    )

  assert config == {'wait_for_completion': False, 'memory_id': 'mem-123'}


def test_build_create_memory_config_omits_memory_id_when_none():
  with mock.patch.object(
      memory_service_module,
      '_get_create_memory_config_keys',
      return_value=frozenset({'wait_for_completion', 'memory_id'}),
  ):
    config = memory_service_module._build_create_memory_config(
        None, memory_id=None
    )

  assert config == {'wait_for_completion': False}


def test_build_create_memory_config_ignores_memory_id_when_unsupported():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_create_memory_config_keys',
          return_value=frozenset({'wait_for_completion'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_create_memory_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_create_memory_config(
        None, memory_id='mem-123'
    )

  assert config == {'wait_for_completion': False}


def test_build_create_memory_config_custom_metadata_memory_id_wins():
  with (
      mock.patch.object(
          memory_service_module,
          '_get_create_memory_config_keys',
          return_value=frozenset({'wait_for_completion', 'memory_id'}),
      ),
      mock.patch.object(
          memory_service_module,
          '_supports_create_memory_metadata',
          return_value=False,
      ),
  ):
    config = memory_service_module._build_create_memory_config(
        {'memory_id': 'explicit'}, memory_id='from-entry'
    )

  assert config['memory_id'] == 'explicit'


@pytest.fixture
def mock_agentplatform_client():
  with mock.patch('agentplatform.Client') as mock_client_constructor:
    mock_async_client = mock.MagicMock()
    mock_async_client.memory_banks.memories.generate = mock.AsyncMock()
    mock_async_client.memory_banks.memories.create = mock.AsyncMock()
    mock_async_client.memory_banks.memories.retrieve = mock.AsyncMock()
    mock_async_client.memory_banks.memories.retrieve_profiles = mock.AsyncMock()
    mock_async_client.memory_banks.ingest_events = mock.AsyncMock()

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


def test_get_api_client_passes_credentials_through():
  mock_credentials = mock.MagicMock(spec=Credentials)
  memory_service = mock_vertex_ai_memory_bank_service(
      credentials=mock_credentials
  )

  with mock.patch('agentplatform.Client') as mock_client_constructor:
    memory_service._get_api_client()

  mock_client_constructor.assert_called_once_with(
      project='test-project',
      location='test-location',
      credentials=mock_credentials,
  )


def test_get_api_client_defaults_credentials_to_none():
  memory_service = mock_vertex_ai_memory_bank_service()

  with mock.patch('agentplatform.Client') as mock_client_constructor:
    memory_service._get_api_client()

  mock_client_constructor.assert_called_once_with(
      project='test-project',
      location='test-location',
      credentials=None,
  )


def test_get_api_client_reuses_one_client_within_an_event_loop():
  """Reuses one Vertex AI client across calls in the same event loop."""
  memory_service = mock_vertex_ai_memory_bank_service()

  async def add_events_twice():
    await memory_service.add_events_to_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        events=[MOCK_SESSION.events[0]],
    )
    await memory_service.add_events_to_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        events=[MOCK_SESSION.events[0]],
    )

  with mock.patch(
      'agentplatform.Client',
      side_effect=lambda **_: mock.MagicMock(aio=mock.AsyncMock()),
  ) as mock_client_constructor:
    asyncio.run(add_events_twice())

  assert mock_client_constructor.call_count == 1


def test_get_api_client_builds_a_separate_client_per_event_loop():
  """Builds a separate Vertex AI client for each event loop."""
  memory_service = mock_vertex_ai_memory_bank_service()

  with mock.patch(
      'agentplatform.Client',
      side_effect=lambda **_: mock.MagicMock(aio=mock.AsyncMock()),
  ) as mock_client_constructor:
    asyncio.run(
        memory_service.add_events_to_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            events=[MOCK_SESSION.events[0]],
        )
    )
    asyncio.run(
        memory_service.add_events_to_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            events=[MOCK_SESSION.events[0]],
        )
    )

  assert mock_client_constructor.call_count == 2


@pytest.mark.asyncio
async def test_add_session_to_memory(mock_agentplatform_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION)

  # Allow the fire-and-forget task to complete.
  await asyncio.sleep(0)

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.ingest_events.assert_awaited_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.ingest_events.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 2
  assert source.events[0].event_id == '444'
  assert source.events[0].content.parts[0].text == 'test_content'
  assert source.events[0].event_time == datetime.datetime.fromtimestamp(
      12345, tz=datetime.timezone.utc
  )
  assert source.events[1].event_id == '666'
  assert source.events[1].content.parts[0].function_call.name == 'test_function'


@pytest.mark.asyncio
async def test_add_events_to_memory_with_explicit_events_and_metadata(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={
          'ttl': '6000s',
          'source': 'agent',
      },
  )

  expected_config = {
      'wait_for_completion': False,
      'revision_ttl': '6000s',
  }
  if _supports_generate_memories_metadata():
    expected_config['metadata'] = {'source': {'string_value': 'agent'}}

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['config'] == expected_config
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].content.parts[0].text == 'test_content'
  agentplatform_types.GenerateMemoriesConfig(**call_kwargs['config'])


@pytest.mark.asyncio
async def test_add_events_to_memory_without_session_id(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={'revision_ttl': '3600s'},
  )

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['config'] == {
      'wait_for_completion': False,
      'revision_ttl': '3600s',
  }
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].content.parts[0].text == 'test_content'
  agentplatform_types.GenerateMemoriesConfig(**call_kwargs['config'])
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_events_to_memory_merges_metadata_field_and_unknown_keys(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['config'] == expected_config
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].content.parts[0].text == 'test_content'
  agentplatform_types.GenerateMemoriesConfig(**call_kwargs['config'])


@pytest.mark.asyncio
async def test_add_events_to_memory_none_wait_for_completion_keeps_default(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={
          'wait_for_completion': None,
      },
  )

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['config'] == {'wait_for_completion': False}
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].content.parts[0].text == 'test_content'
  agentplatform_types.GenerateMemoriesConfig(**call_kwargs['config'])


@pytest.mark.asyncio
async def test_add_events_to_memory_ttl_used_when_revision_ttl_is_none(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['config'] == {
      'wait_for_completion': False,
      'revision_ttl': '6000s',
  }
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].content.parts[0].text == 'test_content'
  agentplatform_types.GenerateMemoriesConfig(**call_kwargs['config'])


@pytest.mark.asyncio
async def test_add_events_to_memory_with_filtered_events_skips_rpc(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      session_id=MOCK_SESSION.id,
      events=[MOCK_SESSION.events[1]],
      custom_metadata={'revision_ttl': '3600s'},
  )

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_events_to_memory_via_ingest(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      events=[MOCK_SESSION.events[0]],
      custom_metadata={
          'stream_id': 'stream-123',
          'force_flush': True,
          'generation_trigger_config': {
              'generation_rule': {'idle_duration': '60s'},
          },
      },
  )

  # Allow the fire-and-forget task to complete.
  await asyncio.sleep(0)

  mock_agentplatform_client.memory_banks.ingest_events.assert_awaited_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.ingest_events.call_args.kwargs
  )
  assert call_kwargs['name'] == 'reasoningEngines/123'
  assert call_kwargs['scope'] == {
      'app_name': MOCK_APP_NAME,
      'user_id': MOCK_USER_ID,
  }
  assert call_kwargs['stream_id'] == 'stream-123'
  assert call_kwargs['config'] == {'force_flush': True}
  assert call_kwargs['generation_trigger_config'] == {
      'generation_rule': {'idle_duration': '60s'},
  }
  source = call_kwargs['direct_contents_source']
  assert len(source.events) == 1
  assert source.events[0].event_id == '444'
  assert source.events[0].content.parts[0].text == 'test_content'
  assert source.events[0].event_time == datetime.datetime.fromtimestamp(
      12345, tz=datetime.timezone.utc
  )


@pytest.mark.asyncio
async def test_add_events_to_memory_via_ingest_no_events(
    mock_agentplatform_client,
):
  """No-events requests are valid for trigger config updates."""
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_events_to_memory(
      app_name=MOCK_SESSION.app_name,
      user_id=MOCK_SESSION.user_id,
      events=[],
      custom_metadata={
          'generation_trigger_config': {
              'generation_rule': {'idle_duration': '60s'},
          },
      },
  )

  # Allow the fire-and-forget task to complete.
  await asyncio.sleep(0)

  mock_agentplatform_client.memory_banks.ingest_events.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      generation_trigger_config={
          'generation_rule': {'idle_duration': '60s'},
      },
  )


@pytest.mark.asyncio
async def test_add_memory_calls_create(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_has_awaits([
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
  assert mock_agentplatform_client.memory_banks.memories.create.await_count == 2

  create_config = (
      mock_agentplatform_client.memory_banks.memories.create.call_args.kwargs[
          'config'
      ]
  )
  agentplatform_types.MemoryConfig(**create_config)


@pytest.mark.asyncio
async def test_add_memory_enable_consolidation_calls_generate_direct_source(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once_with(
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
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()

  generate_config = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs[
          'config'
      ]
  )
  agentplatform_types.GenerateMemoriesConfig(**generate_config)


@pytest.mark.asyncio
async def test_add_memory_enable_consolidation_batches_generate_calls(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_has_awaits([
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
  assert (
      mock_agentplatform_client.memory_banks.memories.generate.await_count == 2
  )
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_invalid_enable_consolidation_type_raises(
    mock_agentplatform_client,
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
  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_calls_create_with_memory_entry_metadata(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_awaited_once_with(
      name='reasoningEngines/123',
      fact='fact one',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      config=expected_config,
  )
  create_config = (
      mock_agentplatform_client.memory_banks.memories.create.call_args.kwargs[
          'config'
      ]
  )
  agentplatform_types.MemoryConfig(**create_config)


@pytest.mark.asyncio
async def test_add_events_to_memory_allowed_topics_routes_to_generate(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with mock.patch.object(
      memory_service_module,
      '_get_generate_memories_config_keys',
      return_value=frozenset({'wait_for_completion', 'allowed_topics'}),
  ):
    await memory_service.add_events_to_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        events=[MOCK_SESSION.events[0]],
        custom_metadata={'allowed_topics': ['USER_PREFERENCES']},
    )

  mock_agentplatform_client.memory_banks.ingest_events.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.generate.assert_called_once()
  call_kwargs = (
      mock_agentplatform_client.memory_banks.memories.generate.call_args.kwargs
  )
  assert call_kwargs['config']['allowed_topics'] == ['USER_PREFERENCES']


@pytest.mark.asyncio
async def test_add_memory_forwards_entry_id_as_memory_id(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with mock.patch.object(
      memory_service_module,
      '_get_create_memory_config_keys',
      return_value=frozenset({'wait_for_completion', 'memory_id'}),
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                id='mem-123',
                content=types.Content(parts=[types.Part(text='fact one')]),
            )
        ],
    )

  create_config = (
      mock_agentplatform_client.memory_banks.memories.create.call_args.kwargs[
          'config'
      ]
  )
  assert create_config['memory_id'] == 'mem-123'


@pytest.mark.asyncio
async def test_add_memory_custom_metadata_memory_id_overrides_entry_id(
    mock_agentplatform_client,
):
  memory_service = mock_vertex_ai_memory_bank_service()
  with mock.patch.object(
      memory_service_module,
      '_get_create_memory_config_keys',
      return_value=frozenset({'wait_for_completion', 'memory_id'}),
  ):
    await memory_service.add_memory(
        app_name=MOCK_SESSION.app_name,
        user_id=MOCK_SESSION.user_id,
        memories=[
            MemoryEntry(
                id='from-entry',
                content=types.Content(parts=[types.Part(text='fact one')]),
            )
        ],
        custom_metadata={'memory_id': 'explicit'},
    )

  create_config = (
      mock_agentplatform_client.memory_banks.memories.create.call_args.kwargs[
          'config'
      ]
  )
  assert create_config['memory_id'] == 'explicit'


@pytest.mark.asyncio
async def test_add_memory_calls_create_with_multimodal_content(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_missing_text_raises(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_whitespace_only_text_raises(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_whitespace_and_non_text_parts_raises(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_missing_memories_raises(
    mock_agentplatform_client,
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
  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_invalid_memory_type_raises(
    mock_agentplatform_client,
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
  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_with_content_type_raises(
    mock_agentplatform_client,
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

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.memories.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_empty_session_to_memory(mock_agentplatform_client):
  memory_service = mock_vertex_ai_memory_bank_service()
  await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

  # Allow the fire-and-forget task to complete.
  await asyncio.sleep(0)

  mock_agentplatform_client.memory_banks.memories.generate.assert_not_called()
  mock_agentplatform_client.memory_banks.ingest_events.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
  )


class _LegacyShapedMemoryClient:
  """Shaped like a 2.x `vertexai` client: Memory Bank exists only under
  `agent_engines.memories`, `ingest_events` included, and there is no
  top-level `memory_banks` attribute."""

  def __init__(self) -> None:
    self.memories = SimpleNamespace(
        generate=mock.AsyncMock(),
        create=mock.AsyncMock(),
        retrieve=mock.AsyncMock(),
        retrieve_profiles=mock.AsyncMock(),
        ingest_events=mock.AsyncMock(),
    )
    self.agent_engines = SimpleNamespace(memories=self.memories)


@pytest.mark.asyncio
async def test_falls_back_to_agent_engines_for_a_vertexai_shaped_client():
  """A subclass returning a `vertexai` client keeps working.

  Several subclasses override `_get_api_client` to return
  `vertexai.Client(...).aio`, which in 2.x only has `agent_engines.memories`.
  Memory operations have to reach that path rather than raise AttributeError
  on `.memory_banks`.
  """
  legacy_client = _LegacyShapedMemoryClient()
  assert getattr(legacy_client, 'memory_banks', None) is None
  retrieved_memory = mock.MagicMock()
  retrieved_memory.memory.fact = 'legacy_content'
  retrieved_memory.memory.update_time = datetime.datetime(2024, 12, 12)
  legacy_client.memories.retrieve.return_value = _AsyncListIterator(
      [retrieved_memory]
  )
  legacy_client.memories.retrieve_profiles.return_value = (
      agentplatform_types.RetrieveProfilesResponse(profiles=None)
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  with mock.patch.object(
      memory_service, '_get_api_client', return_value=legacy_client
  ):
    searched = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
    )
    profiles = await memory_service.retrieve_profiles(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID
    )
    await memory_service.add_session_to_memory(MOCK_SESSION)

  assert searched.memories[0].content.parts[0].text == 'legacy_content'
  assert profiles == []
  legacy_client.memories.retrieve.assert_awaited_once()
  legacy_client.memories.retrieve_profiles.assert_awaited_once()
  # add_session_to_memory defaults to ingest_events, which the service fires
  # as a background task: the call is made immediately, the await later.
  legacy_client.memories.ingest_events.assert_called_once()


def test_ingest_events_and_memories_resolve_for_both_client_shapes():
  """Both accessors pick the v2 path when present and the legacy one otherwise.

  `ingest_events` is the one call whose location differs between the paths: it
  sits on `memory_banks` itself, but under `memories` on the legacy client.
  """
  # pylint: disable=protected-access
  from google.adk.memory import vertex_ai_memory_bank_service as module

  legacy_client = _LegacyShapedMemoryClient()
  assert module._memories_api(legacy_client) is legacy_client.memories
  assert (
      module._ingest_events_api(legacy_client)
      is legacy_client.memories.ingest_events
  )

  v2_client = SimpleNamespace(
      memory_banks=SimpleNamespace(memories=object(), ingest_events=object())
  )
  assert module._memories_api(v2_client) is v2_client.memory_banks.memories
  assert (
      module._ingest_events_api(v2_client)
      is v2_client.memory_banks.ingest_events
  )


@pytest.mark.asyncio
async def test_search_memory(mock_agentplatform_client):
  retrieved_memory = mock.MagicMock()
  retrieved_memory.memory.fact = 'test_content'
  retrieved_memory.memory.update_time = datetime.datetime(
      2024, 12, 12, 12, 12, 12, 123456
  )

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([retrieved_memory])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  mock_agentplatform_client.memory_banks.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'test_content'
  assert result.memories[0].custom_metadata == {}


@pytest.mark.asyncio
async def test_search_memory_returns_custom_metadata(mock_agentplatform_client):
  """`search_memory` must round-trip `custom_metadata`."""
  timestamp = datetime.datetime(2024, 12, 12, 12, 12, 12, 123456)
  retrieved_memory = mock.MagicMock()
  retrieved_memory.memory.fact = 'test_content'
  retrieved_memory.memory.update_time = timestamp
  retrieved_memory.memory.metadata = {
      'a_bool': agentplatform_types.MemoryMetadataValue(bool_value=True),
      'a_double': agentplatform_types.MemoryMetadataValue(double_value=1.5),
      'a_string': agentplatform_types.MemoryMetadataValue(
          string_value='record-123'
      ),
      'a_timestamp': agentplatform_types.MemoryMetadataValue(
          timestamp_value=timestamp
      ),
      'a_mapping': {'string_value': 'mapping-val'},
      'a_mapping_with_none': {
          'bool_value': None,
          'double_value': None,
          'string_value': 'none-mapping-val',
          'timestamp_value': None,
      },
  }

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([retrieved_memory])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].custom_metadata == {
      'a_bool': True,
      'a_double': 1.5,
      'a_string': 'record-123',
      'a_timestamp': timestamp,
      'a_mapping': 'mapping-val',
      'a_mapping_with_none': 'none-mapping-val',
  }


@pytest.mark.asyncio
async def test_search_memory_when_memory_has_no_metadata_attr(
    mock_agentplatform_client,
):
  """A memory object missing the metadata attribute returns an empty dict."""
  retrieved_memory = mock.MagicMock()
  retrieved_memory.memory.fact = 'test_content'
  retrieved_memory.memory.update_time = None
  del retrieved_memory.memory.metadata

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([retrieved_memory])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'test_content'
  assert result.memories[0].custom_metadata == {}


@pytest.mark.asyncio
async def test_search_memory_empty_results(mock_agentplatform_client):
  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  mock_agentplatform_client.memory_banks.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )

  assert len(result.memories) == 0


@pytest.mark.asyncio
async def test_retrieve_profiles(mock_agentplatform_client, caplog):
  """Returns the structured profiles for the scope as a list."""
  retrieve_profiles_response = agentplatform_types.RetrieveProfilesResponse(
      profiles={
          'user-profile': agentplatform_types.MemoryProfile(
              schema_id='user-profile',
              profile={'name': 'Kim'},
          )
      }
  )
  mock_agentplatform_client.memory_banks.memories.retrieve_profiles.return_value = (
      retrieve_profiles_response
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  with caplog.at_level(logging.INFO):
    result = await memory_service.retrieve_profiles(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
    )

  mock_agentplatform_client.memory_banks.memories.retrieve_profiles.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
  )
  assert 'Retrieved 1 memory profiles.' in caplog.text
  assert result == [
      agentplatform_types.MemoryProfile(
          schema_id='user-profile',
          profile={'name': 'Kim'},
      )
  ]


@pytest.mark.asyncio
async def test_retrieve_profiles_empty_results(
    mock_agentplatform_client, caplog
):
  """Returns an empty list when the scope has no profiles."""
  retrieve_profiles_response = agentplatform_types.RetrieveProfilesResponse(
      profiles=None
  )
  mock_agentplatform_client.memory_banks.memories.retrieve_profiles.return_value = (
      retrieve_profiles_response
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  with caplog.at_level(logging.INFO):
    result = await memory_service.retrieve_profiles(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
    )

  mock_agentplatform_client.memory_banks.memories.retrieve_profiles.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
  )
  assert 'Retrieved no memory profiles.' in caplog.text
  assert not result


async def test_search_memory_uses_async_client_path():
  sync_client = mock.MagicMock()
  sync_client.memory_banks.memories.retrieve.side_effect = AssertionError(
      'sync retrieve should not be called'
  )

  async_client = mock.MagicMock()
  async_client.memory_banks.memories.retrieve = mock.AsyncMock(
      return_value=_AsyncListIterator([])
  )

  with mock.patch('agentplatform.Client') as mock_client_constructor:
    mock_client_constructor.return_value = mock.MagicMock(
        aio=async_client,
        memory_banks=sync_client.memory_banks,
    )
    memory_service = mock_vertex_ai_memory_bank_service()
    await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='query',
    )

  async_client.memory_banks.memories.retrieve.assert_awaited_once_with(
      name='reasoningEngines/123',
      scope={'app_name': MOCK_APP_NAME, 'user_id': MOCK_USER_ID},
      similarity_search_params={'search_query': 'query'},
  )
  sync_client.memory_banks.memories.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_skips_entry_with_none_memory(
    mock_agentplatform_client,
):
  bad_entry = mock.MagicMock()
  bad_entry.memory = None

  good_entry = mock.MagicMock()
  good_entry.memory.fact = 'good fact'
  good_entry.memory.update_time = datetime.datetime(2024, 1, 1)

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([bad_entry, good_entry])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'good fact'


@pytest.mark.asyncio
async def test_search_memory_skips_entry_with_empty_fact(
    mock_agentplatform_client,
):
  for empty_fact in [None, '']:
    bad_entry = mock.MagicMock()
    bad_entry.memory.fact = empty_fact
    bad_entry.memory.update_time = datetime.datetime(2024, 1, 1)

    mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
        _AsyncListIterator([bad_entry])
    )
    memory_service = mock_vertex_ai_memory_bank_service()

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
    )

    assert len(result.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_handles_missing_update_time(
    mock_agentplatform_client,
):
  entry = mock.MagicMock()
  entry.memory.fact = 'some fact'
  entry.memory.update_time = None

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([entry])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'some fact'
  assert result.memories[0].timestamp is None


@pytest.mark.asyncio
async def test_search_memory_skips_malformed_entry(mock_agentplatform_client):
  malformed = mock.MagicMock(spec=[])  # no attributes → AttributeError

  good_entry = mock.MagicMock()
  good_entry.memory.fact = 'good fact'
  good_entry.memory.update_time = datetime.datetime(2024, 1, 1)

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      _AsyncListIterator([malformed, good_entry])
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'good fact'


@pytest.mark.asyncio
async def test_search_memory_returns_partial_results_on_iterator_error(
    mock_agentplatform_client,
):
  good_entry = mock.MagicMock()
  good_entry.memory.fact = 'good fact'
  good_entry.memory.update_time = datetime.datetime(2024, 1, 1)

  async def failing_async_iterator():
    yield good_entry
    raise RuntimeError('API stream error')

  mock_agentplatform_client.memory_banks.memories.retrieve.return_value = (
      failing_async_iterator()
  )
  memory_service = mock_vertex_ai_memory_bank_service()

  result = await memory_service.search_memory(
      app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='query'
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == 'good fact'
