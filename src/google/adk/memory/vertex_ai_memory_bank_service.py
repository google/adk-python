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

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
import logging
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..utils.vertex_ai_utils import get_express_mode_api_key
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  import vertexai

  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)

_GENERATE_MEMORIES_CONFIG_KEYS = frozenset({
    'disable_consolidation',
    'disable_memory_revisions',
    'http_options',
    'metadata',
    'metadata_merge_strategy',
    'revision_expire_time',
    'revision_labels',
    'revision_ttl',
    'wait_for_completion',
})


def _supports_generate_memories_metadata() -> bool:
  """Returns whether installed Vertex SDK supports config.metadata."""
  try:
    from vertexai._genai.types import common as vertex_common_types
  except ImportError:
    return False
  return (
      'metadata'
      in vertex_common_types.GenerateAgentEngineMemoriesConfig.model_fields
  )


class VertexAiMemoryBankService(BaseMemoryService):
  """Implementation of the BaseMemoryService using Vertex AI Memory Bank."""

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
      *,
      express_mode_api_key: Optional[str] = None,
  ):
    """Initializes a VertexAiMemoryBankService.

    Args:
      project: The project ID of the Memory Bank to use.
      location: The location of the Memory Bank to use.
      agent_engine_id: The ID of the agent engine to use for the Memory Bank,
        e.g. '456' in
        'projects/my-project/locations/us-central1/reasoningEngines/456'. To
        extract from api_resource.name, use:
        ``agent_engine.api_resource.name.split('/')[-1]``
      express_mode_api_key: The API key to use for Express Mode. If not
        provided, the API key from the GOOGLE_API_KEY environment variable will
        be used. It will only be used if GOOGLE_GENAI_USE_VERTEXAI is true. Do
        not use Google AI Studio API key for this field. For more details, visit
        https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
    """
    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id
    self._express_mode_api_key = get_express_mode_api_key(
        project, location, express_mode_api_key
    )

    if agent_engine_id and '/' in agent_engine_id:
      logger.warning(
          "agent_engine_id appears to be a full resource path: '%s'. "
          "Expected just the ID (e.g., '456'). "
          "Extract the ID using: agent_engine.api_resource.name.split('/')[-1]",
          agent_engine_id,
      )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self._add_events_to_memory_from_events(
        app_name=session.app_name,
        user_id=session.user_id,
        events_to_process=session.events,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    _ = session_id
    await self._add_events_to_memory_from_events(
        app_name=app_name,
        user_id=user_id,
        events_to_process=events,
        custom_metadata=custom_metadata,
    )

  async def _add_events_to_memory_from_events(
      self,
      *,
      app_name: str,
      user_id: str,
      events_to_process: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    if not self._agent_engine_id:
      raise ValueError('Agent Engine ID is required for Memory Bank.')

    direct_events = []
    for event in events_to_process:
      if _should_filter_out_event(event.content):
        continue
      if event.content:
        direct_events.append({
            'content': event.content.model_dump(exclude_none=True, mode='json')
        })
    if direct_events:
      api_client = self._get_api_client()
      config = _build_generate_memories_config(custom_metadata)
      operation = await api_client.agent_engines.memories.generate(
          name='reasoningEngines/' + self._agent_engine_id,
          direct_contents_source={'events': direct_events},
          scope={
              'app_name': app_name,
              'user_id': user_id,
          },
          config=config,
      )
      logger.info('Generate memory response received.')
      logger.debug('Generate memory response: %s', operation)
    else:
      logger.info('No events to add to memory.')

  @override
  async def search_memory(self, *, app_name: str, user_id: str, query: str):
    if not self._agent_engine_id:
      raise ValueError('Agent Engine ID is required for Memory Bank.')

    api_client = self._get_api_client()
    retrieved_memories_iterator = (
        await api_client.agent_engines.memories.retrieve(
            name='reasoningEngines/' + self._agent_engine_id,
            scope={
                'app_name': app_name,
                'user_id': user_id,
            },
            similarity_search_params={
                'search_query': query,
            },
        )
    )

    logger.info('Search memory response received.')

    memory_events: list[MemoryEntry] = []
    async for retrieved_memory in retrieved_memories_iterator:
      # TODO: add more complex error handling
      logger.debug('Retrieved memory: %s', retrieved_memory)
      memory_events.append(
          MemoryEntry(
              author='user',
              content=types.Content(
                  parts=[types.Part(text=retrieved_memory.memory.fact)],
                  role='user',
              ),
              timestamp=retrieved_memory.memory.update_time.isoformat(),
          )
      )
    return SearchMemoryResponse(memories=memory_events)

  def _get_api_client(self) -> vertexai.AsyncClient:
    """Instantiates an API client for the given project and location.

    It needs to be instantiated inside each request so that the event loop
    management can be properly propagated.
    Returns:
      An async API client for the given project and location or express mode api
      key.
    """
    import vertexai

    return vertexai.Client(
        project=self._project,
        location=self._location,
        api_key=self._express_mode_api_key,
    ).aio


def _should_filter_out_event(content: types.Content) -> bool:
  """Returns whether the event should be filtered out."""
  if not content or not content.parts:
    return True
  for part in content.parts:
    if part.text or part.inline_data or part.file_data:
      return False
  return True


def _build_generate_memories_config(
    custom_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
  """Builds a valid memories.generate config from caller metadata."""
  config: dict[str, object] = {'wait_for_completion': False}
  supports_metadata = _supports_generate_memories_metadata()
  if not custom_metadata:
    return config

  logger.debug('Memory generation metadata: %s', custom_metadata)

  metadata_by_key: dict[str, object] = {}
  for key, value in custom_metadata.items():
    if key == 'ttl':
      if value is None:
        continue
      if custom_metadata.get('revision_ttl') is None:
        config['revision_ttl'] = value
      continue
    if key == 'metadata':
      if value is None:
        continue
      if not supports_metadata:
        logger.warning(
            'Ignoring metadata because installed Vertex SDK does not support'
            ' config.metadata.'
        )
        continue
      if isinstance(value, Mapping):
        config['metadata'] = _build_vertex_metadata(value)
      else:
        logger.warning(
            'Ignoring metadata because custom_metadata["metadata"] is not a'
            ' mapping.'
        )
      continue
    if key in _GENERATE_MEMORIES_CONFIG_KEYS:
      if value is None:
        continue
      config[key] = value
    else:
      metadata_by_key[key] = value

  if not metadata_by_key:
    return config

  if not supports_metadata:
    logger.warning(
        'Ignoring custom metadata keys %s because installed Vertex SDK does '
        'not support config.metadata.',
        sorted(metadata_by_key.keys()),
    )
    return config

  existing_metadata = config.get('metadata')
  if existing_metadata is None:
    config['metadata'] = _build_vertex_metadata(metadata_by_key)
    return config

  if isinstance(existing_metadata, Mapping):
    merged_metadata = dict(existing_metadata)
    merged_metadata.update(_build_vertex_metadata(metadata_by_key))
    config['metadata'] = merged_metadata
    return config

  logger.warning(
      'Ignoring custom metadata keys %s because config.metadata is not a'
      ' mapping.',
      sorted(metadata_by_key.keys()),
  )
  return config


def _build_vertex_metadata(
    metadata_by_key: Mapping[str, object],
) -> dict[str, object]:
  """Converts metadata values to Vertex MemoryMetadataValue objects."""
  vertex_metadata: dict[str, object] = {}
  for key, value in metadata_by_key.items():
    converted_value = _to_vertex_metadata_value(key, value)
    if converted_value is None:
      continue
    vertex_metadata[key] = converted_value
  return vertex_metadata


def _to_vertex_metadata_value(
    key: str,
    value: object,
) -> dict[str, object] | None:
  """Converts a metadata value to Vertex MemoryMetadataValue shape."""
  if isinstance(value, bool):
    return {'bool_value': value}
  if isinstance(value, (int, float)):
    return {'double_value': float(value)}
  if isinstance(value, str):
    return {'string_value': value}
  if isinstance(value, datetime):
    return {'timestamp_value': value}
  if isinstance(value, Mapping):
    if value.keys() <= {
        'bool_value',
        'double_value',
        'string_value',
        'timestamp_value',
    }:
      return dict(value)
    return {'string_value': str(dict(value))}
  if value is None:
    logger.warning(
        'Ignoring custom metadata key %s because its value is None.',
        key,
    )
    return None
  return {'string_value': str(value)}
