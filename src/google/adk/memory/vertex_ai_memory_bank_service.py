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

from __future__ import annotations

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
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


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
  async def add_session_to_memory(self, session: Session):
    if not self._agent_engine_id:
      raise ValueError('Agent Engine ID is required for Memory Bank.')

    events = []
    for event in session.events:
      if _should_filter_out_event(event.content):
        continue
      if event.content:
        events.append({
            'content': event.content.model_dump(exclude_none=True, mode='json')
        })
    if events:
      client = self._get_api_client()
      operation = client.agent_engines.memories.generate(
          name='reasoningEngines/' + self._agent_engine_id,
          direct_contents_source={'events': events},
          scope={
              'app_name': session.app_name,
              'user_id': session.user_id,
          },
          config={'wait_for_completion': False},
      )
      logger.info('Generate memory response received.')
      logger.debug('Generate memory response: %s', operation)
    else:
      logger.info('No events to add to memory.')

  @override
  async def search_memory(self, *, app_name: str, user_id: str, query: str):
    if not self._agent_engine_id:
      raise ValueError('Agent Engine ID is required for Memory Bank.')

    client = self._get_api_client()
    retrieved_memories_iterator = client.agent_engines.memories.retrieve(
        name='reasoningEngines/' + self._agent_engine_id,
        scope={
            'app_name': app_name,
            'user_id': user_id,
        },
        similarity_search_params={
            'search_query': query,
        },
    )

    logger.info('Search memory response received.')

    memory_events = []
    for retrieved_memory in retrieved_memories_iterator:
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

  def _get_api_client(self):
    """Instantiates an API client for the given project and location.

    It needs to be instantiated inside each request so that the event loop
    management can be properly propagated.
    Returns:
      An API client for the given project and location or express mode api key.
    """
    import vertexai

    return vertexai.Client(
        project=self._project,
        location=self._location,
        api_key=self._express_mode_api_key,
    )


def _should_filter_out_event(content: types.Content) -> bool:
  """Returns whether the event should be filtered out."""
  if not content or not content.parts:
    return True
  for part in content.parts:
    if part.text or part.inline_data or part.file_data:
      return False
  return True
