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


import json
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING


from google.genai import Client
from google.genai import types
from typing_extensions import override


from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry


if TYPE_CHECKING:
  from ..sessions.session import Session


logger = logging.getLogger('google_adk.' + __name__)


class VertexAiMemoryBankService(BaseMemoryService):
  """Implementation of the BaseMemoryService using Vertex AI Memory Bank.
  
  IMPORTANT - Agent Engine ID Extraction (Issue #2940):
  
  When creating an Agent Engine, the `api_resource.name` returns the FULL resource path,
  but this service requires only the Agent Engine ID (the last segment of the path).
  
  Common Error:
      # This will fail - uses full resource path
      agent_engine = client.agent_engines.create()
      agent_engine_id = agent_engine.api_resource.name
      # Returns: "projects/my-project/locations/us-central1/reasoningEngines/123456"
  
  Correct Usage:
      # This works - extract only the ID 
      agent_engine = client.agent_engines.create()
      agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
      # Returns: "123456"
      
      memory_service = VertexAiMemoryBankService(
          project="my-project",
          location="us-central1", 
          agent_engine_id=agent_engine_id  # Use extracted ID
      )
  
  Complete Working Example:
      import vertexai
      from google.adk.memory import VertexAiMemoryBankService
      
      # Create Vertex AI client and Agent Engine
      client = vertexai.Client(project="your-project", location="us-central1")
      agent_engine = client.agent_engines.create()
      
      # CRITICAL: Extract Agent ID from resource name (Issue #2940 fix)
      agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
      
      # Initialize Memory Bank Service with extracted ID
      memory_service = VertexAiMemoryBankService(
          project="your-project",
          location="us-central1",
          agent_engine_id=agent_engine_id  # Use extracted ID, not full path
      )
  
  Note: The agent_engine_id should be just the numeric/alphanumeric ID, not the full
  resource path. If you're getting errors about "Cannot find agent id", make sure
  you're extracting the ID correctly using .split("/")[-1] on the resource name.
  """

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
  ):
    """Initializes a VertexAiMemoryBankService.

    Args:
      project: The project ID of the Memory Bank to use.
      location: The location of the Memory Bank to use.
      agent_engine_id: The ID of the agent engine to use for the Memory Bank.
        IMPORTANT: Use only the agent engine ID, not the full resource path.
        
        Example: Use '456' (correct) instead of 
        'projects/my-project/locations/us-central1/reasoningEngines/456' (incorrect).
        
        To extract the correct ID from api_resource.name:
        agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
    """
    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id

    # Validate agent_engine_id format to help users catch the common mistake
    if agent_engine_id and "/" in agent_engine_id:
      logger.warning(
        f"Agent Engine ID '{agent_engine_id}' contains '/' which suggests it might be "
        "a full resource path instead of just the ID. If you're getting errors, "
        "try extracting the ID using: agent_engine.api_resource.name.split('/')[-1]"
      )

  @override
  async def add_session_to_memory(self, session: Session):
    api_client = self._get_api_client()

    if not self._agent_engine_id:
      raise ValueError(
        'Agent Engine ID is required for Memory Bank. '
        'Make sure to extract the ID from agent_engine.api_resource.name.split("/")[-1]'
      )

    events = []
    for event in session.events:
      if _should_filter_out_event(event.content):
        continue
      if event.content:
        events.append({
            'content': event.content.model_dump(exclude_none=True, mode='json')
        })
    request_dict = {
        'direct_contents_source': {
            'events': events,
        },
        'scope': {
            'app_name': session.app_name,
            'user_id': session.user_id,
        },
    }

    if events:
      try:
        api_response = await api_client.async_request(
            http_method='POST',
            path=f'reasoningEngines/{self._agent_engine_id}/memories:generate',
            request_dict=request_dict,
        )
        logger.info('Generate memory response received.')
        logger.debug('Generate memory response: %s', api_response)
      except Exception as e:
        if "not found" in str(e).lower() or "invalid" in str(e).lower():
          raise ValueError(
            f"Failed to generate memory with agent_engine_id='{self._agent_engine_id}'. "
            "This might be because the agent_engine_id is the full resource path instead of just the ID. "
            "Try using: agent_engine.api_resource.name.split('/')[-1]"
          ) from e
        raise
    else:
      logger.info('No events to add to memory.')

  @override
  async def search_memory(self, *, app_name: str, user_id: str, query: str):
    api_client = self._get_api_client()

    try:
      api_response = await api_client.async_request(
          http_method='POST',
          path=f'reasoningEngines/{self._agent_engine_id}/memories:retrieve',
          request_dict={
              'scope': {
                  'app_name': app_name,
                  'user_id': user_id,
              },
              'similarity_search_params': {
                  'search_query': query,
              },
          },
      )
    except Exception as e:
      if "not found" in str(e).lower() or "invalid" in str(e).lower():
        raise ValueError(
          f"Failed to search memory with agent_engine_id='{self._agent_engine_id}'. "
          "This might be because the agent_engine_id is the full resource path instead of just the ID. "
          "Try using: agent_engine.api_resource.name.split('/')[-1]"
        ) from e
      raise
      
    api_response = _convert_api_response(api_response)
    logger.info('Search memory response received.')
    logger.debug('Search memory response: %s', api_response)

    if not api_response or not api_response.get('retrievedMemories', None):
      return SearchMemoryResponse()

    memory_events = []
    for memory in api_response.get('retrievedMemories', []):
      # TODO: add more complex error handling
      memory_events.append(
          MemoryEntry(
              author='user',
              content=types.Content(
                  parts=[types.Part(text=memory.get('memory').get('fact'))],
                  role='user',
              ),
              timestamp=memory.get('updateTime'),
          )
      )
    return SearchMemoryResponse(memories=memory_events)

  def _get_api_client(self):
    """Instantiates an API client for the given project and location.

    It needs to be instantiated inside each request so that the event loop
    management can be properly propagated.

    Returns:
      An API client for the given project and location.
    """
    client = Client(
        vertexai=True, project=self._project, location=self._location
    )
    return client._api_client


def _convert_api_response(api_response) -> Dict[str, Any]:
  """Converts the API response to a JSON object based on the type."""
  if hasattr(api_response, 'body'):
    return json.loads(api_response.body)
  return api_response


def _should_filter_out_event(content: types.Content) -> bool:
  """Returns whether the event should be filtered out."""
  if not content or not content.parts:
    return True
  for part in content.parts:
    if part.text or part.inline_data or part.file_data:
      return False
  return True
