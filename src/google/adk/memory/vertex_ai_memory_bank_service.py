"""Implementation of the ADK BaseMemorySerivce using Memory Bank."""

from __future__ import annotations

import logging
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from google import genai

from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)


class VertexAiMemoryBankService(BaseMemoryService):
  """Implementation of the ADK BaseMemorySerivce using Memory Bank."""

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
      is_vertex_session: bool = True,
  ):
    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id
    self._is_vertex_session = is_vertex_session

  @override
  async def add_session_to_memory(self, session: Session):
    api_client = self._get_api_client()

    if not self._agent_engine_id:
      raise ValueError("Agent Engine ID is required for Memory Bank.")

    if self._is_vertex_session:
      request_dict = {
          "vertex_session_source": {
              "session": f"projects/{self._project}/locations/{self._location}/reasoningEngines/{self._agent_engine_id}/sessions/{session.id}"
          },
          "scope": {
              "app_name": session.app_name,
              "user_id": session.user_id,
          },
      }
    else:
      raise ValueError("Unsupported session source.")

    api_response = await api_client.async_request(
        http_method="POST",
        path=f"reasoningEngines/{self._agent_engine_id}/memories:generate",
        request_dict=request_dict,
    )
    logger.info(f"Add session to memory response: {api_response}")
    return api_response

  @override
  async def search_memory(self, *, app_name: str, user_id: str, query: str):
    api_client = self._get_api_client()

    api_response = await api_client.async_request(
        http_method="POST",
        path=f"projects/{self._project}/locations/{self._location}/reasoningEngines/{self._agent_engine_id}/memories:retrieve",
        request_dict={
            "scope": {
                "app_name": app_name,
                "user_id": user_id,
            },
            "similarity_search_params": {
                "search_query": query,
            },
        },
    )

    # Handles empty response case
    if not api_response.get("retrievedMemories", None):
      return SearchMemoryResponse()

    logger.info(f"Search memory response: {api_response}")

    memory_events = []
    for memory in api_response.get("retrievedMemories", []):
      memory_events.append(
          MemoryEntry(
              author="user",
              content=genai.types.Content(
                  parts=[
                      genai.types.Part(text=memory.get("memory").get("fact"))
                  ],
                  role="user",
              ),
              timestamp=memory.get("updateTime"),
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
    client = genai.Client(
        vertexai=True, project=self._project, location=self._location
    )
    return client._api_client
