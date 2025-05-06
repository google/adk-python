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

from typing import Optional
from typing import TYPE_CHECKING

from ..agents.callback_context import CallbackContext
from ..auth.auth_credential import AuthCredential
from ..auth.auth_handler import AuthHandler
from ..auth.auth_tool import AuthConfig

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event_actions import EventActions
  from ..memory.base_memory_service import SearchMemoryResponse


class ToolContext(CallbackContext):
  """The context of the tool.

  This class provides the context for a tool invocation, including access to
  the invocation context, function call ID, event actions, and authentication
  response. It also provides methods for requesting credentials, retrieving
  authentication responses, listing artifacts, and searching memory.

  Attributes:
    invocation_context: The invocation context of the tool.
    function_call_id: The function call id of the current tool call. This id was
      returned in the function call event from LLM to identify a function call.
      If LLM didn't return this id, ADK will assign one to it. This id is used
      to map function call response to the original function call.
    event_actions: The event actions of the current tool call.
  """

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      function_call_id: Optional[str] = None,
      event_actions: Optional[EventActions] = None,
  ):
    super().__init__(invocation_context, event_actions=event_actions)
    self.function_call_id = function_call_id

  @property
  def actions(self) -> EventActions:
    return self._event_actions

  def request_credential(self, auth_config: AuthConfig) -> None:
    if not self.function_call_id:
      raise ValueError('function_call_id is not set.')
    self._event_actions.requested_auth_configs[self.function_call_id] = (
        AuthHandler(auth_config).generate_auth_request()
    )

  def get_auth_response(self, auth_config: AuthConfig) -> AuthCredential:
    return AuthHandler(auth_config).get_auth_response(self.state)

  async def list_artifacts(self) -> list[str]:
    """Lists the filenames of the artifacts attached to the current session."""
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.list_artifact_keys(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
    )

  async def search_memory(self, query: str) -> SearchMemoryResponse:
    """Searches the memory of the current user."""
    if self._invocation_context.memory_service is None:
      raise ValueError('Memory service is not available.')
    return await self._invocation_context.memory_service.search_memory(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        query=query,
    )
