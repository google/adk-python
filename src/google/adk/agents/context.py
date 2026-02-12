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
from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from .readonly_context import ReadonlyContext

if TYPE_CHECKING:
  from google.genai import types

  from ..artifacts.base_artifact_service import ArtifactVersion
  from ..auth.auth_credential import AuthCredential
  from ..auth.auth_tool import AuthConfig
  from ..events.event import Event
  from ..events.event_actions import EventActions
  from ..memory.base_memory_service import SearchMemoryResponse
  from ..sessions.state import State
  from ..tools.tool_confirmation import ToolConfirmation
  from .invocation_context import InvocationContext


class Context(ReadonlyContext):
  """The context within an agent run."""

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      event_actions: EventActions | None = None,
      function_call_id: str | None = None,
      tool_confirmation: ToolConfirmation | None = None,
  ) -> None:
    """Initializes the Context.

    Args:
      invocation_context: The invocation context.
      event_actions: The event actions for state and artifact deltas.
      function_call_id: The function call id of the current tool call. Required
        for tool-specific methods like request_credential and
        request_confirmation.
      tool_confirmation: The tool confirmation of the current tool call.
    """
    super().__init__(invocation_context)

    from ..events.event_actions import EventActions
    from ..sessions.state import State

    self._event_actions = event_actions or EventActions()
    self._state = State(
        value=invocation_context.session.state,
        delta=self._event_actions.state_delta,
    )
    self._function_call_id = function_call_id
    self._tool_confirmation = tool_confirmation

  @property
  def function_call_id(self) -> str | None:
    """The function call id of the current tool call."""
    return self._function_call_id

  @property
  def tool_confirmation(self) -> ToolConfirmation | None:
    """The tool confirmation of the current tool call."""
    return self._tool_confirmation

  @property
  @override
  def state(self) -> State:
    """The delta-aware state of the current session.

    For any state change, you can mutate this object directly,
    e.g. `ctx.state['foo'] = 'bar'`
    """
    return self._state

  @property
  def actions(self) -> EventActions:
    """The event actions for the current context."""
    return self._event_actions

  # ============================================================================
  # Artifact methods
  # ============================================================================

  async def load_artifact(
      self, filename: str, version: int | None = None
  ) -> types.Part | None:
    """Loads an artifact attached to the current session.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    return await self._invocation_context.artifact_service.load_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def save_artifact(
      self,
      filename: str,
      artifact: types.Part,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    """Saves an artifact and records it as delta for the current session.

    Args:
      filename: The filename of the artifact.
      artifact: The artifact to save.
      custom_metadata: Custom metadata to associate with the artifact.

    Returns:
     The version of the artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    version = await self._invocation_context.artifact_service.save_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        artifact=artifact,
        custom_metadata=custom_metadata,
    )
    self._event_actions.artifact_delta[filename] = version
    return version

  async def get_artifact_version(
      self, filename: str, version: int | None = None
  ) -> ArtifactVersion | None:
    """Gets artifact version info.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact version info.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    return await self._invocation_context.artifact_service.get_artifact_version(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def list_artifacts(self) -> list[str]:
    """Lists the filenames of the artifacts attached to the current session."""
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    return await self._invocation_context.artifact_service.list_artifact_keys(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
    )

  # ============================================================================
  # Credential methods
  # ============================================================================

  async def save_credential(self, auth_config: AuthConfig) -> None:
    """Saves a credential to the credential service.

    Args:
      auth_config: The authentication configuration containing the credential.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError("Credential service is not initialized.")
    await self._invocation_context.credential_service.save_credential(
        auth_config, self
    )

  async def load_credential(
      self, auth_config: AuthConfig
  ) -> AuthCredential | None:
    """Loads a credential from the credential service.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The loaded credential, or None if not found.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError("Credential service is not initialized.")
    return await self._invocation_context.credential_service.load_credential(
        auth_config, self
    )

  def get_auth_response(self, auth_config: AuthConfig) -> AuthCredential | None:
    """Gets the auth response credential from session state.

    This method retrieves an authentication credential that was previously
    stored in session state after a user completed an OAuth flow or other
    authentication process.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The auth credential from the auth response, or None if not found.
    """
    from ..auth.auth_handler import AuthHandler

    return AuthHandler(auth_config).get_auth_response(self.state)

  def request_credential(self, auth_config: AuthConfig) -> None:
    """Requests a credential for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set. For callback contexts, use save_credential/load_credential instead.

    Args:
      auth_config: The authentication configuration for the credential.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..auth.auth_handler import AuthHandler

    if not self.function_call_id:
      raise ValueError(
          "request_credential requires function_call_id. "
          "This method can only be used in a tool context, not a callback "
          "context. Consider using save_credential/load_credential instead."
      )
    self._event_actions.requested_auth_configs[self.function_call_id] = (
        AuthHandler(auth_config).generate_auth_request()
    )

  # ============================================================================
  # Tool methods
  # ============================================================================

  def request_confirmation(
      self,
      *,
      hint: str | None = None,
      payload: Any | None = None,
  ) -> None:
    """Requests confirmation for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set.

    Args:
      hint: A hint to the user on how to confirm the tool call.
      payload: The payload used to confirm the tool call.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..tools.tool_confirmation import ToolConfirmation

    if not self.function_call_id:
      raise ValueError(
          "request_confirmation requires function_call_id. "
          "This method can only be used in a tool context."
      )
    self._event_actions.requested_tool_confirmations[self.function_call_id] = (
        ToolConfirmation(
            hint=hint,
            payload=payload,
        )
    )

  # ============================================================================
  # Memory methods
  # ============================================================================

  async def add_session_to_memory(self) -> None:
    """Triggers memory generation for the current session.

    This method saves the current session's events to the memory service,
    enabling the agent to recall information from past interactions.

    Raises:
      ValueError: If memory service is not available.

    Example:
      ```python
      async def my_after_agent_callback(ctx: Context):
          # Save conversation to memory at the end of each interaction
          await ctx.add_session_to_memory()
      ```
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          "Cannot add session to memory: memory service is not available."
      )
    await self._invocation_context.memory_service.add_session_to_memory(
        self._invocation_context.session
    )

  async def add_events_to_memory(
      self,
      *,
      events: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds an explicit list of events to the memory service.

    Uses this callback's current session identifiers as memory scope.

    Args:
      events: Explicit events to add to memory.
      custom_metadata: Optional standard metadata for memory generation.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          "Cannot add events to memory: memory service is not available."
      )
    await self._invocation_context.memory_service.add_events_to_memory(
        app_name=self._invocation_context.session.app_name,
        user_id=self._invocation_context.session.user_id,
        session_id=self._invocation_context.session.id,
        events=events,
        custom_metadata=custom_metadata,
    )

  async def search_memory(self, query: str) -> SearchMemoryResponse:
    """Searches the memory of the current user.

    Args:
      query: The search query.

    Returns:
      The search results from the memory service.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError("Memory service is not available.")
    return await self._invocation_context.memory_service.search_memory(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        query=query,
    )
