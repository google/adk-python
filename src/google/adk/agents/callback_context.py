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

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from .readonly_context import ReadonlyContext

if TYPE_CHECKING:
  from google.genai import types

  from ..artifacts.base_artifact_service import ArtifactVersion
  from ..auth.auth_credential import AuthCredential
  from ..auth.auth_tool import AuthConfig
  from ..events.event_actions import EventActions
  from ..sessions.state import State
  from .invocation_context import InvocationContext


class CallbackContext(ReadonlyContext):
  """The context of various callbacks within an agent run."""

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      event_actions: Optional[EventActions] = None,
  ) -> None:
    super().__init__(invocation_context)

    from ..events.event_actions import EventActions
    from ..sessions.state import State

    self._event_actions = event_actions or EventActions()
    self._state = State(
        value=invocation_context.session.state,
        delta=self._event_actions.state_delta,
    )

  @property
  @override
  def state(self) -> State:
    """The delta-aware state of the current session.

    For any state change, you can mutate this object directly,
    e.g. `ctx.state['foo'] = 'bar'`
    """
    return self._state

  async def load_artifact(
      self, filename: str, version: Optional[int] = None
  ) -> Optional[types.Part]:
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
      custom_metadata: Optional[dict[str, Any]] = None,
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
      self, filename: str, version: Optional[int] = None
  ) -> Optional[ArtifactVersion]:
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
  ) -> Optional[AuthCredential]:
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

  async def add_session_to_memory(self) -> None:
    """Triggers memory generation for the current session.

    This method saves the current session's events to the memory service,
    enabling the agent to recall information from past interactions.

    Raises:
      ValueError: If memory service is not available.

    Example:
      ```python
      async def my_after_agent_callback(callback_context: CallbackContext):
          # Save conversation to memory at the end of each interaction
          await callback_context.add_session_to_memory()
      ```
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          "Cannot add session to memory: memory service is not available."
      )
    await self._invocation_context.memory_service.add_session_to_memory(
        self._invocation_context.session
    )
