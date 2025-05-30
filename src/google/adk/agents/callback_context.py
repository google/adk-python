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

from typing_extensions import override

from .readonly_context import ReadonlyContext

if TYPE_CHECKING:
  from google.genai import types

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

    # TODO(weisun): make this public for Agent Development Kit, but private for
    # users.
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

  async def save_artifact(self, filename: str, artifact: types.Part) -> int:
    """Saves an artifact and records it as delta for the current session.

    Args:
      filename: The filename of the artifact.
      artifact: The artifact to save.

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
    )
    self._event_actions.artifact_delta[filename] = version
    return version
