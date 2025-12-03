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

from google.genai import types
from typing_extensions import override

from ..artifacts.base_artifact_service import ArtifactVersion
from ..artifacts.base_artifact_service import BaseArtifactService

if TYPE_CHECKING:
  from .tool_context import ToolContext


class ForwardingArtifactService(BaseArtifactService):
  """Artifact service that forwards to the parent tool context."""

  def __init__(self, tool_context: ToolContext):
    self.tool_context = tool_context
    self._invocation_context = tool_context._invocation_context

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: types.Part,
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    return await self.tool_context.save_artifact(
        filename=filename,
        artifact=artifact,
        custom_metadata=custom_metadata,
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    return await self.tool_context.load_artifact(
        filename=filename, version=version
    )

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    return await self.tool_context.list_artifacts()

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    del app_name, user_id, session_id
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    await self._invocation_context.artifact_service.delete_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
    )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    del app_name, user_id, session_id
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    return await self._invocation_context.artifact_service.list_versions(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
    )

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    raise NotImplementedError("list_artifact_versions is not implemented yet.")

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    raise NotImplementedError("get_artifact_version is not implemented yet.")
