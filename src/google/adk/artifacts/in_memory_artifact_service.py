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

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


class InMemoryArtifactService(BaseArtifactService, BaseModel):
  """An in-memory implementation of the artifact service.

  It is not suitable for multi-threaded production environments. Use it for
  testing and development only.
  """

  artifacts: dict[str, list[types.Part]] = Field(default_factory=dict)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _artifact_path(
      self, app_name: str, user_id: str, session_id: str, filename: str
  ) -> str:
    """Constructs the artifact path.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.

    Returns:
        The constructed artifact path.
    """
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}"
    return f"{app_name}/{user_id}/{session_id}/{filename}"

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    if path not in self.artifacts:
      self.artifacts[path] = []
    version = len(self.artifacts[path])
    self.artifacts[path].append(artifact)
    return version

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    versions = self.artifacts.get(path)
    if not versions:
      return None
    if version is None:
      version = -1
    return versions[version]

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    session_prefix = f"{app_name}/{user_id}/{session_id}/"
    usernamespace_prefix = f"{app_name}/{user_id}/user/"
    filenames = []
    for path in self.artifacts:
      if path.startswith(session_prefix):
        filename = path.removeprefix(session_prefix)
        filenames.append(filename)
      elif path.startswith(usernamespace_prefix):
        filename = path.removeprefix(usernamespace_prefix)
        filenames.append(filename)
    return sorted(filenames)

  @override
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    if not self.artifacts.get(path):
      return None
    self.artifacts.pop(path, None)

  @override
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    versions = self.artifacts.get(path)
    if not versions:
      return []
    return list(range(len(versions)))
