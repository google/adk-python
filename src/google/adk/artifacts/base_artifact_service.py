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

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import Field


class ArtifactVersion(BaseModel):
  """Represents the metadata of a specific version of an artifact."""

  version: int
  """The version number of the artifact."""
  canonical_uri: str
  """The canonical URI of the artifact version."""
  custom_metadata: dict[str, Any] = Field(default_factory=dict)
  """A dictionary of custom metadata associated with the artifact version."""
  create_time: float = Field(default_factory=lambda: datetime.now().timestamp())
  """The creation time of the artifact version."""
  mime_type: Optional[str] = None
  """The MIME type of the artifact version."""


class BaseArtifactService(ABC):
  """Abstract base class for artifact services."""

  @abstractmethod
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
    """Saves an artifact to the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename. After saving the artifact, a revision ID is returned to identify
    the artifact version.

    Args:
      app_name: The app name.
      user_id: The user ID.
      filename: The filename of the artifact.
      artifact: The artifact to save. If the artifact consists of `file_data`,
        the artifact service assumes its content has been uploaded separately,
        and this method will associate the `file_data` with the artifact if
        necessary.
      session_id: The session ID. If `None`, the artifact is user-scoped.
      custom_metadata: custom metadata to associate with the artifact.

    Returns:
      The revision ID. The first version of the artifact has a revision ID of 0.
      This is incremented by 1 after each successful save.
    """

  @abstractmethod
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    """Gets an artifact from the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename.

    Args:
      app_name: The app name.
      user_id: The user ID.
      filename: The filename of the artifact.
      session_id: The session ID. If `None`, load the user-scoped artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact or None if not found.
    """

  @abstractmethod
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    """Lists all the artifact filenames within a session.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.

    Returns:
        A list of artifact filenames. If `session_id` is provided, returns
        both session-scoped and user-scoped artifact filenames. If `session_id`
        is `None`, returns
        user-scoped artifact filenames.
    """

  @abstractmethod
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    """Deletes an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        session_id: The ID of the session. If `None`, delete the user-scoped
          artifact.
    """

  @abstractmethod
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    """Lists all versions of an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        session_id: The ID of the session. If `None`, only list the user-scoped
          artifacts versions.

    Returns:
        A list of all available versions of the artifact.
    """

  @abstractmethod
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    """Lists all versions and their metadata for a specific artifact.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      filename: The name of the artifact file.
      session_id: The ID of the session. If `None`, lists versions of the
        user-scoped artifact. Otherwise, lists versions of the artifact within
        the specified session.

    Returns:
      A list of ArtifactVersion objects, each representing a version of the
      artifact and its associated metadata.
    """

  @abstractmethod
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    """Gets the metadata for a specific version of an artifact.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      filename: The name of the artifact file.
      session_id: The ID of the session. If `None`, the artifact will be fetched
        from the user-scoped artifacts. Otherwise, it will be fetched from the
        specified session.
      version: The version number of the artifact to retrieve. If `None`, the
        latest version will be returned.

    Returns:
      An ArtifactVersion object containing the metadata of the specified
      artifact version, or `None` if the artifact version is not found.
    """
