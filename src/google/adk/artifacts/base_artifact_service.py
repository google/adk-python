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


from abc import ABC
from abc import abstractmethod
from typing import Optional

from google.genai import types


class BaseArtifactService(ABC):
  """Abstract base class for artifact services."""

  @abstractmethod
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    """Saves an artifact to the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename. After saving the artifact, a revision ID is returned to identify
    the artifact version.

    Args:
      app_name: The app name.
      user_id: The user ID.
      session_id: The session ID.
      filename: The filename of the artifact.
      artifact: The artifact to save.

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
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    """Gets an artifact from the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename.

    Args:
      app_name: The app name.
      user_id: The user ID.
      session_id: The session ID.
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact or None if not found.
    """

  @abstractmethod
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    """Lists all the artifact filenames within a session.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.

    Returns:
        A list of all artifact filenames within a session.
    """

  @abstractmethod
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    """Deletes an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
    """

  @abstractmethod
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    """Lists all versions of an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.

    Returns:
        A list of all available versions of the artifact.
    """
