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

"""An artifact service implementation using Google Cloud Storage (GCS)."""

import logging
from typing import Optional

from google.cloud import storage
from google.genai import types
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


class GcsArtifactService(BaseArtifactService):
  """An artifact service implementation using Google Cloud Storage (GCS)."""

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsArtifactService.

    Args:
        bucket_name: The name of the bucket to use.
        **kwargs: Keyword arguments to pass to the Google Cloud Storage client.
    """
    self.bucket_name = bucket_name
    self.storage_client = storage.Client(**kwargs)
    self.bucket = self.storage_client.bucket(self.bucket_name)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _get_blob_name(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: int,
  ) -> str:
    """Constructs the blob name in GCS.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
        version: The version of the artifact.

    Returns:
        The constructed blob name in GCS.
    """
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}/{version}"
    return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

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
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    version = 0 if not versions else max(versions) + 1

    blob_name = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    blob = self.bucket.blob(blob_name)

    blob.upload_from_string(
        data=artifact.inline_data.data,
        content_type=artifact.inline_data.mime_type,
    )

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
    if version is None:
      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    blob_name = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    blob = self.bucket.blob(blob_name)

    artifact_bytes = blob.download_as_bytes()
    if not artifact_bytes:
      return None
    artifact = types.Part.from_bytes(
        data=artifact_bytes, mime_type=blob.content_type
    )
    return artifact

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    filenames = set()

    session_prefix = f"{app_name}/{user_id}/{session_id}/"
    session_blobs = self.storage_client.list_blobs(
        self.bucket, prefix=session_prefix
    )
    for blob in session_blobs:
      _, _, _, filename, _ = blob.name.split("/")
      filenames.add(filename)

    user_namespace_prefix = f"{app_name}/{user_id}/user/"
    user_namespace_blobs = self.storage_client.list_blobs(
        self.bucket, prefix=user_namespace_prefix
    )
    for blob in user_namespace_blobs:
      _, _, _, filename, _ = blob.name.split("/")
      filenames.add(filename)

    return sorted(list(filenames))

  @override
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    for version in versions:
      blob_name = self._get_blob_name(
          app_name, user_id, session_id, filename, version
      )
      blob = self.bucket.blob(blob_name)
      blob.delete()
    return

  @override
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    prefix = self._get_blob_name(app_name, user_id, session_id, filename, "")
    blobs = self.storage_client.list_blobs(self.bucket, prefix=prefix)
    versions = []
    for blob in blobs:
      _, _, _, _, version = blob.name.split("/")
      versions.append(int(version))
    return versions
