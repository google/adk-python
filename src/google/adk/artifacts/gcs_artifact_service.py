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

"""An artifact service implementation using Google Cloud Storage (GCS).

The blob name format used depends on whether the filename has a user namespace:
  - For files with user namespace (starting with "user:"):
    {app_name}/{user_id}/user/{filename}/{version}
  - For regular session-scoped files:
    {app_name}/{user_id}/{session_id}/{filename}/{version}
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Optional

from google.cloud import storage
from google.genai import types
from typing_extensions import override

from .base_artifact_service import ArtifactVersion
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
    return await asyncio.to_thread(
        self._save_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        artifact,
        custom_metadata,
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
    return await asyncio.to_thread(
        self._load_artifact,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    return await asyncio.to_thread(
        self._list_artifact_keys,
        app_name,
        user_id,
        session_id,
    )

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    return await asyncio.to_thread(
        self._delete_artifact,
        app_name,
        user_id,
        session_id,
        filename,
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
    return await asyncio.to_thread(
        self._list_versions,
        app_name,
        user_id,
        session_id,
        filename,
    )

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _get_blob_prefix(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> str:
    """Constructs the blob name prefix in GCS for a given artifact."""
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}"

    if session_id is None:
      raise ValueError(
          "Session ID must be provided for session-scoped artifacts."
      )
    return f"{app_name}/{user_id}/{session_id}/{filename}"

  def _get_blob_name(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      version: int,
      session_id: Optional[str] = None,
  ) -> str:
    """Constructs the blob name in GCS.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        version: The version of the artifact.
        session_id: The ID of the session.

    Returns:
        The constructed blob name in GCS.
    """
    return (
        f"{self._get_blob_prefix(app_name, user_id, filename, session_id)}/{version}"
    )

  def _save_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      artifact: types.Part,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    versions = self._list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    version = 0 if not versions else max(versions) + 1

    blob_name = self._get_blob_name(
        app_name, user_id, filename, version, session_id
    )
    blob = self.bucket.blob(blob_name)
    if custom_metadata:
      blob.metadata = {k: str(v) for k, v in custom_metadata.items()}

    if artifact.inline_data:
      blob.upload_from_string(
          data=artifact.inline_data.data,
          content_type=artifact.inline_data.mime_type,
      )
    elif artifact.text:
      blob.upload_from_string(
          data=artifact.text,
          content_type="text/plain",
      )
    elif artifact.file_data:
      raise NotImplementedError(
          "Saving artifact with file_data is not supported yet in"
          " GcsArtifactService."
      )
    else:
      raise ValueError("Artifact must have either inline_data or text.")

    return version

  def _load_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    if version is None:
      versions = self._list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    blob_name = self._get_blob_name(
        app_name, user_id, filename, version, session_id
    )
    blob = self.bucket.blob(blob_name)

    artifact_bytes = blob.download_as_bytes()
    if not artifact_bytes:
      return None
    artifact = types.Part.from_bytes(
        data=artifact_bytes, mime_type=blob.content_type
    )
    return artifact

  def _list_artifact_keys(
      self, app_name: str, user_id: str, session_id: Optional[str]
  ) -> list[str]:
    filenames = set()

    if session_id:
      session_prefix = f"{app_name}/{user_id}/{session_id}/"
      session_blobs = self.storage_client.list_blobs(
          self.bucket, prefix=session_prefix
      )
      for blob in session_blobs:
        # blob.name is like session_prefix/filename/version
        # or session_prefix/path/to/filename/version
        # we need to extract filename including slashes, but remove prefix
        # and /version
        fn_and_version = blob.name[len(session_prefix) :]
        filename = "/".join(fn_and_version.split("/")[:-1])
        filenames.add(filename)

    user_namespace_prefix = f"{app_name}/{user_id}/user/"
    user_namespace_blobs = self.storage_client.list_blobs(
        self.bucket, prefix=user_namespace_prefix
    )
    for blob in user_namespace_blobs:
      # blob.name is like user_namespace_prefix/filename/version
      fn_and_version = blob.name[len(user_namespace_prefix) :]
      filename = "/".join(fn_and_version.split("/")[:-1])
      filenames.add(filename)

    return sorted(list(filenames))

  def _delete_artifact(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> None:
    versions = self._list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    for version in versions:
      blob_name = self._get_blob_name(
          app_name, user_id, filename, version, session_id
      )
      blob = self.bucket.blob(blob_name)
      blob.delete()
    return

  def _list_versions(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[int]:
    """Lists all available versions of an artifact.

    This method retrieves all versions of a specific artifact by querying GCS
    blobs
    that match the constructed blob name prefix.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user who owns the artifact.
        session_id: The ID of the session (ignored for user-namespaced files).
        filename: The name of the artifact file.

    Returns:
        A list of version numbers (integers) available for the specified
        artifact.
        Returns an empty list if no versions are found.
    """
    prefix = self._get_blob_prefix(app_name, user_id, filename, session_id)
    blobs = self.storage_client.list_blobs(self.bucket, prefix=f"{prefix}/")
    versions = []
    for blob in blobs:
      *_, version = blob.name.split("/")
      versions.append(int(version))
    return versions

  def _get_artifact_version_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    if version is None:
      versions = self._list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    blob_name = self._get_blob_name(
        app_name, user_id, filename, version, session_id
    )
    blob = self.bucket.get_blob(blob_name)

    if not blob:
      return None

    canonical_uri = f"gs://{self.bucket_name}/{blob.name}"

    return ArtifactVersion(
        version=version,
        canonical_uri=canonical_uri,
        create_time=blob.time_created.timestamp(),
        mime_type=blob.content_type,
        custom_metadata=blob.metadata if blob.metadata else {},
    )

  def _list_artifact_versions_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[ArtifactVersion]:
    """Lists all versions and their metadata of an artifact."""
    prefix = self._get_blob_prefix(app_name, user_id, filename, session_id)
    blobs = self.storage_client.list_blobs(self.bucket, prefix=f"{prefix}/")
    artifact_versions = []
    for blob in blobs:
      try:
        version = int(blob.name.split("/")[-1])
      except ValueError:
        logger.warning(
            "Skipping blob %s because it does not end with a version number.",
            blob.name,
        )
        continue

      canonical_uri = f"gs://{self.bucket_name}/{blob.name}"
      av = ArtifactVersion(
          version=version,
          canonical_uri=canonical_uri,
          create_time=blob.time_created.timestamp(),
          mime_type=blob.content_type,
          custom_metadata=blob.metadata if blob.metadata else {},
      )
      artifact_versions.append(av)

    artifact_versions.sort(key=lambda x: x.version)
    return artifact_versions

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    return await asyncio.to_thread(
        self._list_artifact_versions_sync,
        app_name,
        user_id,
        session_id,
        filename,
    )

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
    return await asyncio.to_thread(
        self._get_artifact_version_sync,
        app_name,
        user_id,
        session_id,
        filename,
        version,
    )
