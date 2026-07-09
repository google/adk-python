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
from typing import Union

from google.genai import types
from typing_extensions import override

from . import artifact_util
from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService
from .base_artifact_service import ensure_part

logger = logging.getLogger("google_adk." + __name__)

_GCS_DISPLAY_NAME_METADATA_KEY = "adkDisplayName"
_GCS_IS_TEXT_METADATA_KEY = "adkIsText"
_GCS_FILE_URI_METADATA_KEY = "adkFileUri"
_GCS_FILE_MIME_TYPE_METADATA_KEY = "adkFileMimeType"


class GcsArtifactService(BaseArtifactService):
  """An artifact service implementation using Google Cloud Storage (GCS)."""

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsArtifactService.

    Args:
        bucket_name: The name of the bucket to use.
        **kwargs: Keyword arguments to pass to the Google Cloud Storage client.
    """
    from google.cloud import storage

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
      artifact: Union[types.Part, dict[str, Any]],
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
      raise InputValidationError(
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
      artifact: Union[types.Part, dict[str, Any]],
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    artifact = ensure_part(artifact)
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
    blob_metadata = {k: str(v) for k, v in (custom_metadata or {}).items()}
    if artifact.inline_data and artifact.inline_data.display_name:
      blob_metadata[_GCS_DISPLAY_NAME_METADATA_KEY] = (
          artifact.inline_data.display_name
      )
    elif artifact.inline_data is None and artifact.text is not None:
      # Flag text artifacts so they can be reconstructed as Part(text=...) on
      # load instead of Part.from_bytes() (which would only populate
      # inline_data).
      blob_metadata[_GCS_IS_TEXT_METADATA_KEY] = "true"
    if blob_metadata:
      blob.metadata = blob_metadata

    if artifact.inline_data:
      blob.upload_from_string(
          data=artifact.inline_data.data,
          content_type=artifact.inline_data.mime_type,
      )
    elif artifact.text is not None:
      blob.upload_from_string(
          data=artifact.text,
          content_type="text/plain",
      )
    elif artifact.file_data:
      file_data = artifact.file_data
      assert file_data is not None
      file_uri = file_data.file_uri
      if not file_uri:
        raise InputValidationError("Artifact file_data must have a file_uri.")
      if artifact_util.is_artifact_ref(artifact):
        parsed_uri = artifact_util.parse_artifact_uri(file_uri)
        if not parsed_uri:
          raise InputValidationError(
              f"Invalid artifact reference URI: {file_uri}"
          )
        artifact_util.validate_artifact_reference_scope(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            parsed_uri=parsed_uri,
        )
      # Store the URI and mime_type (if any) as blob metadata; no content to upload.
      metadata = {
          **(blob.metadata or {}),
          _GCS_FILE_URI_METADATA_KEY: file_uri,
      }
      if file_data.mime_type:
        metadata[_GCS_FILE_MIME_TYPE_METADATA_KEY] = file_data.mime_type
      blob.metadata = metadata
      blob.upload_from_string(
          b"",
          content_type=file_data.mime_type or None,
      )
    else:
      raise InputValidationError(
          "Artifact must have either inline_data or text."
      )

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
    blob = self.bucket.get_blob(blob_name)
    if not blob:
      return None

    # If the artifact was saved as a file_data URI reference, restore or resolve it.
    file_uri = None
    if blob.metadata:
      file_uri = blob.metadata.get(
          _GCS_FILE_URI_METADATA_KEY
      ) or blob.metadata.get("file_uri")

    if file_uri:
      if file_uri.startswith("artifact://"):
        parsed_uri = artifact_util.parse_artifact_uri(file_uri)
        if not parsed_uri:
          raise InputValidationError(
              f"Invalid artifact reference URI: {file_uri}"
          )
        artifact_util.validate_artifact_reference_scope(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            parsed_uri=parsed_uri,
        )
        return self._load_artifact(
            app_name=parsed_uri.app_name,
            user_id=parsed_uri.user_id,
            session_id=parsed_uri.session_id,
            filename=parsed_uri.filename,
            version=parsed_uri.version,
        )
      mime_type = None
      if blob.metadata:
        mime_type = blob.metadata.get(_GCS_FILE_MIME_TYPE_METADATA_KEY)
      if mime_type is None:
        mime_type = blob.content_type or None
      return types.Part(
          file_data=types.FileData(
              file_uri=file_uri,
              mime_type=mime_type,
          )
      )

    artifact_bytes = blob.download_as_bytes()
    if blob.metadata and blob.metadata.get(_GCS_IS_TEXT_METADATA_KEY) == "true":
      return types.Part(text=artifact_bytes.decode("utf-8"))
    display_name = None
    if blob.metadata:
      display_name = blob.metadata.get(_GCS_DISPLAY_NAME_METADATA_KEY)
    if display_name:
      return types.Part(
          inline_data=types.Blob(
              mime_type=blob.content_type,
              data=artifact_bytes,
              display_name=display_name,
          )
      )
    return types.Part.from_bytes(
        data=artifact_bytes, mime_type=blob.content_type
    )

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
