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

"""An artifact service implementation using Amazon S3.

The object key format used depends on whether the filename has a user namespace:
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
from urllib.parse import quote
from urllib.parse import unquote

from google.genai import types
from typing_extensions import override

from ..errors.input_validation_error import InputValidationError
from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


class S3ArtifactService(BaseArtifactService):
  """An artifact service implementation using Amazon S3."""

  def __init__(
      self,
      bucket_name: str,
      region_name: Optional[str] = None,
      **kwargs,
  ):
    """Initializes the S3ArtifactService.

    Args:
        bucket_name: The name of the S3 bucket to use.
        region_name: AWS region name (optional).
        **kwargs: Additional keyword arguments to pass to boto3.client().
    """
    try:
      import boto3
    except ImportError as exc:
      raise ImportError(
          "boto3 is required to use S3ArtifactService. "
          "Install it with: pip install boto3"
      ) from exc

    self.bucket_name = bucket_name
    client_kwargs = dict(kwargs)
    if region_name:
      client_kwargs["region_name"] = region_name

    self.s3_client = boto3.client("s3", **client_kwargs)

    # Verify bucket access
    try:
      self.s3_client.head_bucket(Bucket=self.bucket_name)
      logger.info("S3ArtifactService initialized with bucket: %s", bucket_name)
    except Exception as e:
      logger.error("Cannot access S3 bucket '%s': %s", bucket_name, e)
      raise

  def _encode_filename(self, filename: str) -> str:
    """URL-encode filename to handle special characters.

    Args:
        filename: The filename to encode.

    Returns:
        The URL-encoded filename.
    """
    return quote(filename, safe="")

  def _decode_filename(self, encoded_filename: str) -> str:
    """URL-decode filename to restore original filename.

    Args:
        encoded_filename: The encoded filename to decode.

    Returns:
        The decoded filename.
    """
    return unquote(encoded_filename)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _get_object_key_prefix(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> tuple[str, str]:
    """Constructs the S3 object key prefix and encoded filename.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        session_id: The ID of the session.

    Returns:
        A tuple of (prefix, encoded_filename).
    """
    if self._file_has_user_namespace(filename):
      # Remove "user:" prefix before encoding
      actual_filename = filename[5:]  # len("user:") == 5
      encoded_filename = self._encode_filename(actual_filename)
      return f"{app_name}/{user_id}/user", encoded_filename

    if session_id is None:
      raise InputValidationError(
          "Session ID must be provided for session-scoped artifacts."
      )
    encoded_filename = self._encode_filename(filename)
    return f"{app_name}/{user_id}/{session_id}", encoded_filename

  def _get_object_key(
      self,
      app_name: str,
      user_id: str,
      filename: str,
      version: int,
      session_id: Optional[str] = None,
  ) -> str:
    """Constructs the full S3 object key.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        filename: The name of the artifact file.
        version: The version of the artifact.
        session_id: The ID of the session.

    Returns:
        The constructed S3 object key.
    """
    prefix, encoded_filename = self._get_object_key_prefix(
        app_name, user_id, filename, session_id
    )
    return f"{prefix}/{encoded_filename}/{version}"

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
        self._save_artifact_sync,
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
        self._load_artifact_sync,
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
        self._list_artifact_keys_sync,
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
        self._delete_artifact_sync,
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
        self._list_versions_sync,
        app_name,
        user_id,
        session_id,
        filename,
    )

  def _save_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      artifact: types.Part,
      custom_metadata: Optional[dict[str, Any]],
  ) -> int:
    """Synchronous implementation of save_artifact."""
    # Get next version number
    versions = self._list_versions_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    version = 0 if not versions else max(versions) + 1

    object_key = self._get_object_key(
        app_name, user_id, filename, version, session_id
    )

    # Prepare data and content type
    if artifact.inline_data:
      data = artifact.inline_data.data
      content_type = artifact.inline_data.mime_type or "application/octet-stream"
    elif artifact.text:
      data = artifact.text.encode("utf-8")
      content_type = "text/plain; charset=utf-8"
    else:
      raise InputValidationError(
          "Artifact must have either inline_data or text content."
      )

    # Prepare put_object arguments
    put_kwargs: dict[str, Any] = {
        "Bucket": self.bucket_name,
        "Key": object_key,
        "Body": data,
        "ContentType": content_type,
    }

    # Add custom metadata if provided
    if custom_metadata:
      put_kwargs["Metadata"] = {str(k): str(v) for k, v in custom_metadata.items()}

    try:
      self.s3_client.put_object(**put_kwargs)
      logger.debug(
          "Saved artifact %s version %d to S3 key %s",
          filename,
          version,
          object_key,
      )
      return version
    except Exception as e:
      logger.error("Failed to save artifact '%s' to S3: %s", filename, e)
      raise

  def _load_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int],
  ) -> Optional[types.Part]:
    """Synchronous implementation of load_artifact."""
    if version is None:
      versions = self._list_versions_sync(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    object_key = self._get_object_key(
        app_name, user_id, filename, version, session_id
    )

    try:
      response = self.s3_client.get_object(
          Bucket=self.bucket_name, Key=object_key
      )
      content_type = response.get("ContentType", "application/octet-stream")
      data = response["Body"].read()

      if not data:
        return None

      artifact = types.Part.from_bytes(data=data, mime_type=content_type)
      logger.debug(
          "Loaded artifact %s version %d from S3 key %s",
          filename,
          version,
          object_key,
      )
      return artifact

    except self.s3_client.exceptions.NoSuchKey:
      logger.debug(
          "Artifact %s version %d not found in S3", filename, version
      )
      return None
    except Exception as e:
      logger.error("Failed to load artifact '%s' from S3: %s", filename, e)
      raise

  def _list_artifact_keys_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
  ) -> list[str]:
    """Synchronous implementation of list_artifact_keys."""
    filenames: set[str] = set()

    # List session-scoped artifacts
    if session_id:
      session_prefix = f"{app_name}/{user_id}/{session_id}/"
      try:
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=session_prefix
        )
        if "Contents" in response:
          for obj in response["Contents"]:
            # Parse: {prefix}/{encoded_filename}/{version}
            key = obj["Key"]
            parts = key[len(session_prefix) :].split("/")
            if len(parts) >= 2:
              encoded_filename = parts[0]
              filename = self._decode_filename(encoded_filename)
              filenames.add(filename)
      except Exception as e:
        logger.error(
            "Failed to list session artifacts for %s: %s", session_id, e
        )

    # List user-scoped artifacts
    user_prefix = f"{app_name}/{user_id}/user/"
    try:
      response = self.s3_client.list_objects_v2(
          Bucket=self.bucket_name, Prefix=user_prefix
      )
      if "Contents" in response:
        for obj in response["Contents"]:
          # Parse: {prefix}/{encoded_filename}/{version}
          key = obj["Key"]
          parts = key[len(user_prefix) :].split("/")
          if len(parts) >= 2:
            encoded_filename = parts[0]
            filename = self._decode_filename(encoded_filename)
            filenames.add(f"user:{filename}")
    except Exception as e:
      logger.error("Failed to list user artifacts for %s: %s", user_id, e)

    return sorted(list(filenames))

  def _delete_artifact_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> None:
    """Synchronous implementation of delete_artifact."""
    versions = self._list_versions_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    for version in versions:
      object_key = self._get_object_key(
          app_name, user_id, filename, version, session_id
      )
      try:
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
        logger.debug("Deleted S3 object: %s", object_key)
      except Exception as e:
        logger.error("Failed to delete S3 object %s: %s", object_key, e)

  def _list_versions_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[int]:
    """Lists all available versions of an artifact.

    This method retrieves all versions of a specific artifact by querying S3
    objects that match the constructed object key prefix.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user who owns the artifact.
        session_id: The ID of the session (ignored for user-namespaced files).
        filename: The name of the artifact file.

    Returns:
        A list of version numbers (integers) available for the specified
        artifact. Returns an empty list if no versions are found.
    """
    prefix, encoded_filename = self._get_object_key_prefix(
        app_name, user_id, filename, session_id
    )
    full_prefix = f"{prefix}/{encoded_filename}/"

    try:
      response = self.s3_client.list_objects_v2(
          Bucket=self.bucket_name, Prefix=full_prefix
      )
      versions: list[int] = []
      if "Contents" in response:
        for obj in response["Contents"]:
          # Extract version from key: {prefix}/{encoded_filename}/{version}
          key = obj["Key"]
          version_str = key.split("/")[-1]
          if version_str.isdigit():
            versions.append(int(version_str))
      return sorted(versions)
    except Exception as e:
      logger.error("Failed to list versions for '%s': %s", filename, e)
      return []

  def _get_artifact_version_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: Optional[int],
  ) -> Optional[ArtifactVersion]:
    """Synchronous implementation of get_artifact_version."""
    if version is None:
      versions = self._list_versions_sync(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    object_key = self._get_object_key(
        app_name, user_id, filename, version, session_id
    )

    try:
      response = self.s3_client.head_object(
          Bucket=self.bucket_name, Key=object_key
      )

      metadata = response.get("Metadata", {}) or {}
      last_modified = response.get("LastModified")
      create_time = (
          last_modified.timestamp()
          if hasattr(last_modified, "timestamp")
          else None
      )

      canonical_uri = f"s3://{self.bucket_name}/{object_key}"

      return ArtifactVersion(
          version=version,
          canonical_uri=canonical_uri,
          custom_metadata={str(k): str(v) for k, v in metadata.items()},
          create_time=create_time,
          mime_type=response.get("ContentType"),
      )
    except self.s3_client.exceptions.NoSuchKey:
      logger.debug(
          "Artifact %s version %d not found in S3", filename, version
      )
      return None
    except Exception as e:
      logger.error(
          "Failed to get artifact version for '%s' version %d: %s",
          filename,
          version,
          e,
      )
      return None

  def _list_artifact_versions_sync(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> list[ArtifactVersion]:
    """Lists all versions and their metadata of an artifact."""
    versions = self._list_versions_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    artifact_versions: list[ArtifactVersion] = []
    for version in versions:
      artifact_version = self._get_artifact_version_sync(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
          version=version,
      )
      if artifact_version:
        artifact_versions.append(artifact_version)

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

