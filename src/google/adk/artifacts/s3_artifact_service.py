"""An artifact service implementation using Amazon S3 or other S3-compatible services.

The blob/key name format depends on whether the filename has a user namespace:
  - For files with user namespace (starting with "user:"):
    {app_name}/{user_id}/user/{filename}/{version}
  - For regular session-scoped files:
    {app_name}/{user_id}/{session_id}/{filename}/{version}

This service supports storing and retrieving artifacts with inline data or text.
Artifacts can also have optional custom metadata, which is serialized as JSON
when stored in S3.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from typing import override

from google.genai import types
from pydantic import BaseModel

from .base_artifact_service import ArtifactVersion
from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


class S3ArtifactService(BaseArtifactService, BaseModel):
  """An artifact service implementation using Amazon S3 or other S3-compatible services.

  Attributes:
      bucket_name: The name of the S3 bucket to use for storing and retrieving artifacts.
      aws_configs: A dictionary of AWS configuration options to pass to the boto3 client.
      save_artifact_max_retries: The maximum number of retries to attempt when saving an artifact with version conflicts.
          If set to -1, the service will retry indefinitely.
  """

  bucket_name: str
  aws_configs: dict[str, Any] = {}
  save_artifact_max_retries: int = -1
  _s3_client: Any = None

  async def _client(self):
    """Creates or returns the aioboto3 S3 client."""
    import aioboto3

    if self._s3_client is None:
      self._s3_client = (
          await aioboto3.Session()
          .client(service_name="s3", **self.aws_configs)
          .__aenter__()
      )
    return self._s3_client

  async def close(self):
    """Closes the underlying S3 client session."""
    if self._s3_client:
      await self._s3_client.__aexit__(None, None, None)
      self._s3_client = None

  def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, str]:
    return {k: json.dumps(v) for k, v in (metadata or {}).items()}

  def _unflatten_metadata(self, metadata: dict[str, str]) -> dict[str, Any]:
    return {k: json.loads(v) for k, v in (metadata or {}).items()}

  def _file_has_user_namespace(self, filename: str) -> bool:
    return filename.startswith("user:")

  def _get_blob_prefix(
      self, app_name: str, user_id: str, session_id: str | None, filename: str
  ) -> str:
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}"
    if session_id:
      return f"{app_name}/{user_id}/{session_id}/{filename}"
    raise ValueError("session_id is required for session-scoped artifacts.")

  def _get_blob_name(
      self,
      app_name: str,
      user_id: str,
      session_id: str | None,
      filename: str,
      version: int,
  ) -> str:
    return (
        f"{self._get_blob_prefix(app_name, user_id, session_id, filename)}/{version}"
    )

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: types.Part,
      session_id: str | None = None,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    """Saves an artifact to S3 with atomic versioning using If-None-Match."""
    from botocore.exceptions import ClientError

    s3 = await self._client()

    if self.save_artifact_max_retries < 0:
      retry_iter = iter(int, 1)
    else:
      retry_iter = range(self.save_artifact_max_retries + 1)
    for _ in retry_iter:
      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      version = 0 if not versions else max(versions) + 1
      key = self._get_blob_name(
          app_name, user_id, session_id, filename, version
      )
      if artifact.inline_data:
        body = artifact.inline_data.data
        mime_type = artifact.inline_data.mime_type
      elif artifact.text:
        body = artifact.text
        mime_type = "text/plain"
      elif artifact.file_data:
        raise NotImplementedError(
            "Saving artifact with file_data is not supported yet in"
            " S3ArtifactService."
        )
      else:
        raise ValueError("Artifact must have either inline_data or text.")

      try:
        await s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=body,
            ContentType=mime_type,
            Metadata=self._flatten_metadata(custom_metadata),
            IfNoneMatch="*",
        )
        return version
      except ClientError as e:
        if e.response["Error"]["Code"] in (
            "PreconditionFailed",
            "ObjectAlreadyExists",
        ):
          continue
        raise e
    raise RuntimeError(
        "Failed to save artifact due to version conflicts after retries"
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
      version: int | None = None,
  ) -> types.Part | None:
    """Loads a specific version of an artifact from S3.

    If version is not provided, the latest version is loaded.

    Returns:
        A types.Part instance (always with inline_data), or None if the artifact does not exist.
    """
    from botocore.exceptions import ClientError

    s3 = await self._client()
    if version is None:
      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      if not versions:
        return None
      version = max(versions)

    key = self._get_blob_name(app_name, user_id, session_id, filename, version)
    try:
      response = await s3.get_object(Bucket=self.bucket_name, Key=key)
      async with response["Body"] as stream:
        data = await stream.read()
      mime_type = response["ContentType"]
    except ClientError as e:
      if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
        return None
      raise
    return types.Part.from_bytes(data=data, mime_type=mime_type)

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str | None = None
  ) -> list[str]:
    """Lists all artifact keys for a user, optionally filtered by session."""
    s3 = await self._client()
    keys = set()
    prefixes = [
        f"{app_name}/{user_id}/{session_id}/" if session_id else None,
        f"{app_name}/{user_id}/user/",
    ]

    for prefix in filter(None, prefixes):
      paginator = s3.get_paginator("list_objects_v2")
      async for page in paginator.paginate(
          Bucket=self.bucket_name, Prefix=prefix
      ):
        for obj in page.get("Contents", []):
          relative = obj["Key"][len(prefix) :]
          filename = "/".join(relative.split("/")[:-1])
          keys.add(filename)
    return sorted(keys)

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> None:
    """Deletes all versions of a specified artifact efficiently using batch delete."""
    s3 = await self._client()
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )
    if not versions:
      return

    keys_to_delete = [
        {"Key": self._get_blob_name(app_name, user_id, session_id, filename, v)}
        for v in versions
    ]
    for i in range(0, len(keys_to_delete), 1000):
      batch = keys_to_delete[i : i + 1000]
      await s3.delete_objects(
          Bucket=self.bucket_name, Delete={"Objects": batch}
      )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> list[int]:
    """Lists all available versions of a specified artifact."""
    s3 = await self._client()
    prefix = (
        self._get_blob_prefix(app_name, user_id, session_id, filename) + "/"
    )
    versions = []
    paginator = s3.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=self.bucket_name, Prefix=prefix
    ):
      for obj in page.get("Contents", []):
        try:
          versions.append(int(obj["Key"].split("/")[-1]))
        except ValueError:
          continue
    return sorted(versions)

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> list[ArtifactVersion]:
    """Lists all artifact versions with their metadata."""
    s3 = await self._client()
    prefix = (
        self._get_blob_prefix(app_name, user_id, session_id, filename) + "/"
    )
    results: list[ArtifactVersion] = []

    paginator = s3.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=self.bucket_name, Prefix=prefix
    ):
      for obj in page.get("Contents", []):
        try:
          version = int(obj["Key"].split("/")[-1])
        except ValueError:
          continue

        head = await s3.head_object(Bucket=self.bucket_name, Key=obj["Key"])
        mime_type = head["ContentType"]
        metadata = head.get("Metadata", {})

        canonical_uri = f"s3://{self.bucket_name}/{obj['Key']}"

        results.append(
            ArtifactVersion(
                version=version,
                canonical_uri=canonical_uri,
                custom_metadata=self._unflatten_metadata(metadata),
                create_time=obj["LastModified"].timestamp(),
                mime_type=mime_type,
            )
        )

    return sorted(results, key=lambda a: a.version)

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
      version: int | None = None,
  ) -> ArtifactVersion | None:
    """Retrieves a specific artifact version, or the latest if version is None."""
    s3 = await self._client()
    if version is None:
      all_versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      if not all_versions:
        return None
      version = max(all_versions)

    key = self._get_blob_name(app_name, user_id, session_id, filename, version)

    from botocore.exceptions import ClientError

    try:
      head = await s3.head_object(Bucket=self.bucket_name, Key=key)
    except ClientError as e:
      if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
        return None
      raise

    return ArtifactVersion(
        version=version,
        canonical_uri=f"s3://{self.bucket_name}/{key}",
        custom_metadata=self._unflatten_metadata(head.get("Metadata", {})),
        create_time=head["LastModified"].timestamp(),
        mime_type=head["ContentType"],
    )
