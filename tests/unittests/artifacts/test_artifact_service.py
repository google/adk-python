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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Tests for the artifact service."""

from datetime import datetime
import enum
import json
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
from unittest import mock
from unittest.mock import patch
from urllib.parse import unquote
from urllib.parse import urlparse

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types
import pytest

Enum = enum.Enum

# Define a fixed datetime object to be returned by datetime.now()
FIXED_DATETIME = datetime(2025, 1, 1, 12, 0, 0)


class ArtifactServiceType(Enum):
  FILE = "FILE"
  IN_MEMORY = "IN_MEMORY"
  GCS = "GCS"


class MockBlob:
  """Mocks a GCS Blob object.

  This class provides mock implementations for a few common GCS Blob methods,
  allowing the user to test code that interacts with GCS without actually
  connecting to a real bucket.
  """

  def __init__(self, name: str) -> None:
    """Initializes a MockBlob.

    Args:
        name: The name of the blob.
    """
    self.name = name
    self.content: Optional[bytes] = None
    self.content_type: Optional[str] = None
    self.time_created = FIXED_DATETIME
    self.metadata: dict[str, Any] = {}

  def upload_from_string(
      self, data: Union[str, bytes], content_type: Optional[str] = None
  ) -> None:
    """Mocks uploading data to the blob (from a string or bytes).

    Args:
        data: The data to upload (string or bytes).
        content_type:  The content type of the data (optional).
    """
    if isinstance(data, str):
      self.content = data.encode("utf-8")
    elif isinstance(data, bytes):
      self.content = data
    else:
      raise TypeError("data must be str or bytes")

    if content_type:
      self.content_type = content_type

  def download_as_bytes(self) -> bytes:
    """Mocks downloading the blob's content as bytes.

    Returns:
        bytes: The content of the blob as bytes.

    Raises:
        Exception: If the blob doesn't exist (hasn't been uploaded to).
    """
    if self.content is None:
      return b""
    return self.content

  def delete(self) -> None:
    """Mocks deleting a blob."""
    self.content = None
    self.content_type = None


class MockBucket:
  """Mocks a GCS Bucket object."""

  def __init__(self, name: str) -> None:
    """Initializes a MockBucket.

    Args:
        name: The name of the bucket.
    """
    self.name = name
    self.blobs: dict[str, MockBlob] = {}

  def blob(self, blob_name: str) -> MockBlob:
    """Mocks getting a Blob object (doesn't create it in storage).

    Args:
        blob_name: The name of the blob.

    Returns:
        A MockBlob instance.
    """
    if blob_name not in self.blobs:
      self.blobs[blob_name] = MockBlob(blob_name)
    return self.blobs[blob_name]

  def get_blob(self, blob_name: str) -> Optional[MockBlob]:
    """Mocks getting a blob from storage if it exists and has content."""
    blob = self.blobs.get(blob_name)
    if blob and blob.content is not None:
      return blob
    return None


class MockClient:
  """Mocks the GCS Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    self.buckets: dict[str, MockBucket] = {}

  def bucket(self, bucket_name: str) -> MockBucket:
    """Mocks getting a Bucket object."""
    if bucket_name not in self.buckets:
      self.buckets[bucket_name] = MockBucket(bucket_name)
    return self.buckets[bucket_name]

  def list_blobs(self, bucket: MockBucket, prefix: Optional[str] = None):
    """Mocks listing blobs in a bucket, optionally with a prefix."""
    if prefix:
      return [
          blob
          for name, blob in bucket.blobs.items()
          if name.startswith(prefix) and blob.content is not None
      ]
    return [blob for blob in bucket.blobs.values() if blob.content is not None]


def mock_gcs_artifact_service():
  with mock.patch("google.cloud.storage.Client", return_value=MockClient()):
    return GcsArtifactService(bucket_name="test_bucket")


@pytest.fixture
def artifact_service_factory(tmp_path: Path):
  """Provides an artifact service constructor bound to the test tmp path."""

  def factory(
      service_type: ArtifactServiceType = ArtifactServiceType.IN_MEMORY,
  ):
    if service_type == ArtifactServiceType.GCS:
      return mock_gcs_artifact_service()
    if service_type == ArtifactServiceType.FILE:
      return FileArtifactService(root_dir=tmp_path / "artifacts")
    return InMemoryArtifactService()

  return factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.IN_MEMORY,
        ArtifactServiceType.GCS,
        ArtifactServiceType.FILE,
    ],
)
async def test_load_empty(service_type, artifact_service_factory):
  """Tests loading an artifact when none exists."""
  artifact_service = artifact_service_factory(service_type)
  assert not await artifact_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.IN_MEMORY,
        ArtifactServiceType.GCS,
        ArtifactServiceType.FILE,
    ],
)
async def test_save_load_delete(service_type, artifact_service_factory):
  """Tests saving, loading, and deleting an artifact."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert (
      await artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      == artifact
  )

  # Attempt to load a version that doesn't exist
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )

  await artifact_service.delete_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.IN_MEMORY,
        ArtifactServiceType.GCS,
        ArtifactServiceType.FILE,
    ],
)
async def test_list_keys(service_type, artifact_service_factory):
  """Tests listing keys in the artifact service."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  filenames = [filename + str(i) for i in range(5)]

  for f in filenames:
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  assert (
      await artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      == filenames
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.IN_MEMORY,
        ArtifactServiceType.GCS,
        ArtifactServiceType.FILE,
    ],
)
async def test_list_versions(service_type, artifact_service_factory):
  """Tests listing versions of an artifact."""
  artifact_service = artifact_service_factory(service_type)

  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "with/slash/filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(3)
  ]
  versions.append(types.Part.from_text(text="hello"))

  for i in range(4):
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
    )

  response_versions = await artifact_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert response_versions == list(range(4))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.IN_MEMORY,
        ArtifactServiceType.GCS,
        ArtifactServiceType.FILE,
    ],
)
async def test_list_keys_preserves_user_prefix(
    service_type, artifact_service_factory
):
  """Tests that list_artifact_keys preserves 'user:' prefix in returned names."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  # Save artifacts with "user:" prefix (cross-session artifacts)
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:document.pdf",
      artifact=artifact,
  )

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:image.png",
      artifact=artifact,
  )

  # Save session-scoped artifact without prefix
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="session_file.txt",
      artifact=artifact,
  )

  # List artifacts should return names with "user:" prefix for user-scoped artifacts
  artifact_keys = await artifact_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  # Should contain prefixed names and session file
  expected_keys = ["user:document.pdf", "user:image.png", "session_file.txt"]
  assert sorted(artifact_keys) == sorted(expected_keys)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_list_artifact_versions_and_get_artifact_version(
    service_type, artifact_service_factory
):
  """Tests listing artifact versions and getting a specific version."""
  artifact_service = artifact_service_factory(service_type)
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      custom_metadata = {"key": "value" + str(i)}
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
          artifact=versions[i],
          custom_metadata=custom_metadata,
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    expected_artifact_versions = []
    for i in range(4):
      metadata = {"key": "value" + str(i)}
      if service_type == ArtifactServiceType.GCS:
        uri = (
            f"gs://test_bucket/{app_name}/{user_id}/{session_id}/{filename}/{i}"
        )
      else:
        uri = f"memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{i}"
      expected_artifact_versions.append(
          ArtifactVersion(
              version=i,
              canonical_uri=uri,
              custom_metadata=metadata,
              mime_type="text/plain",
              create_time=FIXED_DATETIME.timestamp(),
          )
      )
    assert artifact_versions == expected_artifact_versions

    # Get latest artifact version when version is not specified
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        == expected_artifact_versions[-1]
    )

    # Get artifact version by version number
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=2,
        )
        == expected_artifact_versions[2]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_list_artifact_versions_with_user_prefix(
    service_type, artifact_service_factory
):
  """Tests listing artifact versions with user prefix."""
  artifact_service = artifact_service_factory(service_type)
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  user_scoped_filename = "user:document.pdf"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      custom_metadata = {"key": "value" + str(i)}
      # Save artifacts with "user:" prefix (cross-session artifacts)
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=user_scoped_filename,
          artifact=versions[i],
          custom_metadata=custom_metadata,
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=user_scoped_filename,
    )

    expected_artifact_versions = []
    for i in range(4):
      metadata = {"key": "value" + str(i)}
      if service_type == ArtifactServiceType.GCS:
        uri = f"gs://test_bucket/{app_name}/{user_id}/user/{user_scoped_filename}/{i}"
      else:
        uri = f"memory://apps/{app_name}/users/{user_id}/artifacts/{user_scoped_filename}/versions/{i}"
      expected_artifact_versions.append(
          ArtifactVersion(
              version=i,
              canonical_uri=uri,
              custom_metadata=metadata,
              mime_type="text/plain",
              create_time=FIXED_DATETIME.timestamp(),
          )
      )
    assert artifact_versions == expected_artifact_versions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_get_artifact_version_artifact_does_not_exist(
    service_type, artifact_service_factory
):
  """Tests getting an artifact version when artifact does not exist."""
  artifact_service = artifact_service_factory(service_type)
  assert not await artifact_service.get_artifact_version(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS]
)
async def test_get_artifact_version_out_of_index(
    service_type, artifact_service_factory
):
  """Tests loading an artifact with an out-of-index version."""
  artifact_service = artifact_service_factory(service_type)
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Attempt to get a version that doesn't exist
  assert not await artifact_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )


@pytest.mark.asyncio
async def test_file_metadata_camelcase(tmp_path, artifact_service_factory):
  """Ensures FileArtifactService writes camelCase metadata without newlines."""
  artifact_service = artifact_service_factory(ArtifactServiceType.FILE)
  artifact = types.Part.from_bytes(
      data=b"binary-content", mime_type="application/octet-stream"
  )
  await artifact_service.save_artifact(
      app_name="myapp",
      user_id="user123",
      session_id="sess789",
      filename="docs/report.txt",
      artifact=artifact,
  )

  metadata_path = (
      tmp_path
      / "artifacts"
      / "apps"
      / "myapp"
      / "users"
      / "user123"
      / "sessions"
      / "sess789"
      / "artifacts"
      / "docs"
      / "report.txt"
      / "versions"
      / "0"
      / "metadata.json"
  )
  raw_metadata = metadata_path.read_text(encoding="utf-8")
  assert "\n" not in raw_metadata

  metadata = json.loads(raw_metadata)
  payload_path = (metadata_path.parent / "report.txt").resolve()
  expected_canonical_uri = payload_path.as_uri()
  create_time = metadata.pop("createTime", None)
  assert create_time is not None
  assert metadata == {
      "fileName": "docs/report.txt",
      "mimeType": "application/octet-stream",
      "canonicalUri": expected_canonical_uri,
      "version": 0,
      "customMetadata": {},
  }
  parsed_canonical = urlparse(metadata["canonicalUri"])
  canonical_path = Path(unquote(parsed_canonical.path))
  assert canonical_path.name == "report.txt"
  assert canonical_path.read_bytes() == b"binary-content"


@pytest.mark.asyncio
async def test_file_list_artifact_versions(tmp_path, artifact_service_factory):
  """FileArtifactService exposes canonical URIs and metadata for each version."""
  artifact_service = artifact_service_factory(ArtifactServiceType.FILE)
  artifact = types.Part.from_bytes(
      data=b"binary-content", mime_type="application/octet-stream"
  )
  custom_metadata = {"origin": "unit-test"}
  await artifact_service.save_artifact(
      app_name="myapp",
      user_id="user123",
      session_id="sess789",
      filename="docs/report.txt",
      artifact=artifact,
      custom_metadata=custom_metadata,
  )

  versions = await artifact_service.list_artifact_versions(
      app_name="myapp",
      user_id="user123",
      session_id="sess789",
      filename="docs/report.txt",
  )
  assert len(versions) == 1
  version_meta = versions[0]
  assert version_meta.version == 0
  version_payload_path = (
      tmp_path
      / "artifacts"
      / "apps"
      / "myapp"
      / "users"
      / "user123"
      / "sessions"
      / "sess789"
      / "artifacts"
      / "docs"
      / "report.txt"
      / "versions"
      / "0"
      / "report.txt"
  ).resolve()
  assert version_meta.canonical_uri == version_payload_path.as_uri()
  assert version_meta.custom_metadata == custom_metadata
  parsed_version_uri = urlparse(version_meta.canonical_uri)
  version_uri_path = Path(unquote(parsed_version_uri.path))
  assert version_uri_path.read_bytes() == b"binary-content"

  fetched = await artifact_service.get_artifact_version(
      app_name="myapp",
      user_id="user123",
      session_id="sess789",
      filename="docs/report.txt",
      version=0,
  )
  assert fetched is not None
  assert fetched.version == version_meta.version
  assert fetched.canonical_uri == version_meta.canonical_uri
  assert fetched.custom_metadata == version_meta.custom_metadata

  latest = await artifact_service.get_artifact_version(
      app_name="myapp",
      user_id="user123",
      session_id="sess789",
      filename="docs/report.txt",
  )
  assert latest is not None
  assert latest.version == version_meta.version
  assert latest.canonical_uri == version_meta.canonical_uri
  assert latest.custom_metadata == version_meta.custom_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "session_id"),
    [
        ("../escape.txt", "sess123"),
        ("user:../escape.txt", "sess123"),
        ("/absolute/path.txt", "sess123"),
        ("user:/absolute/path.txt", None),
    ],
)
async def test_file_save_artifact_rejects_out_of_scope_paths(
    tmp_path, filename, session_id
):
  """FileArtifactService prevents path traversal outside of its storage roots."""
  artifact_service = FileArtifactService(root_dir=tmp_path / "artifacts")
  part = types.Part(text="content")
  with pytest.raises(ValueError):
    await artifact_service.save_artifact(
        app_name="myapp",
        user_id="user123",
        session_id=session_id,
        filename=filename,
        artifact=part,
    )


@pytest.mark.asyncio
async def test_file_save_artifact_rejects_absolute_path_within_scope(tmp_path):
  """Absolute filenames are rejected even when they point inside the scope."""
  artifact_service = FileArtifactService(root_dir=tmp_path / "artifacts")
  absolute_in_scope = (
      tmp_path
      / "artifacts"
      / "apps"
      / "myapp"
      / "users"
      / "user123"
      / "artifacts"
      / "diagram.png"
  )
  part = types.Part(text="content")
  with pytest.raises(ValueError):
    await artifact_service.save_artifact(
        app_name="myapp",
        user_id="user123",
        session_id=None,
        filename=str(absolute_in_scope),
        artifact=part,
    )
