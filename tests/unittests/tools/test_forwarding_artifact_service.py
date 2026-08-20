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
# pylint: disable=missing-module-docstring
# pylint: disable=protected-access

from typing import Any

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
from google.genai import types
import pytest


class _StubToolContext:
  """Stub for ToolContext to record calls."""

  def __init__(self):
    self.saved_artifacts = []
    self.loaded_artifacts = []
    self.listed_artifacts = False
    self._invocation_context = _StubInvocationContext()

  async def save_artifact(
      self,
      *,
      filename: str,
      artifact: types.Part,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    self.saved_artifacts.append((filename, artifact, custom_metadata))
    return len(self.saved_artifacts) - 1

  async def load_artifact(
      self, *, filename: str, version: int | None = None
  ) -> types.Part | None:
    self.loaded_artifacts.append((filename, version))
    return types.Part(text=f"content_of_{filename}_v{version}")

  async def list_artifacts(self) -> list[str]:
    self.listed_artifacts = True
    return ["art1", "art2"]


class _StubArtifactService:
  """Stub for ArtifactService to record calls."""

  def __init__(self):
    self.deleted_artifacts = []
    self.listed_versions = []
    self.listed_artifact_versions = []
    self.got_artifact_versions = []

  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> None:
    self.deleted_artifacts.append((app_name, user_id, filename, session_id))

  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> list[int]:
    self.listed_versions.append((app_name, user_id, filename, session_id))
    return [1, 2]

  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
  ) -> list[ArtifactVersion]:
    self.listed_artifact_versions.append(
        (app_name, user_id, filename, session_id)
    )
    return [ArtifactVersion(version=1, canonical_uri="uri1")]

  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: str | None = None,
      version: int | None = None,
  ) -> ArtifactVersion | None:
    """Gets the metadata for a specific version of an artifact."""
    self.got_artifact_versions.append(
        (app_name, user_id, filename, session_id, version)
    )
    return ArtifactVersion(version=version or 1, canonical_uri="uri_spec")


class _StubSession:
  """Stub for Session."""

  def __init__(self):
    self.id = "fake_session_id"


class _StubInvocationContext:
  """Stub for InvocationContext."""

  def __init__(self):
    self.artifact_service = _StubArtifactService()
    self.app_name = "fake_app"
    self.user_id = "fake_user"
    self.session = _StubSession()


@pytest.mark.asyncio
async def test_save_artifact_forwards_to_tool_context():
  """Verifies save_artifact forwards to tool_context."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)
  part = types.Part(text="test")

  version = await service.save_artifact(
      app_name="ignored",
      user_id="ignored",
      filename="test.txt",
      artifact=part,
      custom_metadata={"key": "val"},
  )

  assert version == 0
  assert tool_context.saved_artifacts == [("test.txt", part, {"key": "val"})]


@pytest.mark.asyncio
async def test_load_artifact_forwards_to_tool_context():
  """Verifies load_artifact forwards to tool_context."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  part = await service.load_artifact(
      app_name="ignored", user_id="ignored", filename="test.txt", version=2
  )

  assert part.text == "content_of_test.txt_v2"
  assert tool_context.loaded_artifacts == [("test.txt", 2)]


@pytest.mark.asyncio
async def test_list_artifact_keys_forwards_to_tool_context():
  """Verifies list_artifact_keys forwards to tool_context."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  keys = await service.list_artifact_keys(app_name="ignored", user_id="ignored")

  assert keys == ["art1", "art2"]
  assert tool_context.listed_artifacts


@pytest.mark.asyncio
async def test_delete_artifact_forwards_to_invocation_context_artifact_service():
  """Verifies delete_artifact forwards to invocation_context.artifact_service."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  await service.delete_artifact(
      app_name="ignored", user_id="ignored", filename="test.txt"
  )

  stub_service = tool_context._invocation_context.artifact_service
  assert stub_service.deleted_artifacts == [
      ("fake_app", "fake_user", "test.txt", "fake_session_id")
  ]


@pytest.mark.asyncio
async def test_delete_artifact_raises_value_error_if_no_service():
  """Verifies delete_artifact raises ValueError if artifact_service is None."""
  tool_context = _StubToolContext()
  tool_context._invocation_context.artifact_service = None
  service = ForwardingArtifactService(tool_context)

  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await service.delete_artifact(
        app_name="ignored", user_id="ignored", filename="test.txt"
    )


@pytest.mark.asyncio
async def test_list_versions_forwards_to_invocation_context_artifact_service():
  """Verifies list_versions forwards to invocation_context.artifact_service."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  versions = await service.list_versions(
      app_name="ignored", user_id="ignored", filename="test.txt"
  )

  assert versions == [1, 2]
  stub_service = tool_context._invocation_context.artifact_service
  assert stub_service.listed_versions == [
      ("fake_app", "fake_user", "test.txt", "fake_session_id")
  ]


@pytest.mark.asyncio
async def test_list_versions_raises_value_error_if_no_service():
  """Verifies list_versions raises ValueError if artifact_service is None."""
  tool_context = _StubToolContext()
  tool_context._invocation_context.artifact_service = None
  service = ForwardingArtifactService(tool_context)

  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await service.list_versions(
        app_name="ignored", user_id="ignored", filename="test.txt"
    )


@pytest.mark.asyncio
async def test_list_artifact_versions_forwards_to_invocation_context_artifact_service():
  """Verifies list_artifact_versions forwards to invocation_context.artifact_service."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  versions = await service.list_artifact_versions(
      app_name="ignored", user_id="ignored", filename="test.txt"
  )

  assert len(versions) == 1
  assert versions[0].version == 1
  assert versions[0].canonical_uri == "uri1"
  stub_service = tool_context._invocation_context.artifact_service
  assert stub_service.listed_artifact_versions == [
      ("fake_app", "fake_user", "test.txt", "fake_session_id")
  ]


@pytest.mark.asyncio
async def test_list_artifact_versions_raises_value_error_if_no_service():
  """Verifies list_artifact_versions raises ValueError if artifact_service is None."""
  tool_context = _StubToolContext()
  tool_context._invocation_context.artifact_service = None
  service = ForwardingArtifactService(tool_context)

  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await service.list_artifact_versions(
        app_name="ignored", user_id="ignored", filename="test.txt"
    )


@pytest.mark.asyncio
async def test_get_artifact_version_forwards_to_invocation_context_artifact_service():
  """Verifies get_artifact_version forwards to invocation_context.artifact_service."""
  tool_context = _StubToolContext()
  service = ForwardingArtifactService(tool_context)

  version = await service.get_artifact_version(
      app_name="ignored", user_id="ignored", filename="test.txt", version=3
  )

  assert version.version == 3
  assert version.canonical_uri == "uri_spec"
  stub_service = tool_context._invocation_context.artifact_service
  assert stub_service.got_artifact_versions == [
      ("fake_app", "fake_user", "test.txt", "fake_session_id", 3)
  ]


@pytest.mark.asyncio
async def test_get_artifact_version_raises_value_error_if_no_service():
  """Verifies get_artifact_version raises ValueError if artifact_service is None."""
  tool_context = _StubToolContext()
  tool_context._invocation_context.artifact_service = None
  service = ForwardingArtifactService(tool_context)

  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await service.get_artifact_version(
        app_name="ignored", user_id="ignored", filename="test.txt", version=3
    )
