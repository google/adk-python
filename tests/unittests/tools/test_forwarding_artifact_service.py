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

from unittest.mock import AsyncMock

from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
from google.genai import types
from pytest import mark
from pytest import raises


class _StubSession:

  def __init__(self, session_id: str):
    self.id = session_id


class _StubInvocationContext:
  """Minimal InvocationContext stub for ForwardingArtifactService tests."""

  def __init__(self, artifact_service=None):
    self.app_name = 'test_app'
    self.user_id = 'test_user'
    self.session = _StubSession('test_session')
    self.artifact_service = artifact_service


class _StubToolContext:
  """Minimal ToolContext stub for ForwardingArtifactService tests."""

  def __init__(self, invocation_context):
    self._invocation_context = invocation_context
    self.save_artifact = AsyncMock(return_value=1)
    self.load_artifact = AsyncMock(return_value=None)
    self.list_artifacts = AsyncMock(return_value=[])


def _create_service(artifact_service=None):
  invocation_context = _StubInvocationContext(artifact_service)
  tool_context = _StubToolContext(invocation_context)
  return (
      ForwardingArtifactService(tool_context),
      tool_context,
      invocation_context,
  )


@mark.asyncio
async def test_save_artifact_delegates_to_tool_context():
  service, tool_context, _ = _create_service()
  artifact = types.Part(text='hello')

  result = await service.save_artifact(
      app_name='ignored_app',
      user_id='ignored_user',
      filename='file.txt',
      artifact=artifact,
      custom_metadata={'k': 'v'},
  )

  assert result == 1
  tool_context.save_artifact.assert_awaited_once_with(
      filename='file.txt',
      artifact=artifact,
      custom_metadata={'k': 'v'},
  )


@mark.asyncio
async def test_load_artifact_delegates_to_tool_context():
  service, tool_context, _ = _create_service()
  artifact = types.Part(text='hello')
  tool_context.load_artifact.return_value = artifact

  result = await service.load_artifact(
      app_name='ignored_app',
      user_id='ignored_user',
      filename='file.txt',
      version=2,
  )

  assert result is artifact
  tool_context.load_artifact.assert_awaited_once_with(
      filename='file.txt', version=2
  )


@mark.asyncio
async def test_list_artifact_keys_delegates_to_tool_context():
  service, tool_context, _ = _create_service()
  tool_context.list_artifacts.return_value = ['a.txt', 'b.txt']

  result = await service.list_artifact_keys(
      app_name='ignored_app', user_id='ignored_user'
  )

  assert result == ['a.txt', 'b.txt']
  tool_context.list_artifacts.assert_awaited_once_with()


@mark.asyncio
async def test_delete_artifact_delegates_to_invocation_context_service():
  root_artifact_service = AsyncMock()
  service, _, invocation_context = _create_service(root_artifact_service)

  await service.delete_artifact(
      app_name='ignored_app', user_id='ignored_user', filename='file.txt'
  )

  root_artifact_service.delete_artifact.assert_awaited_once_with(
      app_name=invocation_context.app_name,
      user_id=invocation_context.user_id,
      session_id=invocation_context.session.id,
      filename='file.txt',
  )


@mark.asyncio
async def test_delete_artifact_raises_when_no_root_artifact_service():
  service, _, _ = _create_service(artifact_service=None)

  with raises(ValueError, match='Artifact service is not initialized.'):
    await service.delete_artifact(
        app_name='ignored_app', user_id='ignored_user', filename='file.txt'
    )


@mark.asyncio
async def test_list_versions_delegates_to_invocation_context_service():
  root_artifact_service = AsyncMock()
  root_artifact_service.list_versions.return_value = [1, 2, 3]
  service, _, invocation_context = _create_service(root_artifact_service)

  result = await service.list_versions(
      app_name='ignored_app', user_id='ignored_user', filename='file.txt'
  )

  assert result == [1, 2, 3]
  root_artifact_service.list_versions.assert_awaited_once_with(
      app_name=invocation_context.app_name,
      user_id=invocation_context.user_id,
      session_id=invocation_context.session.id,
      filename='file.txt',
  )


@mark.asyncio
async def test_list_versions_raises_when_no_root_artifact_service():
  service, _, _ = _create_service(artifact_service=None)

  with raises(ValueError, match='Artifact service is not initialized.'):
    await service.list_versions(
        app_name='ignored_app', user_id='ignored_user', filename='file.txt'
    )


@mark.asyncio
async def test_list_artifact_versions_raises_not_implemented():
  service, _, _ = _create_service()

  with raises(NotImplementedError):
    await service.list_artifact_versions(
        app_name='ignored_app', user_id='ignored_user', filename='file.txt'
    )


@mark.asyncio
async def test_get_artifact_version_raises_not_implemented():
  service, _, _ = _create_service()

  with raises(NotImplementedError):
    await service.get_artifact_version(
        app_name='ignored_app', user_id='ignored_user', filename='file.txt'
    )
