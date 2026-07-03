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

"""Tests for DaytonaEnvironment."""

from unittest import mock

import daytona
from daytona import CreateSandboxFromImageParams
from daytona import CreateSandboxFromSnapshotParams
from daytona import DaytonaError
from daytona import DaytonaNotFoundError
from google.adk.integrations.daytona._daytona_environment import DaytonaEnvironment
import pytest


def _make_sandbox() -> mock.MagicMock:
  """Build a mock AsyncSandbox with async method stubs."""
  sandbox = mock.MagicMock(name="AsyncSandbox")
  sandbox.delete = mock.AsyncMock()
  sandbox.process.exec = mock.AsyncMock()
  sandbox.fs.download_file = mock.AsyncMock()
  sandbox.fs.upload_file = mock.AsyncMock()
  sandbox.fs.create_folder = mock.AsyncMock()
  sandbox.refresh_activity = mock.AsyncMock()
  return sandbox


@pytest.fixture(name="sandbox")
def _sandbox() -> mock.MagicMock:
  return _make_sandbox()


@pytest.fixture(name="daytona_patch")
def _daytona_patch(sandbox: mock.MagicMock):
  """Patch AsyncDaytona to return a mock client."""
  mock_client = mock.MagicMock(name="AsyncDaytona")
  mock_client.create = mock.AsyncMock(return_value=sandbox)

  with mock.patch.object(daytona, "AsyncDaytona", autospec=True) as mock_class:
    mock_class.return_value = mock_client
    yield mock_class


async def test_initialize_creates_sandbox(daytona_patch, sandbox):
  env = DaytonaEnvironment(image="custom-image", env_vars={"A": "1"})
  assert env.is_initialized is False
  await env.initialize()
  assert env.is_initialized is True

  daytona_patch.assert_called_once()
  client = daytona_patch.return_value
  client.create.assert_awaited_once()

  args, _ = client.create.call_args
  params = args[0]
  assert isinstance(params, CreateSandboxFromImageParams)
  assert params.image == "custom-image"
  assert params.env_vars == {"A": "1"}
  assert params.auto_stop_interval == 5
  assert params.auto_delete_interval == 0
  assert env._sandbox is sandbox


async def test_initialize_creates_sandbox_default(daytona_patch, sandbox):
  env = DaytonaEnvironment(env_vars={"B": "2"})
  await env.initialize()

  daytona_patch.assert_called_once()
  client = daytona_patch.return_value
  client.create.assert_awaited_once()

  args, _ = client.create.call_args
  params = args[0]
  assert isinstance(params, CreateSandboxFromSnapshotParams)
  assert params.language == "python"
  assert params.env_vars == {"B": "2"}
  assert params.auto_stop_interval == 5
  assert params.auto_delete_interval == 0
  assert env._sandbox is sandbox


async def test_initialize_is_idempotent(daytona_patch, sandbox):
  env = DaytonaEnvironment()
  await env.initialize()
  await env.initialize()
  client = daytona_patch.return_value
  client.create.assert_awaited_once()


async def test_close_deletes_sandbox_and_is_idempotent(daytona_patch, sandbox):
  env = DaytonaEnvironment()
  await env.initialize()
  assert env.is_initialized is True
  await env.close()
  sandbox.delete.assert_awaited_once()
  assert env._sandbox is None
  assert env.is_initialized is False

  # Second close is a no-op.
  await env.close()
  sandbox.delete.assert_awaited_once()


async def test_working_dir_requires_initialize():
  env = DaytonaEnvironment()
  with pytest.raises(RuntimeError):
    _ = env.working_dir


async def test_execute_before_initialize_raises():
  env = DaytonaEnvironment()
  with pytest.raises(RuntimeError):
    await env.execute("echo hi")


async def test_execute_success(daytona_patch, sandbox):
  class MockArtifacts:
    stdout = "out"

  class MockResponse:
    exit_code = 0
    artifacts = MockArtifacts()

  sandbox.process.exec.return_value = MockResponse()

  env = DaytonaEnvironment()
  await env.initialize()

  result = await env.execute("echo out")

  assert result.exit_code == 0
  assert result.stdout == "out"
  assert result.stderr == ""
  assert result.timed_out is False
  sandbox.refresh_activity.assert_awaited_once()


async def test_execute_timeout(daytona_patch, sandbox):
  # Simulate a Daytona timeout error
  sandbox.process.exec.side_effect = DaytonaError("timeout occurred")
  env = DaytonaEnvironment()
  await env.initialize()

  result = await env.execute("sleep 999")

  assert result.timed_out is True


async def test_read_file_returns_bytes(daytona_patch, sandbox):
  sandbox.fs.download_file.return_value = b"data"
  env = DaytonaEnvironment()
  await env.initialize()

  data = await env.read_file("notes.txt")

  assert data == b"data"
  sandbox.fs.download_file.assert_awaited_once_with("/workspaces/notes.txt")
  sandbox.refresh_activity.assert_awaited_once()


async def test_read_file_absolute_path_passthrough(daytona_patch, sandbox):
  sandbox.fs.download_file.return_value = b"x"
  env = DaytonaEnvironment()
  await env.initialize()

  await env.read_file("/etc/hostname")

  sandbox.fs.download_file.assert_awaited_once_with("/etc/hostname")


async def test_read_file_missing_raises(daytona_patch, sandbox):
  sandbox.fs.download_file.return_value = None
  env = DaytonaEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError):
    await env.read_file("missing.txt")


async def test_write_file_resolves_relative_path(daytona_patch, sandbox):
  env = DaytonaEnvironment()
  await env.initialize()

  await env.write_file("sub/out.txt", "hello")
  sandbox.refresh_activity.assert_awaited_once()

  sandbox.fs.upload_file.assert_awaited_once_with(
      b"hello", "/workspaces/sub/out.txt"
  )


async def test_initialize_propagates_api_key_and_url(daytona_patch, sandbox):
  from daytona import DaytonaConfig

  env = DaytonaEnvironment(api_key="my-key", api_url="my-url")
  await env.initialize()

  daytona_patch.assert_called_once()
  _, kwargs = daytona_patch.call_args
  config = kwargs.get("config")
  assert isinstance(config, DaytonaConfig)
  assert config.api_key == "my-key"
  assert config.api_url == "my-url"


async def test_write_file_creates_parent_directory(daytona_patch, sandbox):
  env = DaytonaEnvironment()
  await env.initialize()

  await env.write_file("sub/nested/file.txt", "content")

  sandbox.fs.create_folder.assert_has_calls([
      mock.call("/workspaces", mode="755"),
      mock.call("/workspaces/sub", mode="755"),
      mock.call("/workspaces/sub/nested", mode="755"),
  ])
  sandbox.fs.upload_file.assert_awaited_once_with(
      b"content", "/workspaces/sub/nested/file.txt"
  )


async def test_read_file_raises_file_not_found_on_daytona_not_found(
    daytona_patch, sandbox
):
  sandbox.fs.download_file.side_effect = DaytonaNotFoundError("not found")
  env = DaytonaEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError):
    await env.read_file("missing.txt")
