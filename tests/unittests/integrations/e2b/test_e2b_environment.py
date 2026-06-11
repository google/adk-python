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

"""Tests for E2BEnvironment."""

from unittest import mock

from e2b import CommandExitException
from e2b import CommandResult
from e2b import FileNotFoundException
from e2b import TimeoutException
from google.adk.integrations.e2b._e2b_environment import E2BEnvironment
import pytest


def _make_sandbox(*, running: bool = True) -> mock.MagicMock:
  """Build a mock AsyncSandbox with async method stubs."""
  sandbox = mock.MagicMock(name='AsyncSandbox')
  sandbox.is_running = mock.AsyncMock(return_value=running)
  sandbox.set_timeout = mock.AsyncMock()
  sandbox.kill = mock.AsyncMock(return_value=True)
  sandbox.commands.run = mock.AsyncMock()
  sandbox.files.read = mock.AsyncMock()
  sandbox.files.write = mock.AsyncMock()
  return sandbox


@pytest.fixture(name='sandbox')
def _sandbox() -> mock.MagicMock:
  return _make_sandbox()


@pytest.fixture(name='create_patch')
def _create_patch(sandbox: mock.MagicMock):
  """Patch AsyncSandbox.create to return the mock sandbox."""
  with mock.patch(
      'e2b.AsyncSandbox.create', new=mock.AsyncMock(return_value=sandbox)
  ) as create:
    yield create


@pytest.mark.asyncio
async def test_initialize_creates_sandbox(create_patch, sandbox):
  env = E2BEnvironment(image='custom', timeout=120, env_vars={'A': '1'})
  await env.initialize()

  create_patch.assert_awaited_once()
  _, kwargs = create_patch.call_args
  assert kwargs['template'] == 'custom'
  assert kwargs['timeout'] == 120
  assert kwargs['envs'] == {'A': '1'}
  assert env._sandbox is sandbox


@pytest.mark.asyncio
async def test_initialize_is_idempotent(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()
  await env.initialize()
  create_patch.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_kills_sandbox_and_is_idempotent(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()
  await env.close()
  sandbox.kill.assert_awaited_once()
  assert env._sandbox is None
  # Second close is a no-op.
  await env.close()
  sandbox.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_working_dir_requires_initialize():
  env = E2BEnvironment()
  with pytest.raises(RuntimeError):
    _ = env.working_dir


@pytest.mark.asyncio
async def test_execute_before_initialize_raises():
  env = E2BEnvironment()
  with pytest.raises(RuntimeError):
    await env.execute('echo hi')


@pytest.mark.asyncio
async def test_execute_success(create_patch, sandbox):
  sandbox.commands.run.return_value = CommandResult(
      stdout='out', stderr='err', exit_code=0, error=None
  )
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('echo out')

  assert result.exit_code == 0
  assert result.stdout == 'out'
  assert result.stderr == 'err'
  assert result.timed_out is False
  sandbox.set_timeout.assert_awaited()  # keepalive


@pytest.mark.asyncio
async def test_execute_nonzero_exit_is_normal_result(create_patch, sandbox):
  exc = CommandExitException(
      stdout='partial', stderr='boom', exit_code=2, error='failed'
  )
  sandbox.commands.run.side_effect = exc
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('false')

  assert result.exit_code == 2
  assert result.stdout == 'partial'
  assert result.stderr == 'boom'
  assert result.timed_out is False


@pytest.mark.asyncio
async def test_execute_timeout(create_patch, sandbox):
  sandbox.commands.run.side_effect = TimeoutException('too slow')
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('sleep 999')

  assert result.timed_out is True


@pytest.mark.asyncio
async def test_read_file_returns_bytes(create_patch, sandbox):
  sandbox.files.read.return_value = b'data'
  env = E2BEnvironment()
  await env.initialize()

  data = await env.read_file('notes.txt')

  assert data == b'data'
  sandbox.files.read.assert_awaited_once_with(
      '/home/user/notes.txt', format='bytes'
  )


@pytest.mark.asyncio
async def test_read_file_absolute_path_passthrough(create_patch, sandbox):
  sandbox.files.read.return_value = b'x'
  env = E2BEnvironment()
  await env.initialize()

  await env.read_file('/etc/hostname')

  sandbox.files.read.assert_awaited_once_with('/etc/hostname', format='bytes')


@pytest.mark.asyncio
async def test_read_file_missing_raises(create_patch, sandbox):
  sandbox.files.read.side_effect = FileNotFoundException('nope')
  env = E2BEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError):
    await env.read_file('missing.txt')


@pytest.mark.asyncio
async def test_write_file_resolves_relative_path(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()

  await env.write_file('sub/out.txt', 'hello')

  sandbox.files.write.assert_awaited_once_with(
      '/home/user/sub/out.txt', 'hello'
  )


@pytest.mark.asyncio
async def test_keepalive_extends_timeout_when_running(create_patch, sandbox):
  sandbox.files.read.return_value = b'1'
  env = E2BEnvironment(timeout=200)
  await env.initialize()

  await env.read_file('a.txt')

  sandbox.set_timeout.assert_awaited_with(200)


@pytest.mark.asyncio
async def test_lazy_recreate_when_expired(sandbox):
  expired = _make_sandbox(running=False)
  fresh = _make_sandbox(running=True)
  fresh.files.read.return_value = b'fresh'

  with mock.patch(
      'e2b.AsyncSandbox.create',
      new=mock.AsyncMock(side_effect=[expired, fresh]),
  ) as create:
    env = E2BEnvironment()
    await env.initialize()  # -> expired
    data = await env.read_file('a.txt')  # detects dead, recreates -> fresh

  assert data == b'fresh'
  assert create.await_count == 2
  assert env._sandbox is fresh
  expired.set_timeout.assert_not_awaited()
