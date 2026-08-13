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

"""Tests for OpenSandboxEnvironment."""

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import gc
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest import mock

from google.adk.integrations.opensandbox import OpenSandboxEnvironment
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import SandboxApiException
import pytest


def _renew_response(timeout: timedelta) -> SimpleNamespace:
  return SimpleNamespace(expires_at=datetime.now(timezone.utc) + timeout)


def _make_sandbox(*, sandbox_id: str = "sandbox-1") -> mock.MagicMock:
  """Build a mock async OpenSandbox client."""

  async def _renew(timeout):
    return _renew_response(timeout)

  sandbox = mock.MagicMock(name="Sandbox")
  sandbox.id = sandbox_id
  sandbox.destroy = mock.AsyncMock()
  sandbox.close = mock.AsyncMock()
  sandbox.renew = mock.AsyncMock(side_effect=_renew)
  sandbox.commands.run = mock.AsyncMock()
  sandbox.files.create_directories = mock.AsyncMock()
  sandbox.files.read_bytes = mock.AsyncMock()
  sandbox.files.write_file = mock.AsyncMock()
  return sandbox


def _execution(
    *,
    exit_code: int | None = 0,
    stdout: tuple[str, ...] = (),
    stderr: tuple[str, ...] = (),
    results: tuple[str, ...] = (),
    error: object | None = None,
) -> SimpleNamespace:
  """Build the observable result returned by the OpenSandbox SDK boundary."""
  return SimpleNamespace(
      exit_code=exit_code,
      error=error,
      result=[SimpleNamespace(text=value) for value in results],
      logs=SimpleNamespace(
          stdout=[SimpleNamespace(text=value) for value in stdout],
          stderr=[SimpleNamespace(text=value) for value in stderr],
      ),
  )


@pytest.fixture(name="sandbox")
def _sandbox() -> mock.MagicMock:
  return _make_sandbox()


@pytest.fixture(name="create_patch")
def _create_patch(sandbox: mock.MagicMock):
  """Patch Sandbox.create to return the mock sandbox."""
  with mock.patch(
      "opensandbox.Sandbox.create", new=mock.AsyncMock(return_value=sandbox)
  ) as create:
    yield create


@pytest.fixture(name="connect_patch")
def _connect_patch(sandbox: mock.MagicMock):
  """Patch Sandbox.connect to return the mock sandbox."""
  with mock.patch(
      "opensandbox.Sandbox.connect", new=mock.AsyncMock(return_value=sandbox)
  ) as connect:
    yield connect


@pytest.mark.asyncio
async def test_initialize_creates_sandbox_with_configuration(
    create_patch: mock.AsyncMock, sandbox: mock.MagicMock
):
  """Initialization forwards lifecycle and connection settings."""
  env = OpenSandboxEnvironment(
      image="custom:latest",
      timeout=120,
      ready_timeout=12.5,
      working_dir="/work",
      env_vars={"A": "1"},
      metadata={"team": "adk"},
      api_key="key",
      domain="sandbox.example:8080",
      protocol="https",
      request_timeout=8,
      use_server_proxy=True,
  )

  await env.initialize()

  assert env.is_initialized is True
  assert env.working_dir == Path("/work")
  args, kwargs = create_patch.call_args
  assert args == ("custom:latest",)
  assert kwargs["timeout"] == timedelta(seconds=120)
  assert kwargs["ready_timeout"] == timedelta(seconds=12.5)
  assert kwargs["env"] == {"A": "1"}
  assert kwargs["metadata"] == {
      "framework": "google-adk",
      "integration": "google-adk-opensandbox",
      "team": "adk",
  }
  config = kwargs["connection_config"]
  assert config.api_key == "key"
  assert config.domain == "sandbox.example:8080"
  assert config.protocol == "https"
  assert config.request_timeout == timedelta(seconds=8)
  assert config.use_server_proxy is True
  directory = sandbox.files.create_directories.await_args.args[0][0]
  assert directory.path == "/work"
  assert directory.mode == 755


@pytest.mark.asyncio
async def test_initialize_uses_snapshot_without_default_image(create_patch):
  """A snapshot startup does not also send the default image."""
  env = OpenSandboxEnvironment(snapshot_id="snapshot-1")

  await env.initialize()

  args, kwargs = create_patch.call_args
  assert args == (None,)
  assert kwargs["snapshot_id"] == "snapshot-1"


@pytest.mark.asyncio
async def test_initialize_uses_complete_connection_config(create_patch):
  """A caller-supplied ConnectionConfig is passed through unchanged."""
  config = ConnectionConfig(domain="sandbox.example:8080")
  env = OpenSandboxEnvironment(connection_config=config)

  await env.initialize()

  assert create_patch.await_args.kwargs["connection_config"] is config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain",
    ["https://sandbox.example", "https://sandbox.example:8443"],
)
async def test_initialize_derives_https_protocol_from_domain(
    create_patch, domain
):
  """A URL scheme also configures HTTPS for sandbox service endpoints."""
  env = OpenSandboxEnvironment(domain=domain)

  await env.initialize()

  config = create_patch.await_args.kwargs["connection_config"]
  assert config.protocol == "https"


@pytest.mark.asyncio
async def test_initialize_derives_protocol_from_environment(
    create_patch, monkeypatch
):
  """The environment domain scheme applies to lifecycle and sandbox calls."""
  monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", "https://sandbox.example")
  env = OpenSandboxEnvironment()

  await env.initialize()

  config = create_patch.await_args.kwargs["connection_config"]
  assert config.protocol == "https"


@pytest.mark.asyncio
async def test_initialize_is_idempotent(create_patch):
  """Repeated initialization creates one remote sandbox."""
  env = OpenSandboxEnvironment()

  await env.initialize()
  await env.initialize()

  create_patch.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_initialize_creates_one_sandbox(create_patch):
  """Concurrent initialization is serialized without leaking a sandbox."""
  env = OpenSandboxEnvironment()

  await asyncio.gather(env.initialize(), env.initialize(), env.initialize())

  create_patch.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_destroys_owned_sandbox_and_is_idempotent(
    create_patch, sandbox
):
  """Closing a created environment destroys the remote resource once."""
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.close()
  await env.close()

  sandbox.destroy.assert_awaited_once()
  sandbox.close.assert_not_awaited()
  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_close_accepts_already_expired_owned_sandbox(
    create_patch, sandbox
):
  """An already absent remote resource still counts as successfully closed."""
  sandbox.destroy.side_effect = SandboxApiException(status_code=404)
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.close()

  with mock.patch(
      "opensandbox.Sandbox.connect", new=mock.AsyncMock()
  ) as connect:
    await env.close()

  connect.assert_not_awaited()
  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_close_attached_sandbox_releases_only_local_resources(
    connect_patch, sandbox
):
  """Closing an attached environment leaves the caller-owned sandbox alive."""
  env = OpenSandboxEnvironment(sandbox_id="existing-1")
  await env.initialize()

  await env.close()

  connect_patch.assert_awaited_once()
  assert connect_patch.await_args.args == ("existing-1",)
  sandbox.close.assert_awaited_once()
  sandbox.destroy.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_waits_for_active_operation(create_patch, sandbox):
  """Closing never destroys a sandbox while a command is using it."""
  renew_started = asyncio.Event()
  allow_renew = asyncio.Event()

  async def _renew(timeout):
    renew_started.set()
    await allow_renew.wait()
    return _renew_response(timeout)

  sandbox.renew.side_effect = _renew
  sandbox.commands.run.return_value = _execution(stdout=("done",))
  env = OpenSandboxEnvironment()
  await env.initialize()

  execute_task = asyncio.create_task(env.execute("printf done"))
  await renew_started.wait()
  close_task = asyncio.create_task(env.close())
  await asyncio.sleep(0)

  assert close_task.done() is False
  sandbox.destroy.assert_not_awaited()

  allow_renew.set()
  result, _ = await asyncio.gather(execute_task, close_task)

  assert result.stdout == "done"
  sandbox.destroy.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_rejects_operations_while_draining(create_patch, sandbox):
  """A close in progress prevents new operations from delaying cleanup."""
  renew_started = asyncio.Event()
  allow_renew = asyncio.Event()

  async def _renew(timeout):
    renew_started.set()
    await allow_renew.wait()
    return _renew_response(timeout)

  sandbox.renew.side_effect = _renew
  sandbox.commands.run.return_value = _execution(stdout=("done",))
  env = OpenSandboxEnvironment()
  await env.initialize()

  execute_task = asyncio.create_task(env.execute("printf done"))
  await renew_started.wait()
  close_task = asyncio.create_task(env.close())
  await asyncio.sleep(0)

  with pytest.raises(RuntimeError, match="closing"):
    await env.read_file("late.txt")

  allow_renew.set()
  result, _ = await asyncio.gather(execute_task, close_task)

  assert result.stdout == "done"
  sandbox.destroy.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_waits_for_close_then_creates_new_sandbox(
    create_patch, sandbox
):
  """Initialization concurrent with close returns a new live sandbox."""
  renew_started = asyncio.Event()
  allow_renew = asyncio.Event()

  async def _renew(timeout):
    renew_started.set()
    await allow_renew.wait()
    return _renew_response(timeout)

  sandbox.renew.side_effect = _renew
  sandbox.commands.run.return_value = _execution(stdout=("done",))
  env = OpenSandboxEnvironment()
  await env.initialize()

  execute_task = asyncio.create_task(env.execute("printf done"))
  await renew_started.wait()
  close_task = asyncio.create_task(env.close())
  await asyncio.sleep(0)
  initialize_task = asyncio.create_task(env.initialize())
  await asyncio.sleep(0)

  assert initialize_task.done() is False
  assert create_patch.await_count == 1

  allow_renew.set()
  await asyncio.gather(execute_task, close_task, initialize_task)

  assert env.is_initialized is True
  assert create_patch.await_count == 2
  sandbox.destroy.assert_awaited_once()

  await env.close()


@pytest.mark.asyncio
async def test_concurrent_close_does_not_destroy_reinitialized_sandbox():
  """Concurrent close calls coalesce instead of closing the next generation."""
  old_sandbox = _make_sandbox(sandbox_id="old-sandbox")
  new_sandbox = _make_sandbox(sandbox_id="new-sandbox")
  destroy_started = asyncio.Event()
  allow_destroy = asyncio.Event()

  async def _destroy_old():
    destroy_started.set()
    await allow_destroy.wait()

  old_sandbox.destroy.side_effect = _destroy_old
  create = mock.AsyncMock(side_effect=[old_sandbox, new_sandbox])
  env = OpenSandboxEnvironment()

  with mock.patch("opensandbox.Sandbox.create", new=create):
    await env.initialize()
    first_close = asyncio.create_task(env.close())
    await destroy_started.wait()
    initialize = asyncio.create_task(env.initialize())
    second_close = asyncio.create_task(env.close())
    await asyncio.sleep(0)

    allow_destroy.set()
    await asyncio.gather(first_close, initialize, second_close)

  assert env.is_initialized is True
  assert create.await_count == 2
  old_sandbox.destroy.assert_awaited_once()
  new_sandbox.destroy.assert_not_awaited()
  new_sandbox.files.read_bytes.return_value = b"live"
  assert await env.read_file("live.txt") == b"live"

  await env.close()


@pytest.mark.asyncio
async def test_cancelled_close_does_not_cancel_concurrent_close(
    create_patch, sandbox
):
  """Cancelling one close waiter does not abandon shared cleanup."""
  renew_started = asyncio.Event()
  allow_renew = asyncio.Event()

  async def _renew(timeout):
    renew_started.set()
    await allow_renew.wait()
    return _renew_response(timeout)

  sandbox.renew.side_effect = _renew
  sandbox.commands.run.return_value = _execution(stdout=("done",))
  env = OpenSandboxEnvironment()
  await env.initialize()

  execute = asyncio.create_task(env.execute("printf done"))
  await renew_started.wait()
  first_close = asyncio.create_task(env.close())
  second_close = asyncio.create_task(env.close())
  await asyncio.sleep(0)
  first_close.cancel()

  with pytest.raises(asyncio.CancelledError):
    await first_close
  assert second_close.done() is False

  allow_renew.set()
  result, _ = await asyncio.gather(execute, second_close)

  assert result.stdout == "done"
  assert env.is_initialized is False
  sandbox.destroy.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_only_close_waiter_observes_cleanup_failure(
    create_patch, sandbox
):
  """A background cleanup failure is retrieved after its waiter is cancelled."""
  destroy_started = asyncio.Event()
  allow_destroy = asyncio.Event()
  loop_errors = []

  async def _destroy():
    destroy_started.set()
    await allow_destroy.wait()
    raise RuntimeError("delete failed")

  sandbox.destroy.side_effect = _destroy
  env = OpenSandboxEnvironment()
  await env.initialize()
  loop = asyncio.get_running_loop()
  old_exception_handler = loop.get_exception_handler()
  loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
  try:
    close_waiter = asyncio.create_task(env.close())
    await destroy_started.wait()
    close_waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
      await close_waiter

    allow_destroy.set()
    deadline = monotonic() + 5.0
    while env._close_task is not None:
      assert monotonic() < deadline, "close task did not finish"
      await asyncio.sleep(0.001)
    gc.collect()
    await asyncio.sleep(0)

    retry_sandbox = _make_sandbox(sandbox_id="sandbox-1")
    with mock.patch(
        "opensandbox.Sandbox.connect",
        new=mock.AsyncMock(return_value=retry_sandbox),
    ):
      await env.close()

    assert loop_errors == []
    retry_sandbox.destroy.assert_awaited_once()
  finally:
    loop.set_exception_handler(old_exception_handler)


@pytest.mark.asyncio
async def test_close_retries_failed_owned_cleanup(create_patch, sandbox):
  """A second close reconnects to retry a failed remote destroy."""
  sandbox.destroy.side_effect = RuntimeError("delete failed")
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(RuntimeError, match="delete failed"):
    await env.close()

  retry_sandbox = _make_sandbox(sandbox_id="sandbox-1")
  with mock.patch(
      "opensandbox.Sandbox.connect",
      new=mock.AsyncMock(return_value=retry_sandbox),
  ) as connect:
    await env.close()

  connect.assert_awaited_once()
  assert connect.await_args.args == ("sandbox-1",)
  assert connect.await_args.kwargs["skip_health_check"] is True
  retry_sandbox.destroy.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_failure_destroys_owned_sandbox(create_patch, sandbox):
  """A working-directory failure does not leak a newly created sandbox."""
  sandbox.files.create_directories.side_effect = RuntimeError("setup failed")
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="setup failed"):
    await env.initialize()

  sandbox.destroy.assert_awaited_once()
  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_initialize_failure_does_not_destroy_attached_sandbox(
    connect_patch, sandbox
):
  """A failed attached setup closes its client without killing the sandbox."""
  sandbox.files.create_directories.side_effect = RuntimeError("setup failed")
  env = OpenSandboxEnvironment(sandbox_id="existing-1")

  with pytest.raises(RuntimeError, match="setup failed"):
    await env.initialize()

  sandbox.close.assert_awaited_once()
  sandbox.destroy.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_initialize_error_and_retries(
    create_patch, sandbox
):
  """Cleanup errors preserve setup failure and leave remote cleanup retryable."""
  sandbox.files.create_directories.side_effect = RuntimeError("setup failed")
  sandbox.destroy.side_effect = RuntimeError("cleanup failed")
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="setup failed"):
    await env.initialize()

  retry_sandbox = _make_sandbox(sandbox_id="sandbox-1")
  with mock.patch(
      "opensandbox.Sandbox.connect",
      new=mock.AsyncMock(return_value=retry_sandbox),
  ) as connect:
    await env.close()

  connect.assert_awaited_once()
  assert connect.await_args.kwargs["skip_health_check"] is True
  retry_sandbox.destroy.assert_awaited_once()


def test_working_dir_requires_initialize():
  """The working directory is unavailable before initialization."""
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    _ = env.working_dir


@pytest.mark.asyncio
async def test_execute_requires_initialize():
  """Commands cannot run before initialization."""
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    await env.execute("echo hi")


@pytest.mark.asyncio
async def test_read_requires_initialize():
  """Files cannot be read before initialization."""
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    await env.read_file("a.txt")


@pytest.mark.asyncio
async def test_write_requires_initialize():
  """Files cannot be written before initialization."""
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    await env.write_file("a.txt", "data")


@pytest.mark.asyncio
async def test_execute_returns_separate_streams_and_result_text(
    create_patch, sandbox
):
  """Command output keeps stdout and stderr separate."""
  sandbox.commands.run.return_value = _execution(
      stdout=("first\n", "second"),
      stderr=("warning\n",),
      results=("result\n",),
  )
  env = OpenSandboxEnvironment(env_vars={"A": "1"})
  await env.initialize()

  result = await env.execute("printf test")

  assert result.exit_code == 0
  assert result.stdout == "first\nsecond\nresult"
  assert result.stderr == "warning"
  assert result.timed_out is False
  opts = sandbox.commands.run.await_args.kwargs["opts"]
  assert opts.working_directory == "/workspace"
  assert opts.timeout is None
  assert opts.envs == {"A": "1"}
  sandbox.renew.assert_awaited_once_with(timedelta(seconds=300))


@pytest.mark.asyncio
async def test_execute_preserves_nonzero_exit(create_patch, sandbox):
  """A nonzero process exit is a normal execution result."""
  sandbox.commands.run.return_value = _execution(
      exit_code=7, stderr=("failed",)
  )
  env = OpenSandboxEnvironment()
  await env.initialize()

  result = await env.execute("exit 7")

  assert result.exit_code == 7
  assert result.stderr == "failed"


@pytest.mark.asyncio
async def test_execute_renews_past_command_deadline(create_patch, sandbox):
  """A finite command deadline cannot outlive the owned sandbox TTL."""
  sandbox.commands.run.return_value = _execution(stdout=("done",))
  env = OpenSandboxEnvironment(timeout=60)
  await env.initialize()

  await env.execute("sleep 120", timeout=120)

  sandbox.renew.assert_awaited_once_with(timedelta(seconds=150))


@pytest.mark.asyncio
async def test_execute_without_deadline_renews_while_running(
    create_patch, sandbox, monkeypatch
):
  """An unlimited command receives heartbeat renewals until it completes."""

  async def _run(*_args, **_kwargs):
    await asyncio.sleep(0.13)
    return _execution(stdout=("done",))

  monkeypatch.setattr(
      "google.adk.integrations.opensandbox._opensandbox_environment."
      "_MAX_RENEW_INTERVAL_SECONDS",
      0.05,
  )
  sandbox.commands.run.side_effect = _run
  env = OpenSandboxEnvironment(timeout=60)
  await env.initialize()

  result = await env.execute("sleep 0.13")

  assert result.stdout == "done"
  assert sandbox.renew.await_count >= 2
  sandbox.renew.assert_awaited_with(timedelta(seconds=60))


@pytest.mark.asyncio
async def test_short_operation_does_not_shorten_long_command_ttl(
    create_patch, sandbox
):
  """Concurrent operations serialize renewals without reducing expiration."""
  command_started = asyncio.Event()
  allow_command = asyncio.Event()

  async def _run(*_args, **_kwargs):
    command_started.set()
    await allow_command.wait()
    return _execution(stdout=("done",))

  sandbox.commands.run.side_effect = _run
  sandbox.files.read_bytes.return_value = b"data"
  env = OpenSandboxEnvironment(timeout=60)
  await env.initialize()

  execute = asyncio.create_task(env.execute("sleep 600", timeout=600))
  await command_started.wait()
  assert await env.read_file("data.bin") == b"data"

  sandbox.renew.assert_awaited_once_with(timedelta(seconds=630))
  allow_command.set()
  result = await execute

  assert result.stdout == "done"


@pytest.mark.asyncio
async def test_execute_marks_server_deadline_as_timed_out(
    create_patch, sandbox
):
  """A deadline-length exit minus one is reported as a best-effort timeout."""
  sandbox.commands.run.return_value = _execution(exit_code=-1)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with mock.patch(
      "google.adk.integrations.opensandbox._opensandbox_environment.monotonic",
      side_effect=[10.0, 11.1],
  ):
    result = await env.execute("sleep 30", timeout=1)

  assert result.timed_out is True
  opts = sandbox.commands.run.await_args.kwargs["opts"]
  assert opts.timeout == timedelta(seconds=1)


@pytest.mark.asyncio
async def test_execute_does_not_mislabel_early_sigkill(create_patch, sandbox):
  """An immediate exit minus one is not called a timeout before its deadline."""
  sandbox.commands.run.return_value = _execution(exit_code=-1)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with mock.patch(
      "google.adk.integrations.opensandbox._opensandbox_environment.monotonic",
      side_effect=[10.0, 10.1],
  ):
    result = await env.execute("kill -9 $$", timeout=5)

  assert result.timed_out is False


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, 0.0001])
async def test_execute_uses_minimum_one_millisecond_timeout(
    create_patch, sandbox, timeout
):
  """Zero and sub-millisecond deadlines never become unlimited commands."""
  sandbox.commands.run.return_value = _execution(exit_code=-1)
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.execute("sleep 30", timeout=timeout)

  opts = sandbox.commands.run.await_args.kwargs["opts"]
  assert opts.timeout == timedelta(milliseconds=1)


@pytest.mark.asyncio
async def test_execute_rejects_negative_timeout(create_patch):
  """Negative command deadlines fail before reaching the SDK."""
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(ValueError, match="non-negative"):
    await env.execute("echo hi", timeout=-1)


@pytest.mark.asyncio
async def test_execute_rejects_missing_exit_code(create_patch, sandbox):
  """An incomplete SDK execution cannot be reported as success."""
  error = SimpleNamespace(name="ProtocolError", value="stream ended")
  sandbox.commands.run.return_value = _execution(exit_code=None, error=error)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(RuntimeError, match="ProtocolError: stream ended"):
    await env.execute("echo hi")


@pytest.mark.asyncio
async def test_read_file_resolves_relative_path(create_patch, sandbox):
  """Relative file reads use the configured working directory."""
  sandbox.files.read_bytes.return_value = b"data"
  env = OpenSandboxEnvironment(working_dir="/work")
  await env.initialize()

  content = await env.read_file(Path("nested/data.bin"))

  assert content == b"data"
  sandbox.files.read_bytes.assert_awaited_once_with("/work/nested/data.bin")


@pytest.mark.asyncio
async def test_read_file_preserves_absolute_path(create_patch, sandbox):
  """Absolute remote paths pass through unchanged."""
  sandbox.files.read_bytes.return_value = b"host"
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.read_file("/etc/hostname")

  sandbox.files.read_bytes.assert_awaited_once_with("/etc/hostname")


@pytest.mark.asyncio
async def test_read_file_maps_not_found(create_patch, sandbox):
  """OpenSandbox HTTP 404 errors become FileNotFoundError."""
  sandbox.files.read_bytes.side_effect = SandboxApiException(status_code=404)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError, match="missing.txt"):
    await env.read_file("missing.txt")


@pytest.mark.asyncio
async def test_read_file_preserves_other_api_errors(create_patch, sandbox):
  """Non-404 API failures retain their OpenSandbox error type."""
  error = SandboxApiException(status_code=403)
  sandbox.files.read_bytes.side_effect = error
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(SandboxApiException) as caught:
    await env.read_file("private.txt")

  assert caught.value is error


@pytest.mark.asyncio
async def test_write_file_resolves_path_and_uses_regular_file_mode(
    create_patch, sandbox
):
  """Writes create regular non-executable files below the workspace."""
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.write_file("nested/out.txt", "hello")

  sandbox.files.write_file.assert_awaited_once_with(
      "/workspace/nested/out.txt", "hello", mode=644
  )


@pytest.mark.asyncio
async def test_write_file_preserves_binary_content(create_patch, sandbox):
  """Binary file content reaches the SDK without text conversion."""
  env = OpenSandboxEnvironment()
  await env.initialize()
  content = b"\x00\xff"

  await env.write_file("/tmp/data.bin", content)

  sandbox.files.write_file.assert_awaited_once_with(
      "/tmp/data.bin", content, mode=644
  )


@pytest.mark.asyncio
async def test_attached_operations_do_not_renew_caller_ttl(
    connect_patch, sandbox
):
  """Using an attached sandbox does not modify its owner's expiration policy."""
  sandbox.files.read_bytes.return_value = b"data"
  env = OpenSandboxEnvironment(sandbox_id="existing-1")
  await env.initialize()

  await env.read_file("data.bin")

  sandbox.renew.assert_not_awaited()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": "python:3.11", "snapshot_id": "snap"}, "mutually"),
        ({"sandbox_id": "id", "image": "python:3.11"}, "attaching"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": 59}, "at least 60"),
        ({"ready_timeout": 0}, "ready_timeout"),
        ({"request_timeout": 0}, "request_timeout"),
        ({"working_dir": "relative"}, "absolute"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
  """Invalid lifecycle and path combinations fail during construction."""
  with pytest.raises(ValueError, match=message):
    OpenSandboxEnvironment(**kwargs)


def test_constructor_rejects_mixed_connection_configuration():
  """Complete and individual connection settings cannot be mixed."""
  config = ConnectionConfig()

  with pytest.raises(TypeError, match="cannot be combined"):
    OpenSandboxEnvironment(connection_config=config, domain="localhost:8080")


@pytest.mark.asyncio
async def test_initialize_explains_missing_optional_dependency():
  """Missing OpenSandbox installs produce an actionable extra hint."""
  env = OpenSandboxEnvironment()

  with mock.patch.dict("sys.modules", {"opensandbox": None}):
    with pytest.raises(ImportError, match=r"google-adk\[opensandbox\]"):
      await env.initialize()
