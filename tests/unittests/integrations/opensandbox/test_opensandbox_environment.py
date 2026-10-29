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

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from google.adk.integrations.opensandbox import OpenSandboxEnvironment
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import SandboxApiException
import pytest


def _make_sandbox() -> mock.MagicMock:
  sandbox = mock.MagicMock(name="Sandbox")
  sandbox.destroy = mock.AsyncMock()
  sandbox.close = mock.AsyncMock()
  sandbox.renew = mock.AsyncMock()
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
  with mock.patch(
      "opensandbox.Sandbox.create", new=mock.AsyncMock(return_value=sandbox)
  ) as create:
    yield create


@pytest.fixture(name="connect_patch")
def _connect_patch(sandbox: mock.MagicMock):
  with mock.patch(
      "opensandbox.Sandbox.connect", new=mock.AsyncMock(return_value=sandbox)
  ) as connect:
    yield connect


@pytest.mark.asyncio
async def test_initialize_creates_sandbox_with_configuration(
    create_patch: mock.AsyncMock, sandbox: mock.MagicMock
):
  config = ConnectionConfig(domain="sandbox.example:8080")
  env = OpenSandboxEnvironment(
      image="custom:latest",
      timeout=120,
      ready_timeout=12.5,
      env_vars={"A": "1"},
      metadata={"team": "adk"},
      connection_config=config,
  )

  await env.initialize()

  args, kwargs = create_patch.call_args
  assert args == ("custom:latest",)
  assert kwargs == {
      "snapshot_id": None,
      "timeout": timedelta(seconds=120),
      "ready_timeout": timedelta(seconds=12.5),
      "env": {"A": "1"},
      "metadata": {
          "framework": "google-adk",
          "integration": "google-adk-opensandbox",
          "team": "adk",
      },
      "connection_config": config,
      "network_policy": None,
      "secure_access": False,
  }
  directory = sandbox.files.create_directories.await_args.args[0][0]
  assert directory.path == "/workspace"
  assert directory.mode == 755
  assert env.working_dir == Path("/workspace")
  assert env.is_initialized is True


@pytest.mark.asyncio
async def test_initialize_passes_network_policy_and_secure_access(
    create_patch: mock.AsyncMock,
):
  policy = mock.MagicMock(name="NetworkPolicy")
  env = OpenSandboxEnvironment(network_policy=policy, secure_access=True)

  await env.initialize()

  assert create_patch.await_args.kwargs["network_policy"] is policy
  assert create_patch.await_args.kwargs["secure_access"] is True


@pytest.mark.asyncio
async def test_initialize_uses_snapshot_without_default_image(create_patch):
  env = OpenSandboxEnvironment(snapshot_id="snapshot-1")

  await env.initialize()

  assert create_patch.await_args.args == (None,)
  assert create_patch.await_args.kwargs["snapshot_id"] == "snapshot-1"


@pytest.mark.asyncio
async def test_initialize_attaches_to_existing_sandbox(connect_patch):
  config = ConnectionConfig(domain="sandbox.example:8080")
  env = OpenSandboxEnvironment(
      sandbox_id="existing-1", connection_config=config
  )

  await env.initialize()

  connect_patch.assert_awaited_once_with(
      "existing-1",
      connection_config=config,
      connect_timeout=timedelta(seconds=30),
  )


@pytest.mark.asyncio
async def test_initialize_defaults_connection_config_to_http_for_localhost(
    create_patch,
):
  env = OpenSandboxEnvironment()

  await env.initialize()

  connection_config = create_patch.await_args.kwargs["connection_config"]
  assert isinstance(connection_config, ConnectionConfig)
  assert connection_config.protocol == "http"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("domain", "expected_protocol"),
    [
        ("sandbox.example.com", "https"),
        ("http://10.0.0.5:8080", "http"),
        ("https://localhost:8443", "https"),
        ("[::1]:8080", "http"),
    ],
)
async def test_initialize_resolves_connection_config_protocol_from_domain(
    create_patch, monkeypatch, domain, expected_protocol
):
  monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", domain)
  env = OpenSandboxEnvironment()

  await env.initialize()

  connection_config = create_patch.await_args.kwargs["connection_config"]
  assert isinstance(connection_config, ConnectionConfig)
  assert connection_config.protocol == expected_protocol


@pytest.mark.asyncio
async def test_initialize_connect_defaults_protocol_to_http_for_localhost(
    connect_patch,
):
  env = OpenSandboxEnvironment(sandbox_id="existing-1")

  await env.initialize()

  connection_config = connect_patch.await_args.kwargs["connection_config"]
  assert isinstance(connection_config, ConnectionConfig)
  assert connection_config.protocol == "http"


@pytest.mark.asyncio
async def test_initialize_is_idempotent(create_patch):
  env = OpenSandboxEnvironment()

  await env.initialize()
  await env.initialize()

  create_patch.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_failure_cleans_up_owned_sandbox(
    create_patch, sandbox
):
  sandbox.files.create_directories.side_effect = RuntimeError("setup failed")
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="setup failed"):
    await env.initialize()

  sandbox.destroy.assert_awaited_once()
  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_initialize_failure_closes_attached_sandbox(
    connect_patch, sandbox
):
  sandbox.files.create_directories.side_effect = RuntimeError("setup failed")
  env = OpenSandboxEnvironment(sandbox_id="existing-1")

  with pytest.raises(RuntimeError, match="setup failed"):
    await env.initialize()

  sandbox.close.assert_awaited_once()
  sandbox.destroy.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_destroys_owned_sandbox_and_is_idempotent(
    create_patch, sandbox
):
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.close()
  await env.close()

  sandbox.destroy.assert_awaited_once()
  sandbox.close.assert_not_awaited()
  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_close_accepts_already_absent_owned_sandbox(
    create_patch, sandbox
):
  sandbox.destroy.side_effect = SandboxApiException(status_code=404)
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.close()

  assert env.is_initialized is False


@pytest.mark.asyncio
async def test_close_resets_state_even_if_cleanup_raises(create_patch, sandbox):
  sandbox.destroy.side_effect = RuntimeError("delete failed")
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(RuntimeError, match="delete failed"):
    await env.close()

  assert env.is_initialized is False
  with pytest.raises(RuntimeError, match="initialize"):
    _ = env.working_dir
  await env.close()
  sandbox.destroy.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_attached_sandbox_releases_only_local_resources(
    connect_patch, sandbox
):
  env = OpenSandboxEnvironment(sandbox_id="existing-1")
  await env.initialize()

  await env.close()

  sandbox.close.assert_awaited_once()
  sandbox.destroy.assert_not_awaited()


def test_working_dir_requires_initialize():
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    _ = env.working_dir


@pytest.mark.asyncio
async def test_execute_requires_initialize():
  env = OpenSandboxEnvironment()

  with pytest.raises(RuntimeError, match="initialize"):
    await env.execute("pwd")


@pytest.mark.asyncio
async def test_execute_maps_command_options_and_result(create_patch, sandbox):
  sandbox.commands.run.return_value = _execution(
      exit_code=7,
      stdout=("out-1\n",),
      stderr=("err\n",),
      results=("out-2",),
  )
  env = OpenSandboxEnvironment(env_vars={"A": "1"})
  await env.initialize()

  result = await env.execute("run-me", timeout=1.25)

  assert result.exit_code == 7
  assert result.stdout == "out-1\nout-2"
  assert result.stderr == "err\n"
  assert result.timed_out is False
  opts = sandbox.commands.run.await_args.kwargs["opts"]
  assert opts.working_directory == "/workspace"
  assert opts.timeout == timedelta(seconds=1.25)
  assert opts.envs == {"A": "1"}
  sandbox.renew.assert_awaited_once_with(timedelta(seconds=300))


@pytest.mark.asyncio
async def test_execute_joins_raw_streamed_chunks_safely(create_patch, sandbox):
  sandbox.commands.run.return_value = _execution(
      exit_code=0,
      stdout=("line 1\n", None, "partial ", "line 2\n\n"),
      stderr=("error 1: ", "details\n", None),
      results=(None, "res-1\n"),
  )
  env = OpenSandboxEnvironment()
  await env.initialize()

  result = await env.execute("stream")

  assert result.stdout == "line 1\npartial line 2\n\nres-1\n"
  assert result.stderr == "error 1: details\n"


@pytest.mark.asyncio
async def test_execute_renews_past_long_command_deadline(create_patch, sandbox):
  sandbox.commands.run.return_value = _execution()
  env = OpenSandboxEnvironment(timeout=60)
  await env.initialize()

  await env.execute("long-command", timeout=120)

  sandbox.renew.assert_awaited_once_with(timedelta(seconds=150))


@pytest.mark.asyncio
async def test_execute_marks_server_deadline_as_timed_out(
    create_patch, sandbox
):
  sandbox.commands.run.return_value = _execution(exit_code=-1)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with mock.patch(
      "google.adk.integrations.opensandbox._opensandbox_environment.monotonic",
      side_effect=[10.0, 11.1],
  ):
    result = await env.execute("sleep 30", timeout=1)

  assert result.timed_out is True


@pytest.mark.asyncio
async def test_execute_does_not_mislabel_early_sigkill(create_patch, sandbox):
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
async def test_execute_uses_minimum_one_millisecond_timeout(
    create_patch, sandbox
):
  sandbox.commands.run.return_value = _execution()
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.execute("true", timeout=0)

  opts = sandbox.commands.run.await_args.kwargs["opts"]
  assert opts.timeout == timedelta(milliseconds=1)


@pytest.mark.asyncio
async def test_execute_rejects_negative_timeout(create_patch):
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(ValueError, match="non-negative"):
    await env.execute("true", timeout=-1)


@pytest.mark.asyncio
async def test_execute_handles_missing_exit_code_without_raising(
    create_patch, sandbox
):
  sandbox.commands.run.return_value = _execution(
      exit_code=None,
      stdout=("partial output\n",),
      stderr=("captured error\n",),
      error=SimpleNamespace(name="OOMKilled", value="Process killed"),
  )
  env = OpenSandboxEnvironment()
  await env.initialize()

  result = await env.execute("broken")

  assert result.exit_code == -1
  assert result.stdout == "partial output\n"
  assert "captured error\nOOMKilled: Process killed" in result.stderr
  assert result.timed_out is False


@pytest.mark.asyncio
async def test_execute_handles_missing_exit_code_and_no_error_event(
    create_patch, sandbox
):
  sandbox.commands.run.return_value = _execution(
      exit_code=None,
      stdout=("some output",),
  )
  env = OpenSandboxEnvironment()
  await env.initialize()

  result = await env.execute("broken")

  assert result.exit_code == -1
  assert result.stdout == "some output"
  assert result.stderr == "command completed without an exit code"
  assert result.timed_out is False


@pytest.mark.asyncio
async def test_read_and_write_files(create_patch, sandbox):
  sandbox.files.read_bytes.return_value = b"\x00data"
  env = OpenSandboxEnvironment()
  await env.initialize()

  content = await env.read_file("nested/input.bin")
  await env.write_file(Path("/tmp/output.bin"), b"\xffdata")

  assert content == b"\x00data"
  sandbox.files.read_bytes.assert_awaited_once_with(
      "/workspace/nested/input.bin"
  )
  sandbox.files.write_file.assert_awaited_once_with(
      "/tmp/output.bin", b"\xffdata", mode=644
  )
  sandbox.renew.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_recreates_expired_owned_sandbox(create_patch, sandbox):
  fresh_sandbox = _make_sandbox()
  fresh_sandbox.commands.run.return_value = _execution(
      exit_code=0, stdout=("ok",)
  )
  create_patch.side_effect = [sandbox, fresh_sandbox]
  sandbox.renew.side_effect = SandboxApiException(status_code=404)

  env = OpenSandboxEnvironment()
  await env.initialize()

  result = await env.execute("echo ok")
  assert result.stdout == "ok"
  assert create_patch.await_count == 2
  sandbox.close.assert_awaited_once()
  fresh_sandbox.close.assert_not_awaited()
  fresh_sandbox.files.create_directories.assert_awaited_once()
  fresh_sandbox.renew.assert_not_awaited()
  assert env.working_dir == Path("/workspace")


@pytest.mark.asyncio
async def test_execute_recreates_expired_owned_sandbox_renews_to_timeout(
    create_patch, sandbox
):
  fresh_sandbox = _make_sandbox()
  fresh_sandbox.commands.run.return_value = _execution(
      exit_code=0, stdout=("ok",)
  )
  create_patch.side_effect = [sandbox, fresh_sandbox]
  sandbox.renew.side_effect = SandboxApiException(status_code=404)

  env = OpenSandboxEnvironment(timeout=60)
  await env.initialize()

  result = await env.execute("echo ok", timeout=120)
  assert result.stdout == "ok"
  assert create_patch.await_count == 2
  sandbox.close.assert_awaited_once()
  fresh_sandbox.close.assert_not_awaited()
  fresh_sandbox.renew.assert_awaited_once_with(timedelta(seconds=150))


@pytest.mark.asyncio
async def test_write_file_creates_parent_directories(create_patch, sandbox):
  env = OpenSandboxEnvironment()
  await env.initialize()

  await env.write_file("nested/sub/file.txt", "content")

  entries = sandbox.files.create_directories.await_args.args[0]
  assert [entry.path for entry in entries] == [
      "/workspace",
      "/workspace/nested",
      "/workspace/nested/sub",
  ]
  assert all(entry.mode == 755 for entry in entries)
  sandbox.files.write_file.assert_awaited_once_with(
      "/workspace/nested/sub/file.txt", "content", mode=644
  )


@pytest.mark.asyncio
async def test_write_file_root_does_not_create_parent_directory(
    create_patch, sandbox
):
  env = OpenSandboxEnvironment()
  await env.initialize()
  sandbox.files.create_directories.reset_mock()

  await env.write_file("/file.txt", "content")

  sandbox.files.create_directories.assert_not_awaited()
  sandbox.files.write_file.assert_awaited_once_with(
      "/file.txt", "content", mode=644
  )


@pytest.mark.asyncio
async def test_read_file_maps_not_found(create_patch, sandbox):
  sandbox.files.read_bytes.side_effect = SandboxApiException(status_code=404)
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError, match="missing.txt"):
    await env.read_file("missing.txt")


@pytest.mark.asyncio
async def test_read_file_preserves_other_api_errors(create_patch, sandbox):
  error = SandboxApiException(status_code=500)
  sandbox.files.read_bytes.side_effect = error
  env = OpenSandboxEnvironment()
  await env.initialize()

  with pytest.raises(SandboxApiException) as exc_info:
    await env.read_file("unavailable.txt")

  assert exc_info.value is error


@pytest.mark.asyncio
async def test_attached_operations_do_not_renew_caller_ttl(
    connect_patch, sandbox
):
  sandbox.commands.run.return_value = _execution()
  env = OpenSandboxEnvironment(sandbox_id="existing-1")
  await env.initialize()

  await env.execute("true")
  await env.write_file("output.txt", "done")

  sandbox.renew.assert_not_awaited()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": "python", "snapshot_id": "snapshot"}, "mutually"),
        ({"sandbox_id": "existing", "image": "python"}, "attaching"),
        (
            {"sandbox_id": "existing", "network_policy": mock.MagicMock()},
            "attaching",
        ),
        ({"sandbox_id": "existing", "secure_access": True}, "attaching"),
        ({"timeout": 59}, "at least 60"),
        ({"ready_timeout": 0}, "positive"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
  with pytest.raises(ValueError, match=message):
    OpenSandboxEnvironment(**kwargs)


@pytest.mark.asyncio
async def test_initialize_explains_missing_optional_dependency():
  env = OpenSandboxEnvironment()

  with mock.patch.dict("sys.modules", {"opensandbox": None}):
    with pytest.raises(ImportError, match=r"google-adk\[opensandbox\]"):
      await env.initialize()
