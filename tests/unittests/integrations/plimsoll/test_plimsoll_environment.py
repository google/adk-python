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

"""Tests for PlimsollEnvironment, with the Plimsoll client mocked."""

from __future__ import annotations

import asyncio
from pathlib import Path
from pathlib import PureWindowsPath
import threading
from types import SimpleNamespace
from unittest import mock

from google.adk.integrations.plimsoll import PlimsollEnvironment
from google.adk.tools.environment import EnvironmentToolset
import plimsoll_client
import pytest

_ROOT = "/work"


def _info(*, sessions: bool = True, max_timeout_ms: int = 300_000):
  return SimpleNamespace(
      provider="docker",
      supports_sessions=sessions,
      session_environment=SimpleNamespace(max_timeout_ms=max_timeout_ms),
  )


def _outcome(
    *,
    exit_code: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    timed_out: bool = False,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
):
  return SimpleNamespace(
      exit_code=exit_code,
      stdout_text=stdout.decode(),
      stderr_text=stderr.decode(),
      timed_out=timed_out,
      stdout_truncated=stdout_truncated,
      stderr_truncated=stderr_truncated,
      result=SimpleNamespace(
          isolation="kernel", record=SimpleNamespace(sha256="ab" * 32)
      ),
  )


@pytest.fixture(name="session")
def _session() -> mock.MagicMock:
  session = mock.MagicMock(name="Session")
  session.isolation = "kernel"
  return session


@pytest.fixture(name="client")
def _client(session: mock.MagicMock) -> mock.MagicMock:
  client = mock.MagicMock(name="Client")
  client.describe.return_value = _info()
  client.open_session.return_value = session
  return client


@pytest.fixture(name="workspace")
def _workspace() -> mock.MagicMock:
  workspace = mock.MagicMock(name="Workspace")
  workspace.root = _ROOT
  workspace.run.return_value = _outcome(stdout=b"42\n")
  workspace.read_whole.return_value = b"data"
  return workspace


@pytest.fixture(name="workspace_class")
def _workspace_class(workspace: mock.MagicMock):
  with mock.patch.object(
      plimsoll_client, "Workspace", return_value=workspace
  ) as workspace_class:
    yield workspace_class


@pytest.fixture(name="env")
def _env(client: mock.MagicMock, workspace_class: mock.MagicMock):
  del workspace_class
  return PlimsollEnvironment(client=client)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"minimum_isolation": "none"}, "minimum_isolation"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": True}, "timeout"),
        ({"timeout": float("nan")}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
        ({"env_vars": {"A\n": "1"}}, "shell variable name"),
        ({"env_vars": {"A-B": "1"}}, "shell variable name"),
        ({"env_vars": {"A": "x\x00y"}}, "must be text"),
        ({"client": mock.MagicMock(), "url": "http://h"}, "client"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
  with pytest.raises(ValueError, match=message):
    PlimsollEnvironment(**kwargs)


def test_constructor_contacts_nothing(client: mock.MagicMock):
  env = PlimsollEnvironment(client=client)

  client.describe.assert_not_called()
  client.open_session.assert_not_called()
  assert env.working_dir is None
  assert env.isolation is None
  assert env.is_initialized is False


async def test_initialize_opens_session_at_the_floor(
    env: PlimsollEnvironment,
    client: mock.MagicMock,
    session: mock.MagicMock,
    workspace_class: mock.MagicMock,
):
  await env.initialize()

  client.open_session.assert_called_once_with(
      minimum_isolation="kernel", lifetime=None, idle_timeout=None, owner=None
  )
  workspace_class.assert_called_once_with(
      session, timeout=120.0, max_timeout=300.0, confined=False
  )
  assert env.working_dir == Path(_ROOT)
  assert env.isolation == "kernel"
  assert env.is_initialized is True


async def test_initialize_is_idempotent(
    env: PlimsollEnvironment, client: mock.MagicMock
):
  await env.initialize()
  await env.initialize()

  client.open_session.assert_called_once()


async def test_initialize_passes_session_options(
    client: mock.MagicMock, workspace_class: mock.MagicMock
):
  del workspace_class
  env = PlimsollEnvironment(
      client=client,
      minimum_isolation="vm",
      lifetime=600,
      idle_timeout=60,
      owner="user-1",
  )
  client.open_session.return_value.isolation = "vm"

  await env.initialize()

  client.open_session.assert_called_once_with(
      minimum_isolation="vm", lifetime=600, idle_timeout=60, owner="user-1"
  )


async def test_initialize_refuses_a_daemon_without_sessions(
    env: PlimsollEnvironment, client: mock.MagicMock
):
  client.describe.return_value = _info(sessions=False)

  with pytest.raises(plimsoll_client.UnsupportedError) as raised:
    await env.initialize()

  assert raised.value.not_dispatched == "unsupported"
  client.open_session.assert_not_called()
  assert env.is_initialized is False


async def test_initialize_refuses_and_closes_a_session_below_the_floor(
    env: PlimsollEnvironment,
    session: mock.MagicMock,
    workspace_class: mock.MagicMock,
):
  session.isolation = "container"

  with pytest.raises(plimsoll_client.InsufficientIsolationError) as raised:
    await env.initialize()

  assert raised.value.not_dispatched == "isolation"
  session.close.assert_called_once()
  workspace_class.assert_not_called()
  assert env.working_dir is None


async def test_initialize_closes_the_session_when_the_workspace_fails(
    env: PlimsollEnvironment,
    session: mock.MagicMock,
    workspace: mock.MagicMock,
):
  type(workspace).root = mock.PropertyMock(side_effect=OSError("gone"))

  with pytest.raises(OSError, match="gone"):
    await env.initialize()

  session.close.assert_called_once()
  assert env.is_initialized is False


async def test_initialize_builds_a_client_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    client: mock.MagicMock,
    workspace_class: mock.MagicMock,
):
  del workspace_class
  monkeypatch.setenv("PLIMSOLL_URL", "http://127.0.0.1:9999")
  monkeypatch.setenv("PLIMSOLL_TOKEN", "t" * 32)
  with mock.patch.object(
      plimsoll_client, "Client", return_value=client
  ) as client_class:
    await PlimsollEnvironment().initialize()

  client_class.assert_called_once_with("http://127.0.0.1:9999", token="t" * 32)


async def test_cancelled_initialize_closes_the_session_it_opened(
    env: PlimsollEnvironment, client: mock.MagicMock, session: mock.MagicMock
):
  opened = threading.Event()
  release = threading.Event()
  closed = threading.Event()
  session.close.side_effect = lambda: closed.set()

  def slow_open(**kwargs):
    del kwargs
    opened.set()
    release.wait(5)
    return session

  client.open_session.side_effect = slow_open
  task = asyncio.create_task(env.initialize())
  await asyncio.to_thread(opened.wait, 5)
  task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await task
  release.set()

  assert await asyncio.to_thread(closed.wait, 5)
  assert env.is_initialized is False

  # Nothing was open when the open was cancelled, so the next one may open.
  client.open_session.side_effect = None
  await env.initialize()
  assert client.open_session.call_count == 2


async def test_methods_before_initialize_raise(env: PlimsollEnvironment):
  with pytest.raises(RuntimeError, match="initialize"):
    await env.execute("true")
  with pytest.raises(RuntimeError, match="initialize"):
    await env.read_file(Path("a.txt"))
  with pytest.raises(RuntimeError, match="initialize"):
    await env.write_file(Path("a.txt"), "x")


async def test_execute_runs_the_command_in_a_shell(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  env = PlimsollEnvironment(
      client=env._client, env_vars={"GREETING": "hi"}, timeout=30
  )
  await env.initialize()

  result = await env.execute("echo 42", timeout=5)

  args, kwargs = workspace.run.call_args
  assert args == (["sh", "-c", "echo 42"],)
  assert kwargs["env"] == {"GREETING": "hi"}
  assert kwargs["timeout"] == 5.0
  assert result.exit_code == 0
  assert result.stdout == "42\n"
  assert result.timed_out is False


async def test_execute_without_timeout_uses_the_daemon_ceiling(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  await env.initialize()

  await env.execute("true")

  assert workspace.run.call_args.kwargs["timeout"] == 300.0


async def test_execute_without_timeout_or_ceiling_uses_the_timeout(
    client: mock.MagicMock,
    workspace: mock.MagicMock,
    workspace_class: mock.MagicMock,
):
  del workspace_class
  client.describe.return_value = _info(max_timeout_ms=0)
  env = PlimsollEnvironment(client=client, timeout=45)
  await env.initialize()

  await env.execute("true")

  assert workspace.run.call_args.kwargs["timeout"] == 45.0


async def test_execute_maps_failure_and_timeout(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  workspace.run.return_value = _outcome(
      exit_code=124, stderr=b"slow", timed_out=True
  )
  await env.initialize()

  result = await env.execute("sleep 99", timeout=1)

  assert result.exit_code == 124
  assert result.stderr == "slow"
  assert result.timed_out is True


async def test_execute_with_a_zero_timeout_sends_the_shortest_budget(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  # As OpenSandboxEnvironment does: zero is a valid budget, sent as the
  # shortest one the daemon takes (1 ms), not a result made up here.
  await env.initialize()

  result = await env.execute("true", timeout=0)

  assert workspace.run.call_args.kwargs["timeout"] == 0.001
  assert result.exit_code == 0


@pytest.mark.parametrize(
    "timeout", [-5, -0.5, float("nan"), float("inf"), float("-inf"), True, "5"]
)
async def test_execute_rejects_an_invalid_timeout(
    env: PlimsollEnvironment, workspace: mock.MagicMock, timeout
):
  await env.initialize()

  with pytest.raises(ValueError, match="timeout"):
    await env.execute("true", timeout=timeout)

  workspace.run.assert_not_called()


async def test_execute_reports_truncated_streams(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  workspace.run.return_value = _outcome(
      stderr=b"warn", stdout_truncated=True, stderr_truncated=True
  )
  await env.initialize()

  result = await env.execute("yes")

  assert result.stderr == (
      "warn\n"
      "plimsoll: the daemon cut stdout at its output limit\n"
      "plimsoll: the daemon cut stderr at its output limit\n"
  )


async def test_unusable_answer_raises_without_sandbox_text(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  workspace.run.side_effect = plimsoll_client.WorkspaceCallError(
      "guest text", SimpleNamespace(outcome="runner_failed")
  )
  await env.initialize()

  with pytest.raises(RuntimeError) as raised:
    await env.execute("true")

  assert "runner_failed" in str(raised.value)
  assert "may have run" in str(raised.value)
  assert "guest text" not in str(raised.value)


async def test_refusals_pass_through_with_their_mark(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  workspace.run.side_effect = plimsoll_client.SessionEndedError(
      "ended", reason="expired", not_dispatched="request"
  )
  await env.initialize()

  with pytest.raises(plimsoll_client.SessionEndedError) as raised:
    await env.execute("true")

  assert raised.value.not_dispatched == "request"


async def test_cancelled_execute_cuts_the_call(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  started = threading.Event()
  handles = []

  def slow_run(argv, *, env, timeout, cancel):
    del argv, env, timeout
    handles.append(cancel)
    started.set()
    for _ in range(500):
      if cancel.cancelled:
        break
      threading.Event().wait(0.01)
    return _outcome()

  workspace.run.side_effect = slow_run
  await env.initialize()
  task = asyncio.create_task(env.execute("sleep 9"))
  await asyncio.to_thread(started.wait, 5)
  task.cancel()

  with pytest.raises(asyncio.CancelledError):
    await task
  assert handles[0].cancelled is True


async def test_read_file_returns_bytes_for_a_posix_path(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  await env.initialize()

  data = await env.read_file(PureWindowsPath("dir\\a.txt"))

  assert data == b"data"
  assert workspace.read_whole.call_args.args == ("dir/a.txt",)


async def test_read_file_missing_raises_file_not_found(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  workspace.read_whole.side_effect = FileNotFoundError("a.txt")
  await env.initialize()

  with pytest.raises(FileNotFoundError):
    await env.read_file(Path("a.txt"))


@pytest.mark.parametrize(
    "content, sent",
    [("héllo\n", "héllo\n".encode()), (b"\x00\xff", b"\x00\xff")],
)
async def test_write_file_sends_bytes(
    env: PlimsollEnvironment, workspace: mock.MagicMock, content, sent
):
  await env.initialize()

  await env.write_file(Path("sub/dir/a.bin"), content)

  assert workspace.write.call_args.args == ("sub/dir/a.bin", sent)


async def test_write_file_rejects_other_content(env: PlimsollEnvironment):
  await env.initialize()

  with pytest.raises(TypeError, match="str or bytes"):
    await env.write_file(Path("a"), 3)


async def test_close_closes_the_session_and_is_idempotent(
    env: PlimsollEnvironment, client: mock.MagicMock, session: mock.MagicMock
):
  await env.initialize()

  await env.close()
  await env.close()

  session.close.assert_called_once()
  assert env.is_initialized is False
  assert env.working_dir is None
  with pytest.raises(RuntimeError, match="was closed"):
    await env.execute("true")
  with pytest.raises(RuntimeError, match="was closed"):
    await env.initialize()
  client.open_session.assert_called_once()


async def test_reopen_after_close_opens_a_new_session(
    client: mock.MagicMock, workspace_class: mock.MagicMock
):
  del workspace_class
  env = PlimsollEnvironment(client=client, reopen_after_close=True)
  await env.initialize()
  await env.close()

  await env.initialize()

  assert client.open_session.call_count == 2
  assert env.is_initialized is True


async def test_close_before_any_open_is_harmless(
    env: PlimsollEnvironment, client: mock.MagicMock
):
  await env.close()

  await env.initialize()

  client.open_session.assert_called_once()


async def test_a_failed_close_still_refuses_a_reopen(
    env: PlimsollEnvironment, client: mock.MagicMock, session: mock.MagicMock
):
  session.close.side_effect = OSError("daemon gone")
  await env.initialize()

  with pytest.raises(OSError, match="daemon gone"):
    await env.close()
  with pytest.raises(RuntimeError, match="was closed"):
    await env.initialize()

  client.open_session.assert_called_once()


async def test_toolset_drives_the_environment(
    env: PlimsollEnvironment, workspace: mock.MagicMock
):
  toolset = EnvironmentToolset(environment=env)

  tools = await toolset.get_tools()
  await env.initialize()
  await toolset.close()

  assert {"Execute", "ReadFile", "WriteFile", "EditFile"} <= {
      tool.name for tool in tools
  }
  assert env.is_initialized is False
  del workspace
