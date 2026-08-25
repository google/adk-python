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

from __future__ import annotations

import asyncio
import functools
import os
import signal
import subprocess
import sys
from unittest.mock import AsyncMock
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import anyio
from fastapi.testclient import TestClient
from google.adk.cli import dev_server
from google.adk.cli.fast_api import get_fast_api_app
import pytest

_LIFECYCLE_TEST_TIMEOUT_SECONDS = 10

# Two processes that ignore SIGTERM. The grandchild announces itself on an
# inherited pipe, so EOF on that pipe means the whole group is gone.
_SIGTERM_PROOF_GRANDCHILD = (
    "import os, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "os.write(int(sys.argv[1]), b'up')\n"
    "time.sleep(30)\n"
)
_SIGTERM_PROOF_PARENT = (
    "import os, signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "fd = int(sys.argv[1])\n"
    "subprocess.Popen([sys.executable, '-c', sys.argv[2], str(fd)],"
    " pass_fds=(fd,))\n"
    "os.close(fd)\n"
    "time.sleep(30)\n"
)


def _bounded(test_fn):
  """Fails a deadlocked lifecycle test instead of hanging the suite.

  asyncio.wait_for is not enough: a task parked inside a shielded cancel scope
  never sees the cancellation, and wait_for then waits on it forever. This
  abandons the task instead of awaiting it.
  """

  @functools.wraps(test_fn)
  async def wrapper(*args, **kwargs):
    task = asyncio.ensure_future(test_fn(*args, **kwargs))
    done, _ = await asyncio.wait(
        {task}, timeout=_LIFECYCLE_TEST_TIMEOUT_SECONDS
    )
    if not done:
      task.cancel()
      pytest.fail(
          f"{test_fn.__name__} did not finish within"
          f" {_LIFECYCLE_TEST_TIMEOUT_SECONDS}s"
      )
    await task

  return wrapper


async def _read_pipe(read_fd: int, size: int, timeout: float) -> bytes | None:
  """Reads from a non-blocking pipe; b"" means every writer is gone."""
  loop = asyncio.get_running_loop()
  deadline = loop.time() + timeout
  while loop.time() < deadline:
    try:
      return os.read(read_fd, size)
    except BlockingIOError:
      await asyncio.sleep(0.01)
  return None


@pytest.fixture
def test_client(tmp_path):
  """Client with a temporary agents directory."""
  app = get_fast_api_app(
      agents_dir=str(tmp_path),
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )
  return TestClient(app)


def test_list_tests_empty(test_client):
  response = test_client.get("/dev/apps/test_app/tests")
  assert response.status_code == 200
  assert response.json() == []


def test_create_test(test_client, tmp_path):
  # Create agent dir so it exists
  agent_dir = tmp_path / "test_app"
  agent_dir.mkdir()

  payload = {"session_data": {"events": []}}

  response = test_client.put(
      "/dev/apps/test_app/tests/my_test.json", json=payload
  )
  assert response.status_code == 200
  assert response.json() == {"status": "success", "file": "my_test.json"}

  # Verify file exists
  assert (agent_dir / "tests" / "my_test.json").exists()


def test_create_test_preserves_non_ascii(test_client, tmp_path):
  """Saved tests keep non-ASCII event text readable, not \\uXXXX escapes."""
  agent_dir = tmp_path / "test_app"
  agent_dir.mkdir()

  payload = {
      "session_data": {
          "events": [{
              "author": "user",
              "content": {"parts": [{"text": "日本語の質問"}]},
          }]
      }
  }

  response = test_client.put(
      "/dev/apps/test_app/tests/unicode.json", json=payload
  )
  assert response.status_code == 200

  saved = (agent_dir / "tests" / "unicode.json").read_text(encoding="utf-8")
  assert "日本語の質問" in saved
  assert "\\u" not in saved


def test_list_tests_not_empty(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  (tests_dir / "test1.json").write_text("{}")
  (tests_dir / "test2.json").write_text("{}")

  response = test_client.get("/dev/apps/test_app/tests")
  assert response.status_code == 200
  assert response.json() == ["test1.json", "test2.json"]


def test_delete_test(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  test_file = tests_dir / "test1.json"
  test_file.write_text("{}")

  response = test_client.delete("/dev/apps/test_app/tests/test1.json")
  assert response.status_code == 200
  assert response.json() == {"status": "success"}
  assert not test_file.exists()


def test_get_test_content(test_client, tmp_path):
  agent_dir = tmp_path / "test_app"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  test_file = tests_dir / "test_get.json"
  test_file.write_text('{"foo": "bar"}')

  response = test_client.get("/dev/apps/test_app/tests/test_get.json")
  assert response.status_code == 200
  assert response.json() == {"foo": "bar"}


def test_get_test_content_not_found(test_client):
  response = test_client.get("/dev/apps/test_app/tests/non_existent.json")
  assert response.status_code == 404


def test_rebuild_tests(test_client):
  with patch("google.adk.cli.dev_server.asyncio.to_thread") as mock_to_thread:
    mock_to_thread.return_value = None
    response = test_client.post("/dev/apps/test_app/tests/rebuild", json={})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_to_thread.assert_called_once()


def test_rebuild_single_test(test_client):
  with patch("google.adk.cli.dev_server.asyncio.to_thread") as mock_to_thread:
    mock_to_thread.return_value = None
    response = test_client.post(
        "/dev/apps/test_app/tests/rebuild?test_name=my_test.json", json={}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_to_thread.assert_called_once()
    args, kwargs = mock_to_thread.call_args
    test_dir, test_name = os.path.split(args[1])
    assert (os.path.basename(test_dir), test_name) == ("tests", "my_test.json")


def test_run_tests(test_client):
  mock_process = MagicMock()
  mock_process.returncode = 0
  mock_process.stdout.read = AsyncMock(
      side_effect=[b"line1\n", b"line2\n", b""]
  )
  mock_process.wait = AsyncMock(return_value=0)

  with patch(
      "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
      new_callable=AsyncMock,
  ) as mock_create_subprocess:
    mock_create_subprocess.return_value = mock_process

    response = test_client.post("/dev/apps/test_app/tests/run", json={})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    # Read stream
    content = response.content
    assert b"line1\n" in content
    assert b"line2\n" in content


@pytest.mark.asyncio
@_bounded
async def test_stream_test_output_starts_new_session_on_posix(monkeypatch):
  monkeypatch.setattr(dev_server, "_IS_WINDOWS", False)
  mock_process = MagicMock()
  mock_process.returncode = 0
  mock_process.stdout.read = AsyncMock(return_value=b"")
  mock_process.wait = AsyncMock(return_value=0)
  mock_create_subprocess = AsyncMock(return_value=mock_process)

  with (
      patch(
          "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
          new=mock_create_subprocess,
      ),
      patch(
          "google.adk.cli.dev_server._terminate_process_tree",
          new=AsyncMock(),
      ),
  ):
    async for _ in dev_server._stream_test_output(
        agent_dir="agent", test_name=None
    ):
      pass

  # Signalling the group only reaches descendants if the child leads its own.
  assert mock_create_subprocess.await_args.kwargs["start_new_session"] is True


@pytest.mark.asyncio
@_bounded
async def test_stream_test_output_shields_cleanup_from_cancel_scope():
  read_started = anyio.Event()
  cleanup_finished = anyio.Event()
  mock_process = MagicMock()
  mock_process.returncode = None

  async def read_output(_size):
    read_started.set()
    await anyio.sleep_forever()

  async def cleanup(_process):
    await anyio.sleep(0)
    cleanup_finished.set()

  mock_process.stdout.read = AsyncMock(side_effect=read_output)

  with (
      patch(
          "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
          new=AsyncMock(return_value=mock_process),
      ),
      patch(
          "google.adk.cli.dev_server._terminate_process_tree",
          new=AsyncMock(side_effect=cleanup),
      ),
  ):

    async def consume_output():
      async for _ in dev_server._stream_test_output(
          agent_dir="agent", test_name=None
      ):
        pass

    async with anyio.create_task_group() as task_group:
      task_group.start_soon(consume_output)
      await read_started.wait()
      task_group.cancel_scope.cancel()

  assert cleanup_finished.is_set()


@pytest.mark.asyncio
@_bounded
async def test_stream_test_output_cleans_up_after_partial_output():
  mock_process = MagicMock()
  mock_process.returncode = None
  mock_process.stdout.read = AsyncMock(return_value=b"partial output")
  mock_cleanup = AsyncMock()

  with (
      patch(
          "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
          new=AsyncMock(return_value=mock_process),
      ),
      patch(
          "google.adk.cli.dev_server._terminate_process_tree",
          new=mock_cleanup,
      ),
  ):
    output = dev_server._stream_test_output(agent_dir="agent", test_name=None)
    assert await anext(output) == b"partial output"
    await output.aclose()

  mock_cleanup.assert_awaited_once_with(mock_process)


@pytest.mark.asyncio
@_bounded
async def test_stream_test_output_shields_cleanup_during_process_wait():
  wait_started = anyio.Event()
  cleanup_finished = anyio.Event()
  mock_process = MagicMock()
  mock_process.returncode = None
  mock_process.stdout.read = AsyncMock(return_value=b"")

  async def wait_for_exit():
    wait_started.set()
    await anyio.sleep_forever()

  async def cleanup(_process):
    await anyio.sleep(0)
    cleanup_finished.set()

  mock_process.wait = AsyncMock(side_effect=wait_for_exit)

  with (
      patch(
          "google.adk.cli.dev_server.asyncio.create_subprocess_exec",
          new=AsyncMock(return_value=mock_process),
      ),
      patch(
          "google.adk.cli.dev_server._terminate_process_tree",
          new=AsyncMock(side_effect=cleanup),
      ),
  ):

    async def consume_output():
      async for _ in dev_server._stream_test_output(
          agent_dir="agent", test_name=None
      ):
        pass

    async with anyio.create_task_group() as task_group:
      task_group.start_soon(consume_output)
      await wait_started.wait()
      task_group.cancel_scope.cancel()

  assert cleanup_finished.is_set()


@pytest.mark.asyncio
@_bounded
async def test_terminate_process_tree_stops_after_graceful_exit():
  mock_process = MagicMock()
  mock_process.returncode = None
  mock_process.wait = AsyncMock(return_value=0)
  mock_signal = AsyncMock()

  with patch("google.adk.cli.dev_server._signal_process_tree", new=mock_signal):
    await dev_server._terminate_process_tree(mock_process)

  mock_signal.assert_awaited_once_with(mock_process, force=False)
  mock_process.wait.assert_awaited_once()


@pytest.mark.asyncio
@_bounded
async def test_terminate_process_tree_escalates_after_timeout():
  mock_process = MagicMock()
  mock_process.returncode = None
  mock_process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
  mock_signal = AsyncMock()

  with patch("google.adk.cli.dev_server._signal_process_tree", new=mock_signal):
    await dev_server._terminate_process_tree(mock_process)

  assert mock_signal.await_args_list == [
      call(mock_process, force=False),
      call(mock_process, force=True),
  ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.asyncio
@_bounded
async def test_terminate_process_tree_kills_real_grandchild():
  read_fd, write_fd = os.pipe()
  os.set_blocking(read_fd, False)
  process = await asyncio.create_subprocess_exec(
      sys.executable,
      "-c",
      _SIGTERM_PROOF_PARENT,
      str(write_fd),
      _SIGTERM_PROOF_GRANDCHILD,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      pass_fds=(write_fd,),
      start_new_session=True,
  )
  os.close(write_fd)
  try:
    assert await _read_pipe(read_fd, 2, 5) == b"up"

    await dev_server._terminate_process_tree(process)

    # Both processes ignore SIGTERM, so only the forced escalation can land.
    assert process.returncode == -signal.SIGKILL
    # The grandchild is the last holder of the write end; EOF means it is gone.
    assert await _read_pipe(read_fd, 1, 3) == b""
  finally:
    os.close(read_fd)
    if process.returncode is None:
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      await process.wait()


@pytest.mark.asyncio
@_bounded
async def test_signal_process_tree_falls_back_to_direct_child(monkeypatch):
  mock_process = MagicMock()
  mock_process.pid = 123
  mock_process.returncode = None
  monkeypatch.setattr(dev_server, "_IS_WINDOWS", False)
  monkeypatch.setattr(
      dev_server.os,
      "killpg",
      MagicMock(side_effect=ProcessLookupError),
      raising=False,
  )

  await dev_server._signal_process_tree(mock_process, force=True)

  mock_process.kill.assert_called_once_with()


@pytest.mark.asyncio
@_bounded
async def test_signal_process_tree_targets_posix_process_group(monkeypatch):
  mock_process = MagicMock()
  mock_process.pid = 123
  mock_killpg = MagicMock()
  monkeypatch.setattr(dev_server, "_IS_WINDOWS", False)
  monkeypatch.setattr(dev_server.os, "killpg", mock_killpg, raising=False)

  await dev_server._signal_process_tree(mock_process, force=False)

  mock_killpg.assert_called_once_with(123, dev_server.signal.SIGTERM)


@pytest.mark.asyncio
@_bounded
async def test_signal_process_tree_targets_windows_descendants(monkeypatch):
  mock_process = MagicMock()
  mock_process.pid = 123
  mock_terminator = MagicMock()
  mock_terminator.wait = AsyncMock(return_value=0)
  mock_create_subprocess = AsyncMock(return_value=mock_terminator)
  monkeypatch.setattr(dev_server, "_IS_WINDOWS", True)
  monkeypatch.setattr(
      dev_server.asyncio,
      "create_subprocess_exec",
      mock_create_subprocess,
  )

  await dev_server._signal_process_tree(mock_process, force=True)

  mock_create_subprocess.assert_awaited_once_with(
      "taskkill",
      "/PID",
      "123",
      "/T",
      "/F",
      stdout=asyncio.subprocess.DEVNULL,
      stderr=asyncio.subprocess.DEVNULL,
  )
