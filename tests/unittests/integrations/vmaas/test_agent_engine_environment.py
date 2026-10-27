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

"""Tests for AgentEngineEnvironment.

Verifies the environment's sandbox lifecycle, command execution, and file
access against a fake Agent Platform client whose shell runs commands in a
local bash.
"""

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.adk.integrations.vmaas import _agent_engine_environment
from google.adk.integrations.vmaas import AgentEngineEnvironment
from google.genai.errors import APIError
from google.genai.errors import ClientError
from google.genai.errors import ServerError
import pytest

_ENGINE = "projects/p/locations/us-central1/reasoningEngines/1"
_TEMPLATE = f"{_ENGINE}/sandboxEnvironmentTemplates/7"


class _FakeSandboxes:
  """In-memory sandboxes API whose shell runs commands in a local bash."""

  def __init__(self):
    self.templates = MagicMock()
    self.templates.list.return_value = []
    self.templates.create.return_value = SimpleNamespace(
        response=SimpleNamespace(name=_TEMPLATE, state="ACTIVE"), error=None
    )
    self.states = {}
    self.create_calls = []
    self.commands = []
    self.pause_calls = []
    self.resume_calls = []

  def create(self, *, name, config, poll_interval_seconds):
    self.create_calls.append(config)
    sandbox_name = f"{name}/sandboxEnvironments/{len(self.create_calls)}"
    self.states[sandbox_name] = "STATE_RUNNING"
    return SimpleNamespace(
        response=SimpleNamespace(name=sandbox_name, state="STATE_RUNNING"),
        error=None,
    )

  def get(self, *, name):
    if name not in self.states:
      raise ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND"}})
    return SimpleNamespace(name=name, state=self.states[name])

  def pause(self, *, name, poll_interval_seconds):
    self.pause_calls.append(name)
    self.states[name] = "STATE_PAUSED"
    return SimpleNamespace(error=None)

  def resume(self, *, name, poll_interval_seconds):
    self.resume_calls.append(name)
    self.states[name] = "STATE_RUNNING"
    return SimpleNamespace(error=None)

  def execute_bash(self, *, name, command, timeout):
    self.commands.append({"command": command, "timeout": timeout})
    if self.states[name] != "STATE_RUNNING":
      raise ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
    process = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )
    return {
        "stdout": process.stdout,
        "stderr": process.stderr,
        "returncode": process.returncode,
        "duration_ms": 1,
    }


def _make_client():
  client = MagicMock()
  client.runtimes.create.return_value = SimpleNamespace(
      api_resource=SimpleNamespace(name=_ENGINE)
  )
  client.sandboxes = _FakeSandboxes()
  return client


async def _make_environment(client=None, **kwargs):
  """Returns an initialized environment on a fake client."""
  environment = AgentEngineEnvironment(
      vertexai_client=client or _make_client(), **kwargs
  )
  await environment.initialize()
  return environment


async def test_initialize_creates_sandbox_from_shell_template():
  """Initializing creates one sandbox from the agent engine's shell template."""
  client = _make_client()

  environment = await _make_environment(client)

  [config] = client.sandboxes.create_calls
  assert config["sandbox_environment_template"] == _TEMPLATE
  assert environment.sandbox_name == f"{_ENGINE}/sandboxEnvironments/1"
  assert environment.is_initialized


async def test_initialize_uses_given_template():
  """A given template is used without resolving a shell template."""
  client = _make_client()
  template = f"{_ENGINE}/sandboxEnvironmentTemplates/given"

  await _make_environment(client, sandbox_template_name=template)

  [config] = client.sandboxes.create_calls
  assert config["sandbox_environment_template"] == template
  client.sandboxes.templates.create.assert_not_called()


async def test_initialize_resumes_given_sandbox_without_creating_one():
  """A given paused sandbox is resumed instead of creating a new one."""
  client = _make_client()
  sandbox = f"{_ENGINE}/sandboxEnvironments/existing"
  client.sandboxes.states[sandbox] = "STATE_PAUSED"

  environment = await _make_environment(client, sandbox_name=sandbox)

  assert environment.sandbox_name == sandbox
  assert client.sandboxes.resume_calls == [sandbox]
  assert not client.sandboxes.create_calls


async def test_concurrent_initialize_creates_one_sandbox():
  """Initializing concurrently creates one sandbox."""
  client = _make_client()
  environment = AgentEngineEnvironment(vertexai_client=client)

  await asyncio.gather(environment.initialize(), environment.initialize())

  assert len(client.sandboxes.create_calls) == 1


async def test_initialize_is_idempotent():
  """Initializing twice creates one sandbox."""
  client = _make_client()
  environment = await _make_environment(client)

  await environment.initialize()

  assert len(client.sandboxes.create_calls) == 1


async def test_close_pauses_sandbox_and_initialize_resumes_it():
  """Closing pauses the sandbox, and initializing again resumes the same one."""
  client = _make_client()
  environment = await _make_environment(client)
  sandbox = environment.sandbox_name

  await environment.close()
  paused_state = client.sandboxes.states[sandbox]
  await environment.initialize()

  assert paused_state == "STATE_PAUSED"
  assert client.sandboxes.resume_calls == [sandbox]
  assert len(client.sandboxes.create_calls) == 1


async def test_initialize_replaces_own_sandbox_that_expired():
  """A sandbox this environment created that has since expired is replaced."""
  client = _make_client()
  environment = await _make_environment(client)
  expired = environment.sandbox_name
  await environment.close()
  del client.sandboxes.states[expired]

  await environment.initialize()

  assert environment.sandbox_name not in (None, expired)
  assert len(client.sandboxes.create_calls) == 2


async def test_initialize_with_missing_given_sandbox_raises():
  """A given sandbox that does not exist raises instead of being replaced."""
  client = _make_client()
  environment = AgentEngineEnvironment(
      vertexai_client=client,
      sandbox_name=f"{_ENGINE}/sandboxEnvironments/missing",
  )

  with pytest.raises(RuntimeError, match="does not exist"):
    await environment.initialize()

  assert not client.sandboxes.create_calls


async def test_close_before_initialize_does_nothing():
  """Closing an environment that was never initialized pauses nothing."""
  client = _make_client()

  await AgentEngineEnvironment(vertexai_client=client).close()

  assert not client.sandboxes.pause_calls


async def test_execute_reports_output_and_exit_code():
  """A command's output and exit code are reported, not raised."""
  environment = await _make_environment()

  result = await environment.execute("echo out; echo err >&2; exit 3")

  assert (result.stdout, result.stderr, result.exit_code) == (
      "out\n",
      "err\n",
      3,
  )
  assert not result.timed_out


async def test_execute_passes_timeout_in_whole_seconds():
  """The timeout is rounded up to whole seconds for the sandbox."""
  client = _make_client()
  environment = await _make_environment(client)

  await environment.execute("true", timeout=2.5)

  assert client.sandboxes.commands[-1]["timeout"] == 3


@pytest.mark.parametrize(
    "error",
    [
        ServerError(504, {"error": {"code": 504}}),
        APIError(0, {"error": {"status": "DEADLINE_EXCEEDED"}}),
    ],
    ids=["http_504", "deadline_exceeded"],
)
async def test_execute_reports_timeout(error):
  """A command the sandbox kills for running too long is reported timed out."""
  client = _make_client()
  environment = await _make_environment(client)
  client.sandboxes.execute_bash = MagicMock(side_effect=error)

  result = await environment.execute("sleep 999", timeout=1)

  assert result.timed_out


async def test_timeout_reported_as_failed_precondition_runs_command_once():
  """A timeout reported as FAILED_PRECONDITION is timed out and not retried.

  The service reports a command the shell server killed for running too long
  as FAILED_PRECONDITION with DEADLINE_EXCEEDED in the message. Even if the
  sandbox is no longer running when the timeout returns, the command is not
  retried and the sandbox state is not queried.
  """
  client = _make_client()
  environment = await _make_environment(client)
  client.sandboxes.states[environment.sandbox_name] = "STATE_PAUSED"
  client.sandboxes.get = MagicMock(wraps=client.sandboxes.get)
  client.sandboxes.execute_bash = MagicMock(
      side_effect=ClientError(
          400,
          {
              "error": {
                  "code": 400,
                  "status": "FAILED_PRECONDITION",
                  "message": (
                      "Execution Failed. Error: DEADLINE_EXCEEDED on URL"
                      " `https://sandbox/exec`."
                  ),
              }
          },
      )
  )

  result = await environment.execute("sleep 999", timeout=2)

  assert result.timed_out
  client.sandboxes.execute_bash.assert_called_once()
  client.sandboxes.get.assert_not_called()
  assert not client.sandboxes.resume_calls


async def test_error_from_running_sandbox_is_raised_without_retry():
  """A FAILED_PRECONDITION from a running sandbox raises and does not rerun."""
  client = _make_client()
  environment = await _make_environment(client)
  error = ClientError(
      400, {"error": {"code": 400, "status": "FAILED_PRECONDITION"}}
  )
  client.sandboxes.execute_bash = MagicMock(side_effect=error)

  with pytest.raises(ClientError):
    await environment.execute("echo once")

  client.sandboxes.execute_bash.assert_called_once()


def _unreachable_error():
  return ClientError(
      400,
      {
          "error": {
              "code": 400,
              "status": "FAILED_PRECONDITION",
              "message": (
                  "Execution Failed. Error: INTERNAL on URL"
                  " `https://sandbox/exec`. Error Details: Bad Gateway:"
                  " Unable to reach the sandbox environment."
              ),
          }
      },
  )


async def test_command_the_sandbox_cannot_receive_yet_is_resent(monkeypatch):
  """A command the service could not deliver to the sandbox is sent again."""
  monkeypatch.setattr(
      _agent_engine_environment, "_UNREACHABLE_RETRY_INTERVAL_SECONDS", 0
  )
  client = _make_client()
  environment = await _make_environment(client)
  execute_bash = client.sandboxes.execute_bash
  calls = []

  def unreachable_once(**kwargs):
    calls.append(kwargs)
    if len(calls) == 1:
      raise _unreachable_error()
    return execute_bash(**kwargs)

  client.sandboxes.execute_bash = unreachable_once

  result = await environment.execute("echo hi")

  assert result.stdout == "hi\n"
  assert len(calls) == 2


async def test_sandbox_that_stays_unreachable_raises(monkeypatch):
  """A sandbox the service cannot reach for too long raises without resuming."""
  monkeypatch.setattr(
      _agent_engine_environment, "_UNREACHABLE_RETRY_SECONDS", 0
  )
  client = _make_client()
  environment = await _make_environment(client)
  client.sandboxes.states[environment.sandbox_name] = "STATE_PAUSED"
  client.sandboxes.get = MagicMock(wraps=client.sandboxes.get)
  client.sandboxes.execute_bash = MagicMock(side_effect=_unreachable_error())

  with pytest.raises(ClientError, match="Unable to reach"):
    await environment.execute("echo hi")

  client.sandboxes.execute_bash.assert_called_once()
  client.sandboxes.get.assert_not_called()
  assert not client.sandboxes.resume_calls


async def test_execute_on_paused_sandbox_resumes_and_runs():
  """A command sent to a paused sandbox resumes it and then runs."""
  client = _make_client()
  environment = await _make_environment(client)
  client.sandboxes.states[environment.sandbox_name] = "STATE_PAUSED"

  result = await environment.execute("echo hi")

  assert result.stdout == "hi\n"
  assert client.sandboxes.resume_calls == [environment.sandbox_name]


async def test_execute_on_terminated_sandbox_raises():
  """A command sent to a terminated sandbox raises instead of waiting."""
  client = _make_client()
  environment = await _make_environment(client)
  client.sandboxes.states[environment.sandbox_name] = "STATE_TERMINATED"

  with pytest.raises(RuntimeError, match="STATE_TERMINATED"):
    await environment.execute("echo hi")


async def test_execute_before_initialize_raises():
  """Running a command before initializing raises."""
  environment = AgentEngineEnvironment(vertexai_client=_make_client())

  with pytest.raises(RuntimeError, match="initialize"):
    await environment.execute("echo hi")


async def test_written_file_is_read_back(tmp_path):
  """Bytes written to a file, in a new directory, are read back unchanged."""
  environment = await _make_environment()
  path = tmp_path / "new dir" / "data.bin"
  content = bytes(range(256)) * 4

  await environment.write_file(path, content)
  read_back = await environment.read_file(path)

  assert read_back == content


async def test_text_is_written_as_utf8(tmp_path):
  """Text content is written as UTF-8."""
  environment = await _make_environment()
  path = tmp_path / "notes.txt"

  await environment.write_file(path, "héllo")

  assert path.read_bytes() == "héllo".encode("utf-8")


async def test_reading_missing_file_raises_file_not_found(tmp_path):
  """Reading a file that does not exist raises FileNotFoundError."""
  environment = await _make_environment()

  with pytest.raises(FileNotFoundError):
    await environment.read_file(tmp_path / "missing.txt")


async def test_reading_directory_raises(tmp_path):
  """Reading a directory raises instead of returning empty content."""
  environment = await _make_environment()

  with pytest.raises(OSError, match="Failed to read"):
    await environment.read_file(tmp_path)


async def test_empty_file_is_read(tmp_path):
  """Reading an empty file returns no bytes."""
  environment = await _make_environment()
  path = tmp_path / "empty.txt"
  path.write_bytes(b"")

  assert await environment.read_file(path) == b""


async def test_file_of_one_slice_is_read_in_one_command(tmp_path):
  """A file of exactly one slice, 1 MiB, is read with a single command."""
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "one.bin"
  content = bytes(range(256)) * 4096
  path.write_bytes(content)

  read_back = await environment.read_file(path)

  assert read_back == content
  assert len(client.sandboxes.commands) == 1


async def test_large_binary_file_is_read_in_slices(tmp_path):
  """A 3.5 MiB binary file is read back unchanged, one 1 MiB slice at a time.

  One command finds the file too large for a single read, then four commands
  read its slices.
  """
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "big.bin"
  content = bytes(range(256)) * (14 * 1024)
  path.write_bytes(content)

  read_back = await environment.read_file(path)

  assert read_back == content
  assert len(client.sandboxes.commands) == 5


async def test_large_file_removed_during_read_raises(tmp_path, monkeypatch):
  """A large file that disappears between slices raises OSError."""
  monkeypatch.setattr(_agent_engine_environment, "_READ_CHUNK_BYTES", 4)
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "data.bin"
  path.write_bytes(b"x" * 10)
  execute_bash = client.sandboxes.execute_bash

  def remove_after_first_slice(**kwargs):
    result = execute_bash(**kwargs)
    if len(client.sandboxes.commands) == 2:
      path.unlink()
    return result

  client.sandboxes.execute_bash = remove_after_first_slice

  with pytest.raises(OSError, match="Failed to read"):
    await environment.read_file(path)


async def test_symlink_to_large_file_is_read_in_slices(tmp_path, monkeypatch):
  """A symlink to a file larger than one slice is read in slices."""
  monkeypatch.setattr(_agent_engine_environment, "_READ_CHUNK_BYTES", 4)
  client = _make_client()
  environment = await _make_environment(client)
  target = tmp_path / "target.bin"
  content = b"0123456789"
  target.write_bytes(content)
  link = tmp_path / "link.bin"
  link.symlink_to(target)

  read_back = await environment.read_file(link)

  assert read_back == content
  assert len(client.sandboxes.commands) == 4


async def test_large_file_truncated_during_read_raises(tmp_path, monkeypatch):
  """A large file that shrinks between slices raises OSError."""
  monkeypatch.setattr(_agent_engine_environment, "_READ_CHUNK_BYTES", 4)
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "data.bin"
  path.write_bytes(b"0123456789")
  execute_bash = client.sandboxes.execute_bash

  def truncate_after_first_slice(**kwargs):
    result = execute_bash(**kwargs)
    if len(client.sandboxes.commands) == 2:
      path.write_bytes(b"01234")
    return result

  client.sandboxes.execute_bash = truncate_after_first_slice

  with pytest.raises(OSError, match="expected 10 bytes, got 5"):
    await environment.read_file(path)


async def test_relative_path_resolves_under_workspace():
  """A relative path is resolved against the sandbox workspace."""
  client = _make_client()
  environment = await _make_environment(client)

  with pytest.raises(FileNotFoundError):
    await environment.read_file("sub/notes.txt")

  assert "/workspace/sub/notes.txt" in client.sandboxes.commands[-1]["command"]


async def test_empty_file_is_written(tmp_path):
  """Writing empty content creates an empty file."""
  environment = await _make_environment()
  path = tmp_path / "empty.txt"

  await environment.write_file(path, b"")

  assert path.read_bytes() == b""


@pytest.mark.parametrize(
    ("size", "expected_commands"),
    [(12, 1), (13, 3), (24, 3), (25, 4)],
    ids=[
        "exactly_one_chunk",
        "one_chunk_plus_one",
        "exactly_two_chunks",
        "two_chunks_plus_one",
    ],
)
async def test_write_is_split_at_chunk_boundaries(
    tmp_path, monkeypatch, size, expected_commands
):
  """Content up to one chunk takes one command; more is sent in chunks.

  With 16-character chunks, 12 bytes encode to exactly one chunk. A chunked
  write takes one command per chunk plus one to decode and rename.
  """
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "data.bin"
  content = bytes(range(size))

  await environment.write_file(path, content)

  assert path.read_bytes() == content
  assert len(client.sandboxes.commands) == expected_commands
  assert sorted(tmp_path.iterdir()) == [path]


async def test_large_binary_file_round_trips(tmp_path):
  """A 1 MiB binary file is written in chunks and read back unchanged."""
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "big.bin"
  content = bytes(range(256)) * 4096

  await environment.write_file(path, content)
  read_back = await environment.read_file(path)

  assert read_back == content
  assert len(client.sandboxes.commands) > 2
  assert max(len(c["command"]) for c in client.sandboxes.commands) < 128 * 1024
  assert sorted(tmp_path.iterdir()) == [path]


async def test_failed_chunked_write_keeps_old_file_and_removes_temp_files(
    tmp_path, monkeypatch
):
  """A chunked write that fails partway raises and leaves no trace.

  The existing file keeps its old content, since the target is replaced only by
  the final rename, and the temporary files are removed.
  """
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  client = _make_client()
  environment = await _make_environment(client)
  path = tmp_path / "data.txt"
  path.write_bytes(b"old content")
  execute_bash = client.sandboxes.execute_bash
  calls = []

  def fail_third_command(**kwargs):
    calls.append(kwargs)
    if len(calls) == 3:
      return {"stdout": "", "stderr": "disk full", "returncode": 1}
    return execute_bash(**kwargs)

  client.sandboxes.execute_bash = fail_third_command

  with pytest.raises(OSError, match="disk full"):
    await environment.write_file(path, b"x" * 100)

  assert path.read_bytes() == b"old content"
  assert sorted(tmp_path.iterdir()) == [path]


async def test_chunked_write_replaces_existing_file(tmp_path, monkeypatch):
  """A chunked write replaces an existing, longer file in full."""
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  environment = await _make_environment()
  path = tmp_path / "data.txt"
  path.write_bytes(b"y" * 1000)

  await environment.write_file(path, b"x" * 100)

  assert path.read_bytes() == b"x" * 100
  assert sorted(tmp_path.iterdir()) == [path]


async def test_chunked_write_keeps_existing_file_mode(tmp_path, monkeypatch):
  """A chunked write over an executable file keeps it executable."""
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  environment = await _make_environment()
  path = tmp_path / "run.sh"
  path.write_bytes(b"#!/bin/sh\n")
  path.chmod(0o751)

  await environment.write_file(path, b"#!/bin/sh\necho " + b"x" * 100)

  assert path.stat().st_mode & 0o777 == 0o751


async def test_chunked_write_to_existing_directory_raises(
    tmp_path, monkeypatch
):
  """A chunked write whose target is an existing directory raises OSError."""
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  environment = await _make_environment()
  directory = tmp_path / "dir"
  directory.mkdir()

  with pytest.raises(OSError, match="Failed to write"):
    await environment.write_file(directory, b"x" * 100)

  assert list(directory.iterdir()) == []
  assert sorted(tmp_path.iterdir()) == [directory]


async def test_chunked_write_with_long_multibyte_filename(
    tmp_path, monkeypatch
):
  """A chunked write to a long multibyte filename stays within NAME_MAX."""
  monkeypatch.setattr(_agent_engine_environment, "_WRITE_CHUNK_BYTES", 16)
  environment = await _make_environment()
  path = tmp_path / ("日" * 80)

  await environment.write_file(path, b"x" * 100)

  assert path.read_bytes() == b"x" * 100
  assert sorted(tmp_path.iterdir()) == [path]


async def test_writing_file_over_size_limit_raises(tmp_path, monkeypatch):
  """A file over the size limit is rejected before anything is sent."""
  monkeypatch.setattr(_agent_engine_environment, "_MAX_WRITE_BYTES", 10)
  client = _make_client()
  environment = await _make_environment(client)

  with pytest.raises(ValueError, match="not supported"):
    await environment.write_file(tmp_path / "big.bin", b"x" * 11)

  assert not client.sandboxes.commands


async def test_working_dir_is_workspace():
  """The working directory is the sandbox workspace."""
  environment = await _make_environment()

  assert str(environment.working_dir) == "/workspace"
