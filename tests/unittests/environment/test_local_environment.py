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

"""Tests for LocalEnvironment file access and command execution."""

import asyncio
import os
from pathlib import Path

from google.adk.environment._local_environment import LocalEnvironment
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(name="env")
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestReadFileWriteFile:
  """Verify read_file and write_file accept both str and Path arguments."""

  @pytest.mark.asyncio
  async def test_write_and_read_with_str(self, env: LocalEnvironment):
    """Round-trip a file using str paths."""
    await env.write_file("hello.txt", "hello world")
    data = await env.read_file("hello.txt")
    assert data == b"hello world"

  @pytest.mark.asyncio
  async def test_write_and_read_with_path(self, env: LocalEnvironment):
    """Round-trip a file using Path objects."""
    await env.write_file(Path("path_obj.txt"), "path content")
    data = await env.read_file(Path("path_obj.txt"))
    assert data == b"path content"

  @pytest.mark.asyncio
  async def test_write_str_read_path(self, env: LocalEnvironment):
    """Write with str, read with Path."""
    await env.write_file("mixed.txt", "mixed")
    data = await env.read_file(Path("mixed.txt"))
    assert data == b"mixed"

  @pytest.mark.asyncio
  async def test_write_path_read_str(self, env: LocalEnvironment):
    """Write with Path, read with str."""
    await env.write_file(Path("mixed2.txt"), "mixed2")
    data = await env.read_file("mixed2.txt")
    assert data == b"mixed2"

  @pytest.mark.asyncio
  async def test_write_bytes_content(self, env: LocalEnvironment):
    """Write raw bytes and read them back."""
    raw = b"\x00\x01\x02\xff"
    await env.write_file(Path("binary.bin"), raw)
    data = await env.read_file("binary.bin")
    assert data == raw

  @pytest.mark.asyncio
  async def test_write_preserves_explicit_crlf(self, env: LocalEnvironment):
    """Explicit CRLF sequences are written without newline translation."""
    await env.write_file("crlf.txt", "first\r\nsecond\r\n")

    data = await env.read_file("crlf.txt")

    assert data == b"first\r\nsecond\r\n"

  @pytest.mark.asyncio
  async def test_write_creates_parent_dirs(self, env: LocalEnvironment):
    """Parent directories are created automatically."""
    await env.write_file(Path("sub/dir/file.txt"), "nested")
    data = await env.read_file("sub/dir/file.txt")
    assert data == b"nested"

  @pytest.mark.asyncio
  async def test_absolute_path_inside_working_dir(self, env: LocalEnvironment):
    """Absolute paths are accepted when they stay inside the workspace."""
    path = env.working_dir / "absolute.txt"
    await env.write_file(path, "absolute")
    data = await env.read_file(path)
    assert data == b"absolute"

  @pytest.mark.asyncio
  async def test_rejects_relative_path_escape(self, env: LocalEnvironment):
    """Parent traversal cannot escape the workspace."""
    outside = env.working_dir.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.read_file(Path("..") / outside.name)

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.write_file(Path("..") / "write-outside.txt", "nope")

    assert not (env.working_dir.parent / "write-outside.txt").exists()

  @pytest.mark.asyncio
  async def test_rejects_absolute_path_outside_working_dir(
      self, env: LocalEnvironment
  ):
    """Absolute paths outside the workspace are rejected."""
    outside = env.working_dir.parent / "outside-absolute.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.read_file(outside)

  @pytest.mark.asyncio
  async def test_read_nonexistent_raises(self, env: LocalEnvironment):
    """Reading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      await env.read_file(Path("does_not_exist.txt"))


class TestExecuteTimeout:
  """Timeout and cancellation must reach the whole process tree."""

  # A background command that keeps appending to a file, started by a command
  # that then blocks. The background one inherits stdout/stderr, so it holds
  # those pipes open for as long as it runs -- which is what used to make a
  # timeout wait forever. Shell, not Python: the test runner embeds its
  # interpreter and leaves `sys.executable` empty.
  _HEARTBEAT = "while true; do echo x >> beat; sleep 0.05; done\n"
  _COMMAND = "sh heartbeat.sh & sleep 60"

  @pytest.fixture(name="spawning_command")
  def _spawning_command(self, env: LocalEnvironment) -> str:
    """Writes the background script and returns the command that starts it."""
    (env.working_dir / "heartbeat.sh").write_text(self._HEARTBEAT)
    return self._COMMAND

  @staticmethod
  async def _wait_for_beat(env: LocalEnvironment) -> Path:
    """Waits until the background command is definitely running."""
    beat = env.working_dir / "beat"
    for _ in range(200):
      if beat.exists() and beat.stat().st_size:
        return beat
      await asyncio.sleep(0.05)
    pytest.fail("the background command never started")

  @staticmethod
  async def _assert_stopped(beat: Path) -> None:
    """Asserts the background command stopped writing, i.e. it is gone."""
    before = beat.stat().st_size
    await asyncio.sleep(0.5)
    size = beat.stat().st_size
    assert size == before, f"still running: {before} -> {size} bytes"

  @pytest.mark.skipif(
      not hasattr(os, "killpg"), reason="needs POSIX process groups"
  )
  @pytest.mark.asyncio
  async def test_timeout_returns_and_reaps_descendants(
      self, env: LocalEnvironment, spawning_command: str
  ):
    """A surviving background command must not hold the timeout open."""
    try:
      result = await asyncio.wait_for(
          env.execute(spawning_command, timeout=0.5), timeout=30
      )
    except asyncio.TimeoutError:
      pytest.fail("execute() hung well past its own 0.5s deadline")

    assert result.timed_out
    await self._assert_stopped(env.working_dir / "beat")

  @pytest.mark.skipif(
      not hasattr(os, "killpg"), reason="needs POSIX process groups"
  )
  @pytest.mark.asyncio
  async def test_cancellation_reaps_descendants(
      self, env: LocalEnvironment, spawning_command: str
  ):
    """Cancelling the caller must clean up the process tree too."""
    task = asyncio.create_task(env.execute(spawning_command))
    beat = await self._wait_for_beat(env)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=30)

    await self._assert_stopped(beat)
