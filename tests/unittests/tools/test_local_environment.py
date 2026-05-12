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

"""Tests for LocalEnvironment.read_file and write_file."""

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
  async def test_write_creates_parent_dirs(self, env: LocalEnvironment):
    """Parent directories are created automatically."""
    await env.write_file(Path("sub/dir/file.txt"), "nested")
    data = await env.read_file("sub/dir/file.txt")
    assert data == b"nested"

  @pytest.mark.asyncio
  async def test_read_nonexistent_raises(self, env: LocalEnvironment):
    """Reading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      await env.read_file(Path("does_not_exist.txt"))
