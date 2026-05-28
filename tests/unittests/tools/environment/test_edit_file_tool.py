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

"""Tests for EditFileTool.

Verifies that EditFileTool correctly handles line break differences.
"""

from pathlib import Path

from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._edit_file_tool import EditFileTool
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(name="env")
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestEditFileTool:
  """Tests for EditFileTool behavior."""

  @pytest.mark.asyncio
  async def test_edit_file_handles_line_breaks_linux_file_windows_search(
      self, env: LocalEnvironment
  ):
    """File has \\n, search string has \\r\\n."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\r\nline2",
        "new_string": "line1_replaced\nline2_replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"line1_replaced\nline2_replaced\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_handles_line_breaks_windows_file_linux_search(
      self, env: LocalEnvironment
  ):
    """File has \\r\\n, search string has \\n."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\r\nline2\r\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "line1_replaced\r\nline2_replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"line1_replaced\r\nline2_replaced\r\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_fails_on_multiple_matches(
      self, env: LocalEnvironment
  ):
    """Tool fails if old_string appears multiple times."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline1\nline2")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "error"
    assert "appears 2 times" in result["error"]

  @pytest.mark.asyncio
  async def test_edit_file_exact_match_works(self, env: LocalEnvironment):
    """Exact match works as before."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"replaced\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_handles_special_regex_chars(
      self, env: LocalEnvironment
  ):
    """Special regex characters in old_string are escaped."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1.content\nline2")

    args = {
        "path": "test.txt",
        "old_string": "line1.content",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"replaced\nline2"
