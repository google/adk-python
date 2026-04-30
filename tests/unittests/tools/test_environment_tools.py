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

"""Tests for environment tools."""

from pathlib import Path

from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._tools import ReadFileTool
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(name='env')
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestReadFileTool:
  """Verifies file reads stay within Python file I/O."""

  @pytest.mark.asyncio
  async def test_ranged_read_returns_selected_lines(
      self, env: LocalEnvironment
  ):
    """Reads the requested line range and preserves line numbers."""
    await env.write_file('sample.txt', 'line1\nline2\nline3\n')

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={'path': 'sample.txt', 'start_line': 2, 'end_line': 3},
        tool_context=None,
    )

    assert result == {
        'status': 'ok',
        'content': '     2\tline2\n     3\tline3\n',
        'total_lines': 3,
    }

  @pytest.mark.asyncio
  async def test_ranged_read_missing_file_returns_error(
      self, env: LocalEnvironment
  ):
    """Returns a missing-file error for ranged reads."""
    tool = ReadFileTool(env)

    result = await tool.run_async(
        args={'path': 'missing.txt', 'start_line': 2},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': 'File not found: missing.txt',
    }

  @pytest.mark.asyncio
  async def test_ranged_read_rejects_non_integer_end_line(
      self, env: LocalEnvironment
  ):
    """Rejects non-integer line numbers without executing shell syntax."""
    await env.write_file('sample.txt', 'line1\nline2\n')
    marker = env.working_dir / 'marker.txt'
    injected_end_line = f"1'; touch {marker}; echo '"

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={'path': 'sample.txt', 'end_line': injected_end_line},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': '`end_line` must be an integer if provided.',
    }
    assert not marker.exists()
