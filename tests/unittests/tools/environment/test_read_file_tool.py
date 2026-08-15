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

"""Tests for ReadFileTool."""

from pathlib import Path
from typing import Optional

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._read_file_tool import ReadFileTool
import pytest
import pytest_asyncio


class _StubEnvironment(BaseEnvironment):
  """Minimal environment double for ReadFileTool tests."""

  def __init__(self, files: dict[str, bytes]):
    self._files = files
    self.execute_calls: list[str] = []

  @property
  def working_dir(self) -> Path:
    return Path('/tmp/adk-test')

  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
  ) -> ExecutionResult:
    del timeout
    self.execute_calls.append(command)
    raise AssertionError('ReadFileTool should not invoke execute().')

  async def read_file(self, path: Path) -> bytes:
    key = str(path)
    if key not in self._files:
      raise FileNotFoundError(key)
    return self._files[key]

  async def write_file(self, path: Path, content: str | bytes) -> None:
    del path, content
    raise NotImplementedError


@pytest.mark.asyncio
async def test_read_file_with_line_range_uses_direct_file_read():
  """ReadFileTool slices lines without shelling out."""
  environment = _StubEnvironment({
      'notes.txt': b'alpha\nbeta\ngamma\ndelta\n',
  })

  result = await ReadFileTool(environment).run_async(
      args={'path': 'notes.txt', 'start_line': 2, 'end_line': 3},
      tool_context=None,
  )

  assert result == {
      'status': 'ok',
      'content': '     2\tbeta\n     3\tgamma\n',
      'total_lines': 4,
  }
  assert environment.execute_calls == []


@pytest.mark.asyncio
async def test_read_file_with_line_range_treats_shell_payload_as_literal_path():
  """Shell metacharacters in the path do not trigger command execution."""
  environment = _StubEnvironment({
      'safe.txt': b'line1\nline2\n',
  })
  payload = (
      '\'; python3 -c "from pathlib import Path;'
      " Path('pwned.txt').write_text('owned')\"; echo '"
  )

  result = await ReadFileTool(environment).run_async(
      args={'path': payload, 'start_line': 1, 'end_line': 2},
      tool_context=None,
  )

  assert result == {
      'status': 'error',
      'error': f'File not found: {payload}',
  }
  assert environment.execute_calls == []


@pytest_asyncio.fixture(name='env')
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestReadFileTool:
  """Tests for ReadFileTool behavior."""

  @pytest.mark.asyncio
  async def test_read_file_with_line_range_returns_selected_lines(
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
  async def test_read_file_with_line_range_missing_file_returns_error(
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
  async def test_read_file_rejects_non_integer_end_line(
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

  @pytest.mark.asyncio
  async def test_read_file_rejects_boolean_line_numbers(
      self, env: LocalEnvironment
  ):
    """Rejects boolean values for start_line and end_line."""
    await env.write_file('sample.txt', 'line1\nline2\n')

    tool = ReadFileTool(env)
    res_start = await tool.run_async(
        args={'path': 'sample.txt', 'start_line': True},
        tool_context=None,
    )
    res_end = await tool.run_async(
        args={'path': 'sample.txt', 'end_line': False},
        tool_context=None,
    )

    assert res_start == {
        'status': 'error',
        'error': '`start_line` must be an integer if provided.',
    }
    assert res_end == {
        'status': 'error',
        'error': '`end_line` must be an integer if provided.',
    }
