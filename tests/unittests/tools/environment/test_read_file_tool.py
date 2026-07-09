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

from pathlib import Path
from typing import Optional

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.tools.environment._tools import ReadFileTool
import pytest


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
