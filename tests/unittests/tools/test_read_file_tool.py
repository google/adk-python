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
from google.adk.tools.environment._tools import ReadFileTool
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
async def test_ranged_read_does_not_use_shell():
  """A ranged read must not shell out (regression: `cat -n | sed` pipeline)."""
  env = _StubEnvironment({'f.txt': b'l1\nl2\nl3\nl4\n'})
  tool = ReadFileTool(env)

  result = await tool.run_async(
      args={'path': 'f.txt', 'start_line': 2, 'end_line': 3},
      tool_context=None,
  )

  assert result['status'] == 'ok'
  assert result['content'] == '     2\tl2\n     3\tl3\n'
  assert not env.execute_calls


@pytest.mark.asyncio
async def test_shell_metacharacters_in_path_are_literal():
  """A path with shell metacharacters is treated as a literal filename."""
  env = _StubEnvironment({'f.txt': b'l1\nl2\n'})
  tool = ReadFileTool(env)

  result = await tool.run_async(
      args={'path': "'; id > /tmp/pwned ; echo '", 'start_line': 2},
      tool_context=None,
  )

  assert result['status'] == 'error'
  assert 'File not found' in result['error']
  assert not env.execute_calls


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['start_line', 'end_line'])
@pytest.mark.parametrize('bad_value', ['2', 2.5, True, None.__class__])
async def test_non_integer_line_numbers_rejected(field, bad_value):
  """Non-integer line arguments are rejected instead of reaching a slice.

  Booleans are excluded explicitly: ``isinstance(True, int)`` is True, so a
  bare int check would let ``True`` through into the slice.
  """
  env = _StubEnvironment({'f.txt': b'l1\nl2\n'})
  tool = ReadFileTool(env)

  result = await tool.run_async(
      args={'path': 'f.txt', field: bad_value},
      tool_context=None,
  )

  assert result['status'] == 'error'
  assert f'`{field}` must be an integer if provided.' == result['error']
  assert not env.execute_calls


@pytest_asyncio.fixture(name='env')
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


@pytest.mark.asyncio
async def test_shell_injection_payload_creates_no_file(
    env: LocalEnvironment, tmp_path: Path
):
  """End-to-end: an injection payload must not execute against a real env."""
  canary = tmp_path.parent / 'adk_injection_canary'
  tool = ReadFileTool(env)

  result = await tool.run_async(
      args={'path': f"'; touch {canary} ; echo '", 'start_line': 2},
      tool_context=None,
  )

  assert result['status'] == 'error'
  assert not canary.exists()


@pytest.mark.asyncio
async def test_ranged_read_still_works_against_real_env(env: LocalEnvironment):
  """Ranged reads keep working after the shell path was removed."""
  await env.write_file('real.txt', 'a\nb\nc\nd\n')
  tool = ReadFileTool(env)

  result = await tool.run_async(
      args={'path': 'real.txt', 'start_line': 2, 'end_line': 3},
      tool_context=None,
  )

  assert result['status'] == 'ok'
  assert result['content'] == '     2\tb\n     3\tc\n'
  assert result['total_lines'] == 4
