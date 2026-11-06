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

from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._read_file_tool import _read_line_range
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

  @pytest.mark.parametrize('content', [b'', b'a\nb\n'])
  @pytest.mark.parametrize(
      'line_range',
      [
          {},
          {'start_line': 1},
          {'start_line': 0},
          {'start_line': -1},
          {'end_line': 0},
          {'end_line': 5},
          {'start_line': 1, 'end_line': 100},
      ],
  )
  async def test_read_whole_file_with_equivalent_line_ranges(
      self, env: LocalEnvironment, content: bytes, line_range: dict[str, int]
  ):
    """Equivalent whole-file ranges succeed for empty and nonempty files."""
    await env.write_file('sample.txt', content)

    result = await ReadFileTool(env).run_async(
        args={'path': 'sample.txt', **line_range}, tool_context=None
    )

    expected = '' if not content else '     1\ta\n     2\tb\n'
    assert result == {'status': 'ok', 'content': expected}

  @pytest.mark.parametrize(
      'line_range', [{'end_line': 1}, {'start_line': 1, 'end_line': 1}]
  )
  async def test_read_empty_file_through_first_line(
      self, env: LocalEnvironment, line_range: dict[str, int]
  ):
    """An explicit first-line boundary also succeeds for an empty file."""
    await env.write_file('empty.txt', b'')

    result = await ReadFileTool(env).run_async(
        args={'path': 'empty.txt', **line_range}, tool_context=None
    )

    assert result == {'status': 'ok', 'content': ''}

  @pytest.mark.parametrize(
      'line_range',
      [{'start_line': 2}, {'start_line': 2, 'end_line': 100}, {'end_line': -1}],
  )
  async def test_read_empty_file_with_line_range_returns_error(
      self, env: LocalEnvironment, line_range: dict[str, int]
  ):
    """Invalid ranges still report that an empty file has no lines."""
    await env.write_file('empty.txt', b'')
    start = line_range.get('start_line', 1)

    result = await ReadFileTool(env).run_async(
        args={'path': 'empty.txt', **line_range}, tool_context=None
    )

    assert result == {
        'status': 'error',
        'error': f'`start_line` {start} exceeds file length (0 lines).',
        'total_lines': 0,
    }

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

  @pytest.mark.asyncio
  async def test_read_file_lines_exceeds_total(self, env: LocalEnvironment):
    """Test start_line exceeding total lines of file."""
    await env.write_file('sample.txt', 'line1\nline2\n')

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={'path': 'sample.txt', 'start_line': 5},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': '`start_line` 5 exceeds file length (2 lines).',
        'total_lines': 2,
    }

  @pytest.mark.asyncio
  async def test_read_file_lines_order_violation(self, env: LocalEnvironment):
    """Test start_line after end_line error handling."""
    await env.write_file('sample.txt', 'line1\nline2\n')

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={'path': 'sample.txt', 'start_line': 3, 'end_line': 2},
        tool_context=None,
    )

    # Note that start_line is 3, which is also exceeding the file length.
    # Therefore, the start_line exceeding check is fired first.
    assert result == {
        'status': 'error',
        'error': '`start_line` 3 exceeds file length (2 lines).',
        'total_lines': 2,
    }

    # Now let's try with start_line = 2, end_line = 1 (valid lines but out of
    # order)
    result_out_of_order = await tool.run_async(
        args={'path': 'sample.txt', 'start_line': 2, 'end_line': 1},
        tool_context=None,
    )
    assert result_out_of_order == {
        'status': 'error',
        'error': '`start_line` (2) is after `end_line` (1).',
        'total_lines': 2,
    }

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'line_range',
      [{'end_line': 0}, {'start_line': 0}, {'start_line': -1, 'end_line': 0}],
  )
  async def test_read_file_lines_zero_or_negative_bounds_read_whole_file(
      self, env: LocalEnvironment, line_range: dict[str, int]
  ):
    """Test start_line below 1 reads from line 1 and end_line=0 reads to EOF."""
    await env.write_file('sample.txt', 'line1\nline2\n')

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={'path': 'sample.txt', **line_range},
        tool_context=None,
    )
    assert result == {
        'status': 'ok',
        'content': '     1\tline1\n     2\tline2\n',
    }

  @pytest.mark.asyncio
  async def test_read_file_tool_reads_through_read_file_stream(self):
    """ReadFileTool reads lines split across stream chunks."""

    class _ChunkedEnvironment(_StubEnvironment):

      def __init__(self):
        super().__init__({})
        self.seen_paths: list[str] = []

      async def read_file(self, path: Path) -> bytes:
        raise AssertionError('read_file should not be called.')

      async def read_file_stream(
          self, path: Path
      ) -> AsyncGenerator[bytes, None]:
        self.seen_paths.append(path)
        for chunk in (b'fir', b'st\nsec', b'ond\nthi', b'rd\n'):
          yield chunk

    environment = _ChunkedEnvironment()
    result = await ReadFileTool(environment).run_async(
        args={'path': 'sample.txt', 'start_line': 2, 'end_line': 2},
        tool_context=None,
    )

    assert result == {
        'status': 'ok',
        'content': '     2\tsecond\n',
        'total_lines': 3,
    }
    assert environment.seen_paths == ['sample.txt']

  @pytest.mark.asyncio
  async def test_read_file_tool_closes_stream_on_mid_read_exit(self):
    """The stream is closed when the tool stops reading partway through."""

    class _StreamingEnvironment(_StubEnvironment):

      def __init__(self):
        super().__init__({})
        self.closed = False

      async def read_file_stream(self, path: Path):
        try:
          yield b'alpha\n'
          yield 'not bytes'  # Makes the tool fail between chunks.
          yield b'never read\n'
        finally:
          self.closed = True

    environment = _StreamingEnvironment()
    result = await ReadFileTool(environment).run_async(
        args={'path': 'sample.txt'},
        tool_context=None,
    )

    assert result['status'] == 'error'
    assert environment.closed

  @pytest.mark.asyncio
  async def test_read_file_tool_read_file_only_env_gets_path_unchanged(self):
    """An environment overriding only read_file gets the path as is."""

    class _RecordingEnvironment(_StubEnvironment):

      def __init__(self):
        super().__init__({'/workspace/notes.txt': b'alpha\n'})
        self.seen_paths = []

      async def read_file(self, path: Path) -> bytes:
        self.seen_paths.append(path)
        return await super().read_file(path)

    environment = _RecordingEnvironment()
    result = await ReadFileTool(environment).run_async(
        args={'path': '/workspace/notes.txt'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'content': '     1\talpha\n'}
    assert environment.seen_paths == ['/workspace/notes.txt']
    assert isinstance(environment.seen_paths[0], str)

  @pytest.mark.asyncio
  async def test_read_file_tool_splits_on_newline_only(self):
    """Lone carriage returns do not split lines."""
    environment = _StubEnvironment({'progress.log': b'step1\rstep2\nstep3\n'})

    result = await ReadFileTool(environment).run_async(
        args={'path': 'progress.log', 'start_line': 2},
        tool_context=None,
    )

    assert result == {
        'status': 'ok',
        'content': '     2\tstep3\n',
        'total_lines': 2,
    }

  @pytest.mark.asyncio
  async def test_read_file_tool_read_file_only_env_bounds_large_file(self):
    """A read_file-only environment reports total_lines when truncated."""
    lines_data = [f'line {i:04d}\n'.encode('utf-8') for i in range(1, 5001)]
    environment = _StubEnvironment({'large_log.txt': b''.join(lines_data)})

    result = await ReadFileTool(environment, max_output_chars=500).run_async(
        args={'path': 'large_log.txt'},
        tool_context=None,
    )

    assert result['status'] == 'ok'
    assert result['total_lines'] == 5000
    assert result['content'].endswith('\n... (truncated, 5000 total lines)')
    assert 'total chars' not in result['content']

  @pytest.mark.asyncio
  async def test_read_file_tool_bounds_large_file(self, env: LocalEnvironment):
    """ReadFileTool bounds reading large file and reports total_lines."""
    lines_data = [f'line {i:04d}\n'.encode('utf-8') for i in range(1, 5001)]
    await env.write_file('large_log.txt', b''.join(lines_data))

    tool = ReadFileTool(env, max_output_chars=500)
    result = await tool.run_async(
        args={'path': 'large_log.txt'},
        tool_context=None,
    )

    assert result['status'] == 'ok'
    assert result['total_lines'] == 5000
    assert result['content'].endswith('\n... (truncated, 5000 total lines)')
    assert 'total chars' not in result['content']

  @pytest.mark.asyncio
  async def test_read_file_tool_bounds_single_line_large_file(
      self, env: LocalEnvironment
  ):
    """ReadFileTool bounds reading a multi-chunk single-line file."""
    await env.write_file('single_line.txt', 'x' * 150_000)

    tool = ReadFileTool(env, max_output_chars=500)
    result = await tool.run_async(
        args={'path': 'single_line.txt'},
        tool_context=None,
    )

    assert result['status'] == 'ok'
    assert result['total_lines'] == 1
    assert result['content'].startswith('     1\t' + 'x' * 493)
    assert result['content'].endswith('\n... (truncated, 1 total lines)')
    assert 'total chars' not in result['content']
    assert len(result['content']) < 600

  @pytest.mark.asyncio
  async def test_read_file_tool_bounds_multibyte_utf8_truncates_content(
      self, env: LocalEnvironment
  ):
    """ReadFileTool bounds multibyte UTF-8 file and truncates content."""
    lines_data = [f'line {i:04d} ' + '中' * 20 + '\n' for i in range(1, 100)]
    await env.write_file('cjk_log.txt', ''.join(lines_data))

    tool = ReadFileTool(env, max_output_chars=500)
    result = await tool.run_async(
        args={'path': 'cjk_log.txt'},
        tool_context=None,
    )

    assert result['status'] == 'ok'
    assert result['total_lines'] == 99
    assert result['content'].endswith('\n... (truncated, 99 total lines)')
    assert 'total chars' not in result['content']

  @pytest.mark.asyncio
  async def test_read_file_tool_counts_lines_past_range_across_chunks(
      self, env: LocalEnvironment
  ):
    """total_lines counts every line, including past the 64 KiB chunks."""
    # A 70000-byte line straddles the first chunk boundary; no trailing \n.
    content = b'first\n' + b'y' * 70_000 + b'\n' + b'z\n' * 50_000 + b'tail'
    await env.write_file('mixed.txt', content)

    result = await ReadFileTool(env).run_async(
        args={'path': 'mixed.txt', 'end_line': 1},
        tool_context=None,
    )

    assert result == {
        'status': 'ok',
        'content': '     1\tfirst\n',
        'total_lines': 50_003,
    }


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
  for part in parts:
    yield part


@pytest.mark.asyncio
async def test_read_line_range_cuts_selected_lines_at_max_bytes():
  """Selected lines stop at max_bytes while every line is still counted."""
  data = b''.join(f'line {i:03d}\n'.encode('utf-8') for i in range(1, 101))

  lines, total = await _read_line_range(
      _chunks(data[:40], data[40:]),
      start_line=1,
      end_line=None,
      max_bytes=25,
  )

  assert lines == [b'line 001\n', b'line 002\n', b'line 00']
  assert total == 100


@pytest.mark.asyncio
async def test_read_line_range_counts_lines_split_across_chunks():
  """Lines split across chunk boundaries are kept and counted once."""
  lines, total = await _read_line_range(
      _chunks(b'first\nyy', b'yy\nz', b'\nz\n', b'tail'),
      start_line=2,
      end_line=2,
      max_bytes=100,
  )

  assert lines == [b'yyyy\n']
  assert total == 5
