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

"""ReadFileTool for reading file contents in the environment."""

from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ..base_tool import BaseTool
from ._constants import MAX_OUTPUT_CHARS
from ._utils import truncate as _truncate

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


def _is_valid_line_number(value: Any) -> bool:
  """Returns True when *value* is a non-bool integer."""
  return isinstance(value, int) and not isinstance(value, bool)


async def _read_line_range(
    chunks: AsyncIterator[bytes],
    *,
    start_line: int,
    end_line: int | None,
    max_bytes: int,
) -> tuple[list[bytes], int]:
  """Returns lines start_line..end_line, cut at max_bytes, and the line count.

  Lines split on newline only. Lines past the range are counted, not kept.
  """
  selected_lines: list[bytes] = []
  total_lines = 0
  total_bytes = 0
  at_line_start = True
  line_buf = bytearray()
  async for chunk in chunks:
    pos = 0
    while pos < len(chunk):
      if total_bytes >= max_bytes or (
          end_line is not None
          and (
              total_lines > end_line
              or (total_lines == end_line and at_line_start)
          )
      ):
        # Nothing left to keep; count the remaining lines without a per-line
        # loop.
        ends_with_nl = chunk.endswith(b'\n')
        total_lines += (
            chunk.count(b'\n', pos) + int(at_line_start) - int(ends_with_nl)
        )
        at_line_start = ends_with_nl
        break
      if at_line_start:
        total_lines += 1
      nl = chunk.find(b'\n', pos)
      seg_end = len(chunk) if nl == -1 else nl + 1
      if start_line <= total_lines:
        seg = chunk[pos:seg_end][: max_bytes - total_bytes]
        line_buf.extend(seg)
        total_bytes += len(seg)
      at_line_start = nl != -1
      if at_line_start and line_buf:
        selected_lines.append(bytes(line_buf))
        line_buf.clear()
      pos = seg_end
  if line_buf:
    selected_lines.append(bytes(line_buf))
  return selected_lines, total_lines


class ReadFileTool(BaseTool):
  """Read a file from the environment."""

  def __init__(
      self,
      environment: BaseEnvironment,
      *,
      max_output_chars: Optional[int] = None,
  ):
    super().__init__(
        name='ReadFile',
        description=(
            'Read the contents of a file in the environment. '
            'Returns the file content with line numbers.'
        ),
    )
    self._environment = environment
    self._max_output_chars = (
        max_output_chars if max_output_chars is not None else MAX_OUTPUT_CHARS
    )

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': (
                        'Path of the file to read within the environment.'
                    ),
                },
                'start_line': {
                    'type': 'integer',
                    'description': (
                        'First line to return (1-based, '
                        'inclusive). Defaults to 1.'
                    ),
                },
                'end_line': {
                    'type': 'integer',
                    'description': (
                        'Last line to return (1-based, '
                        'inclusive). Defaults to end of file.'
                    ),
                },
            },
            'required': ['path'],
        },
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    path = args.get('path', '')
    if not path:
      return {'status': 'error', 'error': '`path` is required.'}
    start_line = args.get('start_line')
    end_line = args.get('end_line')
    for name, value in (('start_line', start_line), ('end_line', end_line)):
      if value is not None and not _is_valid_line_number(value):
        return {
            'status': 'error',
            'error': f'`{name}` must be an integer if provided.',
        }

    try:
      start = max(1, start_line or 1)
      # end_line=0 reads to the end of the file, as before.
      end_line = end_line or None
      max_bytes = self._max_output_chars * 4
      async with contextlib.aclosing(
          self._environment.read_file_stream(path)
      ) as chunks:
        selected_bytes, total = await _read_line_range(
            chunks,
            start_line=start,
            end_line=end_line,
            max_bytes=max_bytes + 1,
        )
      if start > total:
        return {
            'status': 'error',
            'error': (
                f'`start_line` {start} exceeds file length ({total} lines).'
            ),
            'total_lines': total,
        }
      end = total if end_line is None else min(total, end_line)
      if start > end:
        return {
            'status': 'error',
            'error': f'`start_line` ({start}) is after `end_line` ({end}).',
            'total_lines': total,
        }
      lines = [
          line_bytes.decode('utf-8', errors='replace')
          for line_bytes in selected_bytes
      ]
      numbered = ''.join(
          f'{start + i:6d}\t{line}' for i, line in enumerate(lines)
      )
      bounded_by_bytes = (
          len(lines) < (end - start + 1)
          or sum(len(b) for b in selected_bytes) > max_bytes
      )
      if bounded_by_bytes:
        content = (
            numbered[: self._max_output_chars]
            + f'\n... (truncated, {total} total lines)'
        )
      else:
        content = _truncate(
            numbered,
            limit=self._max_output_chars,
        )
      result: dict[str, str | int] = {
          'status': 'ok',
          'content': content,
      }
      if start > 1 or end < total or bounded_by_bytes:
        result['total_lines'] = total
      return result
    except FileNotFoundError:
      return {'status': 'error', 'error': f'File not found: {path}'}
    except Exception as e:
      return {'status': 'error', 'error': str(e)}

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get('status') == 'error':
      return 'TOOL_ERROR'
    return None
