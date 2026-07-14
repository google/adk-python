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

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...utils.feature_decorator import experimental
from ..base_tool import BaseTool
from ._constants import MAX_OUTPUT_CHARS
from ._utils import truncate as _truncate

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


def _is_valid_line_number(value: Any) -> bool:
  """Returns True when *value* is a non-bool integer."""
  return isinstance(value, int) and not isinstance(value, bool)


@experimental
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
      # TODO: Avoid loading the entire file into memory to prevent OOM on large files.
      data_bytes = await self._environment.read_file(path)
      # Slice data_bytes by line boundaries before decoding.
      lines_bytes = data_bytes.splitlines(keepends=True)
      total = len(lines_bytes)
      start = max(1, start_line or 1)
      end = min(total, end_line or total)
      if start > total:
        return {
            'status': 'error',
            'error': (
                f'`start_line` {start} exceeds file length ({total} lines).'
            ),
            'total_lines': total,
        }
      if start > end:
        return {
            'status': 'error',
            'error': f'`start_line` ({start}) is after `end_line` ({end}).',
            'total_lines': total,
        }
      selected_bytes = lines_bytes[start - 1 : end]
      lines = [
          line_bytes.decode('utf-8', errors='replace')
          for line_bytes in selected_bytes
      ]
      numbered = ''.join(
          f'{start + i:6d}\t{line}' for i, line in enumerate(lines)
      )
      result = {
          'status': 'ok',
          'content': _truncate(
              numbered,
              limit=self._max_output_chars,
          ),
      }
      if start > 1 or end < total:
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
