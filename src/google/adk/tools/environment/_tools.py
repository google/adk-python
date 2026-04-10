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

"""Tools available in the environment toolset."""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...utils.feature_decorator import experimental
from ..base_tool import BaseTool
from ._constants import DEFAULT_TIMEOUT
from ._constants import MAX_OUTPUT_CHARS

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
  """Truncate text to *limit* characters with a notice."""
  if len(text) <= limit:
    return text
  return text[:limit] + f'\n... (truncated, {len(text)} total chars)'


_EXECUTE_TOOL_DESCRIPTION = """
Run a shell command in the environment. For running programs, tests, and build
commands ONLY. WARNING: Do NOT use for file reading -- use the ReadFile tool 
instead. Shell commands like 'cat, head, tail will produce inferior results. 
Good: Execute("python3 script.py"), Execute("pytest"), Execute("find ..."). 
Bad: Execute("head ..."), Execute("cat ...").
"""


@experimental
class ExecuteTool(BaseTool):
  """Run a shell command in the environment's working directory."""

  def __init__(self, environment: BaseEnvironment):
    super().__init__(
        name='Execute',
        description=_EXECUTE_TOOL_DESCRIPTION,
    )
    self._environment = environment

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': (
                        'The shell command to execute. Chain dependent commands'
                        ' with &&.'
                    ),
                },
            },
            'required': ['command'],
        },
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    command = args.get('command', '')
    if not command:
      return {'status': 'error', 'error': '`command` is required.'}

    logger.debug('Execute command: %s', command)
    try:
      execution_result: ExecutionResult = await self._environment.execute(
          command, timeout=DEFAULT_TIMEOUT
      )
      logger.debug(
          'Execute result: exit_code=%d, stdout=%r, stderr=%r, timed_out=%r',
          execution_result.exit_code,
          execution_result.stdout[:200] if execution_result.stdout else '',
          execution_result.stderr[:200] if execution_result.stderr else '',
          execution_result.timed_out,
      )
    except Exception as e:
      logger.exception('Execute failed: %s', e)
      return {'status': 'error', 'error': str(e)}

    result: dict[str, Any] = {'status': 'ok'}
    if execution_result.stdout:
      result['stdout'] = _truncate(execution_result.stdout)
    if execution_result.stderr:
      result['stderr'] = _truncate(execution_result.stderr)
    if execution_result.exit_code != 0:
      result['status'] = 'error'
      result['exit_code'] = execution_result.exit_code
    if execution_result.timed_out:
      result['status'] = 'error'
      result['error'] = f'Command timed out after {DEFAULT_TIMEOUT}s.'
    return result


@experimental
class ReadFileTool(BaseTool):
  """Read a file from the environment."""

  def __init__(self, environment: BaseEnvironment):
    super().__init__(
        name='ReadFile',
        description=(
            'Read the contents of a file in the environment. '
            'Returns the file content with line numbers.'
        ),
    )
    self._environment = environment

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

    # Use `sed` to read the file if start_line or end_line are specified.
    if (start_line and start_line > 1) or end_line:
      start = start_line or 1
      if end_line:
        sed_range = f'{start},{end_line}'
      else:
        sed_range = f'{start},$'
      cmd = f"cat -n '{path}' | sed -n '{sed_range}p'"
      res = await self._environment.execute(cmd)
      if res.exit_code == 0:
        return {'status': 'ok', 'content': _truncate(res.stdout)}

    try:
      data_bytes = await self._environment.read_file(path)
      text = data_bytes.decode('utf-8', errors='replace')
      lines = text.splitlines(True)
      total = len(lines)
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
      selected = lines[start - 1 : end]
      numbered = ''.join(
          f'{start + i:6d}\t{line}' for i, line in enumerate(selected)
      )
      result = {'status': 'ok', 'content': _truncate(numbered)}
      if start > 1 or end < total:
        result['total_lines'] = total
      return result
    except FileNotFoundError:
      return {'status': 'error', 'error': f'File not found: {path}'}
    except Exception as e:
      return {'status': 'error', 'error': str(e)}


@experimental
class WriteFileTool(BaseTool):
  """Create or overwrite a file in the environment."""

  def __init__(self, environment: BaseEnvironment):
    super().__init__(
        name='WriteFile',
        description=(
            'Create or overwrite a file in the environment. '
            'Use for new files or full rewrites. For small '
            'changes to existing files, prefer EditFile.'
        ),
    )
    self._environment = environment

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
                    'description': 'Path to the file within the environment.',
                },
                'content': {
                    'type': 'string',
                    'description': 'The full file content to write.',
                },
            },
            'required': ['path', 'content'],
        },
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    path = args.get('path', '')
    content = args.get('content', '')
    if not path:
      return {'status': 'error', 'error': '`path` is required.'}
    try:
      await self._environment.write_file(path, content)
    except Exception as e:
      return {'status': 'error', 'error': str(e)}
    return {'status': 'ok', 'message': f'Wrote {path}'}


@experimental
class EditFileTool(BaseTool):
  """Perform a surgical text replacement in an existing file."""

  def __init__(self, environment: BaseEnvironment):
    super().__init__(
        name='EditFile',
        description=(
            'Replace an exact substring in an existing file '
            'with new text. The old_string must appear exactly '
            'once in the file. To create new files, use the WriteFile tool.'
        ),
    )
    self._environment = environment

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
                        'Path of the file to edit within the environment.'
                    ),
                },
                'old_string': {
                    'type': 'string',
                    'description': (
                        'The exact text to find and replace. Must not be empty.'
                    ),
                },
                'new_string': {
                    'type': 'string',
                    'description': 'The replacement text.',
                },
            },
            'required': ['path', 'old_string', 'new_string'],
        },
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    path = args.get('path', '')
    old_string = args.get('old_string', '')
    new_string = args.get('new_string', '')
    if not path:
      return {'status': 'error', 'error': '`path` is required.'}

    if not old_string:
      return {
          'status': 'error',
          'error': (
              '`old_string` cannot be empty. To create a new '
              'file, use the WriteFile tool.'
          ),
      }

    try:
      data_bytes = await self._environment.read_file(path)
      content = data_bytes.decode('utf-8', errors='replace')
    except FileNotFoundError:
      return {'status': 'error', 'error': f'File not found: {path}'}

    count = content.count(old_string)
    if count == 0:
      return {
          'status': 'error',
          'error': (
              '`old_string` not found in file. Read the file first '
              'to verify contents.'
          ),
      }
    if count > 1:
      return {
          'status': 'error',
          'error': (
              f'`old_string` appears {count} times. Provide more '
              'surrounding context to make it unique.'
          ),
      }

    new_content = content.replace(old_string, new_string, 1)
    await self._environment.write_file(path, new_content)
    return {'status': 'ok', 'message': f'Edited {path}'}
