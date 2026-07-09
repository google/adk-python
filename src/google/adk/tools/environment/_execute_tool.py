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

"""ExecuteTool for running shell commands in the environment."""

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
from ._utils import truncate as _truncate

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


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

  def __init__(
      self,
      environment: BaseEnvironment,
      *,
      max_output_chars: Optional[int] = None,
  ):
    super().__init__(
        name='Execute',
        description=_EXECUTE_TOOL_DESCRIPTION,
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
      result['stdout'] = _truncate(
          execution_result.stdout,
          limit=self._max_output_chars,
      )
    if execution_result.stderr:
      result['stderr'] = _truncate(
          execution_result.stderr,
          limit=self._max_output_chars,
      )
    if execution_result.exit_code != 0:
      result['status'] = 'error'
      result['exit_code'] = execution_result.exit_code
    if execution_result.timed_out:
      result['status'] = 'error'
      result['error'] = f'Command timed out after {DEFAULT_TIMEOUT}s.'
    return result

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get('status') == 'error':
      return 'TOOL_ERROR'
    return None
