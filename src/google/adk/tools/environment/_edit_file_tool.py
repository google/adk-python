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

"""EditFileTool for performing surgical text replacements in existing files."""

from __future__ import annotations

import logging
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...utils.feature_decorator import experimental
from ..base_tool import BaseTool

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


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

    # Normalize line breaks in old_string to \n and use regex for flexible matching
    normalized_old = old_string.replace('\r\n', '\n')
    pattern = re.escape(normalized_old).replace('\n', '\r?\n')

    matches = re.findall(pattern, content)
    count = len(matches)

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

    new_content = re.sub(pattern, lambda m: new_string, content, count=1)
    await self._environment.write_file(path, new_content)
    return {'status': 'ok', 'message': f'Edited {path}'}

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get('status') == 'error':
      return 'TOOL_ERROR'
    return None
