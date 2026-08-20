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

"""WriteFileTool for creating or overwriting files in the environment."""

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

if TYPE_CHECKING:
  from ..tool_context import ToolContext


logger = logging.getLogger('google_adk.' + __name__)


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

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get('status') == 'error':
      return 'TOOL_ERROR'
    return None
