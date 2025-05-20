# Copyright 2025 Google LLC
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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deprecated import deprecated
from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


@deprecated(
    'No longer supported. Please use the new BuiltInCodeExecutor instead.'
)
class BuiltInCodeExecutionTool(BaseTool):
  """A built-in code execution tool that is automatically invoked by Gemini 2 models.

  This tool operates internally within the model and does not require or perform
  local code execution.
  """

  def __init__(self):
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='code_execution', description='code_execution')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    logger.warning(
        'BuiltInCodeExecutionTool is deprecated and will be removed in 1.1.0.'
        ' Please use the new BuiltInCodeExecutor instead.'
    )
    if llm_request.model and llm_request.model.startswith('gemini-2'):
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(code_execution=types.ToolCodeExecution())
      )
    else:
      raise ValueError(
          f'Code execution tool is not supported for model {llm_request.model}'
      )


built_in_code_execution = BuiltInCodeExecutionTool()
