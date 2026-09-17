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

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..models.llm_request import LlmRequest
from ..utils.model_name_utils import is_gemini_model
from ..utils.model_name_utils import is_gemini_model_id_check_disabled
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult


class BuiltInCodeExecutor(BaseCodeExecutor):
  """A code executor that uses the Model's built-in code executor.

  Currently only supports Gemini models, but will be expanded to
  other models.
  """

  @override
  def execute_code(  # type: ignore[empty-body]
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    # Execution is delegated to the model, so there is nothing to run here.
    # A direct caller gets None; raising instead would change that.
    pass

  def process_llm_request(self, llm_request: LlmRequest) -> None:
    """Pre-process the LLM request for Gemini models to use the code execution tool."""
    from google.genai import types

    model_check_disabled = is_gemini_model_id_check_disabled()
    if is_gemini_model(llm_request.model) or model_check_disabled:
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(code_execution=types.ToolCodeExecution())
      )
      return
    raise ValueError(
        "Gemini code execution tool is not supported for model"
        f" {llm_request.model}"
    )
