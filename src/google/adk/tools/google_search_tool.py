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

from google.genai import types
from typing_extensions import override

from ..utils.model_name_utils import is_gemini_1_model
from ..utils.model_name_utils import is_gemini_model
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class GoogleSearchTool(BaseTool):
  """A built-in tool that is automatically invoked by Gemini 2 models to retrieve search results from Google Search.

  This tool operates internally within the model and does not require or perform
  local code execution.
  """

  def __init__(
      self,
      *,
      bypass_multi_tools_limit: bool = False,
      model: str | None = None,
  ):
    """Initializes the Google search tool.

    Args:
      bypass_multi_tools_limit: Whether to bypass the multi tools limitation,
        so that the tool can be used with other tools in the same agent.
      model: Optional model name to use for processing the LLM request. If
        provided, this model will be used instead of the model from the
        incoming llm_request.
    """

    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='google_search', description='google_search')
    self.bypass_multi_tools_limit = bypass_multi_tools_limit
    self.model = model

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    # If a custom model is specified, use it instead of the original model
    if self.model is not None:
      llm_request.model = self.model

    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.tools = llm_request.config.tools or []
    if is_gemini_1_model(llm_request.model):
      if llm_request.config.tools:
        raise ValueError(
            'Google search tool cannot be used with other tools in Gemini 1.x.'
        )
      llm_request.config.tools.append(
          types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
      )
    elif is_gemini_model(llm_request.model):
      llm_request.config.tools.append(
          types.Tool(google_search=types.GoogleSearch())
      )
    else:
      raise ValueError(
          f'Google search tool is not supported for model {llm_request.model}'
      )


google_search = GoogleSearchTool()
