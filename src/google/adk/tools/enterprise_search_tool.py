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

from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class EnterpriseWebSearchTool(BaseTool):
  """A Gemini 2+ built-in tool using web grounding for Enterprise compliance.

  See the documentation for more details:
  https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/web-grounding-enterprise.
  """

  def __init__(self):
    """Initializes the Vertex AI Search tool."""
    # Name and description are not used because this is a model built-in tool.
    super().__init__(
        name='enterprise_web_search', description='enterprise_web_search'
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    if llm_request.model and llm_request.model.startswith('gemini-'):
      if llm_request.model.startswith('gemini-1') and llm_request.config.tools:
        raise ValueError(
            'Enterprise web search tool can not be used with other tools in'
            ' Gemini 1.x.'
        )
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
      )
    else:
      raise ValueError(
          'Enterprise web search tool is not supported for model'
          f' {llm_request.model}'
      )
