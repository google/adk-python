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

from ..utils.model_name_utils import is_gemini_1_model
from ..utils.model_name_utils import is_gemini_model
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class GoogleMapsGroundingTool(BaseTool):
  """A built-in tool that is automatically invoked by Gemini 2 models to ground query results with Google Maps.

  This tool operates internally within the model and does not require or perform
  local code execution.

  Only available for use with the VertexAI Gemini API (e.g.
  GOOGLE_GENAI_USE_VERTEXAI=TRUE)
  """

  def __init__(self):
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='google_maps', description='google_maps')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.tools = llm_request.config.tools or []
    if is_gemini_1_model(llm_request.model):
      raise ValueError(
          'Google Maps grounding tool cannot be used with Gemini 1.x models.'
      )
    elif is_gemini_model(llm_request.model):
      llm_request.config.tools.append(
          types.Tool(google_maps=types.GoogleMaps())
      )
    else:
      raise ValueError(
          f'Google maps tool is not supported for model {llm_request.model}'
      )


google_maps_grounding = GoogleMapsGroundingTool()
