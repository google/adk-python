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

from typing import Any
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin

PARTS_RETURNED_BY_TOOLS_ID = "temp:PARTS_RETURNED_BY_TOOLS_ID"


class MultimodalToolResultsPlugin(BasePlugin):
  """A plugin that modifies function tool responses to support returning list of parts directly.

  Should be removed in favor of directly supporting FunctionResponsePart when these
  are supported outside of computer use tool.
  For context see: https://github.com/google/adk-python/issues/3064#issuecomment-3463067459
  """

  def __init__(self, name: str = "multimodal_tool_results_plugin"):
    """Initialize the multimodal tool results plugin.

    Args:
      name: The name of the plugin instance.
    """
    super().__init__(name)

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Saves parts returned by the tool in ToolContext.

    Later these are passed to LLM's context as-is.
    No-op if tool doesn't return list[google.genai.types.Part] or google.genai.types.Part.
    """

    if not (
        isinstance(result, types.Part)
        or isinstance(result, list)
        and result
        and isinstance(result[0], types.Part)
    ):
      return result

    parts = [result] if isinstance(result, types.Part) else result[:]

    if PARTS_RETURNED_BY_TOOLS_ID in tool_context.state:
      tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] += parts
    else:
      tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] = parts

    return None

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Attach saved list[google.genai.types.Part] returned by the tool to llm_request."""

    if saved_parts := callback_context.state.get(
        PARTS_RETURNED_BY_TOOLS_ID, None
    ):
      llm_request.contents[-1].parts += saved_parts
      callback_context.state.update({PARTS_RETURNED_BY_TOOLS_ID: []})

    return None
