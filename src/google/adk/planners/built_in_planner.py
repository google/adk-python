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

from typing import List
from typing import Optional

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest
from .base_planner import BasePlanner


class BuiltInPlanner(BasePlanner):
  """The built-in planner that uses model's built-in thinking features.

  Attributes:
      thinking_config: Config for model built-in thinking features. An error
        will be returned if this field is set for models that don't support
        thinking.
  """

  thinking_config: types.ThinkingConfig
  """
  Config for model built-in thinking features. An error will be returned if this
  field is set for models that don't support thinking.
  """

  def __init__(self, *, thinking_config: types.ThinkingConfig):
    """Initializes the built-in planner.

    Args:
      thinking_config: Config for model built-in thinking features. An error
        will be returned if this field is set for models that don't support
        thinking.
    """
    self.thinking_config = thinking_config

  def apply_thinking_config(self, llm_request: LlmRequest) -> None:
    """Applies the thinking config to the LLM request.

    Args:
      llm_request: The LLM request to apply the thinking config to.
    """
    if self.thinking_config:
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.thinking_config = self.thinking_config

  @override
  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> Optional[str]:
    return

  @override
  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    return
