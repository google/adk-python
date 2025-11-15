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
from typing import Callable
from typing import List
from typing import Optional

from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from .base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)


class ContextFilterPlugin(BasePlugin):
  """A plugin that filters the LLM context to reduce its size."""

  def __init__(
      self,
      num_invocations_to_keep: Optional[int] = None,
      custom_filter: Optional[Callable[[List[Event]], List[Event]]] = None,
      name: str = "context_filter_plugin",
  ):
    """Initializes the context management plugin.

    Args:
      num_invocations_to_keep: The number of last invocations to keep. An
        invocation is defined as one or more consecutive user messages followed
        by a model response.
      custom_filter: A function to filter the context.
      name: The name of the plugin instance.
    """
    super().__init__(name)
    self._num_invocations_to_keep = num_invocations_to_keep
    self._custom_filter = custom_filter

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Filters the LLM request's context before it is sent to the model."""
    try:
      contents = llm_request.contents

      if (
          self._num_invocations_to_keep is not None
          and self._num_invocations_to_keep > 0
      ):
        num_model_turns = sum(1 for c in contents if c.role == "model")
        if num_model_turns >= self._num_invocations_to_keep:
          model_turns_to_find = self._num_invocations_to_keep
          split_index = 0
          for i in range(len(contents) - 1, -1, -1):
            if contents[i].role == "model":
              model_turns_to_find -= 1
              if model_turns_to_find == 0:
                start_index = i
                while (
                    start_index > 0 and contents[start_index - 1].role == "user"
                ):
                  start_index -= 1
                split_index = start_index
                break
          contents = contents[split_index:]

      if self._custom_filter:
        contents = self._custom_filter(contents)

      llm_request.contents = contents
    except Exception as e:
      logger.error(f"Failed to reduce context for request: {e}")

    return None
