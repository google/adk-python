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

from collections.abc import Sequence
import logging
from typing import Callable
from typing import List
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from .base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)


def _adjust_split_index_to_avoid_orphaned_function_responses(
    contents: Sequence[types.Content], split_index: int
) -> int:
  """Moves `split_index` left until function calls/responses stay paired.

  When truncating context, we must avoid keeping a `function_response` while
  dropping its matching preceding `function_call`.

  Args:
    contents: Full conversation contents in chronological order.
    split_index: Candidate split index (keep `contents[split_index:]`).

  Returns:
    A (possibly smaller) split index that preserves call/response pairs.
  """
  needed_call_ids = set()
  for i in range(len(contents) - 1, -1, -1):
    parts = contents[i].parts
    if parts:
      for part in reversed(parts):
        if part.function_response and part.function_response.id:
          needed_call_ids.add(part.function_response.id)
        if part.function_call and part.function_call.id:
          needed_call_ids.discard(part.function_call.id)

    if i <= split_index and not needed_call_ids:
      return i

  return 0


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
          # Adjust split_index to avoid orphaned function_responses.
          split_index = (
              _adjust_split_index_to_avoid_orphaned_function_responses(
                  contents, split_index
              )
          )
          contents = contents[split_index:]

      if self._custom_filter:
        contents = self._custom_filter(contents)

      llm_request.contents = contents
    except Exception as e:
      logger.error(f"Failed to reduce context for request: {e}")

    return None
