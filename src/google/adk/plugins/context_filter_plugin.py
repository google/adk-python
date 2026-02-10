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
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
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


def _is_function_response_content(content: types.Content) -> bool:
  """Returns whether a content contains function responses."""
  return bool(content.parts) and any(
      part.function_response is not None for part in content.parts
  )


def _is_human_user_content(content: types.Content) -> bool:
  """Returns whether a content represents user input (not tool output)."""
  return content.role == "user" and not _is_function_response_content(content)


def _get_invocation_start_indices(
    contents: Sequence[types.Content],
) -> list[int]:
  """Returns indices that begin a user-started invocation.

  An invocation begins with one or more consecutive user messages. Tool outputs
  (function responses) are role="user" but are *not* considered invocation
  starts.

  Args:
    contents: Full conversation contents in chronological order.

  Returns:
    A list of indices where each index marks the beginning of an invocation.
  """
  invocation_start_indices = []
  previous_was_human_user = False
  for i, content in enumerate(contents):
    is_human_user = _is_human_user_content(content)
    if is_human_user and not previous_was_human_user:
      invocation_start_indices.append(i)
    previous_was_human_user = is_human_user
  return invocation_start_indices


class ContextFilterPlugin(BasePlugin):
  """A plugin that filters the LLM context to reduce its size."""

  def __init__(
      self,
      num_invocations_to_keep: Optional[int] = None,
      custom_filter: Optional[
          Callable[[list[types.Content]], list[types.Content]]
      ] = None,
      name: str = "context_filter_plugin",
  ):
    """Initializes the context management plugin.

    Args:
      num_invocations_to_keep: The number of last invocations to keep. An
        invocation starts with one or more consecutive user messages and can
        contain multiple model turns (e.g. tool calls) until the next user
        message starts a new invocation.
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
      contents: list[types.Content] = llm_request.contents

      if (
          self._num_invocations_to_keep is not None
          and self._num_invocations_to_keep > 0
      ):
        invocation_start_indices = _get_invocation_start_indices(contents)
        if len(invocation_start_indices) > self._num_invocations_to_keep:
          split_index = invocation_start_indices[-self._num_invocations_to_keep]

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
    except Exception:
      logger.exception("Failed to reduce context for request")

    return None
