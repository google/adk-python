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

"""Tool-aware context filter plugin for managing conversation history.

This plugin extends the standard context filtering to properly handle function
call/response sequences, ensuring they remain atomic during history trimming.

PROBLEM WITH STANDARD ContextFilterPlugin:
==========================================
The standard ContextFilterPlugin treats each model message as a separate
"invocation", but when a model makes a tool call, it creates MULTIPLE model
messages in sequence:
  1. Model message with function_call
  2. User message with function_response (tool result)
  3. Model message with final text response

When filtering to keep N "invocations", the standard plugin can split these
related messages apart, creating orphaned function_responses without their
corresponding function_calls, which violates OpenAI API requirements.

HOW THIS PLUGIN SOLVES IT:
===========================
This plugin groups messages into LOGICAL invocations where a complete cycle is:
  - User query (one or more messages)
  - Model response (possibly with function_call)
  - Function response(s) (if tool was called)
  - Model final response (after tool execution)

All messages in a tool call sequence are kept together as an atomic unit.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

logger = logging.getLogger("google_adk." + __name__)


class ToolAwareContextFilterPlugin(BasePlugin):
  """A plugin that filters LLM context while preserving tool call sequences.

  This plugin extends context filtering to handle function call/response pairs
  correctly, ensuring they are never split during history trimming.
  """

  def __init__(
      self,
      num_invocations_to_keep: Optional[int] = None,
      custom_filter: Optional[
          Callable[[List[types.Content]], List[types.Content]]
      ] = None,
      name: str = "tool_aware_context_filter_plugin",
  ):
    """Initializes the tool-aware context filter plugin.

    Args:
      num_invocations_to_keep: The number of last invocations to keep. An
        invocation is defined as a complete user-model interaction cycle,
        including any tool calls and their responses.
      custom_filter: A function to apply additional filtering to the context.
      name: The name of the plugin instance.
    """
    super().__init__(name)
    self._num_invocations_to_keep = num_invocations_to_keep
    self._custom_filter = custom_filter

  @staticmethod
  def _has_function_call(content) -> bool:
    """Check if a content has a function_call part."""
    if not content.parts:
      return False
    return any(
        hasattr(part, "function_call") and part.function_call
        for part in content.parts
    )

  @staticmethod
  def _has_function_response(content) -> bool:
    """Check if a content has a function_response part."""
    if not content.parts:
      return False
    return any(
        hasattr(part, "function_response") and part.function_response
        for part in content.parts
    )

  def _finalize_invocation_if_complete(
      self,
      current_invocation: List[int],
      invocations: List[List[int]],
      contents: List[types.Content],
  ) -> List[int]:
    """Finalize current invocation if it has a model response.

    Args:
      current_invocation: Current invocation being built.
      invocations: List of completed invocations.
      contents: List of message contents.

    Returns:
      Empty list if invocation was finalized, otherwise current_invocation.
    """
    if current_invocation:
      has_model = any(
          contents[idx].role == "model" for idx in current_invocation
      )
      if has_model:
        invocations.append(current_invocation)
        return []
    return current_invocation

  def _process_user_message(
      self,
      i: int,
      contents: List[types.Content],
      current_invocation: List[int],
      invocations: List[List[int]],
  ) -> tuple[int, List[int]]:
    """Process a user message and update invocation tracking.

    Args:
      i: Current index in contents.
      contents: List of message contents.
      current_invocation: Current invocation being built.
      invocations: List of completed invocations.

    Returns:
      Tuple of (next_index, updated_current_invocation).
    """
    content = contents[i]

    # Check if this is a function_response (part of ongoing tool cycle)
    if self._has_function_response(content):
      # This is a tool response - must be part of current invocation
      current_invocation.append(i)
      return i + 1, current_invocation

    # Regular user message (not a function_response)
    # Only start a NEW invocation if we've completed a previous one
    current_invocation = self._finalize_invocation_if_complete(
        current_invocation, invocations, contents
    )

    # Add this user message to current invocation
    current_invocation.append(i)
    return i + 1, current_invocation

  def _process_model_message_with_tool_call(
      self,
      i: int,
      contents: List[types.Content],
      current_invocation: List[int],
      invocations: List[List[int]],
  ) -> tuple[int, List[int]]:
    """Process a model message with tool call and collect the full cycle.

    Args:
      i: Current index in contents (at model message with function_call).
      contents: List of message contents.
      current_invocation: Current invocation being built.
      invocations: List of completed invocations.

    Returns:
      Tuple of (next_index, empty_invocation_list).
    """
    # Model made a tool call - keep following messages together:
    # 1. This model message (function_call) - already added
    # 2. User message(s) with function_response - collect next
    # 3. Model's final response - collect after tool responses

    i += 1  # Move to next message

    # Collect all function_response messages (usually 1, but could be multiple)
    while (
        i < len(contents)
        and contents[i].role == "user"
        and self._has_function_response(contents[i])
    ):
      current_invocation.append(i)
      i += 1

    # Now collect the model's final response after processing tool results
    if i < len(contents) and contents[i].role == "model":
      current_invocation.append(i)
      i += 1

    # Complete tool cycle collected - this is ONE complete invocation
    invocations.append(current_invocation)
    return i, []

  def _process_model_message(
      self,
      i: int,
      contents: List[types.Content],
      current_invocation: List[int],
      invocations: List[List[int]],
  ) -> tuple[int, List[int]]:
    """Process a model message and update invocation tracking.

    Args:
      i: Current index in contents.
      contents: List of message contents.
      current_invocation: Current invocation being built.
      invocations: List of completed invocations.

    Returns:
      Tuple of (next_index, updated_current_invocation).
    """
    content = contents[i]
    current_invocation.append(i)

    # Check if model is making a tool call
    if self._has_function_call(content):
      return self._process_model_message_with_tool_call(
          i, contents, current_invocation, invocations
      )

    # Model response WITHOUT function call - simple case
    # The invocation is complete (user query → model answer)
    invocations.append(current_invocation)
    return i + 1, []

  def _group_into_invocations(
      self, contents: List[types.Content]
  ) -> List[List[int]]:
    """Group message indices into complete invocations.

    An invocation pattern:
    1. One or more user messages (including consecutive user messages)
    2. Model response (possibly with function_call)
    3. If function_call exists: user message(s) with function_response
    4. If function_call exists: model final response

    Example grouping:
      Messages: [user, user, model, user, model+func_call, user+func_response,
      model] Groups:   [0,1,2] [3,4,5,6]
                ^^^^^^^ ^^^^^^^^^^^
                Inv 1   Inv 2 (includes tool cycle)

    Args:
      contents: List of message contents to group.

    Returns:
      List of invocations, where each invocation is a list of message indices.
    """
    invocations = []
    current_invocation = []
    i = 0

    while i < len(contents):
      content = contents[i]

      if content.role == "user":
        i, current_invocation = self._process_user_message(
            i, contents, current_invocation, invocations
        )
      elif content.role == "model":
        i, current_invocation = self._process_model_message(
            i, contents, current_invocation, invocations
        )
      else:
        # Unknown role - just add to current invocation
        current_invocation.append(i)
        i += 1

    # Add any remaining messages as final invocation
    if current_invocation:
      invocations.append(current_invocation)

    return invocations

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Filters the LLM request's context before it is sent to the model.

    This method groups messages into logical invocations and keeps only the
    most recent N invocations, ensuring tool call sequences remain intact.

    Args:
      callback_context: Context containing invocation and agent information.
      llm_request: The LLM request to filter.

    Returns:
      None - the request is modified in place.
    """
    try:
      contents = llm_request.contents

      if not contents:
        return None

      # Apply invocation-based filtering if configured
      if (
          self._num_invocations_to_keep is not None
          and self._num_invocations_to_keep > 0
      ):
        # Group messages into logical invocations
        invocations = self._group_into_invocations(contents)

        logger.info(
            "ToolAwareContextFilter: Total invocations=%d, keeping last %d",
            len(invocations),
            self._num_invocations_to_keep,
        )

        # Keep only the last N invocations
        if len(invocations) > self._num_invocations_to_keep:
          invocations_to_keep = invocations[-self._num_invocations_to_keep :]

          # Flatten the list of indices
          indices_to_keep = []
          for invocation in invocations_to_keep:
            indices_to_keep.extend(invocation)

          # Filter contents based on indices
          filtered_contents = [contents[i] for i in indices_to_keep]

          logger.info(
              "ToolAwareContextFilter: Reduced from %d messages to %d messages"
              " (kept %d invocations)",
              len(contents),
              len(filtered_contents),
              len(invocations_to_keep),
          )

          contents = filtered_contents

      # Apply custom filter if provided
      if self._custom_filter:
        contents = self._custom_filter(contents)

      llm_request.contents = contents

    except Exception as e:
      logger.error("ToolAwareContextFilter: Failed to filter context: %s", e)

    return None