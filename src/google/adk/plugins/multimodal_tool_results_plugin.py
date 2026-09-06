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

from typing import Any
from typing import Literal
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin

PARTS_RETURNED_BY_TOOLS_ID = "temp:PARTS_RETURNED_BY_TOOLS_ID"
# Deliberately NOT "temp:"-prefixed: the session layer treats "temp:" state
# as invocation-scoped and strips it before persisting an event (see
# BaseSessionService._trim_temp_delta_state). retention="session" needs the
# saved parts to survive into later invocations (i.e. later conversational
# turns), so it is stored under this session-scoped key instead.
SESSION_PARTS_RETURNED_BY_TOOLS_ID = (
    "multimodal_tool_results_plugin:PARTS_RETURNED_BY_TOOLS_ID"
)
_CURRENT_TURN_PARTS_ID = (
    "temp:multimodal_tool_results_plugin:current_turn_parts"
)
_SESSION_UPDATED_KEY = (
    "temp:multimodal_tool_results_plugin:updated_in_invocation"
)


class MultimodalToolResultsPlugin(BasePlugin):
  """A plugin that modifies function tool responses to support returning list of parts directly.

  Should be removed in favor of directly supporting FunctionResponsePart when these
  are supported outside of computer use tool.
  """

  def __init__(
      self,
      name: str = "multimodal_tool_results_plugin",
      *,
      retention: Literal["next_model_call", "session"] = "next_model_call",
  ):
    """Initialize the multimodal tool results plugin.

    Args:
      name: The name of the plugin instance.
      retention: How long tool-returned parts stay attached to model
        requests. "next_model_call" (default) attaches the saved parts once
        and then clears them. "session" keeps re-attaching the latest saved
        parts to every subsequent model request for the rest of the
        session, so follow-up turns can still reference them. Note that only
        file_data and text parts are retained across turns (in session mode);
        inline_data parts (such as image or audio bytes) are always one-shot.

    Raises:
      ValueError: If retention is not 'next_model_call' or 'session'.
    """
    if retention not in ("next_model_call", "session"):
      raise ValueError(
          f"retention must be 'next_model_call' or 'session', got {retention}"
      )
    super().__init__(name)
    self._retention = retention

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> Optional[dict[str, Any]]:
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

    if self._retention == "session":
      session_parts = []
      for p in parts:
        if isinstance(p, types.Part) and p.inline_data is not None:
          pass
        else:
          session_parts.append(p)

      session_key = SESSION_PARTS_RETURNED_BY_TOOLS_ID
      updated_key = _SESSION_UPDATED_KEY

      if session_parts:
        serialized_session_parts = [
            p.model_dump(mode="json") if isinstance(p, types.Part) else p
            for p in session_parts
        ]
        if updated_key in tool_context.state:
          tool_context.state[session_key] += serialized_session_parts
        else:
          tool_context.state[updated_key] = True
          tool_context.state[session_key] = serialized_session_parts

      # Accumulate ALL parts of current turn in order
      current_turn_key = _CURRENT_TURN_PARTS_ID
      if current_turn_key in tool_context.state:
        tool_context.state[current_turn_key] += parts
      else:
        tool_context.state[current_turn_key] = parts
    else:
      if PARTS_RETURNED_BY_TOOLS_ID in tool_context.state:
        tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] += parts
      else:
        tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] = parts

    return None

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Attach saved list[google.genai.types.Part] returned by the tool to llm_request."""

    if not llm_request.contents:
      return None

    if self._retention == "session":
      session_key = SESSION_PARTS_RETURNED_BY_TOOLS_ID
      current_turn_key = _CURRENT_TURN_PARTS_ID

      session_parts = []
      if saved_parts := callback_context.state.get(session_key, None):
        for p in saved_parts:
          if isinstance(p, dict):
            session_parts.append(types.Part.model_validate(p))
          else:
            session_parts.append(p)

      current_parts = []
      if current_turn_key in callback_context.state:
        if parts := callback_context.state.get(current_turn_key, None):
          current_parts = parts

      # Skip session parts that are already in current_parts to avoid duplication.
      filtered_session_parts = [
          p for p in session_parts if p not in current_parts
      ]

      parts_to_attach = filtered_session_parts + current_parts

      if current_parts:
        callback_context.state.update({current_turn_key: []})

      if parts_to_attach:
        llm_request.contents[-1].parts += parts_to_attach

    else:
      # Default mode
      temp_key = PARTS_RETURNED_BY_TOOLS_ID
      if temp_parts := callback_context.state.get(temp_key, None):
        llm_request.contents[-1].parts += temp_parts
        callback_context.state.update({temp_key: []})

    return None
