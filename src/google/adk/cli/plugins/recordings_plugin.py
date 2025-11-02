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

"""Recording plugin for ADK conformance testing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override
import yaml

from ...agents.callback_context import CallbackContext
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...plugins.base_plugin import BasePlugin
from ...utils.yaml_utils import dump_pydantic_to_yaml
from .recordings_schema import LlmRecording
from .recordings_schema import Recording
from .recordings_schema import Recordings
from .recordings_schema import ToolRecording

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...tools.base_tool import BaseTool
  from ...tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class _InvocationRecordingState(BaseModel):
  """Per-invocation recording state to isolate concurrent runs."""

  test_case_path: str
  user_message_index: int
  records: Recordings

  # Track pending recordings per agent/call
  # key: agent_name
  pending_llm_recordings: dict[str, Recording] = Field(default_factory=dict)
  # key: function_call_id
  pending_tool_recordings: dict[str, Recording] = Field(default_factory=dict)

  # Ordered list of pending recordings to maintain chronological order
  pending_recordings_order: list[Recording] = Field(default_factory=list)


class RecordingsPlugin(BasePlugin):
  """Plugin for recording ADK agent interactions."""

  def __init__(self, *, name: str = "adk_recordings") -> None:
    super().__init__(name=name)

    # Track recording state per invocation to support concurrent runs
    # key: invocation_id -> _InvocationRecordingState
    self._invocation_states: dict[str, _InvocationRecordingState] = {}

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Always create fresh per-invocation recording state when enabled."""
    ctx = CallbackContext(invocation_context)
    if self._is_record_mode_on(ctx):
      # Always create/overwrite the state for this invocation
      self._create_invocation_state(ctx)
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Create pending LLM recording awaiting response.

    Uses per-invocation recording state. Assumes state was created in
    before_run; raises if missing to surface misuse.
    """
    if not self._is_record_mode_on(callback_context):
      return None

    if (state := self._get_invocation_state(callback_context)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    pending_recording = Recording(
        user_message_index=state.user_message_index,
        agent_name=callback_context.agent_name,
        llm_recording=LlmRecording(
            llm_request=llm_request,
            llm_response=None,
        ),
    )

    # Store in both lookup dict and chronological list
    state.pending_llm_recordings[callback_context.agent_name] = (
        pending_recording
    )
    state.pending_recordings_order.append(pending_recording)

    logger.debug(
        "Created pending LLM recording for agent %s: model=%s, contents=%d",
        callback_context.agent_name,
        llm_request.model,
        len(llm_request.contents),
    )

    return None  # Continue LLM execution

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Complete pending LLM recording for the invocation specified in session state."""
    if not self._is_record_mode_on(callback_context):
      return None

    if (state := self._get_invocation_state(callback_context)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    agent_name = callback_context.agent_name
    if pending_recording := state.pending_llm_recordings.pop(agent_name, None):
      if pending_recording.llm_recording is not None:
        pending_recording.llm_recording.llm_response = llm_response
        logger.debug("Completed LLM recording for agent %s", agent_name)
    else:
      logger.warning(
          "No pending LLM recording found for agent %s, skipping response",
          agent_name,
      )

    return None  # Continue LLM execution

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Create pending tool recording for the invocation specified in session state."""
    if not self._is_record_mode_on(tool_context):
      return None

    if not (function_call_id := tool_context.function_call_id):
      logger.warning(
          "No function_call_id provided for tool %s, skipping recording",
          tool.name,
      )
      return None  # Continue tool execution

    if (state := self._get_invocation_state(tool_context)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    pending_recording = Recording(
        user_message_index=state.user_message_index,
        agent_name=tool_context.agent_name,
        tool_recording=ToolRecording(
            tool_call=types.FunctionCall(
                id=function_call_id, name=tool.name, args=tool_args
            ),
            tool_response=None,
        ),
    )

    # Store in both lookup dict and chronological list
    state.pending_tool_recordings[function_call_id] = pending_recording
    state.pending_recordings_order.append(pending_recording)

    logger.debug(
        "Created pending tool recording for agent %s: tool=%s, id=%s",
        tool_context.agent_name,
        tool.name,
        function_call_id,
    )

    return None  # Continue tool execution

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Complete pending tool recording for the invocation specified in session state."""
    if not self._is_record_mode_on(tool_context):
      return None

    if not (function_call_id := tool_context.function_call_id):
      logger.warning(
          "No function_call_id provided for tool %s result, skipping"
          " completion",
          tool.name,
      )
      return None  # Continue tool execution

    if (state := self._get_invocation_state(tool_context)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    if pending_recording := state.pending_tool_recordings.pop(
        function_call_id, None
    ):
      if pending_recording.tool_recording is not None:
        pending_recording.tool_recording.tool_response = types.FunctionResponse(
            id=function_call_id,
            name=tool.name,
            response=result if isinstance(result, dict) else {"result": result},
        )
      logger.debug(
          "Completed tool recording for agent %s: tool=%s, id=%s",
          pending_recording.agent_name,
          tool.name,
          function_call_id,
      )
    else:
      logger.warning(
          "No pending tool recording found for id %s, skipping result",
          function_call_id,
      )

    return None  # Continue tool execution

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    """Handle tool error callback with state guard.

    Recording schema does not yet capture errors; we only validate state.
    """
    if not self._is_record_mode_on(tool_context):
      return None

    if (state := self._get_invocation_state(tool_context)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    logger.debug(
        "Tool error occurred for agent %s: tool=%s, id=%s, error=%s",
        tool_context.agent_name,
        tool.name,
        tool_context.function_call_id,
        str(error),
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Finalize and persist recordings, then clean per-invocation state."""
    ctx = CallbackContext(invocation_context)
    if not self._is_record_mode_on(ctx):
      return None

    if (state := self._get_invocation_state(ctx)) is None:
      raise ValueError(
          "Recording state not initialized. Ensure before_run created it."
      )

    try:
      for pending in state.pending_recordings_order:
        if pending.llm_recording is not None:
          if pending.llm_recording.llm_response is not None:
            state.records.recordings.append(pending)
          else:
            logger.warning(
                "Incomplete LLM recording for agent %s, skipping",
                pending.agent_name,
            )
        elif pending.tool_recording is not None:
          if pending.tool_recording.tool_response is not None:
            state.records.recordings.append(pending)
          else:
            logger.warning(
                "Incomplete tool recording for agent %s, skipping",
                pending.agent_name,
            )

      dump_pydantic_to_yaml(
          state.records,
          f"{state.test_case_path}/generated-recordings.yaml",
          sort_keys=False,
      )
      logger.info(
          "Saved %d recordings to %s/generated-recordings.yaml",
          len(state.records.recordings),
          state.test_case_path,
      )
    except Exception as e:
      logger.error("Failed to save interactions: %s", e)
    finally:
      # Cleanup per-invocation recording state
      self._invocation_states.pop(ctx.invocation_id, None)

  # Private helpers (placed after public callbacks)
  def _is_record_mode_on(self, callback_context: CallbackContext) -> bool:
    """Check if recording mode is enabled for this invocation.

    Args:
      callback_context: The callback context containing session state.

    Returns:
      True if recording mode is enabled, False otherwise.
    """
    # TODO: Investigate how to support with `temp:` states.
    session_state = callback_context.state
    if not (config := session_state.get("_adk_recordings_config")):
      return False

    case_dir = config.get("dir")
    msg_index = config.get("user_message_index")

    return case_dir and msg_index is not None

  def _get_invocation_state(
      self, callback_context: CallbackContext
  ) -> Optional[_InvocationRecordingState]:
    """Get existing recording state for this invocation."""
    invocation_id = callback_context.invocation_id
    return self._invocation_states.get(invocation_id)

  def _create_invocation_state(
      self, callback_context: CallbackContext
  ) -> _InvocationRecordingState:
    """Create and store recording state for this invocation."""
    invocation_id = callback_context.invocation_id
    session_state = callback_context.state

    config = session_state.get("_adk_recordings_config", {})
    case_dir = config.get("dir")
    msg_index = config.get("user_message_index")

    if not case_dir or msg_index is None:
      raise ValueError("Recording parameters are missing from session state")

    # Load or create recordings
    recordings_file = Path(case_dir) / "generated-recordings.yaml"

    if recordings_file.exists():
      try:
        with recordings_file.open("r", encoding="utf-8") as f:
          recordings_data = yaml.safe_load(f)
        records = Recordings.model_validate(recordings_data)
      except Exception as e:
        logger.error(
            "Failed to load recordings from %s: %s", recordings_file, e
        )
        records = Recordings(recordings=[])
    else:
      records = Recordings(recordings=[])

    # Create and store invocation state
    state = _InvocationRecordingState(
        test_case_path=case_dir,
        user_message_index=msg_index,
        records=records,
    )
    self._invocation_states[invocation_id] = state
    logger.debug(
        "Created recording state for invocation %s: case_dir=%s, msg_index=%s",
        invocation_id,
        case_dir,
        msg_index,
    )
    return state
