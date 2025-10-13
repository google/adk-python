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

"""Replay plugin for ADK conformance testing."""

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
from .recordings_schema import LlmRecording
from .recordings_schema import Recording
from .recordings_schema import Recordings
from .recordings_schema import ToolRecording

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...tools.base_tool import BaseTool
  from ...tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class ReplayVerificationError(Exception):
  """Exception raised when replay verification fails."""

  pass


class ReplayConfigError(Exception):
  """Exception raised when replay configuration is invalid or missing."""

  pass


class _InvocationReplayState(BaseModel):
  """Per-invocation replay state to isolate concurrent runs."""

  test_case_path: str
  user_message_index: int
  recordings: Recordings

  # Per-agent replay indices for parallel execution
  # key: agent_name -> current replay index for that agent
  agent_replay_indices: dict[str, int] = Field(default_factory=dict)


class ReplayPlugin(BasePlugin):
  """Plugin for replaying ADK agent interactions from recordings."""

  def __init__(self, *, name: str = "adk_replay") -> None:
    super().__init__(name=name)

    # Track replay state per invocation to support concurrent runs
    # key: invocation_id -> _InvocationReplayState
    self._invocation_states: dict[str, _InvocationReplayState] = {}

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Load replay recordings when enabled."""
    ctx = CallbackContext(invocation_context)
    if self._is_replay_mode_on(ctx):
      # Load the replay state for this invocation
      self._load_invocation_state(ctx)
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Replay LLM response from recordings instead of making real call."""
    if not self._is_replay_mode_on(callback_context):
      return None

    if (state := self._get_invocation_state(callback_context)) is None:
      raise ReplayConfigError(
          "Replay state not initialized. Ensure before_run created it."
      )

    agent_name = callback_context.agent_name

    # Verify and get the next LLM recording for this specific agent
    recording = self._verify_and_get_next_llm_recording_for_agent(
        state, agent_name, llm_request
    )

    logger.debug("Verified and replaying LLM response for agent %s", agent_name)

    # Return the recorded response
    return recording.llm_response

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Replay tool response from recordings instead of executing tool."""
    if not self._is_replay_mode_on(tool_context):
      return None

    if (state := self._get_invocation_state(tool_context)) is None:
      raise ReplayConfigError(
          "Replay state not initialized. Ensure before_run created it."
      )

    agent_name = tool_context.agent_name

    # Verify and get the next tool recording for this specific agent
    recording = self._verify_and_get_next_tool_recording_for_agent(
        state, agent_name, tool.name, tool_args
    )

    from google.adk.tools.agent_tool import AgentTool

    if not isinstance(tool, AgentTool):
      # TODO: support replay requests and responses from AgentTool.
      await tool.run_async(args=tool_args, tool_context=tool_context)

    logger.debug(
        "Verified and replaying tool response for agent %s: tool=%s",
        agent_name,
        tool.name,
    )

    # Return the recorded response
    return recording.tool_response.response

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Clean up replay state after invocation completes."""
    ctx = CallbackContext(invocation_context)
    if not self._is_replay_mode_on(ctx):
      return None

    # Clean up per-invocation replay state
    self._invocation_states.pop(ctx.invocation_id, None)
    logger.debug("Cleaned up replay state for invocation %s", ctx.invocation_id)

  # Private helpers
  def _is_replay_mode_on(self, callback_context: CallbackContext) -> bool:
    """Check if replay mode is enabled for this invocation."""
    session_state = callback_context.state
    if not (config := session_state.get("_adk_replay_config")):
      return False

    case_dir = config.get("dir")
    msg_index = config.get("user_message_index")

    return case_dir and msg_index is not None

  def _get_invocation_state(
      self, callback_context: CallbackContext
  ) -> Optional[_InvocationReplayState]:
    """Get existing replay state for this invocation."""
    invocation_id = callback_context.invocation_id
    return self._invocation_states.get(invocation_id)

  def _load_invocation_state(
      self, callback_context: CallbackContext
  ) -> _InvocationReplayState:
    """Load and store replay state for this invocation."""
    invocation_id = callback_context.invocation_id
    session_state = callback_context.state

    config = session_state.get("_adk_replay_config", {})
    case_dir = config.get("dir")
    msg_index = config.get("user_message_index")

    if not case_dir or msg_index is None:
      raise ReplayConfigError(
          "Replay parameters are missing from session state"
      )

    # Load recordings
    recordings_file = Path(case_dir) / "generated-recordings.yaml"

    if not recordings_file.exists():
      raise ReplayConfigError(f"Recordings file not found: {recordings_file}")

    try:
      with recordings_file.open("r", encoding="utf-8") as f:
        recordings_data = yaml.safe_load(f)
      recordings = Recordings.model_validate(recordings_data)
    except Exception as e:
      raise ReplayConfigError(
          f"Failed to load recordings from {recordings_file}: {e}"
      ) from e

    # Load and store invocation state
    state = _InvocationReplayState(
        test_case_path=case_dir,
        user_message_index=msg_index,
        recordings=recordings,
    )
    self._invocation_states[invocation_id] = state
    logger.debug(
        "Loaded replay state for invocation %s: case_dir=%s, msg_index=%s, "
        "recordings=%d",
        invocation_id,
        case_dir,
        msg_index,
        len(recordings.recordings),
    )
    return state

  def _get_next_recording_for_agent(
      self,
      state: _InvocationReplayState,
      agent_name: str,
  ) -> Recording:
    """Get the next recording for the specific agent in strict order."""
    # Get current agent index
    current_agent_index = state.agent_replay_indices.get(agent_name, 0)

    # Filter ALL recordings for this agent and user message index (strict order)
    agent_recordings = [
        recording
        for recording in state.recordings.recordings
        if (
            recording.agent_name == agent_name
            and recording.user_message_index == state.user_message_index
        )
    ]

    # Check if we have enough recordings for this agent
    if current_agent_index >= len(agent_recordings):
      raise ReplayVerificationError(
          f"Runtime sent more requests than expected for agent '{agent_name}'"
          f" at user_message_index {state.user_message_index}. Expected"
          f" {len(agent_recordings)}, but got request at index"
          f" {current_agent_index}"
      )

    # Get the expected recording
    expected_recording = agent_recordings[current_agent_index]

    # Advance agent index
    state.agent_replay_indices[agent_name] = current_agent_index + 1

    return expected_recording

  def _verify_and_get_next_llm_recording_for_agent(
      self,
      state: _InvocationReplayState,
      agent_name: str,
      llm_request: LlmRequest,
  ) -> LlmRecording:
    """Verify and get the next LLM recording for the specific agent."""
    current_agent_index = state.agent_replay_indices.get(agent_name, 0)
    expected_recording = self._get_next_recording_for_agent(state, agent_name)

    # Verify this is an LLM recording
    if not expected_recording.llm_recording:
      raise ReplayVerificationError(
          f"Expected LLM recording for agent '{agent_name}' at index "
          f"{current_agent_index}, but found tool recording"
      )

    # Strict verification of LLM request
    self._verify_llm_request_match(
        expected_recording.llm_recording.llm_request,
        llm_request,
        agent_name,
        current_agent_index,
    )

    return expected_recording.llm_recording

  def _verify_and_get_next_tool_recording_for_agent(
      self,
      state: _InvocationReplayState,
      agent_name: str,
      tool_name: str,
      tool_args: dict[str, Any],
  ) -> ToolRecording:
    """Verify and get the next tool recording for the specific agent."""
    current_agent_index = state.agent_replay_indices.get(agent_name, 0)
    expected_recording = self._get_next_recording_for_agent(state, agent_name)

    # Verify this is a tool recording
    if not expected_recording.tool_recording:
      raise ReplayVerificationError(
          f"Expected tool recording for agent '{agent_name}' at index "
          f"{current_agent_index}, but found LLM recording"
      )

    # Strict verification of tool call
    self._verify_tool_call_match(
        expected_recording.tool_recording.tool_call,
        tool_name,
        tool_args,
        agent_name,
        current_agent_index,
    )

    return expected_recording.tool_recording

  def _verify_llm_request_match(
      self,
      recorded_request: LlmRequest,
      current_request: LlmRequest,
      agent_name: str,
      agent_index: int,
  ) -> None:
    """Verify that the current LLM request exactly matches the recorded one."""
    # Comprehensive exclude dict for all fields that can differ between runs
    excluded_fields = {
        "live_connect_config": True,
        "config": {  # some config fields can vary per run
            "http_options": True,
            "labels": True,
        },
    }

    # Compare using model dumps with nested exclude dict
    recorded_dict = recorded_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )
    current_dict = current_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )

    if recorded_dict != current_dict:
      raise ReplayVerificationError(
          f"""LLM request mismatch for agent '{agent_name}' (index {agent_index}):
recorded: {recorded_dict}
current: {current_dict}"""
      )

  def _verify_tool_call_match(
      self,
      recorded_call: types.FunctionCall,
      tool_name: str,
      tool_args: dict[str, Any],
      agent_name: str,
      agent_index: int,
  ) -> None:
    """Verify that the current tool call exactly matches the recorded one."""
    if recorded_call.name != tool_name:
      raise ReplayVerificationError(
          f"""Tool name mismatch for agent '{agent_name}' at index {agent_index}:
recorded: '{recorded_call.name}'
current: '{tool_name}'"""
      )

    if recorded_call.args != tool_args:
      raise ReplayVerificationError(
          f"""Tool args mismatch for agent '{agent_name}' at index {agent_index}:
recorded: {recorded_call.args}
current: {tool_args}"""
      )
