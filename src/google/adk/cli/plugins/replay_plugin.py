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
from ...plugins.base_plugin import BasePlugin
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
  # key: agent_name -> current tool replay index for that agent
  agent_tool_replay_indices: dict[str, int] = Field(default_factory=dict)


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
    streaming_mode = config.get("streaming_mode")

    if not case_dir or msg_index is None:
      raise ReplayConfigError(
          "Replay parameters are missing from session state"
      )

    # Load recordings
    if streaming_mode == "sse":
      recordings_file = Path(case_dir) / "generated-recordings-sse.yaml"
    elif streaming_mode == "none":
      recordings_file = Path(case_dir) / "generated-recordings.yaml"
    else:
      raise ValueError(f"Unsupported streaming mode: {streaming_mode}")

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

    # Store recordings in session state for BaseLlmFlow to access
    config["_adk_replay_recordings"] = recordings

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

  def _get_next_tool_recording_for_agent(
      self,
      state: _InvocationReplayState,
      agent_name: str,
  ) -> ToolRecording:
    """Get the next tool recording for the specific agent."""
    # Get current agent index
    current_agent_index = state.agent_tool_replay_indices.get(agent_name, 0)

    # Filter tool recordings for this agent and user message index
    agent_recordings = [
        recording.tool_recording
        for recording in state.recordings.recordings
        if (
            recording.agent_name == agent_name
            and recording.user_message_index == state.user_message_index
            and recording.tool_recording
        )
    ]

    # Check if we have enough recordings for this agent
    if current_agent_index >= len(agent_recordings):
      raise ReplayVerificationError(
          "Runtime sent more tool requests than expected for agent"
          f" '{agent_name}' at user_message_index {state.user_message_index}."
          f" Expected {len(agent_recordings)}, but got request at index"
          f" {current_agent_index}"
      )

    # Get the expected recording
    expected_recording = agent_recordings[current_agent_index]

    # Advance agent index
    state.agent_tool_replay_indices[agent_name] = current_agent_index + 1

    return expected_recording

  def _verify_and_get_next_tool_recording_for_agent(
      self,
      state: _InvocationReplayState,
      agent_name: str,
      tool_name: str,
      tool_args: dict[str, Any],
  ) -> ToolRecording:
    """Verify and get the next tool recording for the specific agent."""
    current_agent_index = state.agent_tool_replay_indices.get(agent_name, 0)
    expected_recording = self._get_next_tool_recording_for_agent(
        state, agent_name
    )

    # Strict verification of tool call
    self._verify_tool_call_match(
        expected_recording.tool_call,
        tool_name,
        tool_args,
        agent_name,
        current_agent_index,
    )

    return expected_recording

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
