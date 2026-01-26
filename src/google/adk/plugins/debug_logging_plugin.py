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

"""Debug logging plugin for capturing complete interaction data to a file."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override
import yaml

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class _DebugEntry(BaseModel):
  """A single debug log entry."""

  timestamp: str
  entry_type: str
  invocation_id: str | None = None
  agent_name: str | None = None
  data: dict[str, Any] = Field(default_factory=dict)


class _InvocationDebugState(BaseModel):
  """Per-invocation debug state."""

  invocation_id: str
  session_id: str
  app_name: str
  user_id: str | None = None
  start_time: str
  entries: list[_DebugEntry] = Field(default_factory=list)


class DebugLoggingPlugin(BasePlugin):
  """A plugin that captures complete debug information to a file.

  This plugin records detailed interaction data including:
  - LLM requests (model, system instruction, contents, tools)
  - LLM responses (content, usage metadata, errors)
  - Function calls with arguments
  - Function responses with results
  - Events yielded from the runner
  - Session state at the end of each invocation

  The output is written as YAML format for human readability. Each invocation
  is appended to the file as a separate YAML document (separated by ---).
  This format is easy to read and can be shared for debugging purposes.

  Example:
      >>> debug_plugin = DebugLoggingPlugin(output_path="/tmp/adk_debug.yaml")
      >>> runner = Runner(
      ...     agent=my_agent,
      ...     plugins=[debug_plugin],
      ... )

  Attributes:
      output_path: Path to the output file. Defaults to "adk_debug.yaml".
      include_session_state: Whether to include session state in the output.
      include_system_instruction: Whether to include system instructions.
  """

  def __init__(
      self,
      *,
      name: str = "debug_logging_plugin",
      output_path: str = "adk_debug.yaml",
      include_session_state: bool = True,
      include_system_instruction: bool = True,
  ):
    """Initialize the debug logging plugin.

    Args:
      name: The name of the plugin instance.
      output_path: Path to the output file. Defaults to "adk_debug.yaml".
      include_session_state: Whether to include session state snapshot.
      include_system_instruction: Whether to include full system instructions.
    """
    super().__init__(name)
    self._output_path = Path(output_path)
    self._include_session_state = include_session_state
    self._include_system_instruction = include_system_instruction
    self._invocation_states: dict[str, _InvocationDebugState] = {}

  def _get_timestamp(self) -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

  def _serialize_content(
      self, content: types.Content | None
  ) -> dict[str, Any] | None:
    """Serialize Content to a dictionary."""
    if content is None:
      return None

    parts = []
    if content.parts:
      for part in content.parts:
        part_data: dict[str, Any] = {}
        if part.text:
          part_data["text"] = part.text
        if part.function_call:
          part_data["function_call"] = {
              "id": part.function_call.id,
              "name": part.function_call.name,
              "args": part.function_call.args,
          }
        if part.function_response:
          part_data["function_response"] = {
              "id": part.function_response.id,
              "name": part.function_response.name,
              "response": self._safe_serialize(part.function_response.response),
          }
        if part.inline_data:
          part_data["inline_data"] = {
              "mime_type": part.inline_data.mime_type,
              "display_name": getattr(part.inline_data, "display_name", None),
              # Omit actual data to keep file size manageable
              "_data_omitted": True,
          }
        if part.file_data:
          part_data["file_data"] = {
              "file_uri": part.file_data.file_uri,
              "mime_type": part.file_data.mime_type,
          }
        if part.code_execution_result:
          part_data["code_execution_result"] = {
              "outcome": str(part.code_execution_result.outcome),
              "output": part.code_execution_result.output,
          }
        if part.executable_code:
          part_data["executable_code"] = {
              "language": str(part.executable_code.language),
              "code": part.executable_code.code,
          }
        if part_data:
          parts.append(part_data)

    return {"role": content.role, "parts": parts}

  def _safe_serialize(self, obj: Any) -> Any:
    """Safely serialize an object to JSON-compatible format."""
    if obj is None:
      return None
    if isinstance(obj, (str, int, float, bool)):
      return obj
    if isinstance(obj, (list, tuple)):
      return [self._safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
      return {k: self._safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
      try:
        return obj.model_dump(mode="json", exclude_none=True)
      except Exception:
        return str(obj)
    if isinstance(obj, bytes):
      return f"<bytes: {len(obj)} bytes>"
    try:
      return str(obj)
    except Exception:
      return "<unserializable>"

  def _add_entry(
      self,
      invocation_id: str,
      entry_type: str,
      agent_name: str | None = None,
      **data: Any,
  ) -> None:
    """Add a debug entry to the current invocation state."""
    if invocation_id not in self._invocation_states:
      logger.warning(
          "No debug state for invocation %s, skipping entry", invocation_id
      )
      return

    entry = _DebugEntry(
        timestamp=self._get_timestamp(),
        entry_type=entry_type,
        invocation_id=invocation_id,
        agent_name=agent_name,
        data=self._safe_serialize(data),
    )
    self._invocation_states[invocation_id].entries.append(entry)

  @override
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> types.Content | None:
    """Log user message and invocation start."""
    invocation_id = invocation_context.invocation_id

    self._add_entry(
        invocation_id,
        "user_message",
        content=self._serialize_content(user_message),
    )
    return None

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> types.Content | None:
    """Initialize debug state for this invocation."""
    invocation_id = invocation_context.invocation_id
    session = invocation_context.session

    state = _InvocationDebugState(
        invocation_id=invocation_id,
        session_id=session.id,
        app_name=session.app_name,
        user_id=invocation_context.user_id,
        start_time=self._get_timestamp(),
    )
    self._invocation_states[invocation_id] = state

    self._add_entry(
        invocation_id,
        "invocation_start",
        agent_name=getattr(invocation_context.agent, "name", None),
        branch=invocation_context.branch,
    )
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Event | None:
    """Log events yielded from the runner."""
    invocation_id = invocation_context.invocation_id

    event_data: dict[str, Any] = {
        "event_id": event.id,
        "author": event.author,
        "content": self._serialize_content(event.content),
        "is_final_response": event.is_final_response(),
        "partial": event.partial,
        "turn_complete": event.turn_complete,
        "branch": event.branch,
    }

    if event.actions:
      actions_data: dict[str, Any] = {}
      if event.actions.state_delta:
        actions_data["state_delta"] = self._safe_serialize(
            event.actions.state_delta
        )
      if event.actions.artifact_delta:
        # Preserve filename -> version mapping for debugging
        actions_data["artifact_delta"] = dict(event.actions.artifact_delta)
      if event.actions.transfer_to_agent:
        actions_data["transfer_to_agent"] = event.actions.transfer_to_agent
      if event.actions.escalate:
        actions_data["escalate"] = event.actions.escalate
      if event.actions.requested_auth_configs:
        actions_data["requested_auth_configs"] = len(
            event.actions.requested_auth_configs
        )
      if actions_data:
        event_data["actions"] = actions_data

    if event.grounding_metadata:
      event_data["has_grounding_metadata"] = True

    if event.usage_metadata:
      event_data["usage_metadata"] = {
          "prompt_token_count": event.usage_metadata.prompt_token_count,
          "candidates_token_count": event.usage_metadata.candidates_token_count,
          "total_token_count": event.usage_metadata.total_token_count,
      }

    if event.error_code:
      event_data["error_code"] = event.error_code
      event_data["error_message"] = event.error_message

    if event.long_running_tool_ids:
      event_data["long_running_tool_ids"] = list(event.long_running_tool_ids)

    self._add_entry(
        invocation_id,
        "event",
        agent_name=event.author,
        **event_data,
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Finalize and write debug data to file."""
    invocation_id = invocation_context.invocation_id

    if invocation_id not in self._invocation_states:
      logger.warning(
          "No debug state for invocation %s, skipping write", invocation_id
      )
      return

    state = self._invocation_states[invocation_id]

    # Add session state snapshot if enabled
    if self._include_session_state:
      session = invocation_context.session
      self._add_entry(
          invocation_id,
          "session_state_snapshot",
          state=self._safe_serialize(session.state),
          event_count=len(session.events),
      )

    self._add_entry(invocation_id, "invocation_end")

    # Write to file as YAML
    try:
      output_data = state.model_dump(mode="json", exclude_none=True)
      with self._output_path.open("a", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(
            output_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
      logger.debug(
          "Wrote debug data for invocation %s to %s",
          invocation_id,
          self._output_path,
      )
    except Exception as e:
      logger.error("Failed to write debug data: %s", e)
    finally:
      # Cleanup invocation state
      self._invocation_states.pop(invocation_id, None)

  @override
  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    """Log agent execution start."""
    self._add_entry(
        callback_context.invocation_id,
        "agent_start",
        agent_name=callback_context.agent_name,
        branch=callback_context._invocation_context.branch,
    )
    return None

  @override
  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    """Log agent execution completion."""
    self._add_entry(
        callback_context.invocation_id,
        "agent_end",
        agent_name=callback_context.agent_name,
    )
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    """Log LLM request before sending to model."""
    request_data: dict[str, Any] = {
        "model": llm_request.model,
        "content_count": len(llm_request.contents),
        "contents": [self._serialize_content(c) for c in llm_request.contents],
    }

    if llm_request.tools_dict:
      request_data["tools"] = list(llm_request.tools_dict.keys())

    if llm_request.config:
      config = llm_request.config
      config_data: dict[str, Any] = {}

      if self._include_system_instruction and config.system_instruction:
        config_data["system_instruction"] = config.system_instruction
      elif config.system_instruction:
        # Just indicate presence without full content
        si = config.system_instruction
        if isinstance(si, str):
          config_data["system_instruction_length"] = len(si)
        else:
          config_data["has_system_instruction"] = True

      if config.temperature is not None:
        config_data["temperature"] = config.temperature
      if config.top_p is not None:
        config_data["top_p"] = config.top_p
      if config.top_k is not None:
        config_data["top_k"] = config.top_k
      if config.max_output_tokens is not None:
        config_data["max_output_tokens"] = config.max_output_tokens
      if config.response_mime_type:
        config_data["response_mime_type"] = config.response_mime_type
      if config.response_schema:
        config_data["has_response_schema"] = True

      if config_data:
        request_data["config"] = config_data

    self._add_entry(
        callback_context.invocation_id,
        "llm_request",
        agent_name=callback_context.agent_name,
        **request_data,
    )
    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    """Log LLM response after receiving from model."""
    response_data: dict[str, Any] = {
        "content": self._serialize_content(llm_response.content),
        "partial": llm_response.partial,
        "turn_complete": llm_response.turn_complete,
    }

    if llm_response.error_code:
      response_data["error_code"] = llm_response.error_code
      response_data["error_message"] = llm_response.error_message

    if llm_response.usage_metadata:
      response_data["usage_metadata"] = {
          "prompt_token_count": llm_response.usage_metadata.prompt_token_count,
          "candidates_token_count": (
              llm_response.usage_metadata.candidates_token_count
          ),
          "total_token_count": llm_response.usage_metadata.total_token_count,
          "cached_content_token_count": (
              llm_response.usage_metadata.cached_content_token_count
          ),
      }

    if llm_response.grounding_metadata:
      response_data["has_grounding_metadata"] = True

    if llm_response.finish_reason:
      response_data["finish_reason"] = str(llm_response.finish_reason)

    if llm_response.model_version:
      response_data["model_version"] = llm_response.model_version

    self._add_entry(
        callback_context.invocation_id,
        "llm_response",
        agent_name=callback_context.agent_name,
        **response_data,
    )
    return None

  @override
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> LlmResponse | None:
    """Log LLM error."""
    self._add_entry(
        callback_context.invocation_id,
        "llm_error",
        agent_name=callback_context.agent_name,
        error_type=type(error).__name__,
        error_message=str(error),
        model=llm_request.model,
    )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    """Log tool execution start."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_call",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        args=self._safe_serialize(tool_args),
    )
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Log tool execution completion."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_response",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        result=self._safe_serialize(result),
    )
    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, Any] | None:
    """Log tool error."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_error",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        args=self._safe_serialize(tool_args),
        error_type=type(error).__name__,
        error_message=str(error),
    )
    return None
