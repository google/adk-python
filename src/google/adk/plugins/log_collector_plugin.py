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

import asyncio
from collections import defaultdict
from typing import Any, Optional, Dict, List, TYPE_CHECKING

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext


class LogCollectorPlugin(BasePlugin):
  """
  A plugin to programmatically and safely collect execution details from all
  callbacks in async environments, organized by session ID.
  """

  def __init__(self, name: str = "log_collector"):
    super().__init__(name)
    self.logs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    self._lock = asyncio.Lock()

  async def _log_entry(self, session_id: str, callback_type: str, data: Dict[str, Any]):
    log_entry = {"callback_type": callback_type, **data}
    async with self._lock:
      self.logs[session_id].append(log_entry)

  async def on_user_message_callback(
      self, *, invocation_context: "InvocationContext", user_message: types.Content
  ) -> Optional[types.Content]:
    session_id = invocation_context.session.id
    await self._log_entry(
        session_id,
        "on_user_message",
        {
            "invocation_id": invocation_context.invocation_id,
            "user_message": user_message.parts[0].text,
        },
    )
    return None

  async def before_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> Optional[types.Content]:
    session_id = invocation_context.session.id
    await self._log_entry(
        session_id,
        "before_run",
        {
            "invocation_id": invocation_context.invocation_id,
            "agent_name": invocation_context.agent.name,
        },
    )
    return None

  async def after_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    session_id = invocation_context.session.id
    await self._log_entry(
        session_id,
        "after_run",
        {
            "invocation_id": invocation_context.invocation_id,
            "agent_name": invocation_context.agent.name,
        },
    )
    return None

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    session_id = callback_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "before_agent",
        {
            "agent_name": agent.name,
            "invocation_id": callback_context.invocation_id,
        },
    )
    return None

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    session_id = callback_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "after_agent",
        {
            "agent_name": agent.name,
            "invocation_id": callback_context.invocation_id,
        },
    )
    return None

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    session_id = callback_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "before_model",
        {
            "agent_name": callback_context.agent_name,
            "request": llm_request.model_dump(),
        },
    )
    return None

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    session_id = callback_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "after_model",
        {
            "agent_name": callback_context.agent_name,
            "response": llm_response.model_dump(),
        },
    )
    return None

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    session_id = callback_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "on_model_error",
        {
            "agent_name": callback_context.agent_name,
            "request": llm_request.model_dump(),
            "error": str(error),
        },
    )
    return None

  async def on_event_callback(
      self, *, invocation_context: "InvocationContext", event: Event
  ) -> Optional[Event]:
    session_id = invocation_context.session.id
    await self._log_entry(
        session_id,
        "on_event",
        {
            "event_id": event.id,
            "author": event.author,
            "content": event.content.parts[0].text,
            "is_final": event.is_final_response(),
        },
    )
    return None

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: Dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[Dict]:
    session_id = tool_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "before_tool",
        {
            "tool_name": tool.name,
            "agent_name": tool_context.agent_name,
            "function_call_id": tool_context.function_call_id,
            "args": tool_args,
        },
    )
    return None

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: Dict[str, Any],
      tool_context: ToolContext,
      result: Dict,
  ) -> Optional[Dict]:
    session_id = tool_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "after_tool",
        {
            "tool_name": tool.name,
            "agent_name": tool_context.agent_name,
            "function_call_id": tool_context.function_call_id,
            "args": tool_args,
            "result": result,
        },
    )
    return None

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: Dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[Dict]:
    session_id = tool_context._invocation_context.session.id
    await self._log_entry(
        session_id,
        "on_tool_error",
        {
            "tool_name": tool.name,
            "agent_name": tool_context.agent_name,
            "function_call_id": tool_context.function_call_id,
            "args": tool_args,
            "error": str(error),
        },
    )
    return None

  def get_logs_by_session(self, session_id: str) -> List[Dict[str, Any]]:
    """Retrieve all logs for a specific session."""
    return self.logs.get(session_id, [])
