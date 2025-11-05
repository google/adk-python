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
import dataclasses
from datetime import datetime
from datetime import timezone
import json
import logging
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import google.api_core.client_info
import google.auth
from google.auth import exceptions as auth_exceptions
from google.cloud import bigquery
from google.cloud import exceptions as cloud_exceptions
from google.genai import types

from .. import version
from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext


@dataclasses.dataclass
class BigQueryLoggerConfig:
  """Configuration for the BigQueryAgentAnalyticsPlugin.

  Attributes:
      enabled: Whether the plugin is enabled.
      event_allowlist: List of event types to log. If None, all are allowed
        except those in event_denylist.
      event_denylist: List of event types to not log. Takes precedence over
        event_allowlist.
      content_formatter: Function to format or redact the 'content' field before
        logging.
  """

  enabled: bool = True
  event_allowlist: Optional[List[str]] = None
  event_denylist: Optional[List[str]] = None
  content_formatter: Optional[Callable[[Any], str]] = None


def _get_event_type(event: Event) -> str:
  if event.author == "user":
    return "USER_INPUT"
  if event.get_function_calls():
    return "TOOL_CALL"
  if event.get_function_responses():
    return "TOOL_RESULT"
  if event.content and event.content.parts:
    return "MODEL_RESPONSE"
  if event.error_message:
    return "ERROR"
  return "SYSTEM"  # Fallback for other event types


def _format_content(
    content: Optional[types.Content], max_length: int = 200
) -> str:
  """Format content for logging, truncating if too long."""
  if not content or not content.parts:
    return "None"
  parts = []
  for part in content.parts:
    if part.text:
      text = part.text.strip()
      if len(text) > max_length:
        text = text[:max_length] + "..."
      parts.append(f"text: '{text}'")
    elif part.function_call:
      parts.append(f"function_call: {part.function_call.name}")
    elif part.function_response:
      parts.append(f"function_response: {part.function_response.name}")
    elif part.code_execution_result:
      parts.append("code_execution_result")
    else:
      parts.append("other_part")
  return " | ".join(parts)


def _format_args(args: dict[str, Any], max_length: int = 300) -> str:
  """Format arguments dictionary for logging."""
  if not args:
    return "{}"
  formatted = str(args)
  if len(formatted) > max_length:
    formatted = formatted[:max_length] + "...}"
  return formatted


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """A plugin that logs ADK events to a BigQuery table.

  This plugin captures critical events during an agent invocation and logs them
  as structured data to the specified BigQuery table. This allows for
  persistent storage, auditing, and analysis of agent interactions.

  The plugin logs the following information at each callback point:
  - User messages and invocation context
  - Agent execution flow (start and completion)
  - LLM requests and responses (including token usage in content)
  - Tool calls with arguments and results
  - Events yielded by agents
  - Errors during model and tool execution

  Each log entry includes a timestamp, event type, agent name, session ID,
  invocation ID, user ID, content payload, and any error messages.

  Logging behavior can be customized using the BigQueryLoggerConfig.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events",
      config: Optional[BigQueryLoggerConfig] = None,
      **kwargs,
  ):
    super().__init__(name=kwargs.get("name", "BigQueryAgentAnalyticsPlugin"))
    self._project_id = project_id
    self._dataset_id = dataset_id
    self._table_id = table_id
    self._config = config if config else BigQueryLoggerConfig()
    self._bq_client: bigquery.Client | None = None
    self._client_init_lock = threading.Lock()
    self._init_done = False
    self._init_succeeded = False

    if not self._config.enabled:
      logging.info(
          "BigQueryAgentAnalyticsPlugin %s is disabled by configuration.",
          self.name,
      )
      return

    logging.debug(
        "DEBUG: BigQueryAgentAnalyticsPlugin INSTANTIATED (Name: %s)", self.name
    )

  def _ensure_initialized_sync(self):
    """Synchronous initialization of BQ client and table."""
    if not self._config.enabled:
      return

    with self._client_init_lock:
      if self._init_done:
        return
      self._init_done = True
      try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        client_info = google.api_core.client_info.ClientInfo(
            user_agent=f"google-adk-bq-logger/{version.__version__}"
        )
        self._bq_client = bigquery.Client(
            project=self._project_id,
            credentials=credentials,
            client_info=client_info,
        )
        logging.info(
            "BigQuery client initialized for project %s", self._project_id
        )
        dataset_ref = self._bq_client.dataset(self._dataset_id)
        self._bq_client.create_dataset(dataset_ref, exists_ok=True)
        logging.info("Dataset %s ensured to exist.", self._dataset_id)
        table_ref = dataset_ref.table(self._table_id)
        # Schema without separate token columns
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("event_type", "STRING"),
            bigquery.SchemaField("agent", "STRING"),
            bigquery.SchemaField("session_id", "STRING"),
            bigquery.SchemaField("invocation_id", "STRING"),
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("content", "STRING"),
            bigquery.SchemaField("error_message", "STRING"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        self._bq_client.create_table(table, exists_ok=True)
        logging.info("Table %s ensured to exist.", self._table_id)
        self._init_succeeded = True
      except (
          auth_exceptions.GoogleAuthError,
          cloud_exceptions.GoogleCloudError,
      ) as e:
        logging.exception(
            "Failed to initialize BigQuery client or table: %s", e
        )
        self._init_succeeded = False

  async def _log_to_bigquery_async(self, event_dict: dict[str, Any]):
    if not self._config.enabled:
      return

    event_type = event_dict.get("event_type")

    # Check denylist
    if (
        self._config.event_denylist
        and event_type in self._config.event_denylist
    ):
      return

    # Check allowlist
    if (
        self._config.event_allowlist
        and event_type not in self._config.event_allowlist
    ):
      return

    # Apply custom content formatter
    if self._config.content_formatter and "content" in event_dict:
      try:
        event_dict["content"] = self._config.content_formatter(
            event_dict["content"]
        )
      except Exception as e:
        logging.warning(
            "Error applying custom content formatter for event type %s: %s",
            event_type,
            e,
        )
        # Optionally log a generic message or the error

    def _sync_log():
      self._ensure_initialized_sync()
      if not self._init_succeeded or not self._bq_client:
        return
      table_ref = self._bq_client.dataset(self._dataset_id).table(
          self._table_id
      )
      default_row = {
          "timestamp": datetime.now(timezone.utc).isoformat(),
          "event_type": None,
          "agent": None,
          "session_id": None,
          "invocation_id": None,
          "user_id": None,
          "content": None,
          "error_message": None,
      }
      insert_row = {**default_row, **event_dict}

      errors = self._bq_client.insert_rows_json(table_ref, [insert_row])
      if errors:
        logging.error(
            "Errors occurred while inserting to BigQuery table %s.%s: %s",
            self._dataset_id,
            self._table_id,
            errors,
        )

    try:
      await asyncio.to_thread(_sync_log)
    except (
        cloud_exceptions.GoogleCloudError,
        auth_exceptions.GoogleAuthError,
    ) as e:
      logging.exception("Failed to log to BigQuery: %s", e)

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """Log user message and invocation start."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "USER_MESSAGE_RECEIVED",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": f"User Content: {_format_content(user_message)}",
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Log invocation start."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "INVOCATION_STARTING",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": None,
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """Logs event data to BigQuery."""
    event_dict = {
        "timestamp": datetime.fromtimestamp(
            event.timestamp, timezone.utc
        ).isoformat(),
        "event_type": _get_event_type(event),
        "agent": event.author,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": (
            json.dumps(
                [part.model_dump(mode="json") for part in event.content.parts]
            )
            if event.content and event.content.parts
            else None
        ),
        "error_message": event.error_message,
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[None]:
    """Log invocation completion."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "INVOCATION_COMPLETED",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": None,
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Log agent execution start."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "AGENT_STARTING",
        "agent": agent.name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": f"Agent Name: {callback_context.agent_name}",
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Log agent execution completion."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "AGENT_COMPLETED",
        "agent": agent.name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": f"Agent Name: {callback_context.agent_name}",
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Log LLM request before sending to model, including the full system instruction."""

    content_parts = [
        f"Model: {llm_request.model or 'default'}",
    ]

    # Log Full System Instruction
    system_instruction_text = "None"
    if llm_request.config and hasattr(llm_request.config, "system_instruction"):
      si = llm_request.config.system_instruction
      if si:
        if isinstance(si, str):
          system_instruction_text = si
        elif hasattr(si, "__iter__"):  # Handles list, tuple, etc. of parts
          # Join parts together to form the complete system instruction
          system_instruction_text = "".join(
              part.text for part in si if hasattr(part, "text")
          )
        else:
          system_instruction_text = str(si)
      else:
        system_instruction_text = "Empty"

    content_parts.append(f"System Prompt: {system_instruction_text}")

    # Log Generation Config Parameters
    if llm_request.config:
      config = llm_request.config
      params_to_log = {}
      if hasattr(config, "temperature") and config.temperature is not None:
        params_to_log["temperature"] = config.temperature
      if hasattr(config, "top_p") and config.top_p is not None:
        params_to_log["top_p"] = config.top_p
      if hasattr(config, "top_k") and config.top_k is not None:
        params_to_log["top_k"] = config.top_k
      if (
          hasattr(config, "max_output_tokens")
          and config.max_output_tokens is not None
      ):
        params_to_log["max_output_tokens"] = config.max_output_tokens

      if params_to_log:
        params_str = ", ".join([f"{k}={v}" for k, v in params_to_log.items()])
        content_parts.append(f"Params: {{{params_str}}}")

    if llm_request.tools_dict:
      content_parts.append(
          f"Available Tools: {list(llm_request.tools_dict.keys())}"
      )

    final_content = " | ".join(content_parts)

    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "LLM_REQUEST",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": final_content,
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Log LLM response after receiving from model."""
    content_parts = []
    content = llm_response.content
    is_tool_call = False
    if content and content.parts:
      is_tool_call = any(part.function_call for part in content.parts)

    if is_tool_call:
      # Explicitly state Tool Name
      fc_names = []
      if content and content.parts:
        fc_names = [
            part.function_call.name
            for part in content.parts
            if part.function_call
        ]
      content_parts.append(f"Tool Name: {', '.join(fc_names)}")
    else:
      # This is a text response
      text_content = _format_content(
          llm_response.content
      )  # This returns something like "text: 'The actual message...'"
      content_parts.append(f"Tool Name: text_response, {text_content}")

    if llm_response.usage_metadata:
      prompt_tokens = getattr(
          llm_response.usage_metadata, "prompt_token_count", "N/A"
      )
      candidates_tokens = getattr(
          llm_response.usage_metadata, "candidates_token_count", "N/A"
      )
      total_tokens = getattr(
          llm_response.usage_metadata, "total_token_count", "N/A"
      )
      token_usage_str = (
          f"Token Usage: {{prompt: {prompt_tokens}, candidates:"
          f" {candidates_tokens}, total: {total_tokens}}}"
      )
      content_parts.append(token_usage_str)

    final_content = " | ".join(content_parts)

    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "LLM_RESPONSE",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": final_content,
        "error_message": (
            llm_response.error_message if llm_response.error_code else None
        ),
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[None]:
    """Log tool execution start."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "TOOL_STARTING",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": (
            f"Tool Name: {tool.name}, Description: {tool.description},"
            f" Arguments: {_format_args(tool_args)}"
        ),
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> None:
    """Log tool execution completion."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "TOOL_COMPLETED",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": f"Tool Name: {tool.name}, Result: {_format_args(result)}",
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    """Log LLM error."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "LLM_ERROR",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "error_message": str(error),
    }
    await self._log_to_bigquery_async(event_dict)
    return None

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Log tool error."""
    event_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "TOOL_ERROR",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": (
            f"Tool Name: {tool.name}, Arguments: {_format_args(tool_args)}"
        ),
        "error_message": str(error),
    }
    await self._log_to_bigquery_async(event_dict)
    return None
