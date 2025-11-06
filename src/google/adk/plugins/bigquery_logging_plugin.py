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
import warnings

import google.api_core.client_info
import google.auth
from google.auth import exceptions as auth_exceptions
from google.cloud import bigquery
from google.cloud import exceptions as cloud_exceptions
from google.cloud.bigquery import schema as bq_schema
from google.cloud.bigquery_storage_v1 import types as bq_storage_types
from google.cloud.bigquery_storage_v1.services.big_query_write.async_client import BigQueryWriteAsyncClient
from google.genai import types
import pyarrow as pa

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


def _pyarrow_datetime():
  return pa.timestamp("us", tz=None)


def _pyarrow_numeric():
  return pa.decimal128(38, 9)


def _pyarrow_bignumeric():
  return pa.decimal256(76, 38)


def _pyarrow_time():
  return pa.time64("us")


def _pyarrow_timestamp():
  return pa.timestamp("us", tz="UTC")


_BQ_TO_ARROW_SCALARS = {
    "BOOL": pa.bool_,
    "BOOLEAN": pa.bool_,
    "BYTES": pa.binary,
    "DATE": pa.date32,
    "DATETIME": _pyarrow_datetime,
    "FLOAT": pa.float64,
    "FLOAT64": pa.float64,
    "GEOGRAPHY": pa.string,
    "INT64": pa.int64,
    "INTEGER": pa.int64,
    "JSON": pa.string,
    "NUMERIC": _pyarrow_numeric,
    "BIGNUMERIC": _pyarrow_bignumeric,
    "STRING": pa.string,
    "TIME": _pyarrow_time,
    "TIMESTAMP": _pyarrow_timestamp,
}


def _bq_to_arrow_scalars(bq_scalar: str):
  return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


_BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
    "GEOGRAPHY": {
        b"ARROW:extension:name": b"google:sqlType:geography",
        b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
    },
    "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
    "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
}
_STRUCT_TYPES = ("RECORD", "STRUCT")


def _bq_to_arrow_struct_data_type(field):
  arrow_fields = []
  for subfield in field.fields:
    arrow_subfield = _bq_to_arrow_field(subfield)
    if arrow_subfield:
      arrow_fields.append(arrow_subfield)
    else:
      return None
  return pa.struct(arrow_fields)


def _bq_to_arrow_range_data_type(field):
  if field is None:
    raise ValueError("Range element type cannot be None")
  element_type = field.element_type.upper()
  arrow_element_type = _bq_to_arrow_scalars(element_type)()
  return pa.struct([("start", arrow_element_type), ("end", arrow_element_type)])


def _bq_to_arrow_data_type(field):
  if field.mode is not None and field.mode.upper() == "REPEATED":
    inner_type = _bq_to_arrow_data_type(
        bq_schema.SchemaField(field.name, field.field_type, fields=field.fields)
    )
    if inner_type:
      return pa.list_(inner_type)
    return None

  field_type_upper = field.field_type.upper() if field.field_type else ""
  if field_type_upper in _STRUCT_TYPES:
    return _bq_to_arrow_struct_data_type(field)

  if field_type_upper == "RANGE":
    return _bq_to_arrow_range_data_type(field.range_element_type)

  data_type_constructor = _bq_to_arrow_scalars(field_type_upper)
  if data_type_constructor is None:
    return None
  return data_type_constructor()


def _bq_to_arrow_field(bq_field, array_type=None):
  arrow_type = _bq_to_arrow_data_type(bq_field)
  if arrow_type is not None:
    if array_type is not None:
      arrow_type = array_type
    metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
        bq_field.field_type.upper() if bq_field.field_type else ""
    )
    return pa.field(
        bq_field.name,
        arrow_type,
        nullable=False if bq_field.mode.upper() == "REPEATED" else True,
        metadata=metadata,
    )

  warnings.warn(f"Unable to determine Arrow type for field '{bq_field.name}'.")
  return None


def to_arrow_schema(bq_schema_list):
  """Return the Arrow schema, corresponding to a given BigQuery schema."""
  arrow_fields = []
  for bq_field in bq_schema_list:
    arrow_field = _bq_to_arrow_field(bq_field)
    if arrow_field is None:
      return None
    arrow_fields.append(arrow_field)
  return pa.schema(arrow_fields)


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

    self._write_client: BigQueryWriteAsyncClient | None = None
    self._arrow_schema: pa.Schema | None = None
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
            scopes=[
                "https://www.googleapis.com/auth/bigquery",
                "https://www.googleapis.com/auth/cloud-platform",  # For Storage Write
            ]
        )
        client_info = google.api_core.client_info.ClientInfo(
            user_agent=f"google-adk-bq-logger/{version.__version__}"
        )

        # 1. Init BQ Client (for create_dataset/create_table)
        self._bq_client = bigquery.Client(
            project=self._project_id,
            credentials=credentials,
            client_info=client_info,
        )

        # 2. Init BQ Storage Write Client
        self._write_client = BigQueryWriteAsyncClient(
            credentials=credentials, client_info=client_info
        )

        logging.info(
            "BigQuery clients (Core & Storage Write) initialized for"
            " project %s",
            self._project_id,
        )
        dataset_ref = self._bq_client.dataset(self._dataset_id)
        self._bq_client.create_dataset(dataset_ref, exists_ok=True)
        logging.info("Dataset %s ensured to exist.", self._dataset_id)
        table_ref = dataset_ref.table(self._table_id)

        # Schema
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

        # 4. Store Arrow schema for Write API
        self._arrow_schema = to_arrow_schema(schema)  # USE LOCAL VERSION
        # --- self._table_ref_str removed ---

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

    try:
      if not self._init_done:
        await asyncio.to_thread(self._ensure_initialized_sync)

      # Check for all required Storage Write API components
      if not (
          self._init_succeeded and self._write_client and self._arrow_schema
      ):
        logging.warning("BigQuery write client not initialized. Skipping log.")
        return

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

      # --- START MODIFIED STORAGE WRITE API LOGIC (using Default Stream) ---
      # 1. Convert the single row dict to a PyArrow RecordBatch
      #    pa.RecordBatch.from_pydict requires a dict of lists
      pydict = {
          field.name: [insert_row.get(field.name)]
          for field in self._arrow_schema
      }
      batch = pa.RecordBatch.from_pydict(pydict, schema=self._arrow_schema)

      # 2. Create the AppendRowsRequest, pointing to the default stream
      request = bq_storage_types.AppendRowsRequest(
          write_stream=(
              f"projects/{self._project_id}/datasets/{self._dataset_id}"
              f"/tables/{self._table_id}/_default"
          )
      )
      request.arrow_rows.writer_schema.serialized_schema = (
          self._arrow_schema.serialize().to_pybytes()
      )

      request.arrow_rows.rows.serialized_record_batch = (
          batch.serialize().to_pybytes()
      )

      # 3. Send the request and check for errors
      response_iterator = self._write_client.append_rows(requests=[request])
      async for response in response_iterator:
        if response.row_errors:
          logging.error(
              "Errors occurred while writing to BigQuery (Storage Write"
              " API): %s",
              response.row_errors,
          )
          break  # Only one response expected
    except Exception as e:
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
    if llm_request.config and llm_request.config.system_instruction:
      si = llm_request.config.system_instruction
      if isinstance(si, str):
        system_instruction_text = si
      elif isinstance(si, types.Content):
        system_instruction_text = "".join(p.text for p in si.parts if p.text)
      elif isinstance(si, types.Part):
        system_instruction_text = si.text
      elif hasattr(si, "__iter__"):
        texts = []
        for item in si:
          if isinstance(item, str):
            texts.append(item)
          elif isinstance(item, types.Part) and item.text:
            texts.append(item.text)
        system_instruction_text = "".join(texts)
      else:
        system_instruction_text = str(si)
    elif llm_request.config and not llm_request.config.system_instruction:
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
