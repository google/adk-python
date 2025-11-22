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
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from google.api_core.gapic_v1 import client_info as gapic_client_info
import google.auth
from google.cloud import bigquery
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


# --- PyArrow Helper Functions ---
def _pyarrow_datetime():
  """Returns PyArrow type for BigQuery DATETIME."""
  return pa.timestamp("us", tz=None)


def _pyarrow_numeric():
  """Returns PyArrow type for BigQuery NUMERIC."""
  return pa.decimal128(38, 9)


def _pyarrow_bignumeric():
  """Returns PyArrow type for BigQuery BIGNUMERIC."""
  return pa.decimal256(76, 38)


def _pyarrow_time():
  """Returns PyArrow type for BigQuery TIME."""
  return pa.time64("us")


def _pyarrow_timestamp():
  """Returns PyArrow type for BigQuery TIMESTAMP."""
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

_BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
    "GEOGRAPHY": {
        b"ARROW:extension:name": b"google:sqlType:geography",
        b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
    },
    "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
    "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
}
_STRUCT_TYPES = ("RECORD", "STRUCT")


def _bq_to_arrow_scalars(bq_scalar: str):
  """Converts a BigQuery scalar type string to a PyArrow data type constructor."""
  return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


def _bq_to_arrow_struct_data_type(field):
  """Converts a BigQuery STRUCT/RECORD field to a PyArrow struct type."""
  arrow_fields = []
  for subfield in field.fields:
    arrow_subfield = _bq_to_arrow_field(subfield)
    if arrow_subfield:
      arrow_fields.append(arrow_subfield)
    else:
      logging.warning(
          "Failed to convert STRUCT/RECORD field '%s' due to subfield '%s'.",
          field.name,
          subfield.name,
      )
      return None
  return pa.struct(arrow_fields)


def _bq_to_arrow_range_data_type(field):
  """Converts a BigQuery RANGE field to a PyArrow struct type."""
  if field is None:
    raise ValueError("Range element type cannot be None")
  return pa.struct([
      ("start", _bq_to_arrow_scalars(field.element_type.upper())()),
      ("end", _bq_to_arrow_scalars(field.element_type.upper())()),
  ])


def _bq_to_arrow_data_type(field):
  """Converts a BigQuery schema field to a PyArrow data type."""
  if field.mode == "REPEATED":
    inner = _bq_to_arrow_data_type(
        bq_schema.SchemaField(
            field.name,
            field.field_type,
            fields=field.fields,
            range_element_type=getattr(field, "range_element_type", None),
        )
    )
    return pa.list_(inner) if inner else None
  field_type_upper = field.field_type.upper() if field.field_type else ""
  if field_type_upper in _STRUCT_TYPES:
    return _bq_to_arrow_struct_data_type(field)
  if field_type_upper == "RANGE":
    return _bq_to_arrow_range_data_type(field.range_element_type)
  constructor = _bq_to_arrow_scalars(field_type_upper)
  if constructor:
    return constructor()
  else:
    logging.warning(
        "Failed to convert BigQuery field '%s': unsupported type '%s'.",
        field.name,
        field.field_type,
    )
    return None


def _bq_to_arrow_field(bq_field):
  """Converts a BigQuery SchemaField to a PyArrow Field."""
  arrow_type = _bq_to_arrow_data_type(bq_field)
  if arrow_type:
    metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
        bq_field.field_type.upper() if bq_field.field_type else ""
    )
    nullable = bq_field.mode.upper() != "REQUIRED"
    return pa.field(
        bq_field.name,
        arrow_type,
        nullable=nullable,
        metadata=metadata,
    )
  logging.warning(
      "Could not determine Arrow type for field '%s' with type '%s'.",
      bq_field.name,
      bq_field.field_type,
  )
  return None


def to_arrow_schema(bq_schema_list):
  """Converts a list of BigQuery SchemaFields to a PyArrow Schema."""
  arrow_fields = []
  for bq_field in bq_schema_list:
    af = _bq_to_arrow_field(bq_field)
    if af:
      arrow_fields.append(af)
    else:
      logging.warning(
          "Failed to convert schema due to field '%s'.", bq_field.name
      )
      return None
  return pa.schema(arrow_fields)


@dataclasses.dataclass
class BigQueryLoggerConfig:
  """Configuration for BigQueryAgentAnalyticsPlugin.

  Attributes:
    enabled: Whether logging is enabled.
    event_allowlist: A list of event types to log. If None, all events are
      logged except those in event_denylist.
    event_denylist: A list of event types to skip logging.
    content_formatter: An optional function to format event content before
      logging.
    shutdown_timeout: Seconds to wait for logs to flush during shutdown.
    client_close_timeout: Seconds to wait for BQ client to close.
    max_content_length: The maximum length of content parts before truncation.
  """

  enabled: bool = True
  event_allowlist: Optional[List[str]] = None
  event_denylist: Optional[List[str]] = None
  content_formatter: Optional[Callable[[Any], str]] = None
  shutdown_timeout: float = 5.0
  client_close_timeout: float = 2.0
  max_content_length: int = 500


# --- Helper Formatters ---
def _get_event_type(event: Event) -> str:
  """Determines the event type from an Event object."""
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
  return "SYSTEM"


def _format_content(
    content: Optional[types.Content], max_len: int = 500
) -> tuple[str, bool]:
  """Formats an Event content for logging.

  Args:
    content: The Event content to format.
    max_len: The maximum length of the text parts before truncation.

  Returns:
    A tuple containing the formatted content string and a boolean indicating if
    the content was truncated.
  """
  if not content or not content.parts:
    return "None", False
  parts = []
  for p in content.parts:
    if p.text:
      parts.append(
          f"text: '{p.text[:max_len]}...' "
          if len(p.text) > max_len
          else f"text: '{p.text}'"
      )
    elif p.function_call:
      parts.append(f"call: {p.function_call.name}")
    elif p.function_response:
      parts.append(f"resp: {p.function_response.name}")
    else:
      parts.append("other")
  return " | ".join(parts), any(
      len(p.text) > max_len for p in content.parts if p.text
  )


def _format_args(
    args: dict[str, Any], *, max_len: int = 1000
) -> tuple[str, bool]:
  """Formats tool arguments or results for logging.

  Args:
    args: The tool arguments or results dictionary to format.
    max_len: The maximum length of the output string before truncation.

  Returns:
    A tuple containing the JSON formatted string and a boolean indicating if
    the content was truncated.
  """
  if not args:
    return "{}", False
  try:
    s = json.dumps(args)
  except TypeError:
    s = str(args)
  if len(s) > max_len:
    return s[:max_len] + "...", True
  return s, False


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """A plugin that logs agent analytic events to Google BigQuery.

  This plugin captures key events during an agent's lifecycle—such as user
  interactions, tool executions, LLM requests/responses, and errors—and
  streams them to a BigQuery table for analysis and monitoring.

  It uses the BigQuery Write API for efficient, high-throughput streaming
  ingestion and is designed to be non-blocking, ensuring that logging
  operations do not impact agent performance. If the destination table does
  not exist, the plugin will attempt to create it based on a predefined
  schema.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events",
      config: Optional[BigQueryLoggerConfig] = None,
      **kwargs,
  ):
    """Initializes the BigQueryAgentAnalyticsPlugin.

    Args:
      project_id: Google Cloud project ID.
      dataset_id: BigQuery dataset ID.
      table_id: BigQuery table ID for agent events.
      config: Plugin configuration.
      **kwargs: Additional arguments.
    """
    super().__init__(name=kwargs.get("name", "BigQueryAgentAnalyticsPlugin"))
    self._project_id, self._dataset_id, self._table_id = (
        project_id,
        dataset_id,
        table_id,
    )
    self._config = config if config else BigQueryLoggerConfig()
    self._bq_client: bigquery.Client | None = None
    self._write_client: BigQueryWriteAsyncClient | None = None
    self._init_lock: asyncio.Lock | None = None
    self._arrow_schema: pa.Schema | None = None
    self._background_tasks: set[asyncio.Task] = set()
    self._is_shutting_down = False
    self._schema = [
        bigquery.SchemaField(
            "timestamp",
            "TIMESTAMP",
            mode="REQUIRED",
            description="The UTC time at which the event was logged.",
        ),
        bigquery.SchemaField(
            "event_type",
            "STRING",
            mode="NULLABLE",
            description=(
                "Indicates the type of event being logged (e.g., 'LLM_REQUEST',"
                " 'TOOL_COMPLETED')."
            ),
        ),
        bigquery.SchemaField(
            "agent",
            "STRING",
            mode="NULLABLE",
            description=(
                "The name of the ADK agent or author associated with the event."
            ),
        ),
        bigquery.SchemaField(
            "session_id",
            "STRING",
            mode="NULLABLE",
            description=(
                "A unique identifier to group events within a single"
                " conversation or user session."
            ),
        ),
        bigquery.SchemaField(
            "invocation_id",
            "STRING",
            mode="NULLABLE",
            description=(
                "A unique identifier for each individual agent execution or"
                " turn within a session."
            ),
        ),
        bigquery.SchemaField(
            "user_id",
            "STRING",
            mode="NULLABLE",
            description=(
                "The identifier of the user associated with the current"
                " session."
            ),
        ),
        bigquery.SchemaField(
            "content",
            "STRING",
            mode="NULLABLE",
            description=(
                "The event-specific data (payload). Format varies by"
                " event_type."
            ),
        ),
        bigquery.SchemaField(
            "error_message",
            "STRING",
            mode="NULLABLE",
            description=(
                "Populated if an error occurs during the processing of the"
                " event."
            ),
        ),
        bigquery.SchemaField(
            "is_truncated",
            "BOOLEAN",
            mode="NULLABLE",
            description=(
                "Indicates if the content field was truncated due to size"
                " limits."
            ),
        ),
    ]

  def _format_content_safely(
      self, content: Optional[types.Content]
  ) -> tuple[str | None, bool]:
    """Formats content using self._config.content_formatter or _format_content, catching errors."""
    if content is None:
      return None, False
    try:
      if self._config.content_formatter:
        # Custom formatter: we assume no truncation or we can't know.
        return self._config.content_formatter(content), False
      return _format_content(content, max_len=self._config.max_content_length)
    except Exception as e:
      logging.warning("Content formatter failed: %s", e)
      return "[FORMATTING FAILED]", False

  async def _ensure_init(self):
    """Ensures BigQuery clients are initialized."""
    if self._write_client:
      return True
    if not self._init_lock:
      self._init_lock = asyncio.Lock()
    async with self._init_lock:
      if self._write_client:
        return True
      try:
        creds, _ = await asyncio.to_thread(
            google.auth.default,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client_info = gapic_client_info.ClientInfo(
            user_agent=f"google-adk-bq-logger/{version.__version__}"
        )
        self._bq_client = bigquery.Client(
            project=self._project_id, credentials=creds, client_info=client_info
        )

        # Ensure table exists (sync call in thread)
        def create_resources():
          if self._bq_client:
            self._bq_client.create_dataset(self._dataset_id, exists_ok=True)
            table = bigquery.Table(
                f"{self._project_id}.{self._dataset_id}.{self._table_id}",
                schema=self._schema,
            )
            table.time_partitioning = bigquery.TimePartitioning(
                type_="DAY", field="timestamp"
            )
            table.clustering_fields = ["event_type", "agent", "user_id"]
            self._bq_client.create_table(table, exists_ok=True)
            logging.info(
                "BQ Plugin: Dataset %s and Table %s ensured to exist.",
                self._dataset_id,
                self._table_id,
            )

        await asyncio.to_thread(create_resources)

        self._write_client = BigQueryWriteAsyncClient(
            credentials=creds,
            client_info=client_info,
        )
        self._arrow_schema = to_arrow_schema(self._schema)
        if not self._arrow_schema:
          raise RuntimeError("Failed to convert BigQuery schema to Arrow.")
        logging.info("BQ Plugin: Initialized successfully.")
        return True
      except Exception as e:
        logging.error("BQ Plugin: Init Failed:", exc_info=True)
        return False

  async def _perform_write(self, row: dict):
    """Actual async write operation, intended to run as a background task."""
    try:
      if (
          not await self._ensure_init()
          or not self._write_client
          or not self._arrow_schema
      ):
        return

      # Serialize
      pydict = {f.name: [row.get(f.name)] for f in self._arrow_schema}
      batch = pa.RecordBatch.from_pydict(pydict, schema=self._arrow_schema)
      req = bq_storage_types.AppendRowsRequest(
          write_stream=f"projects/{self._project_id}/datasets/{self._dataset_id}/tables/{self._table_id}/_default"
      )
      req.arrow_rows.writer_schema.serialized_schema = (
          self._arrow_schema.serialize().to_pybytes()
      )
      req.arrow_rows.rows.serialized_record_batch = (
          batch.serialize().to_pybytes()
      )

      # Write with protection against immediate cancellation
      async for resp in await asyncio.shield(
          self._write_client.append_rows(iter([req]))
      ):
        if resp.error.code != 0:
          msg = resp.error.message
          # Check for common schema mismatch indicators
          if (
              "schema mismatch" in msg.lower()
              or "field" in msg.lower()
              or "type" in msg.lower()
          ):
            logging.error(
                "BQ Plugin: Schema Mismatch Error. The BigQuery table schema"
                " may be incorrect or out of sync with the plugin. Please"
                " verify the table definition. Details: %s",
                msg,
            )
          else:
            logging.error("BQ Plugin: Write Error: %s", msg)

    except RuntimeError as e:
      if "Event loop is closed" not in str(e) and not self._is_shutting_down:
        logging.error("BQ Plugin: Runtime Error during write:", exc_info=True)
    except asyncio.CancelledError:
      if not self._is_shutting_down:
        logging.warning("BQ Plugin: Write task cancelled unexpectedly.")
    except Exception as e:
      logging.error("BQ Plugin: Write Failed:", exc_info=True)

  async def _log(self, data: dict):
    """Schedules a log entry to be written in the background."""
    if not self._config.enabled:
      return
    event_type = data.get("event_type")
    if (
        self._config.event_denylist
        and event_type in self._config.event_denylist
    ):
      return
    if (
        self._config.event_allowlist
        and event_type not in self._config.event_allowlist
    ):
      return

    # Prepare row immediately (capture current state)
    row = {
        "timestamp": datetime.now(timezone.utc),
        "event_type": None,
        "agent": None,
        "session_id": None,
        "invocation_id": None,
        "user_id": None,
        "content": None,
        "error_message": None,
        "is_truncated": False,
    }
    row.update(data)

    # Fire and forget: Create task and track it
    task = asyncio.create_task(self._perform_write(row))
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def close(self):
    """Flushes pending logs and closes client."""
    if self._is_shutting_down:
      return
    self._is_shutting_down = True
    logging.info("BQ Plugin: Shutdown started.")

    if self._background_tasks:
      logging.info(
          "BQ Plugin: Flushing %s pending logs...", len(self._background_tasks)
      )
      try:
        await asyncio.wait(
            self._background_tasks, timeout=self._config.shutdown_timeout
        )
      except asyncio.TimeoutError:
        logging.warning("BQ Plugin: Timeout waiting for logs to flush.")
      except Exception as e:
        logging.warning("BQ Plugin: Error flushing logs:", exc_info=True)

    # Use getattr for safe access in case transport is not present.
    if self._write_client and getattr(self._write_client, "transport", None):
      try:
        logging.info("BQ Plugin: Closing write client.")
        await asyncio.wait_for(
            self._write_client.transport.close(),
            timeout=self._config.client_close_timeout,
        )
      except Exception as e:
        logging.warning("BQ Plugin: Error closing write client: %s", e)
    if self._bq_client:
      try:
        self._bq_client.close()
      except Exception as e:
        logging.warning("BQ Plugin: Error closing BQ client: %s", e)

    self._write_client = None
    self._bq_client = None
    self._is_shutting_down = False
    logging.info("BQ Plugin: Shutdown complete.")

  # --- Streamlined Callbacks ---
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> None:
    """Callback for user messages.

    Logs the user message details including:
    1. User content (text)

    The content is formatted as 'User Content: {content}'.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    content, truncated = self._format_content_safely(user_message)
    await self._log({
        "event_type": "USER_MESSAGE_RECEIVED",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": f"User Content: {content}",
        "is_truncated": truncated,
    })

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Callback before agent invocation.

    Logs the start of an agent invocation.
    No specific content payload is logged for this event, but standard metadata
    (agent name, session ID, invocation ID, user ID) is captured.
    """
    await self._log({
        "event_type": "INVOCATION_STARTING",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
    })

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> None:
    """Callback for agent events.

    Logs generic agent events including:
    1. Event type (determined from event properties)
    2. Event content (text, function calls, or responses)
    3. Error messages (if any)

    The content is formatted based on the event type.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    content, truncated = self._format_content_safely(event.content)
    await self._log({
        "event_type": _get_event_type(event),
        "agent": event.author,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
        "content": content,
        "error_message": event.error_message,
        "timestamp": datetime.fromtimestamp(event.timestamp, timezone.utc),
        "is_truncated": truncated,
    })

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Callback after agent invocation.

    Logs the completion of an agent invocation.
    No specific content payload is logged for this event, but standard metadata
    (agent name, session ID, invocation ID, user ID) is captured.
    """
    await self._log({
        "event_type": "INVOCATION_COMPLETED",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
    })

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    """Callback before an agent starts.

    Logs the start of a specific agent execution.
    Content includes:
    1. Agent Name (from callback context)
    """
    await self._log({
        "event_type": "AGENT_STARTING",
        "agent": agent.name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": f"Agent Name: {callback_context.agent_name}",
    })

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    """Callback after an agent completes.

    Logs the completion of a specific agent execution.
    Content includes:
    1. Agent Name (from callback context)
    """
    await self._log({
        "event_type": "AGENT_COMPLETED",
        "agent": agent.name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": f"Agent Name: {callback_context.agent_name}",
    })

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    """Callback before LLM call.

    Logs the LLM request details including:
    1. Model name
    2. Configuration parameters (temperature, top_p, top_k, max_output_tokens)
    3. Available tool names
    4. Prompt content (user/model messages)
    5. System instructions

    The content is formatted as a single string with fields separated by ' | '.
    If the total length exceeds `max_content_length`, the string is truncated,
    prioritizing the metadata (Model, Params, Tools) over the Prompt and System
    Prompt.
    """
    content_parts = [
        f"Model: {llm_request.model or 'default'}",
    ]
    is_truncated = False

    # 1. Params
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

    # 2. Tools
    if llm_request.tools_dict:
      content_parts.append(
          f"Available Tools: {list(llm_request.tools_dict.keys())}"
      )

    # 3. Prompt
    if contents := getattr(llm_request, "contents", None):
      prompt_parts = []
      for c in contents:
        c_str, c_trunc = self._format_content_safely(c)
        prompt_parts.append(f"{c.role}: {c_str}")
        if c_trunc:
          is_truncated = True
      prompt_str = " | ".join(prompt_parts)
      content_parts.append(f"Prompt: {prompt_str}")

    # 4. System Prompt
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

    final_content = " | ".join(content_parts)
    max_len = self._config.max_content_length
    if len(final_content) > max_len:
      final_content = final_content[:max_len] + "..."
      is_truncated = True
    await self._log({
        "event_type": "LLM_REQUEST",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": final_content,
        "is_truncated": is_truncated,
    })

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> None:
    """Callback after LLM call.

    Logs the LLM response details including:
    1. Tool calls (if any)
    2. Text response (if no tool calls)
    3. Token usage statistics (prompt, candidates, total)

    The content is formatted as a single string with fields separated by ' | '.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    content_parts = []
    content = llm_response.content
    is_tool_call = False
    is_truncated = False
    if content and content.parts:
      is_tool_call = any(part.function_call for part in content.parts)

    if is_tool_call:
      fc_names = []
      if content and content.parts:
        fc_names = [
            part.function_call.name
            for part in content.parts
            if part.function_call
        ]
      content_parts.append(f"Tool Name: {', '.join(fc_names)}")
    else:
      text_content, truncated = self._format_content_safely(
          llm_response.content
      )
      content_parts.append(f"Tool Name: text_response, {text_content}")
      if truncated:
        is_truncated = True

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
    await self._log({
        "event_type": "LLM_RESPONSE",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "content": final_content,
        "error_message": llm_response.error_message,
        "is_truncated": is_truncated,
    })

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> None:
    """Callback before tool call.

    Logs the tool execution start details including:
    1. Tool name
    2. Tool description
    3. Tool arguments

    The content is formatted as 'Tool Name: ..., Description: ..., Arguments:
    ...'.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    args_str, truncated = _format_args(
        tool_args, max_len=self._config.max_content_length
    )
    content = (
        f"Tool Name: {tool.name}, Description: {tool.description},"
        f" Arguments: {args_str}"
    )
    if len(content) > self._config.max_content_length:
      content = content[: self._config.max_content_length] + "..."
      truncated = True
    await self._log({
        "event_type": "TOOL_STARTING",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": content,
        "is_truncated": truncated,
    })

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> None:
    """Callback after tool call.

    Logs the tool execution result details including:
    1. Tool name
    2. Tool result

    The content is formatted as 'Tool Name: ..., Result: ...'.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    result_str, truncated = _format_args(
        result, max_len=self._config.max_content_length
    )
    content = f"Tool Name: {tool.name}, Result: {result_str}"
    if len(content) > self._config.max_content_length:
      content = content[: self._config.max_content_length] + "..."
      truncated = True
    await self._log({
        "event_type": "TOOL_COMPLETED",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": content,
        "is_truncated": truncated,
    })

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> None:
    """Callback for model errors.

    Logs errors that occur during LLM calls.
    No specific content payload is logged, but the error message is captured
    in the `error_message` field.
    """
    await self._log({
        "event_type": "LLM_ERROR",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "error_message": str(error),
    })

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Callback for tool errors.

    Logs errors that occur during tool execution.
    Content includes:
    1. Tool name
    2. Tool arguments

    The error message is captured in the `error_message` field.
    If the content length exceeds `max_content_length`, it is truncated.
    """
    args_str, truncated = _format_args(
        tool_args, max_len=self._config.max_content_length
    )
    content = f"Tool Name: {tool.name}, Arguments: {args_str}"
    if len(content) > self._config.max_content_length:
      content = content[: self._config.max_content_length] + "..."
      truncated = True
    await self._log({
        "event_type": "TOOL_ERROR",
        "agent": tool_context.agent_name,
        "session_id": tool_context.session.id,
        "invocation_id": tool_context.invocation_id,
        "user_id": tool_context.session.user_id,
        "content": content,
        "error_message": str(error),
        "is_truncated": truncated,
    })
