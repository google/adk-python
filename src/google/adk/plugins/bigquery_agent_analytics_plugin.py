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
import atexit
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import functools
import json
import logging
import mimetypes
import random
import time
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
import uuid
import weakref

from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.exceptions import TooManyRequests
from google.api_core.gapic_v1 import client_info as gapic_client_info
import google.auth
from google.cloud import bigquery
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.bigquery import schema as bq_schema
from google.cloud.bigquery_storage_v1 import types as bq_storage_types
from google.cloud.bigquery_storage_v1.services.big_query_write.async_client import BigQueryWriteAsyncClient
from google.genai import types
import pyarrow as pa

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..version import __version__
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger: logging.Logger = logging.getLogger("google_adk." + __name__)


# gRPC Error Codes
_GRPC_DEADLINE_EXCEEDED = 4
_GRPC_INTERNAL = 13
_GRPC_UNAVAILABLE = 14


# --- Helper Formatters ---
def _format_content(
    content: Optional[types.Content], *, max_len: int = 5000
) -> tuple[str, bool]:
  """Formats an Event content for logging.

  Args:
      content: The content to format.
      max_len: Maximum length for text parts.

  Returns:
      A tuple of (formatted_string, is_truncated).
  """
  if content is None or not content.parts:
    return "None", False
  parts = []
  truncated = False
  for p in content.parts:
    if p.text:
      if max_len != -1 and len(p.text) > max_len:
        parts.append(f"text: '{p.text[:max_len]}...'")
        truncated = True
      else:
        parts.append(f"text: '{p.text}'")
    elif p.function_call:
      parts.append(f"call: {p.function_call.name}")
    elif p.function_response:
      parts.append(f"resp: {p.function_response.name}")
    else:
      parts.append("other")
  return " | ".join(parts), truncated


def _recursive_smart_truncate(obj: Any, max_len: int) -> tuple[Any, bool]:
  """Recursively truncates string values within a dict or list.

  Args:
      obj: The object to truncate.
      max_len: Maximum length for string values.

  Returns:
      A tuple of (truncated_object, is_truncated).
  """
  if isinstance(obj, str):
    if max_len != -1 and len(obj) > max_len:
      return obj[:max_len] + "...[TRUNCATED]", True
    return obj, False
  elif isinstance(obj, dict):
    truncated_any = False
    new_dict = {}
    for k, v in obj.items():
      val, trunc = _recursive_smart_truncate(v, max_len)
      if trunc:
        truncated_any = True
      new_dict[k] = val
    return new_dict, truncated_any
  elif isinstance(obj, (list, tuple)):
    truncated_any = False
    new_list = []
    for i in obj:
      val, trunc = _recursive_smart_truncate(i, max_len)
      if trunc:
        truncated_any = True
      new_list.append(val)
    return type(obj)(new_list), truncated_any
  return obj, False


# --- PyArrow Helper Functions ---
def _pyarrow_datetime() -> pa.DataType:
  return pa.timestamp("us", tz=None)


def _pyarrow_numeric() -> pa.DataType:
  return pa.decimal128(38, 9)


def _pyarrow_bignumeric() -> pa.DataType:
  return pa.decimal256(76, 38)


def _pyarrow_time() -> pa.DataType:
  return pa.time64("us")


def _pyarrow_timestamp() -> pa.DataType:
  return pa.timestamp("us", tz="UTC")


_BQ_TO_ARROW_SCALARS = MappingProxyType({
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
})

_BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
    "GEOGRAPHY": {
        b"ARROW:extension:name": b"google:sqlType:geography",
        b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
    },
    "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
    "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
}
_STRUCT_TYPES = ("RECORD", "STRUCT")


def _bq_to_arrow_scalars(bq_scalar: str) -> Optional[Callable[[], pa.DataType]]:
  """Maps BigQuery scalar types to PyArrow type constructors."""
  return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


def _bq_to_arrow_field(bq_field: bq_schema.SchemaField) -> Optional[pa.Field]:
  """Converts a BigQuery SchemaField to a PyArrow Field."""
  arrow_type = _bq_to_arrow_data_type(bq_field)
  if arrow_type:
    metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
        bq_field.field_type.upper() if bq_field.field_type else ""
    )
    nullable = bq_field.mode.upper() != "REQUIRED"
    return pa.field(
        bq_field.name, arrow_type, nullable=nullable, metadata=metadata
    )
  logger.warning(
      "Could not determine Arrow type for field '%s' with type '%s'.",
      bq_field.name,
      bq_field.field_type,
  )
  return None


def _bq_to_arrow_struct_data_type(
    field: bq_schema.SchemaField,
) -> Optional[pa.StructType]:
  """Converts a BigQuery RECORD/STRUCT field to a PyArrow StructType."""
  arrow_fields = []
  for subfield in field.fields:
    arrow_subfield = _bq_to_arrow_field(subfield)
    if arrow_subfield:
      arrow_fields.append(arrow_subfield)
    else:
      logger.warning(
          "Failed to convert STRUCT/RECORD field '%s' due to subfield '%s'.",
          field.name,
          subfield.name,
      )
      return None
  return pa.struct(arrow_fields)


def _bq_to_arrow_data_type(
    field: bq_schema.SchemaField,
) -> Optional[pa.DataType]:
  """Converts a BigQuery field to a PyArrow DataType."""
  if field.mode == "REPEATED":
    inner = _bq_to_arrow_data_type(
        bq_schema.SchemaField(field.name, field.field_type, fields=field.fields)
    )
    return pa.list_(inner) if inner else None
  field_type_upper = field.field_type.upper() if field.field_type else ""
  if field_type_upper in _STRUCT_TYPES:
    return _bq_to_arrow_struct_data_type(field)
  constructor = _bq_to_arrow_scalars(field_type_upper)
  if constructor:
    return constructor()
  else:
    logger.warning(
        "Failed to convert BigQuery field '%s': unsupported type '%s'.",
        field.name,
        field.field_type,
    )
    return None


def to_arrow_schema(
    bq_schema_list: list[bq_schema.SchemaField],
) -> Optional[pa.Schema]:
  """Converts a list of BigQuery SchemaFields to a PyArrow Schema.

  Args:
      bq_schema_list: list of bigquery.SchemaField objects.

  Returns:
      pa.Schema or None if conversion fails.
  """
  arrow_fields = []
  for bq_field in bq_schema_list:
    af = _bq_to_arrow_field(bq_field)
    if af:
      arrow_fields.append(af)
    else:
      logger.error("Failed to convert schema due to field '%s'.", bq_field.name)
      return None
  return pa.schema(arrow_fields)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class RetryConfig:
  """Configuration for retrying failed BigQuery write operations.

  Attributes:
      max_retries: Maximum number of retry attempts.
      initial_delay: Initial delay between retries in seconds.
      multiplier: Multiplier for exponential backoff.
      max_delay: Maximum delay between retries in seconds.
  """

  max_retries: int = 3
  initial_delay: float = 1.0
  multiplier: float = 2.0
  max_delay: float = 10.0


@dataclass
class BigQueryLoggerConfig:
  """Configuration for the BigQueryAgentAnalyticsPlugin.

  Attributes:
      enabled: Whether logging is enabled.
      event_allowlist: list of event types to log. If None, all are allowed.
      event_denylist: list of event types to ignore.
      max_content_length: Max length for text content before truncation.
      table_id: BigQuery table ID.
      clustering_fields: Fields to cluster the table by.
      log_multi_modal_content: Whether to log detailed content parts.
      retry_config: Retry configuration for writes.
      batch_size: Number of rows per batch.
      batch_flush_interval: Max time to wait before flushing a batch.
      shutdown_timeout: Max time to wait for shutdown.
      queue_max_size: Max size of the in-memory queue.
      content_formatter: Optional custom formatter for content.
  """

  enabled: bool = True

  # V1 Configuration Parity
  event_allowlist: list[str] | None = None
  event_denylist: list[str] | None = None
  max_content_length: int = 500 * 1024  # Defaults to 500KB per text block
  table_id: str = "agent_events_v2"

  # V2 Configuration
  clustering_fields: list[str] = field(
      default_factory=lambda: ["event_type", "agent", "user_id"]
  )
  log_multi_modal_content: bool = True
  retry_config: RetryConfig = field(default_factory=RetryConfig)
  batch_size: int = 1
  batch_flush_interval: float = 1.0
  shutdown_timeout: float = 10.0
  queue_max_size: int = 10000
  content_formatter: Optional[Callable[[Any, str], Any]] = None
  # If provided, large content (images, audio, video, large text) will be offloaded to this GCS bucket.
  gcs_bucket_name: Optional[str] = None
  # If provided, this connection ID will be used as the authorizer for ObjectRef columns.
  # Format: "location.connection_id" (e.g. "us.my-connection")
  connection_id: Optional[str] = None


# ==============================================================================
# HELPER: TRACE MANAGER (Async-Safe with ContextVars)
# ==============================================================================

_trace_id_ctx = contextvars.ContextVar("_bq_analytics_trace_id", default=None)
_span_stack_ctx = contextvars.ContextVar(
    "_bq_analytics_span_stack", default=None
)
_span_times_ctx = contextvars.ContextVar(
    "_bq_analytics_span_times", default=None
)
_span_first_token_times_ctx = contextvars.ContextVar(
    "_bq_analytics_span_first_token_times", default=None
)


class TraceManager:
  """Manages OpenTelemetry-style trace and span context using contextvars."""

  @staticmethod
  def init_trace(callback_context: CallbackContext) -> None:
    if _trace_id_ctx.get() is None:
      _trace_id_ctx.set(callback_context.invocation_id)
      _span_stack_ctx.set([])
      _span_times_ctx.set({})
      _span_first_token_times_ctx.set({})

  @staticmethod
  def get_trace_id(callback_context: CallbackContext) -> Optional[str]:
    # Try contextvars first
    if trace_id := _trace_id_ctx.get():
      return trace_id
    # Fallback to callback_context for existing tests/legacy flows
    return callback_context.state.get("_bq_analytics_trace_id")

  @staticmethod
  def push_span(
      callback_context: CallbackContext, span_id: Optional[str] = None
  ) -> str:
    # Ensure trace is initialized
    if _trace_id_ctx.get() is None:
      TraceManager.init_trace(callback_context)

    span_id = span_id or str(uuid.uuid4())

    stack = _span_stack_ctx.get()
    if stack is None:
      # Should have been called by init_trace, but just in case
      stack = []
      _span_stack_ctx.set(stack)

    stack.append(span_id)

    times = _span_times_ctx.get()
    if times is None:
      times = {}
      _span_times_ctx.set(times)

    first_tokens = _span_first_token_times_ctx.get()
    if first_tokens is None:
      first_tokens = {}
      _span_first_token_times_ctx.set(first_tokens)

    times[span_id] = time.time()
    return span_id

  @staticmethod
  def pop_span() -> tuple[Optional[str], Optional[int]]:
    stack = _span_stack_ctx.get()
    if not stack:
      return None, None
    span_id = stack.pop()

    times = _span_times_ctx.get()
    start_time = times.pop(span_id, None) if times else None

    first_tokens = _span_first_token_times_ctx.get()
    if first_tokens:
      first_tokens.pop(span_id, None)

    duration_ms = int((time.time() - start_time) * 1000) if start_time else None
    return span_id, duration_ms

  @staticmethod
  def get_current_span_and_parent() -> tuple[Optional[str], Optional[str]]:
    stack = _span_stack_ctx.get()
    if not stack:
      return None, None
    return stack[-1], (stack[-2] if len(stack) > 1 else None)

  @staticmethod
  def get_current_span_id() -> Optional[str]:
    stack = _span_stack_ctx.get()
    return stack[-1] if stack else None

  @staticmethod
  def get_start_time(span_id: str) -> Optional[float]:
    times = _span_times_ctx.get()
    return times.get(span_id) if times else None

  @staticmethod
  def record_first_token(span_id: str) -> bool:
    """Records the current time as first token time if not already recorded.

    Returns:
        True if this was the first token (newly recorded), False otherwise.
    """
    first_tokens = _span_first_token_times_ctx.get()
    if first_tokens is None:
      first_tokens = {}
      _span_first_token_times_ctx.set(first_tokens)

    if span_id not in first_tokens:
      first_tokens[span_id] = time.time()
      return True
    return False

  @staticmethod
  def get_first_token_time(span_id: str) -> Optional[float]:
    first_tokens = _span_first_token_times_ctx.get()
    return first_tokens.get(span_id) if first_tokens else None


# ==============================================================================
# HELPER: BATCH PROCESSOR
# ==============================================================================
class BatchProcessor:
  """Handles asynchronous batching and writing of events to BigQuery."""

  def __init__(
      self,
      write_client: BigQueryWriteAsyncClient,
      arrow_schema: pa.Schema,
      write_stream: str,
      batch_size: int,
      flush_interval: float,
      retry_config: RetryConfig,
      queue_max_size: int,
      shutdown_timeout: float,
  ):
    """Initializes the BatchProcessor.

    Args:
        write_client: BigQueryWriteAsyncClient for writing rows.
        arrow_schema: PyArrow schema for serialization.
        write_stream: BigQuery write stream name.
        batch_size: Number of rows per batch.
        flush_interval: Max time to wait before flushing a batch.
        retry_config: Retry configuration.
        queue_max_size: Max size of the in-memory queue.
        shutdown_timeout: Max time to wait for shutdown.
    """
    self.write_client = write_client
    self.arrow_schema = arrow_schema
    self.write_stream = write_stream
    self.batch_size = batch_size
    self.flush_interval = flush_interval
    self.retry_config = retry_config
    self.shutdown_timeout = shutdown_timeout
    self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
        maxsize=queue_max_size
    )
    self._batch_processor_task: Optional[asyncio.Task] = None
    self._shutdown = False

  async def start(self):
    """Starts the batch writer worker task."""
    if self._batch_processor_task is None:
      self._batch_processor_task = asyncio.create_task(self._batch_writer())

  async def append(self, row: dict[str, Any]) -> None:
    """Appends a row to the queue for batching.

    Args:
        row: Dictionary representing a single row.
    """
    try:
      self._queue.put_nowait(row)
    except asyncio.QueueFull:
      logger.warning("BigQuery log queue full, dropping event.")

  def _prepare_arrow_batch(self, rows: list[dict[str, Any]]) -> pa.RecordBatch:
    """Prepares a PyArrow RecordBatch from a list of rows.

    Args:
        rows: list of row dictionaries.

    Returns:
        pa.RecordBatch for writing.
    """
    data = {field.name: [] for field in self.arrow_schema}
    for row in rows:
      for field in self.arrow_schema:
        value = row.get(field.name)
        # JSON fields must be serialized to strings for the Arrow layer
        field_metadata = self.arrow_schema.field(field.name).metadata
        is_json = False
        if field_metadata and b"ARROW:extension:name" in field_metadata:
          if field_metadata[b"ARROW:extension:name"] == b"google:sqlType:json":
            is_json = True

        arrow_field_type = self.arrow_schema.field(field.name).type
        is_struct = pa.types.is_struct(arrow_field_type)
        is_list = pa.types.is_list(arrow_field_type)

        if is_json:
          if value is not None:
            if isinstance(value, (dict, list)):
              try:
                value = json.dumps(value)
              except (TypeError, ValueError):
                value = str(value)
            elif isinstance(value, (str, bytes)):
              if isinstance(value, bytes):
                try:
                  value = value.decode("utf-8")
                except UnicodeDecodeError:
                  value = str(value)

              # Check if it's already a valid JSON object or array to avoid double-encoding
              is_already_json = False
              if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith(("{", "[")) and stripped.endswith(
                    ("}", "]")
                ):
                  try:
                    json.loads(value)
                    is_already_json = True
                  except (ValueError, TypeError):
                    pass

              if not is_already_json:
                try:
                  value = json.dumps(value)
                except (TypeError, ValueError):
                  value = str(value)
              # If is_already_json is True, we keep value as-is
            else:
              # For other types (int, float, bool), serialize to JSON equivalents
              try:
                value = json.dumps(value)
              except (TypeError, ValueError):
                value = str(value)
        elif isinstance(value, (dict, list)) and not is_struct and not is_list:
          if value is not None and not isinstance(value, (str, bytes)):
            try:
              value = json.dumps(value)
            except (TypeError, ValueError):
              value = str(value)
        data[field.name].append(value)
    return pa.RecordBatch.from_pydict(data, schema=self.arrow_schema)

  async def _batch_writer(self) -> None:
    """Worker task that batches and writes rows to BigQuery."""
    while not self._shutdown or not self._queue.empty():
      batch = []
      try:
        if self._shutdown:
          try:
            first_item = self._queue.get_nowait()
          except asyncio.QueueEmpty:
            break
        else:
          first_item = await asyncio.wait_for(
              self._queue.get(), timeout=self.flush_interval
          )

        batch.append(first_item)

        while len(batch) < self.batch_size:
          try:
            item = self._queue.get_nowait()
            batch.append(item)
          except asyncio.QueueEmpty:
            break

        if batch:
          try:
            await self._write_rows_with_retry(batch)
          finally:
            # Mark tasks as done ONLY after processing (write attempt)
            for _ in batch:
              self._queue.task_done()

      except asyncio.TimeoutError:
        continue
      except asyncio.CancelledError:
        logger.info("Batch writer task cancelled.")
        break
      except Exception as e:
        logger.error("Error in batch writer loop: %s", e, exc_info=True)
        await asyncio.sleep(1)

  async def _write_rows_with_retry(self, rows: list[dict[str, Any]]) -> None:
    """Writes a batch of rows to BigQuery with retry logic.

    Args:
        rows: list of row dictionaries to write.
    """
    attempt = 0
    delay = self.retry_config.initial_delay

    try:
      arrow_batch = self._prepare_arrow_batch(rows)
      serialized_schema = self.arrow_schema.serialize().to_pybytes()
      serialized_batch = arrow_batch.serialize().to_pybytes()

      req = bq_storage_types.AppendRowsRequest(
          write_stream=self.write_stream,
          trace_id=f"google-adk-bq-logger/{__version__}",
      )
      req.arrow_rows.writer_schema.serialized_schema = serialized_schema
      req.arrow_rows.rows.serialized_record_batch = serialized_batch
    except Exception as e:
      logger.error(
          "Failed to prepare Arrow batch (Data Loss): %s", e, exc_info=True
      )
      return

    while attempt <= self.retry_config.max_retries:
      try:

        async def requests_iter():
          yield req

        responses = await self.write_client.append_rows(requests_iter())
        async for response in responses:
          error = getattr(response, "error", None)
          error_code = getattr(error, "code", None)
          if error_code and error_code != 0:
            error_message = getattr(error, "message", "Unknown error")
            logger.warning(
                "BigQuery Write API returned error code %s: %s",
                error_code,
                error_message,
            )
            if error_code in [
                _GRPC_DEADLINE_EXCEEDED,
                _GRPC_INTERNAL,
                _GRPC_UNAVAILABLE,
            ]:  # Deadline, Internal, Unavailable
              raise ServiceUnavailable(error_message)
            else:
              if "schema mismatch" in error_message.lower():
                logger.error(
                    "BigQuery Schema Mismatch: %s. This usually means the"
                    " table schema does not match the expected schema.",
                    error_message,
                )
              else:
                logger.error("Non-retryable BigQuery error: %s", error_message)
                row_errors = getattr(response, "row_errors", [])
                if row_errors:
                  for row_error in row_errors:
                    logger.error("Row error details: %s", row_error)
                logger.error("Row content causing error: %s", rows)
              return
        return

      except (ServiceUnavailable, TooManyRequests, InternalServerError) as e:
        attempt += 1
        if attempt > self.retry_config.max_retries:
          logger.error(
              "BigQuery Batch Dropped after %s attempts. Last error: %s",
              self.retry_config.max_retries + 1,
              e,
          )
          return

        sleep_time = min(
            delay * (1 + random.random()), self.retry_config.max_delay
        )
        logger.warning(
            "BigQuery write failed (Attempt %s), retrying in %.2fs..."
            " Error: %s",
            attempt,
            sleep_time,
            e,
        )
        await asyncio.sleep(sleep_time)
        delay *= self.retry_config.multiplier
      except Exception as e:
        logger.error(
            "Unexpected BigQuery Write API error (Dropping batch): %s",
            e,
            exc_info=True,
        )
        return

  async def shutdown(self, timeout: float = 5.0) -> None:
    """Shuts down the BatchProcessor, draining the queue.

    Args:
        timeout: Maximum time to wait for the queue to drain.
    """
    self._shutdown = True
    logger.info("BatchProcessor shutting down, draining queue...")
    if self._batch_processor_task:
      try:
        await asyncio.wait_for(self._batch_processor_task, timeout=timeout)
      except asyncio.TimeoutError:
        logger.warning("BatchProcessor shutdown timed out, cancelling worker.")
        self._batch_processor_task.cancel()
      except Exception as e:
        logger.error("Error during BatchProcessor shutdown: %s", e)

  async def close(self) -> None:
    """Closes the processor and flushes remaining items."""
    if self._shutdown:
      return

    self._shutdown = True
    # Wait for queue to be empty
    try:
      await asyncio.wait_for(self._queue.join(), timeout=self.shutdown_timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
      logger.warning(
          "Timeout waiting for BigQuery batch queue to empty on shutdown."
      )

    # Cancel the writer task if it's still running (it should exit on _shutdown + empty queue)
    if self._batch_processor_task and not self._batch_processor_task.done():
      self._batch_processor_task.cancel()
      try:
        await self._batch_processor_task
      except asyncio.CancelledError:
        pass


# ==============================================================================
# HELPER: CONTENT PARSER (Length Limits Only)
# ==============================================================================
class ContentParser:
  """Parses content for logging with length limits and structure normalization."""

  def __init__(self, max_length: int) -> None:
    """Initializes the ContentParser.

    Args:
        max_length: Maximum length for text content.
    """
    self.max_length = max_length

  def _truncate(self, text: str) -> tuple[str, bool]:
    if self.max_length != -1 and text and len(text) > self.max_length:
      return text[: self.max_length] + "...[TRUNCATED]", True
    return text, False


class GCSOffloader:
  """Offloads content to GCS."""

  def __init__(
      self,
      project_id: str,
      bucket_name: str,
      executor: ThreadPoolExecutor,
      storage_client: Optional[storage.Client] = None,
  ):
    self.client = storage_client or storage.Client(project=project_id)
    self.bucket = self.client.bucket(bucket_name)
    self.executor = executor

  async def upload_content(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    """Async wrapper around blocking GCS upload."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        self.executor,
        functools.partial(self._upload_sync, data, content_type, path),
    )

  def _upload_sync(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    blob = self.bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{self.bucket.name}/{path}"


class HybridContentParser:
  """Parses content and offloads large/binary parts to GCS."""

  def __init__(
      self,
      offloader: Optional[GCSOffloader],
      trace_id: str,
      span_id: str,
      max_length: int = 20000,
      connection_id: Optional[str] = None,
  ):
    self.offloader = offloader
    self.trace_id = trace_id
    self.span_id = span_id
    self.max_length = max_length
    self.connection_id = connection_id
    self.inline_text_limit = 32 * 1024  # 32KB limit

  def _truncate(self, text: str) -> tuple[str, bool]:
    if self.max_length != -1 and len(text) > self.max_length:
      return (
          text[: self.max_length] + "...[TRUNCATED]",
          True,
      )
    return text, False

  async def _parse_content_object(
      self, content: types.Content | types.Part
  ) -> tuple[str, list[dict[str, Any]], bool]:
    """Parses a Content or Part object into summary text and content parts."""
    content_parts = []
    is_truncated = False
    summary_text = []

    parts = content.parts if hasattr(content, "parts") else [content]
    for idx, part in enumerate(parts):
      part_data = {
          "part_index": idx,
          "mime_type": "text/plain",
          "uri": None,
          "text": None,
          "part_attributes": "{}",
          "storage_mode": "INLINE",
          "object_ref": None,
      }

      # CASE A: It is already a URI (e.g. from user input)
      if hasattr(part, "file_data") and part.file_data:
        part_data["storage_mode"] = "EXTERNAL_URI"
        part_data["uri"] = part.file_data.file_uri
        part_data["mime_type"] = part.file_data.mime_type

      # CASE B: It is Binary/Inline Data (Image/Blob)
      elif hasattr(part, "inline_data") and part.inline_data:
        if self.offloader:
          ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".bin"
          path = f"{datetime.now().date()}/{self.trace_id}/{self.span_id}_p{idx}{ext}"
          try:
            uri = await self.offloader.upload_content(
                part.inline_data.data, part.inline_data.mime_type, path
            )
            part_data["storage_mode"] = "GCS_REFERENCE"
            part_data["uri"] = uri
            object_ref = {
                "uri": uri,
                "version": None,
                "authorizer": self.connection_id,
                "details": json.dumps({
                    "gcs_metadata": {"content_type": part.inline_data.mime_type}
                }),
            }
            part_data["object_ref"] = object_ref
            part_data["mime_type"] = part.inline_data.mime_type
            part_data["text"] = "[MEDIA OFFLOADED]"
          except Exception as e:
            logger.warning("Failed to offload content to GCS: %s", e)
            part_data["text"] = "[UPLOAD FAILED]"
        else:
          part_data["text"] = "[BINARY DATA]"

      # CASE C: Text
      elif hasattr(part, "text") and part.text:
        text_len = len(part.text.encode("utf-8"))
        # If max_length is set and smaller than inline limit, use it as threshold
        # to prefer offloading over truncation.
        offload_threshold = self.inline_text_limit
        if self.max_length != -1 and self.max_length < offload_threshold:
          offload_threshold = self.max_length

        if self.offloader and text_len > offload_threshold:
          # Text is too big, treat as file
          path = f"{datetime.now().date()}/{self.trace_id}/{self.span_id}_p{idx}.txt"
          try:
            uri = await self.offloader.upload_content(
                part.text, "text/plain", path
            )
            part_data["storage_mode"] = "GCS_REFERENCE"
            part_data["uri"] = uri
            object_ref = {
                "uri": uri,
                "version": None,
                "authorizer": self.connection_id,
                "details": json.dumps(
                    {"gcs_metadata": {"content_type": "text/plain"}}
                ),
            }
            part_data["object_ref"] = object_ref
            part_data["mime_type"] = "text/plain"
            part_data["text"] = part.text[:200] + "... [OFFLOADED]"
          except Exception as e:
            logger.warning("Failed to offload text to GCS: %s", e)
            clean_text, truncated = self._truncate(part.text)
            if truncated:
              is_truncated = True
            part_data["text"] = clean_text
            summary_text.append(clean_text)
        else:
          # Text is small or no offloader, keep inline
          clean_text, truncated = self._truncate(part.text)
          if truncated:
            is_truncated = True
          part_data["text"] = clean_text
          summary_text.append(clean_text)

      elif hasattr(part, "function_call") and part.function_call:
        part_data["mime_type"] = "application/json"
        part_data["text"] = f"Function: {part.function_call.name}"
        part_data["part_attributes"] = json.dumps(
            {"function_name": part.function_call.name}
        )

      content_parts.append(part_data)

    summary_str, truncated = self._truncate(" | ".join(summary_text))
    if truncated:
      is_truncated = True

    return summary_str, content_parts, is_truncated

  async def parse(self, content: Any) -> tuple[Any, list[dict[str, Any]], bool]:
    """Parses content into JSON payload and content parts, potentially offloading to GCS."""
    json_payload = {}
    content_parts = []
    is_truncated = False

    def process_text(t: str) -> tuple[str, bool]:
      return self._truncate(t)

    if isinstance(content, LlmRequest):
      # Handle Prompt
      messages = []
      contents = (
          content.contents
          if isinstance(content.contents, list)
          else [content.contents]
      )
      for c in contents:
        role = getattr(c, "role", "unknown")
        summary, parts, trunc = await self._parse_content_object(c)
        if trunc:
          is_truncated = True
        content_parts.extend(parts)
        messages.append({"role": role, "content": summary})

      if messages:
        json_payload["prompt"] = messages

      # Handle System Instruction
      if content.config and getattr(content.config, "system_instruction", None):
        si = content.config.system_instruction
        if isinstance(si, str):
          json_payload["system_prompt"] = si
        else:
          summary, parts, trunc = await self._parse_content_object(si)
          if trunc:
            is_truncated = True
          content_parts.extend(parts)
          json_payload["system_prompt"] = summary

    elif isinstance(content, (types.Content, types.Part)):
      summary, parts, trunc = await self._parse_content_object(content)
      return {"text_summary": summary}, parts, trunc

    elif isinstance(content, (dict, list)):
      json_payload, is_truncated = _recursive_smart_truncate(
          content, self.max_length
      )
    elif isinstance(content, str):
      json_payload, is_truncated = process_text(content)
    elif content is None:
      json_payload = None
    else:
      json_payload, is_truncated = process_text(str(content))

    return json_payload, content_parts, is_truncated


def _get_events_schema() -> list[bigquery.SchemaField]:
  """Returns the BigQuery schema for the events table."""
  return [
      bigquery.SchemaField(
          "timestamp",
          "TIMESTAMP",
          mode="REQUIRED",
          description=(
              "The UTC timestamp when the event occurred. Used for ordering"
              " events within a session."
          ),
      ),
      bigquery.SchemaField(
          "event_type",
          "STRING",
          mode="NULLABLE",
          description=(
              "The category of the event (e.g., 'LLM_REQUEST', 'TOOL_CALL',"
              " 'AGENT_RESPONSE'). Helps in filtering specific types of"
              " interactions."
          ),
      ),
      bigquery.SchemaField(
          "agent",
          "STRING",
          mode="NULLABLE",
          description=(
              "The name of the agent that generated this event. Useful for"
              " multi-agent systems."
          ),
      ),
      bigquery.SchemaField(
          "session_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "A unique identifier for the entire conversation session. Used"
              " to group all events belonging to a single user interaction."
          ),
      ),
      bigquery.SchemaField(
          "invocation_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "A unique identifier for a single turn or execution within a"
              " session. Groups related events like LLM request and response."
          ),
      ),
      bigquery.SchemaField(
          "user_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "The identifier of the end-user participating in the session,"
              " if available."
          ),
      ),
      bigquery.SchemaField(
          "trace_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "OpenTelemetry trace ID for distributed tracing across services."
          ),
      ),
      bigquery.SchemaField(
          "span_id",
          "STRING",
          mode="NULLABLE",
          description="OpenTelemetry span ID for this specific operation.",
      ),
      bigquery.SchemaField(
          "parent_span_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "OpenTelemetry parent span ID to reconstruct the operation"
              " hierarchy."
          ),
      ),
      bigquery.SchemaField(
          "content",
          "JSON",
          mode="NULLABLE",
          description=(
              "The primary payload of the event, stored as a JSON string. The"
              " structure depends on the event_type (e.g., prompt text for"
              " LLM_REQUEST, tool output for TOOL_RESPONSE)."
          ),
      ),
      bigquery.SchemaField(
          "content_parts",
          "RECORD",
          mode="REPEATED",
          fields=[
              bigquery.SchemaField(
                  "mime_type",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "The MIME type of the content part (e.g., 'text/plain',"
                      " 'image/png')."
                  ),
              ),
              bigquery.SchemaField(
                  "uri",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "The URI of the content part if stored externally"
                      " (e.g., GCS bucket path)."
                  ),
              ),
              bigquery.SchemaField(
                  "object_ref",
                  "RECORD",
                  mode="NULLABLE",
                  fields=[
                      bigquery.SchemaField(
                          "uri",
                          "STRING",
                          mode="NULLABLE",
                          description="The URI of the object.",
                      ),
                      bigquery.SchemaField(
                          "version",
                          "STRING",
                          mode="NULLABLE",
                          description="The version of the object.",
                      ),
                      bigquery.SchemaField(
                          "authorizer",
                          "STRING",
                          mode="NULLABLE",
                          description="The authorizer for the object.",
                      ),
                      bigquery.SchemaField(
                          "details",
                          "JSON",
                          mode="NULLABLE",
                          description="Additional details about the object.",
                      ),
                  ],
                  description=(
                      "The ObjectRef of the content part if stored externally."
                  ),
              ),
              bigquery.SchemaField(
                  "text",
                  "STRING",
                  mode="NULLABLE",
                  description="The raw text content if the part is text-based.",
              ),
              bigquery.SchemaField(
                  "part_index",
                  "INTEGER",
                  mode="NULLABLE",
                  description=(
                      "The zero-based index of this part within the content."
                  ),
              ),
              bigquery.SchemaField(
                  "part_attributes",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "Additional metadata for this content part as a JSON"
                      " object (serialized to string)."
                  ),
              ),
              bigquery.SchemaField(
                  "storage_mode",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "Indicates how the content part is stored (e.g.,"
                      " 'INLINE', 'GCS_REFERENCE', 'EXTERNAL_URI')."
                  ),
              ),
          ],
          description=(
              "For multi-modal events, contains a list of content parts"
              " (text, images, etc.)."
          ),
      ),
      bigquery.SchemaField(
          "attributes",
          "JSON",
          mode="NULLABLE",
          description=(
              "A JSON object containing arbitrary key-value pairs for"
              " additional event metadata not covered by standard fields."
          ),
      ),
      bigquery.SchemaField(
          "latency_ms",
          "JSON",
          mode="NULLABLE",
          description=(
              "A JSON object containing latency measurements, such as"
              " 'total_ms' and 'time_to_first_token_ms'."
          ),
      ),
      bigquery.SchemaField(
          "status",
          "STRING",
          mode="NULLABLE",
          description="The outcome of the event, typically 'OK' or 'ERROR'.",
      ),
      bigquery.SchemaField(
          "error_message",
          "STRING",
          mode="NULLABLE",
          description="Detailed error message if the status is 'ERROR'.",
      ),
      bigquery.SchemaField(
          "is_truncated",
          "BOOLEAN",
          mode="NULLABLE",
          description=(
              "Boolean flag indicating if the 'content' field was truncated"
              " because it exceeded the maximum allowed size."
          ),
      ),
  ]


# ==============================================================================
# MAIN PLUGIN
# ==============================================================================
_GLOBAL_WRITE_CLIENT: Optional[BigQueryWriteAsyncClient] = None
_GLOBAL_CLIENT_LOCK = asyncio.Lock()


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """BigQuery Agent Analytics Plugin (v2.0 using Write API).

  Logs agent events (LLM requests, tool calls, etc.) to BigQuery for analytics.
  Uses the BigQuery Write API for efficient, asynchronous, and reliable logging.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      *,
      table_id: Optional[str] = None,
      config: Optional[BigQueryLoggerConfig] = None,
      location: str = "US",
  ) -> None:
    """Initializes the BigQueryAgentAnalyticsPlugin.

    Args:
        project_id: Google Cloud project ID.
        dataset_id: BigQuery dataset ID.
        table_id: BigQuery table ID (optional, overrides config).
        config: BigQueryLoggerConfig (optional).
        location: BigQuery location (default: "US").
    """
    super().__init__(name="bigquery_agent_analytics")
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.config = config or BigQueryLoggerConfig()
    self.table_id = table_id or self.config.table_id
    self.location = location

    self._started = False
    self._is_shutting_down = False
    self._setup_lock = None
    self.client = None
    self.write_client = None
    self.write_stream = None
    self.batch_processor = None
    self._executor = None
    self.offloader: Optional[GCSOffloader] = None
    self.parser: Optional[HybridContentParser] = None

  def _format_content_safely(
      self, content: Optional[types.Content]
  ) -> tuple[str, bool]:
    """Formats content using config.content_formatter or default formatter.

    Args:
        content: The content to format.

    Returns:
        A tuple of (formatted_string, is_truncated).
    """
    if content is None:
      return "None", False
    try:
      # If a custom formatter is provided, we could try to use it here too,
      # but it expects (content, event_type). For internal formatting,
      # we stick to the default _format_content but respect max_len.
      return _format_content(content, max_len=self.config.max_content_length)
    except Exception as e:
      logger.warning("Content formatter failed: %s", e)
      return "[FORMATTING FAILED]", False

  async def _lazy_setup(self, **kwargs) -> None:
    """Performs lazy initialization of BigQuery clients and resources."""
    if self._started:
      return
    loop = asyncio.get_running_loop()

    if not self.client:
      if self._executor is None:
        self._executor = ThreadPoolExecutor(max_workers=1)

      self.client = await loop.run_in_executor(
          self._executor,
          lambda: bigquery.Client(
              project=self.project_id, location=self.location
          ),
      )

    self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
    self._schema = _get_events_schema()
    await loop.run_in_executor(self._executor, self._ensure_schema_exists)

    if not self.write_client:
      global _GLOBAL_WRITE_CLIENT
      async with _GLOBAL_CLIENT_LOCK:
        if _GLOBAL_WRITE_CLIENT is None:

          def get_credentials():
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return creds

          creds = await loop.run_in_executor(self._executor, get_credentials)
          client_info = gapic_client_info.ClientInfo(
              user_agent=f"google-adk-bq-logger/{__version__}"
          )
          # Initialize the async client in the current event loop, not in the
          # executor.
          _GLOBAL_WRITE_CLIENT = BigQueryWriteAsyncClient(
              credentials=creds, client_info=client_info
          )
        self.write_client = _GLOBAL_WRITE_CLIENT

      self.write_stream = f"projects/{self.project_id}/datasets/{self.dataset_id}/tables/{self.table_id}/_default"

    if not self.batch_processor:
      self.arrow_schema = to_arrow_schema(self._schema)
      if not self.arrow_schema:
        raise RuntimeError("Failed to convert BigQuery schema to Arrow schema.")

      self.offloader = None
      if self.config.gcs_bucket_name:
        self.offloader = GCSOffloader(
            self.project_id,
            self.config.gcs_bucket_name,
            self._executor,
            storage_client=kwargs.get("storage_client"),
        )

      self.parser = HybridContentParser(
          self.offloader,
          "",
          "",
          max_length=self.config.max_content_length,
          connection_id=self.config.connection_id,
      )
      self.batch_processor = BatchProcessor(
          write_client=self.write_client,
          arrow_schema=self.arrow_schema,
          write_stream=self.write_stream,
          batch_size=self.config.batch_size,
          flush_interval=self.config.batch_flush_interval,
          retry_config=self.config.retry_config,
          queue_max_size=self.config.queue_max_size,
          shutdown_timeout=self.config.shutdown_timeout,
      )
      await self.batch_processor.start()

      # Register cleanup to ensure logs are flushed if user forgets to close
      # Use weakref to avoid circular references that prevent garbage collection
      atexit.register(self._atexit_cleanup, weakref.proxy(self.batch_processor))

  @staticmethod
  @staticmethod
  def _atexit_cleanup(batch_processor: "BatchProcessor") -> None:
    """Clean up batch processor on script exit."""
    # Check if the batch_processor object is still alive
    if batch_processor and not batch_processor._shutdown:
      # Emergency Flush: Rescue any logs remaining in the queue
      remaining_items = []
      try:
        while True:
          remaining_items.append(batch_processor._queue.get_nowait())
      except (asyncio.QueueEmpty, AttributeError):
        pass

      if remaining_items:
        # We need a new loop and client to flush these
        async def rescue_flush():
          try:
            # Create a short-lived client just for this flush
            try:
              # Note: This relies on google.auth.default() working in this context.
              # pylint: disable=g-import-not-at-top
              from google.cloud.bigquery_storage_v1.services.big_query_write.async_client import BigQueryWriteAsyncClient

              # pylint: enable=g-import-not-at-top
              client = BigQueryWriteAsyncClient()
            except Exception as e:
              logger.warning("Could not create rescue client: %s", e)
              return

            # Patch batch_processor.write_client temporarily
            old_client = batch_processor.write_client
            batch_processor.write_client = client
            try:
              # Force a write
              await batch_processor._write_rows_with_retry(remaining_items)
              logger.info("Rescued logs flushed successfully.")
            except Exception as e:
              logger.error("Failed to flush rescued logs: %s", e)
            finally:
              batch_processor.write_client = old_client
          except Exception as e:
            logger.error("Rescue flush failed: %s", e)

        try:
          loop = asyncio.new_event_loop()
          loop.run_until_complete(rescue_flush())
          loop.close()
        except Exception as e:
          logger.error("Failed to run rescue loop: %s", e)

  def _ensure_schema_exists(self) -> None:
    """Ensures the BigQuery table exists with the correct schema."""
    try:
      self.client.get_table(self.full_table_id)
    except cloud_exceptions.NotFound:
      logger.info("Table %s not found, creating table.", self.full_table_id)
      tbl = bigquery.Table(self.full_table_id, schema=self._schema)
      tbl.time_partitioning = bigquery.TimePartitioning(
          type_=bigquery.TimePartitioningType.DAY, field="timestamp"
      )
      tbl.clustering_fields = self.config.clustering_fields
      try:
        self.client.create_table(tbl)
      except cloud_exceptions.Conflict:
        pass
      except Exception as e:
        logger.error(
            "Could not create table %s: %s",
            self.full_table_id,
            e,
            exc_info=True,
        )
    except Exception as e:
      logger.error(
          "Error checking for table %s: %s",
          self.full_table_id,
          e,
          exc_info=True,
      )

  async def shutdown(self, timeout: float | None = None) -> None:
    """Shuts down the plugin and releases resources.

    Args:
        timeout: Maximum time to wait for the queue to drain.
    """
    if self._is_shutting_down:
      return
    self._is_shutting_down = True
    t = timeout if timeout is not None else self.config.shutdown_timeout
    loop = asyncio.get_running_loop()
    try:
      if self.batch_processor:
        await self.batch_processor.shutdown(timeout=t)
      if self.write_client and getattr(self.write_client, "transport", None):
        # Only close the client if it's NOT the global one (unlikely with new logic,
        # but good for safety if injected manually) or if we decide to handle global close differently.
        # For now, we DO NOT close the global client to allow reuse.
        if self.write_client is not _GLOBAL_WRITE_CLIENT:
          await self.write_client.transport.close()
      if self.client:
        if self._executor:
          executor = self._executor
          await loop.run_in_executor(None, lambda: executor.shutdown(wait=True))
          self._executor = None
      self.write_client = None
      self.client = None
      self._is_shutting_down = False
    except Exception as e:
      logger.error("Error during shutdown: %s", e, exc_info=True)
    self._is_shutting_down = False
    self._started = False

  def __getstate__(self):
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    state["_setup_lock"] = None
    state["client"] = None
    state["write_client"] = None
    state["write_stream"] = None
    state["batch_processor"] = None
    state["_executor"] = None
    state["offloader"] = None
    state["parser"] = None
    state["_started"] = False
    state["_is_shutting_down"] = False
    return state

  def __setstate__(self, state):
    """Custom unpickling to restore state."""
    self.__dict__.update(state)

  async def __aenter__(self) -> BigQueryAgentAnalyticsPlugin:
    await self._ensure_started()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.shutdown()

  async def _ensure_started(self, **kwargs) -> None:
    """Ensures that the plugin is started and initialized."""
    if not self._started:
      # Kept original lock name as it was not explicitly changed in the
      if self._setup_lock is None:
        self._setup_lock = asyncio.Lock()
      async with self._setup_lock:
        if not self._started:
          try:
            await self._lazy_setup(**kwargs)
            self._started = True
          except Exception as e:
            logger.error("Failed to initialize BigQuery Plugin: %s", e)

  async def _log_event(
      self,
      event_type: str,
      callback_context: CallbackContext,
      raw_content: Any = None,
      is_truncated: bool = False,
      **kwargs,
  ) -> None:
    """Logs an event to BigQuery.

    Args:
        event_type: The type of event (e.g., 'LLM_REQUEST').
        callback_context: The callback context.
        raw_content: The raw content to log.
        is_truncated: Whether the content is already truncated.
        **kwargs: Additional attributes to log.
    """
    if not self.config.enabled or self._is_shutting_down:
      return
    if self.config.event_denylist and event_type in self.config.event_denylist:
      return
    if (
        self.config.event_allowlist
        and event_type not in self.config.event_allowlist
    ):
      return

    if not self._started:
      await self._ensure_started()
      if not self._started:
        return

    timestamp = datetime.now(timezone.utc)
    if self.config.content_formatter:
      try:
        raw_content = self.config.content_formatter(raw_content, event_type)
      except Exception as e:
        logger.warning("Content formatter failed: %s", e)

    trace_id = TraceManager.get_trace_id(callback_context)
    current_span_id, current_parent_span_id = (
        TraceManager.get_current_span_and_parent()
    )

    span_id = current_span_id
    if "span_id_override" in kwargs:
      val = kwargs.pop("span_id_override")
      if val is not None:
        span_id = val

    parent_span_id = current_parent_span_id
    if "parent_span_id_override" in kwargs:
      val = kwargs.pop("parent_span_id_override")
      if val is not None:
        parent_span_id = val

    # Use HybridContentParser if offloader is available, otherwise use default
    # Re-initialize parser with current trace/span IDs for GCS pathing
    self.parser = HybridContentParser(
        self.offloader,
        trace_id or "no_trace",
        span_id or "no_span",
        self.config.max_content_length,
        connection_id=self.config.connection_id,
    )
    content_json, content_parts, parser_truncated = await self.parser.parse(
        raw_content
    )
    is_truncated = is_truncated or parser_truncated

    total_latency = kwargs.get("latency_ms")
    tfft = kwargs.get("time_to_first_token_ms")
    latency_json = {}
    if total_latency is not None:
      latency_json["total_ms"] = total_latency
    if tfft is not None:
      latency_json["time_to_first_token_ms"] = tfft
    kwargs.pop("latency_ms", None)
    kwargs.pop("time_to_first_token_ms", None)

    # Check if content was truncated by the parser or explicitly passed
    # (Already handled by parser_truncated above, but keeping for safety or if other logic added later)

    status = kwargs.pop("status", "OK")
    error_message = kwargs.pop("error_message", None)

    # Serialize remaining kwargs to JSON string for attributes
    try:
      attributes_json = json.dumps(kwargs)
    except (TypeError, ValueError):
      # Fallback for non-serializable objects
      attributes_json = json.dumps(kwargs, default=str)

    row = {
        "timestamp": timestamp,
        "event_type": event_type,
        "agent": callback_context.agent_name,
        "user_id": callback_context.user_id,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "content": content_json,
        "content_parts": (
            content_parts if self.config.log_multi_modal_content else []
        ),
        "attributes": attributes_json,
        "latency_ms": latency_json if latency_json else None,
        "status": status,
        "error_message": error_message,
        "is_truncated": is_truncated,
    }

    if self.batch_processor:
      await self.batch_processor.append(row)

  # --- UPDATED CALLBACKS FOR V1 PARITY ---

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
      **kwargs,
  ) -> None:
    """Parity with V1: Logs USER_MESSAGE_RECEIVED event.

    Args:
        invocation_context: The context of the current invocation.
        user_message: The message content received from the user.
    """
    await self._log_event(
        "USER_MESSAGE_RECEIVED",
        CallbackContext(invocation_context),
        raw_content=user_message,
    )

  async def before_run_callback(
      self, *, invocation_context: "InvocationContext", **kwargs
  ) -> None:
    """Callback before the agent run starts.

    Args:
        invocation_context: The context of the current invocation.
    """
    await self._ensure_started()
    await self._log_event(
        "INVOCATION_STARTING", CallbackContext(invocation_context)
    )

  async def after_run_callback(
      self, *, invocation_context: "InvocationContext", **kwargs
  ) -> None:
    """Callback after the agent run completes.

    Args:
        invocation_context: The context of the current invocation.
    """
    await self._log_event(
        "INVOCATION_COMPLETED", CallbackContext(invocation_context)
    )

  async def before_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext, **kwargs
  ) -> None:
    """Callback before an agent starts processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    TraceManager.init_trace(callback_context)
    TraceManager.push_span(callback_context)
    await self._log_event(
        "AGENT_STARTING", callback_context, raw_content=agent.instruction
    )

  async def after_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext, **kwargs
  ) -> None:
    """Callback after an agent completes processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    span_id, duration = TraceManager.pop_span()
    # When popping, the current stack now points to parent.
    # The event we are logging ("AGENT_COMPLETED") belongs to the span we just popped.
    # So we must override span_id to be the popped span, and parent to be current top of stack.
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "AGENT_COMPLETED",
        callback_context,
        latency_ms=duration,
        span_id_override=span_id,
        parent_span_id_override=parent_span_id,
    )

  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      **kwargs,
  ) -> None:
    """Callback before LLM call.

    Logs the LLM request details including:
    1. Prompt content
    2. System instruction (if available)

    The content is formatted as 'Prompt: {prompt} | System Prompt:
    {system_prompt}'.
    """

    # 5. Attributes (Config & Tools)
    attributes = {}
    if llm_request.config:
      config_dict = {}
      for field_name in [
          "temperature",
          "top_p",
          "top_k",
          "candidate_count",
          "max_output_tokens",
          "stop_sequences",
      ]:
        if val := getattr(llm_request.config, field_name, None):
          config_dict[field_name] = val
      if config_dict:
        attributes["llm_config"] = config_dict

    if hasattr(llm_request, "tools_dict") and llm_request.tools_dict:
      attributes["tools"] = list(llm_request.tools_dict.keys())

    # Merge any additional kwargs into attributes
    attributes.update(kwargs)

    TraceManager.push_span(callback_context)
    await self._log_event(
        "LLM_REQUEST",
        callback_context,
        raw_content=llm_request,
        **attributes,
    )

  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: "LlmResponse",
      **kwargs,
  ) -> None:
    """Callback after LLM call.

    Logs the LLM response details including:
    1. Response content
    2. Token usage (if available)

    The content is formatted as 'Response: {content} | Usage: {usage}'.

    Args:
        callback_context: The callback context.
        llm_response: The LLM response object.
    """
    content_dict = {}
    is_truncated = False
    if llm_response.content:
      part_str, part_truncated = self._format_content_safely(
          llm_response.content
      )
      if part_str:
        content_dict["response"] = part_str
      if part_truncated:
        is_truncated = True

    if llm_response.usage_metadata:
      usage = llm_response.usage_metadata
      usage_dict = {}
      if hasattr(usage, "prompt_token_count"):
        usage_dict["prompt"] = usage.prompt_token_count
      if hasattr(usage, "candidates_token_count"):
        usage_dict["completion"] = usage.candidates_token_count
      if hasattr(usage, "total_token_count"):
        usage_dict["total"] = usage.total_token_count
      if usage_dict:
        content_dict["usage"] = usage_dict

    if content_dict:
      content_str = content_dict
    else:
      content_str = None

    span_id = TraceManager.get_current_span_id()
    _, parent_span_id = TraceManager.get_current_span_and_parent()

    is_popped = False
    duration = 0
    tfft = None

    if hasattr(llm_response, "partial") and llm_response.partial:
      # Streaming chunk - do NOT pop span yet
      if span_id:
        TraceManager.record_first_token(span_id)
        start_time = TraceManager.get_start_time(span_id)
        first_token = TraceManager.get_first_token_time(span_id)
        if start_time:
          duration = int((time.time() - start_time) * 1000)
        if start_time and first_token:
          tfft = int((first_token - start_time) * 1000)
    else:
      # Final response - pop span
      start_time = None
      if span_id:
        # Ensure we have first token time even if it wasn't streaming (or single chunk)
        TraceManager.record_first_token(span_id)
        start_time = TraceManager.get_start_time(span_id)
        first_token = TraceManager.get_first_token_time(span_id)
        if start_time and first_token:
          tfft = int((first_token - start_time) * 1000)

      # ACTUALLY pop the span
      popped_span_id, duration = TraceManager.pop_span()
      is_popped = True

      # If we popped, the span_id from get_current_span_and_parent() above is correct for THIS event
      # Wait, if we popped, get_current_span_and_parent() now returns parent.
      # But we captured span_id BEFORE popping. So we should use THAT.
      # If is_popped is True, we must override span_id in log_event to use the popped one.
      # Otherwise log_event will fetch current stack (which is parent).
      span_id = popped_span_id or span_id

    extra_kwargs = {}
    if tfft is not None:
      extra_kwargs["time_to_first_token_ms"] = tfft

    await self._log_event(
        "LLM_RESPONSE",
        callback_context,
        raw_content=content_str,
        is_truncated=is_truncated,
        latency_ms=duration,
        span_id_override=span_id if is_popped else None,
        parent_span_id_override=parent_span_id
        if is_popped
        else None,  # Use pre-pop state
        **extra_kwargs,
        **kwargs,
    )

  async def on_model_error_callback(
      self, *, callback_context: CallbackContext, error: Exception, **kwargs
  ) -> None:
    """Callback on LLM error.

    Args:
        callback_context: The callback context.
        error: The exception that occurred.
        **kwargs: Additional arguments.
    """
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()
    await self._log_event(
        "LLM_ERROR",
        callback_context,
        error_message=str(error),
        latency_ms=duration,
        span_id_override=span_id,
        parent_span_id_override=parent_span_id,
    )

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      **kwargs,
  ) -> None:
    """Callback before tool execution.

    Args:
        tool: The tool being executed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
    """
    args_truncated, is_truncated = _recursive_smart_truncate(
        tool_args, self.config.max_content_length
    )
    content_dict = {"tool": tool.name, "args": args_truncated}
    TraceManager.push_span(tool_context)
    await self._log_event(
        "TOOL_STARTING",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
    )

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
      **kwargs,
  ) -> None:
    """Callback after tool execution.

    Args:
        tool: The tool that was executed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
        result: The response from the tool.
    """
    resp_truncated, is_truncated = _recursive_smart_truncate(
        result, self.config.max_content_length
    )
    content_dict = {"tool": tool.name, "result": resp_truncated}
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "TOOL_COMPLETED",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
        latency_ms=duration,
        span_id_override=span_id,
        parent_span_id_override=parent_span_id,
    )

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
      **kwargs,
  ) -> None:
    """Callback on tool error.

    Args:
        tool: The tool that failed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
        error: The exception that occurred.
        **kwargs: Additional arguments.
    """
    args_truncated, is_truncated = _recursive_smart_truncate(
        tool_args, self.config.max_content_length
    )
    content_dict = {"tool": tool.name, "args": args_truncated}
    _, duration = TraceManager.pop_span()
    await self._log_event(
        "TOOL_ERROR",
        tool_context,
        raw_content=content_dict,
        error_message=str(error),
        is_truncated=is_truncated,
        latency_ms=duration,
    )
