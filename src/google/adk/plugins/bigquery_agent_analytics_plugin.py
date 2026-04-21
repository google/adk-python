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

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
import contextvars
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import functools
import json
import logging
import mimetypes
import os

# Enable gRPC fork support so child processes created via os.fork()
# can safely create new gRPC channels.  Must be set before grpc's
# C-core is loaded (which happens through the google.api_core
# imports below).  setdefault respects any explicit user override.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

import random
import time
from types import MappingProxyType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
import uuid
import weakref

from google.api_core import client_options
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
from opentelemetry import trace
import pyarrow as pa

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..utils._telemetry_context import _is_visual_builder
from ..version import __version__
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger: logging.Logger = logging.getLogger("google_adk." + __name__)
tracer = trace.get_tracer(
    "google.adk.plugins.bigquery_agent_analytics", __version__
)

# Bumped when the schema changes (1 → 2 → 3 …). Used as a table
# label for governance and to decide whether auto-upgrade should run.
_SCHEMA_VERSION = "1"
_SCHEMA_VERSION_LABEL_KEY = "adk_schema_version"

_HITL_EVENT_MAP = MappingProxyType({
    "adk_request_credential": "HITL_CREDENTIAL_REQUEST",
    "adk_request_confirmation": "HITL_CONFIRMATION_REQUEST",
    "adk_request_input": "HITL_INPUT_REQUEST",
})

# Track all living plugin instances so the fork handler can reset
# them proactively in the child, before _ensure_started runs.
_LIVE_PLUGINS: weakref.WeakSet = weakref.WeakSet()


def _after_fork_in_child() -> None:
  """Reset every living plugin instance after os.fork()."""
  for plugin in list(_LIVE_PLUGINS):
    try:
      plugin._reset_runtime_state()
    except Exception:
      pass


if hasattr(os, "register_at_fork"):
  os.register_at_fork(after_in_child=_after_fork_in_child)


def _safe_callback(func):
  """Decorator that catches and logs exceptions in plugin callbacks.

  Prevents plugin errors from propagating to the runner and crashing
  the agent run. All callback exceptions are logged and swallowed.
  """

  @functools.wraps(func)
  async def wrapper(self, **kwargs):
    try:
      return await func(self, **kwargs)
    except Exception:
      logger.exception(
          "BigQuery analytics plugin error in %s; skipping.",
          func.__name__,
      )
      return None

  return wrapper


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


def _find_transfer_target(agent, agent_name: str):
  """Find a transfer target agent by name in the accessible agent tree.

  Searches the current agent's sub-agents, parent, and peer agents
  to locate the transfer target.

  Args:
      agent: The current agent executing the transfer.
      agent_name: The name of the transfer target to find.

  Returns:
      The matching agent object, or None if not found.
  """
  for sub in getattr(agent, "sub_agents", []):
    if sub.name == agent_name:
      return sub
  parent = getattr(agent, "parent_agent", None)
  if parent is not None and parent.name == agent_name:
    return parent
  if parent is not None:
    for peer in getattr(parent, "sub_agents", []):
      if peer.name == agent_name and peer.name != agent.name:
        return peer
  return None


def _get_tool_origin(
    tool: "BaseTool",
    tool_args: Optional[dict[str, Any]] = None,
    tool_context: Optional["ToolContext"] = None,
) -> str:
  """Returns the provenance category of a tool.

  Uses lazy imports to avoid circular dependencies.

  For ``TransferToAgentTool`` the classification is **call-level**: when
  *tool_args* and *tool_context* are supplied the selected
  ``agent_name`` is resolved against the agent tree so that transfers
  to a ``RemoteA2aAgent`` are labelled ``TRANSFER_A2A`` rather than
  the generic ``TRANSFER_AGENT``.

  Args:
      tool: The tool instance.
      tool_args: Optional tool arguments, used for call-level classification of
        TransferToAgentTool.
      tool_context: Optional tool context, used to access the agent tree for
        TransferToAgentTool classification.

  Returns:
      One of LOCAL, MCP, A2A, SUB_AGENT, TRANSFER_AGENT,
      TRANSFER_A2A, or UNKNOWN.
  """
  # Import lazily to avoid circular dependencies.
  # pylint: disable=g-import-not-at-top
  from ..tools.agent_tool import AgentTool  # pytype: disable=import-error
  from ..tools.function_tool import FunctionTool  # pytype: disable=import-error
  from ..tools.transfer_to_agent_tool import TransferToAgentTool  # pytype: disable=import-error

  try:
    from ..tools.mcp_tool.mcp_tool import McpTool  # pytype: disable=import-error
  except ImportError:
    McpTool = None

  try:
    from ..agents.remote_a2a_agent import RemoteA2aAgent  # pytype: disable=import-error
  except ImportError:
    RemoteA2aAgent = None

  # Order matters: TransferToAgentTool is a subclass of FunctionTool.
  if McpTool is not None and isinstance(tool, McpTool):
    return "MCP"
  if isinstance(tool, TransferToAgentTool):
    if RemoteA2aAgent is not None and tool_args and tool_context:
      agent_name = tool_args.get("agent_name")
      if agent_name:
        target = _find_transfer_target(
            tool_context._invocation_context.agent,
            agent_name,
        )
        if target is not None and isinstance(target, RemoteA2aAgent):
          return "TRANSFER_A2A"
    return "TRANSFER_AGENT"
  if isinstance(tool, AgentTool):
    if RemoteA2aAgent is not None and isinstance(tool.agent, RemoteA2aAgent):
      return "A2A"
    return "SUB_AGENT"
  if isinstance(tool, FunctionTool):
    return "LOCAL"
  return "UNKNOWN"


_SENSITIVE_KEYS = frozenset({
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "password",
})


def _recursive_smart_truncate(
    obj: Any, max_len: int, seen: Optional[set[int]] = None
) -> tuple[Any, bool]:
  """Recursively truncates string values within a dict or list.

  Redacts sensitive keys corresponding to OAuth tokens and secrets
  prior to serialization into BigQuery JSON strings.

  Args:
      obj: The object to truncate.
      max_len: Maximum length for string values.
      seen: Set of object IDs visited in the current recursion stack.

  Returns:
      A tuple of (truncated_object, is_truncated).
  """
  if seen is None:
    seen = set()

  obj_id = id(obj)
  if obj_id in seen:
    return "[CIRCULAR_REFERENCE]", False

  # Track compound objects to detect cycles
  is_compound = (
      isinstance(obj, (dict, list, tuple))
      or (dataclasses.is_dataclass(obj) and not isinstance(obj, type))
      or hasattr(obj, "model_dump")
      or hasattr(obj, "dict")
      or hasattr(obj, "to_dict")
  )

  if is_compound:
    seen.add(obj_id)

  try:
    if isinstance(obj, str):
      if max_len != -1 and len(obj) > max_len:
        return obj[:max_len] + "...[TRUNCATED]", True
      return obj, False
    elif isinstance(obj, dict):
      truncated_any = False
      # Use dict comprehension for potentially slightly better performance,
      # but explicit loop is fine for clarity given recursive nature.
      new_dict = {}
      for k, v in obj.items():
        if isinstance(k, str):
          k_lower = k.lower()
          if k_lower in _SENSITIVE_KEYS or k_lower.startswith("temp:"):
            new_dict[k] = "[REDACTED]"
            continue

        val, trunc = _recursive_smart_truncate(v, max_len, seen)
        if trunc:
          truncated_any = True
        new_dict[k] = val
      return new_dict, truncated_any
    elif isinstance(obj, (list, tuple)):
      truncated_any = False
      new_list = []
      # Explicit loop to handle flag propagation
      for i in obj:
        val, trunc = _recursive_smart_truncate(i, max_len, seen)
        if trunc:
          truncated_any = True
        new_list.append(val)
      return type(obj)(new_list), truncated_any
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
      # Manually iterate fields to preserve 'seen' context, avoiding dataclasses.asdict recursion
      as_dict = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
      return _recursive_smart_truncate(as_dict, max_len, seen)
    elif hasattr(obj, "model_dump") and callable(obj.model_dump):
      # Pydantic v2
      try:
        return _recursive_smart_truncate(obj.model_dump(), max_len, seen)
      except Exception:
        pass
    elif hasattr(obj, "dict") and callable(obj.dict):
      # Pydantic v1
      try:
        return _recursive_smart_truncate(obj.dict(), max_len, seen)
      except Exception:
        pass
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
      # Common pattern for custom objects
      try:
        return _recursive_smart_truncate(obj.to_dict(), max_len, seen)
      except Exception:
        pass
    elif obj is None or isinstance(obj, (int, float, bool)):
      # Basic types are safe
      return obj, False

    # Fallback for unknown types: Convert to string to ensure JSON validity
    # We return string representation of the object, which is a valid JSON string value.
    return str(obj), False
  finally:
    if is_compound:
      seen.remove(obj_id)


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
      gcs_bucket_name: GCS bucket for offloading large content.
      connection_id: BigQuery connection ID for ObjectRef columns.
      log_session_metadata: Whether to log session metadata.
      custom_tags: Static custom tags to attach to every event.
      auto_schema_upgrade: Whether to auto-add new columns on schema evolution.
      create_views: Whether to auto-create per-event-type views.
      view_prefix: Prefix for auto-created view names. Default ``"v"`` produces
        views like ``v_llm_request``. Set a distinct prefix per table when
        multiple plugin instances share one dataset to avoid view-name
        collisions.
  """

  enabled: bool = True

  # V1 Configuration Parity
  event_allowlist: list[str] | None = None
  event_denylist: list[str] | None = None
  max_content_length: int = 500 * 1024  # Defaults to 500KB per text block
  table_id: str = "agent_events"

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

  # Toggle for session metadata (e.g. gchat thread-id)
  log_session_metadata: bool = True
  # Static custom tags (e.g. {"agent_role": "sales"})
  custom_tags: dict[str, Any] = field(default_factory=dict)
  # Automatically add new columns to existing tables when the plugin
  # schema evolves.  Only additive changes are made (columns are never
  # dropped or altered).  Safe to leave enabled; a version label on the
  # table ensures the diff runs at most once per schema version.
  auto_schema_upgrade: bool = True
  # Automatically create per-event-type BigQuery views that unnest
  # JSON columns into typed, queryable columns.
  create_views: bool = True
  # Prefix for auto-created per-event-type view names.
  # Default "v" produces views like ``v_llm_request``.  Set a distinct
  # prefix per table when multiple plugin instances share one dataset
  # to avoid view-name collisions (e.g. ``"v_staging"`` →
  # ``v_staging_llm_request``).
  view_prefix: str = "v"


# ==============================================================================
# HELPER: TRACE MANAGER (Async-Safe with ContextVars)
# ==============================================================================
# NOTE: These contextvars are module-global, not plugin-instance-scoped.
# This is safe in practice for two reasons:
#   1. PluginManager enforces name-uniqueness, preventing two BQ plugin
#      instances on the same Runner.
#   2. Concurrent asyncio tasks (e.g. two Runners in asyncio.gather) each
#      get an isolated contextvar copy, so they don't interfere.
# The only problematic case would be two plugin instances interleaved
# within the *same* asyncio task without task boundaries — which the
# framework's PluginManager already prevents.

_root_agent_name_ctx = contextvars.ContextVar(
    "_bq_analytics_root_agent_name", default=None
)

# Tracks the invocation_id that owns the current span stack so that
# ensure_invocation_span() can distinguish "same invocation re-entry"
# (idempotent) from "stale records from a previous invocation" (clear).
_active_invocation_id_ctx: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("_bq_analytics_active_invocation_id", default=None)
)


@dataclass
class _SpanRecord:
  """A single record on the unified span stack.

  Consolidates span, id, ownership, and timing into one object
  so all stacks stay in sync by construction.

  Note: The plugin intentionally does NOT attach its spans to the
  ambient OTel context (no ``context.attach``).  This prevents the
  plugin from corrupting the framework's span hierarchy when an
  external OTel exporter (e.g. ``opentelemetry-instrumentation-vertexai``)
  is active.  See https://github.com/google/adk-python/issues/4561.
  """

  span: trace.Span
  span_id: str
  owns_span: bool
  start_time_ns: int
  first_token_time: Optional[float] = None


_span_records_ctx: contextvars.ContextVar[list[_SpanRecord]] = (
    contextvars.ContextVar("_bq_analytics_span_records", default=None)
)


class TraceManager:
  """Manages OpenTelemetry-style trace and span context using contextvars.

  Uses a single stack of _SpanRecord objects to keep span, token, ID,
  ownership, and timing in sync by construction.
  """

  @staticmethod
  def _get_records() -> list[_SpanRecord]:
    """Returns the current records stack, initializing if needed."""
    records = _span_records_ctx.get()
    if records is None:
      records = []
      _span_records_ctx.set(records)
    return records

  @staticmethod
  def init_trace(callback_context: CallbackContext) -> None:
    # Always refresh root_agent_name — it can change between
    # invocations (e.g. different root agents in the same task).
    try:
      root_agent = callback_context._invocation_context.agent.root_agent
      _root_agent_name_ctx.set(root_agent.name)
    except (AttributeError, ValueError):
      pass

    # Ensure records stack is initialized
    TraceManager._get_records()

  @staticmethod
  def get_trace_id(callback_context: CallbackContext) -> Optional[str]:
    """Gets the trace ID from the current span or invocation_id."""
    records = _span_records_ctx.get()
    if records:
      current_span = records[-1].span
      if current_span.get_span_context().is_valid:
        return format(current_span.get_span_context().trace_id, "032x")

    # Fallback to OTel context
    current_span = trace.get_current_span()
    if current_span.get_span_context().is_valid:
      return format(current_span.get_span_context().trace_id, "032x")

    return callback_context.invocation_id

  @staticmethod
  def push_span(
      callback_context: CallbackContext,
      span_name: Optional[str] = "adk-span",
  ) -> str:
    """Starts a new span and pushes it onto the stack.

    The span is created but NOT attached to the ambient OTel context,
    so it cannot corrupt the framework's own span hierarchy.  The
    plugin tracks span_id / parent_span_id internally via its own
    contextvar stack.

    If OTel is not configured (returning non-recording spans), a UUID
    fallback is generated to ensure span_id and parent_span_id are
    populated in BigQuery logs.
    """
    TraceManager.init_trace(callback_context)

    # Create the span without attaching it to the ambient context.
    # This avoids re-parenting framework spans like ``call_llm``
    # or ``execute_tool``.  See #4561.
    #
    # If the internal stack already has a span, create the new span
    # as a child so it shares the same trace_id.  Without this, each
    # ``start_span`` would be an independent root with its own
    # trace_id — causing trace_id fracture (see #4645).
    records = TraceManager._get_records()
    parent_ctx = None
    if records and records[-1].span.get_span_context().is_valid:
      parent_ctx = trace.set_span_in_context(records[-1].span)
    span = tracer.start_span(span_name, context=parent_ctx)

    if span.get_span_context().is_valid:
      span_id_str = format(span.get_span_context().span_id, "016x")
    else:
      span_id_str = uuid.uuid4().hex

    record = _SpanRecord(
        span=span,
        span_id=span_id_str,
        owns_span=True,
        start_time_ns=time.time_ns(),
    )

    new_records = list(records) + [record]
    _span_records_ctx.set(new_records)

    return span_id_str

  @staticmethod
  def attach_current_span(
      callback_context: CallbackContext,
  ) -> str:
    """Records the current OTel span on the stack without owning it.

    The span is NOT re-attached to the ambient context; it is only
    tracked internally for span_id / parent_span_id resolution.
    """
    TraceManager.init_trace(callback_context)

    span = trace.get_current_span()

    if span.get_span_context().is_valid:
      span_id_str = format(span.get_span_context().span_id, "016x")
    else:
      span_id_str = uuid.uuid4().hex

    record = _SpanRecord(
        span=span,
        span_id=span_id_str,
        owns_span=False,
        start_time_ns=time.time_ns(),
    )

    records = TraceManager._get_records()
    new_records = list(records) + [record]
    _span_records_ctx.set(new_records)

    return span_id_str

  @staticmethod
  def ensure_invocation_span(
      callback_context: CallbackContext,
  ) -> None:
    """Ensures a root span exists on the plugin stack for this invocation.

    Must be called before any events are logged so that every event in
    the invocation shares the same trace_id.

    * If the stack has entries for the *current* invocation → no-op
      (idempotent within the same invocation).
    * If the stack has entries from a *different* invocation → clear
      stale records and re-initialise (safety net for abnormal exit).
    * If the ambient OTel span is valid → ``attach_current_span``
      (reuse the runner's span without owning it).
    * Otherwise → ``push_span("invocation")`` (create a new root
      span that will be popped in ``after_run_callback``).
    """
    current_inv = callback_context.invocation_id
    active_inv = _active_invocation_id_ctx.get()

    records = _span_records_ctx.get()
    if records:
      if active_inv == current_inv:
        return  # Already initialised for this invocation.
      # Stale records from a previous invocation that wasn't cleaned
      # up (e.g. exception skipped after_run_callback). Clear and
      # re-init.
      logger.debug(
          "Clearing %d stale span records from previous invocation.",
          len(records),
      )
      TraceManager.clear_stack()

    _active_invocation_id_ctx.set(current_inv)

    # Check for a valid ambient span (e.g. the Runner's invocation span).
    ambient = trace.get_current_span()
    if ambient.get_span_context().is_valid:
      TraceManager.attach_current_span(callback_context)
    else:
      TraceManager.push_span(callback_context, "invocation")

  @staticmethod
  def pop_span() -> tuple[Optional[str], Optional[int]]:
    """Ends the current span and pops it from the stack.

    No ambient OTel context is detached because we never attached
    one in the first place (see ``push_span``).
    """
    records = _span_records_ctx.get()
    if not records:
      return None, None

    new_records = list(records)
    record = new_records.pop()
    _span_records_ctx.set(new_records)

    # Calculate duration
    duration_ms = None
    otel_start = getattr(record.span, "start_time", None)
    if isinstance(otel_start, (int, float)) and otel_start:
      duration_ms = int((time.time_ns() - otel_start) / 1_000_000)
    else:
      duration_ms = int((time.time_ns() - record.start_time_ns) / 1_000_000)

    if record.owns_span:
      record.span.end()

    return record.span_id, duration_ms

  @staticmethod
  def clear_stack() -> None:
    """Clears all span records. Safety net for cross-invocation cleanup."""
    records = _span_records_ctx.get()
    if records:
      # End any owned spans to avoid OTel resource leaks.
      for record in reversed(records):
        if record.owns_span:
          record.span.end()
      _span_records_ctx.set([])

  @staticmethod
  def get_current_span_and_parent() -> tuple[Optional[str], Optional[str]]:
    """Gets current span_id and parent span_id."""
    records = _span_records_ctx.get()
    if not records:
      return None, None

    span_id = records[-1].span_id
    parent_id = None
    for i in range(len(records) - 2, -1, -1):
      if records[i].span_id != span_id:
        parent_id = records[i].span_id
        break
    return span_id, parent_id

  @staticmethod
  def get_current_span_id() -> Optional[str]:
    """Gets current span_id."""
    records = _span_records_ctx.get()
    if records:
      return records[-1].span_id
    return None

  @staticmethod
  def get_root_agent_name() -> Optional[str]:
    return _root_agent_name_ctx.get()

  @staticmethod
  def get_start_time(span_id: str) -> Optional[float]:
    """Gets start time of a span by ID."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          # Try OTel span start_time first
          otel_start = getattr(record.span, "start_time", None)
          if (
              record.span.get_span_context().is_valid
              and isinstance(otel_start, (int, float))
              and otel_start
          ):
            return otel_start / 1_000_000_000.0
          return record.start_time_ns / 1_000_000_000.0
    return None

  @staticmethod
  def record_first_token(span_id: str) -> bool:
    """Records the current time as first token time if not already recorded."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          if record.first_token_time is None:
            record.first_token_time = time.time()
            return True
          return False
    return False

  @staticmethod
  def get_first_token_time(span_id: str) -> Optional[float]:
    """Gets the recorded first token time."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          return record.first_token_time
    return None


# ==============================================================================
# HELPER: BATCH PROCESSOR
# ==============================================================================
_SHUTDOWN_SENTINEL = object()


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
    """Initializes the instance.

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

    self._visual_builder = _is_visual_builder.get()

    self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
        maxsize=queue_max_size
    )
    self._batch_processor_task: Optional[asyncio.Task] = None
    self._shutdown = False

  async def flush(self) -> None:
    """Flushes the queue by waiting for it to be empty."""
    if self._queue.empty():
      return
    # Wait for all items in the queue to be processed
    await self._queue.join()

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

        if first_item is _SHUTDOWN_SENTINEL:
          self._queue.task_done()
          continue

        batch.append(first_item)

        while len(batch) < self.batch_size:
          try:
            item = self._queue.get_nowait()
            if item is _SHUTDOWN_SENTINEL:
              self._queue.task_done()
              continue
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
        # Avoid sleeping if we are shutting down or if the task was cancelled
        if not self._shutdown:
          try:
            await asyncio.sleep(1)
          except (asyncio.CancelledError, RuntimeError):
            break
        else:
          break

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

      trace_id_prefix = (
          "google-adk-bq-logger-visual-builder"
          if self._visual_builder
          else "google-adk-bq-logger"
      )

      req = bq_storage_types.AppendRowsRequest(
          write_stream=self.write_stream,
          trace_id=f"{trace_id_prefix}/{__version__}",
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

        async def perform_write():
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
              ]:
                raise ServiceUnavailable(error_message)

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

        await asyncio.wait_for(perform_write(), timeout=30.0)
        return

      except (
          ServiceUnavailable,
          TooManyRequests,
          InternalServerError,
          asyncio.TimeoutError,
      ) as e:
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

    # Signal the writer to wake up and check shutdown status
    try:
      self._queue.put_nowait(_SHUTDOWN_SENTINEL)
    except asyncio.QueueFull:
      # If queue is full, the writer is active and will check _shutdown soon
      pass

    if self._batch_processor_task:
      try:
        await asyncio.wait_for(self._batch_processor_task, timeout=timeout)
      except asyncio.TimeoutError:
        logger.warning("BatchProcessor shutdown timed out, cancelling worker.")
        self._batch_processor_task.cancel()
        try:
          # Wait for the task to acknowledge cancellation
          await self._batch_processor_task
        except asyncio.CancelledError:
          pass
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
    """Initializes the instance.

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
          truncated_si, trunc = process_text(si)
          if trunc:
            is_truncated = True
          json_payload["system_prompt"] = truncated_si
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
              " additional event metadata. Includes enrichment fields like"
              " 'root_agent_name' (turn orchestration), 'model' (request"
              " model), 'model_version' (response version), and"
              " 'usage_metadata' (detailed token counts)."
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
# ANALYTICS VIEW DEFINITIONS
# ==============================================================================

# Columns included in every per-event-type view.
_VIEW_COMMON_COLUMNS = (
    "timestamp",
    "event_type",
    "agent",
    "session_id",
    "invocation_id",
    "user_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "status",
    "error_message",
    "is_truncated",
)

# Per-event-type column extractions.  Each value is a list of
# ``"SQL_EXPR AS alias"`` strings that will be appended after the
# common columns in the view SELECT.
_EVENT_VIEW_DEFS: dict[str, list[str]] = {
    "USER_MESSAGE_RECEIVED": [],
    "LLM_REQUEST": [
        "JSON_VALUE(attributes, '$.model') AS model",
        "content AS request_content",
        "JSON_QUERY(attributes, '$.llm_config') AS llm_config",
        "JSON_QUERY(attributes, '$.tools') AS tools",
    ],
    "LLM_RESPONSE": [
        "JSON_QUERY(content, '$.response') AS response",
        (
            "CAST(JSON_VALUE(content, '$.usage.prompt')"
            " AS INT64) AS usage_prompt_tokens"
        ),
        (
            "CAST(JSON_VALUE(content, '$.usage.completion')"
            " AS INT64) AS usage_completion_tokens"
        ),
        (
            "CAST(JSON_VALUE(content, '$.usage.total')"
            " AS INT64) AS usage_total_tokens"
        ),
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
        (
            "CAST(JSON_VALUE(latency_ms,"
            " '$.time_to_first_token_ms') AS INT64) AS ttft_ms"
        ),
        "JSON_VALUE(attributes, '$.model_version') AS model_version",
        "JSON_QUERY(attributes, '$.usage_metadata') AS usage_metadata",
    ],
    "LLM_ERROR": [
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "TOOL_STARTING": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
    ],
    "TOOL_COMPLETED": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.result') AS tool_result",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "TOOL_ERROR": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "AGENT_STARTING": [
        "JSON_VALUE(content, '$.text_summary') AS agent_instruction",
    ],
    "AGENT_COMPLETED": [
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "INVOCATION_STARTING": [],
    "INVOCATION_COMPLETED": [],
    "STATE_DELTA": [
        "JSON_QUERY(attributes, '$.state_delta') AS state_delta",
    ],
    "HITL_CREDENTIAL_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "HITL_CONFIRMATION_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "HITL_INPUT_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "A2A_INTERACTION": [
        "content AS response_content",
        (
            "JSON_VALUE(attributes,"
            " '$.a2a_metadata.\"a2a:task_id\"') AS a2a_task_id"
        ),
        (
            "JSON_VALUE(attributes,"
            " '$.a2a_metadata.\"a2a:context_id\"') AS a2a_context_id"
        ),
        (
            "JSON_QUERY(attributes,"
            " '$.a2a_metadata.\"a2a:request\"') AS a2a_request"
        ),
        (
            "JSON_QUERY(attributes,"
            " '$.a2a_metadata.\"a2a:response\"') AS a2a_response"
        ),
    ],
}

_VIEW_SQL_TEMPLATE = """\
CREATE OR REPLACE VIEW `{project}.{dataset}.{view_name}` AS
SELECT
  {columns}
FROM
  `{project}.{dataset}.{table}`
WHERE
  event_type = '{event_type}'
"""


# ==============================================================================
# MAIN PLUGIN
# ==============================================================================
@dataclass
class _LoopState:
  """Holds resources bound to a specific event loop."""

  write_client: BigQueryWriteAsyncClient
  batch_processor: BatchProcessor


@dataclass(kw_only=True)
class EventData:
  """Typed container for structured fields passed to _log_event."""

  span_id_override: Optional[str] = None
  parent_span_id_override: Optional[str] = None
  latency_ms: Optional[int] = None
  time_to_first_token_ms: Optional[int] = None
  model: Optional[str] = None
  model_version: Optional[str] = None
  usage_metadata: Any = None
  status: str = "OK"
  error_message: Optional[str] = None
  extra_attributes: dict[str, Any] = field(default_factory=dict)
  trace_id_override: Optional[str] = None


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """BigQuery Agent Analytics Plugin using Write API.

  Logs agent events (LLM requests, tool calls, etc.) to BigQuery for analytics.
  Uses the BigQuery Write API for efficient, asynchronous, and reliable logging.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: Optional[str] = None,
      config: Optional[BigQueryLoggerConfig] = None,
      location: str = "US",
      **kwargs,
  ) -> None:
    """Initializes the instance.

    Args:
        project_id: Google Cloud project ID.
        dataset_id: BigQuery dataset ID.
        table_id: BigQuery table ID (optional, overrides config).
        config: BigQueryLoggerConfig (optional).
        location: BigQuery location (default: "US").
        **kwargs: Additional configuration parameters for BigQueryLoggerConfig.
    """
    super().__init__(name="bigquery_agent_analytics")
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.config = config or BigQueryLoggerConfig()

    # Override config with kwargs if provided
    for key, value in kwargs.items():
      if hasattr(self.config, key):
        setattr(self.config, key, value)
      else:
        logger.warning(f"Unknown configuration parameter: {key}")

    if not self.config.view_prefix:
      raise ValueError("view_prefix must be a non-empty string.")

    self.table_id = table_id or self.config.table_id
    self.location = location

    self._visual_builder = _is_visual_builder.get()

    self._started = False
    self._startup_error: Optional[Exception] = None
    self._is_shutting_down = False
    self._setup_lock = None
    self.client = None
    self._loop_state_by_loop: dict[asyncio.AbstractEventLoop, _LoopState] = {}
    self._write_stream_name = None  # Resolved stream name
    self._executor = None
    self.offloader: Optional[GCSOffloader] = None
    self.parser: Optional[HybridContentParser] = None
    self._schema = None
    self.arrow_schema = None
    self._init_pid = os.getpid()
    _LIVE_PLUGINS.add(self)

  def _cleanup_stale_loop_states(self) -> None:
    """Removes entries for event loops that have been closed."""
    stale = [loop for loop in self._loop_state_by_loop if loop.is_closed()]
    for loop in stale:
      logger.warning(
          "Cleaning up stale loop state for closed loop %s (id=%s).",
          loop,
          id(loop),
      )
      del self._loop_state_by_loop[loop]

  # API Compatibility: These class-level attributes mask the dynamic
  # properties from static analysis tools (preventing "breaking changes"),
  # while __getattribute__ intercepts instance access to route to the
  # actual property implementations.
  batch_processor = None
  write_client = None
  write_stream = None

  def __getattribute__(self, name: str) -> Any:
    """Intercepts attribute access to support API masking.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The value of the attribute.
    """
    if name == "batch_processor":
      return self._batch_processor_prop
    if name == "write_client":
      return self._write_client_prop
    if name == "write_stream":
      return self._write_stream_prop
    return super().__getattribute__(name)

  @property
  def _batch_processor_prop(self) -> Optional["BatchProcessor"]:
    """The batch processor for the current event loop."""
    try:
      loop = asyncio.get_running_loop()
      self._cleanup_stale_loop_states()
      if loop in self._loop_state_by_loop:
        return self._loop_state_by_loop[loop].batch_processor
    except RuntimeError:
      pass
    return None

  @property
  def _write_client_prop(self) -> Optional["BigQueryWriteAsyncClient"]:
    """The write client for the current event loop."""
    try:
      loop = asyncio.get_running_loop()
      if loop in self._loop_state_by_loop:
        return self._loop_state_by_loop[loop].write_client
    except RuntimeError:
      pass
    return None

  @property
  def _write_stream_prop(self) -> Optional[str]:
    """The write stream for the current event loop."""
    bp = self._batch_processor_prop
    return bp.write_stream if bp else None

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

  async def _get_loop_state(self) -> _LoopState:
    """Gets or creates the state for the current event loop.

    Returns:
        The loop-specific state object containing clients and processors.
    """
    loop = asyncio.get_running_loop()
    self._cleanup_stale_loop_states()
    if loop in self._loop_state_by_loop:
      return self._loop_state_by_loop[loop]

    # grpc.aio clients are loop-bound, so we create one per event loop.

    def get_credentials():
      creds, project_id = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
      return creds, project_id

    creds, project_id = await loop.run_in_executor(
        self._executor, get_credentials
    )
    quota_project_id = getattr(creds, "quota_project_id", None)
    options = (
        client_options.ClientOptions(quota_project_id=quota_project_id)
        if quota_project_id
        else None
    )

    user_agents = [f"google-adk-bq-logger/{__version__}"]
    if self._visual_builder:
      user_agents.append(f"google-adk-visual-builder/{__version__}")

    client_info = gapic_client_info.ClientInfo(user_agent=" ".join(user_agents))

    write_client = BigQueryWriteAsyncClient(
        credentials=creds,
        client_info=client_info,
        client_options=options,
    )

    if not self._write_stream_name:
      self._write_stream_name = f"projects/{self.project_id}/datasets/{self.dataset_id}/tables/{self.table_id}/_default"

    batch_processor = BatchProcessor(
        write_client=write_client,
        arrow_schema=self.arrow_schema,
        write_stream=self._write_stream_name,
        batch_size=self.config.batch_size,
        flush_interval=self.config.batch_flush_interval,
        retry_config=self.config.retry_config,
        queue_max_size=self.config.queue_max_size,
        shutdown_timeout=self.config.shutdown_timeout,
    )
    await batch_processor.start()

    state = _LoopState(write_client, batch_processor)
    self._loop_state_by_loop[loop] = state

    atexit.register(self._atexit_cleanup, weakref.proxy(batch_processor))

    return state

  async def flush(self) -> None:
    """Flushes any pending events to BigQuery.

    Flushes the processor associated with the CURRENT loop.
    """
    try:
      loop = asyncio.get_running_loop()
      self._cleanup_stale_loop_states()
      if loop in self._loop_state_by_loop:
        await self._loop_state_by_loop[loop].batch_processor.flush()
    except RuntimeError:
      # No running loop or other issue
      pass

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
    if not self._schema:
      self._schema = _get_events_schema()
      await loop.run_in_executor(self._executor, self._ensure_schema_exists)

    if not self.parser:
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

    await self._get_loop_state()

  @staticmethod
  def _atexit_cleanup(batch_processor: "BatchProcessor") -> None:
    """Clean up batch processor on script exit.

    Drains any remaining items from the queue and logs a warning.
    Callers should use ``flush()`` before shutdown to ensure all
    events are written; this handler only reports data that would
    otherwise be silently lost.
    """
    try:
      if not batch_processor or batch_processor._shutdown:
        return
    except ReferenceError:
      return

    # Drain remaining items and warn — creating a new event loop and
    # BQ client at interpreter exit is fragile and masks shutdown bugs.
    remaining = 0
    try:
      while True:
        batch_processor._queue.get_nowait()
        remaining += 1
    except (asyncio.QueueEmpty, AttributeError):
      pass

    if remaining:
      logger.warning(
          "%d analytics event(s) were still queued at interpreter exit "
          "and could not be flushed. Call plugin.flush() before shutdown "
          "to avoid data loss.",
          remaining,
      )

  def _ensure_schema_exists(self) -> None:
    """Ensures the BigQuery table exists with the correct schema.

    When ``config.auto_schema_upgrade`` is True and the table already
    exists, missing columns are added automatically (additive only).
    A ``adk_schema_version`` label is written for governance.
    """
    try:
      existing_table = self.client.get_table(self.full_table_id)
      if self.config.auto_schema_upgrade:
        self._maybe_upgrade_schema(existing_table)
      if self.config.create_views:
        self._create_analytics_views()
    except cloud_exceptions.NotFound:
      logger.info("Table %s not found, creating table.", self.full_table_id)
      tbl = bigquery.Table(self.full_table_id, schema=self._schema)
      tbl.time_partitioning = bigquery.TimePartitioning(
          type_=bigquery.TimePartitioningType.DAY,
          field="timestamp",
      )
      tbl.clustering_fields = self.config.clustering_fields
      tbl.labels = {_SCHEMA_VERSION_LABEL_KEY: _SCHEMA_VERSION}
      table_ready = False
      try:
        self.client.create_table(tbl)
        table_ready = True
      except cloud_exceptions.Conflict:
        # Another process created it concurrently — still usable.
        table_ready = True
      except Exception as e:
        logger.error(
            "Could not create table %s: %s",
            self.full_table_id,
            e,
            exc_info=True,
        )
      if table_ready and self.config.create_views:
        self._create_analytics_views()
    except Exception as e:
      logger.error(
          "Error checking for table %s: %s",
          self.full_table_id,
          e,
          exc_info=True,
      )

  @staticmethod
  def _schema_fields_match(
      existing: list[bq_schema.SchemaField],
      desired: list[bq_schema.SchemaField],
  ) -> tuple[
      list[bq_schema.SchemaField],
      list[bq_schema.SchemaField],
  ]:
    """Compares existing vs desired schema fields recursively.

    Returns:
        A tuple of (new_top_level_fields, updated_record_fields).
        ``new_top_level_fields`` are fields in *desired* that are
        entirely absent from *existing*.
        ``updated_record_fields`` are RECORD fields that exist in
        both but have new sub-fields in *desired*; each entry is a
        copy of the existing field with the missing sub-fields
        appended.
    """
    existing_by_name = {f.name: f for f in existing}
    new_fields: list[bq_schema.SchemaField] = []
    updated_records: list[bq_schema.SchemaField] = []

    for desired_field in desired:
      existing_field = existing_by_name.get(desired_field.name)
      if existing_field is None:
        new_fields.append(desired_field)
      elif (
          desired_field.field_type == "RECORD"
          and existing_field.field_type == "RECORD"
          and desired_field.fields
      ):
        # Recurse into nested RECORD fields.
        sub_new, sub_updated = (
            BigQueryAgentAnalyticsPlugin._schema_fields_match(
                list(existing_field.fields),
                list(desired_field.fields),
            )
        )
        if sub_new or sub_updated:
          # Build a merged sub-field list.
          merged_sub = list(existing_field.fields)
          # Replace updated nested records in-place.
          updated_names = {f.name for f in sub_updated}
          merged_sub = [
              next(u for u in sub_updated if u.name == f.name)
              if f.name in updated_names
              else f
              for f in merged_sub
          ]
          # Append entirely new sub-fields.
          merged_sub.extend(sub_new)
          # Rebuild via API representation to preserve all
          # existing field attributes (policy_tags, etc.).
          api_repr = existing_field.to_api_repr()
          api_repr["fields"] = [sf.to_api_repr() for sf in merged_sub]
          updated_records.append(bq_schema.SchemaField.from_api_repr(api_repr))

    return new_fields, updated_records

  def _maybe_upgrade_schema(self, existing_table: bigquery.Table) -> None:
    """Adds missing columns to an existing table (additive only).

    Handles nested RECORD fields by recursing into sub-fields.
    The version label is only stamped after a successful update
    so that a failed attempt is retried on the next run.

    Args:
        existing_table: The current BigQuery table object.
    """
    stored_version = (existing_table.labels or {}).get(
        _SCHEMA_VERSION_LABEL_KEY
    )
    if stored_version == _SCHEMA_VERSION:
      return

    new_fields, updated_records = self._schema_fields_match(
        list(existing_table.schema), list(self._schema)
    )

    if new_fields or updated_records:
      # Build merged top-level schema.
      updated_names = {f.name for f in updated_records}
      merged = [
          next(u for u in updated_records if u.name == f.name)
          if f.name in updated_names
          else f
          for f in existing_table.schema
      ]
      merged.extend(new_fields)
      existing_table.schema = merged

      change_desc = []
      if new_fields:
        change_desc.append(f"new columns {[f.name for f in new_fields]}")
      if updated_records:
        change_desc.append(
            f"updated RECORD fields {[f.name for f in updated_records]}"
        )
      logger.info(
          "Auto-upgrading table %s: %s",
          self.full_table_id,
          ", ".join(change_desc),
      )

    try:
      # Stamp the version label inside the try block so that
      # on failure the label is NOT persisted and the next run
      # retries the upgrade.
      labels = dict(existing_table.labels or {})
      labels[_SCHEMA_VERSION_LABEL_KEY] = _SCHEMA_VERSION
      existing_table.labels = labels

      update_fields = ["schema", "labels"]
      self.client.update_table(existing_table, update_fields)
    except Exception as e:
      logger.error(
          "Schema auto-upgrade failed for %s: %s",
          self.full_table_id,
          e,
          exc_info=True,
      )

  def _create_analytics_views(self) -> None:
    """Creates per-event-type BigQuery views (idempotent).

    Each view filters the events table by ``event_type`` and
    extracts JSON columns into typed, queryable columns.  Uses
    ``CREATE OR REPLACE VIEW`` so it is safe to call repeatedly.
    Errors are logged but never raised.
    """
    for event_type, extra_cols in _EVENT_VIEW_DEFS.items():
      view_name = self.config.view_prefix + "_" + event_type.lower()
      columns = ",\n  ".join(list(_VIEW_COMMON_COLUMNS) + extra_cols)
      sql = _VIEW_SQL_TEMPLATE.format(
          project=self.project_id,
          dataset=self.dataset_id,
          view_name=view_name,
          columns=columns,
          table=self.table_id,
          event_type=event_type,
      )
      try:
        self.client.query(sql).result()
      except cloud_exceptions.Conflict:
        logger.debug(
            "View %s was updated concurrently by another process.",
            view_name,
        )
      except Exception as e:
        logger.error(
            "Failed to create view %s: %s",
            view_name,
            e,
            exc_info=True,
        )

  async def create_analytics_views(self) -> None:
    """Public async helper to (re-)create all analytics views.

    Useful when views need to be refreshed explicitly, for example
    after a schema upgrade.  Ensures the plugin is initialized
    before attempting view creation.
    """
    await self._ensure_started()
    if not self._started:
      raise RuntimeError(
          "Plugin initialization failed; cannot create analytics views."
      ) from self._startup_error
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(self._executor, self._create_analytics_views)

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
      # Correct Multi-Loop Shutdown:
      # 1. Shutdown current loop's processor directly.
      if loop in self._loop_state_by_loop:
        await self._loop_state_by_loop[loop].batch_processor.shutdown(timeout=t)

      # 1b. Drain batch processors on other (non-current) loops.
      for other_loop, state in self._loop_state_by_loop.items():
        if other_loop is loop or other_loop.is_closed():
          continue
        try:
          future = asyncio.run_coroutine_threadsafe(
              state.batch_processor.shutdown(timeout=t),
              other_loop,
          )
          future.result(timeout=t)
        except Exception:
          logger.warning(
              "Could not drain batch processor on loop %s",
              other_loop,
          )

      # 2. Close clients for all states
      for state in self._loop_state_by_loop.values():
        if state.write_client and getattr(
            state.write_client, "transport", None
        ):
          try:
            await state.write_client.transport.close()
          except Exception:
            pass

      self._loop_state_by_loop.clear()

      if self.client:
        if self._executor:
          executor = self._executor
          await loop.run_in_executor(None, lambda: executor.shutdown(wait=True))
          self._executor = None
      self.client = None
    except Exception as e:
      logger.error("Error during shutdown: %s", e, exc_info=True)
    self._is_shutting_down = False
    self._started = False

  def __getstate__(self):
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    state["_setup_lock"] = None
    state["client"] = None
    state["_loop_state_by_loop"] = {}
    state["_write_stream_name"] = None
    state["_executor"] = None
    state["offloader"] = None
    state["parser"] = None
    state["_started"] = False
    state["_startup_error"] = None
    state["_is_shutting_down"] = False
    state["_init_pid"] = 0
    return state

  def __setstate__(self, state):
    """Custom unpickling to restore state."""
    # Backfill keys that may be absent in pickled state from older
    # code versions so _ensure_started does not raise AttributeError.
    state.setdefault("_init_pid", 0)
    self.__dict__.update(state)

  def _reset_runtime_state(self) -> None:
    """Resets all runtime state after a fork.

    gRPC channels and asyncio locks are not safe to use after
    ``os.fork()``.  This method clears them so the next call to
    ``_ensure_started()`` re-initializes everything in the child
    process.  Pure-data fields like ``_schema`` and
    ``arrow_schema`` are kept because they are safe across fork.
    """
    logger.warning(
        "Fork detected (parent PID %s, child PID %s). Resetting"
        " gRPC state for BigQuery analytics plugin.  Note: gRPC"
        " bidirectional streaming (used by the BigQuery Storage"
        " Write API) is not fork-safe.  If writes hang or time"
        " out, configure the 'spawn' start method at your program"
        " entry-point before creating child processes:"
        "  multiprocessing.set_start_method('spawn')",
        self._init_pid,
        os.getpid(),
    )
    # Best-effort: close inherited gRPC channels so broken
    # finalizers don't interfere with newly created channels.
    # For grpc.aio channels, close() is a coroutine.  We cannot
    # await here (called from sync context / fork handler), so
    # we skip async channels and only close sync ones.
    for loop_state in self._loop_state_by_loop.values():
      wc = getattr(loop_state, "write_client", None)
      transport = getattr(wc, "transport", None)
      if transport is not None:
        try:
          channel = getattr(transport, "_grpc_channel", None)
          if channel is not None and hasattr(channel, "close"):
            result = channel.close()
            # If close() returned a coroutine (grpc.aio channel),
            # discard it to avoid unawaited-coroutine warnings.
            if asyncio.iscoroutine(result):
              result.close()
        except Exception:
          pass

    # Clear all runtime state.
    self._setup_lock = None
    self.client = None
    self._loop_state_by_loop = {}
    self._write_stream_name = None
    self._executor = None
    self.offloader = None
    self.parser = None
    self._started = False
    self._startup_error = None
    self._is_shutting_down = False
    self._init_pid = os.getpid()

  async def __aenter__(self) -> BigQueryAgentAnalyticsPlugin:
    await self._ensure_started()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.shutdown()

  async def _ensure_started(self, **kwargs) -> None:
    """Ensures that the plugin is started and initialized."""
    if os.getpid() != self._init_pid:
      self._reset_runtime_state()
    if not self._started:
      # Kept original lock name as it was not explicitly changed.
      if self._setup_lock is None:
        self._setup_lock = asyncio.Lock()
      async with self._setup_lock:
        if not self._started:
          try:
            await self._lazy_setup(**kwargs)
            self._started = True
            self._startup_error = None
          except Exception as e:
            self._startup_error = e
            logger.error("Failed to initialize BigQuery Plugin: %s", e)

  @staticmethod
  def _resolve_ids(
      event_data: EventData,
      callback_context: CallbackContext,
  ) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolves trace_id, span_id, and parent_span_id for a log row.

    Resolution rules:

      * **trace_id** — ambient OTel trace wins (the plugin stack already
        shares the ambient trace when initialised from an ambient span,
        so in practice they agree).
      * **span_id / parent_span_id** — the plugin's internal span stack
        (``TraceManager``) is the preferred source.  Ambient OTel spans
        are only used as a fallback when the plugin stack has no span.
        This ensures every ``parent_span_id`` in BigQuery references a
        ``span_id`` that is also logged to BigQuery, producing a
        self-consistent execution tree.
      * **Explicit overrides** (``EventData``) always win last — they
        are set by post-pop callbacks that have already captured the
        correct plugin-stack values before the pop.

    Priority order (highest first):
      1. Explicit ``EventData`` overrides.
      2. Plugin's internal span stack (``TraceManager``) for
         ``span_id`` / ``parent_span_id``.
      3. Ambient OTel span — always used for ``trace_id``; used for
         ``span_id`` / ``parent_span_id`` only when the plugin stack
         has no span.
      4. ``invocation_id`` fallback for trace_id.

    Returns:
        (trace_id, span_id, parent_span_id)
    """
    # --- Plugin stack: span_id / parent_span_id baseline ---
    trace_id = TraceManager.get_trace_id(callback_context)
    plugin_span_id, plugin_parent_span_id = (
        TraceManager.get_current_span_and_parent()
    )
    span_id = plugin_span_id
    parent_span_id = plugin_parent_span_id

    # --- Ambient OTel: trace_id always; span fallback only ---
    ambient = trace.get_current_span()
    ambient_ctx = ambient.get_span_context()
    if ambient_ctx.is_valid:
      trace_id = format(ambient_ctx.trace_id, "032x")
      # Only use ambient span IDs when the plugin stack has no span.
      # Framework-internal spans (execute_tool, call_llm, etc.) are
      # never written to BQ, so deriving parent_span_id from them
      # creates phantom references.  The plugin stack guarantees
      # that both span_id and parent_span_id reference BQ rows.
      if span_id is None:
        span_id = format(ambient_ctx.span_id, "016x")
        parent_span_id = None
        parent_ctx = getattr(ambient, "parent", None)
        if parent_ctx is not None and parent_ctx.span_id:
          parent_span_id = format(parent_ctx.span_id, "016x")

    # --- Explicit EventData overrides (post-pop callbacks) ---
    if event_data.trace_id_override is not None:
      trace_id = event_data.trace_id_override
    if event_data.span_id_override is not None:
      span_id = event_data.span_id_override
    if event_data.parent_span_id_override is not None:
      parent_span_id = event_data.parent_span_id_override

    return trace_id, span_id, parent_span_id

  @staticmethod
  def _extract_latency(
      event_data: EventData,
  ) -> dict[str, Any] | None:
    """Reads latency fields from EventData and returns a latency dict (or None).

    Returns:
        A dict with ``total_ms`` and/or ``time_to_first_token_ms``, or
        *None* if neither was present.
    """
    latency_json: dict[str, Any] = {}
    if event_data.latency_ms is not None:
      latency_json["total_ms"] = event_data.latency_ms
    if event_data.time_to_first_token_ms is not None:
      latency_json["time_to_first_token_ms"] = event_data.time_to_first_token_ms
    return latency_json or None

  def _enrich_attributes(
      self,
      event_data: EventData,
      callback_context: CallbackContext,
  ) -> dict[str, Any]:
    """Builds the attributes dict from EventData and enrichments.

    Reads ``model``, ``model_version``, and ``usage_metadata`` from
    *event_data*, copies ``extra_attributes``, then adds session metadata
    and custom tags.

    Returns:
        A new dict ready for JSON serialization into the attributes column.
    """
    attrs: dict[str, Any] = dict(event_data.extra_attributes)

    attrs["root_agent_name"] = TraceManager.get_root_agent_name()
    if event_data.model:
      attrs["model"] = event_data.model
    if event_data.model_version:
      attrs["model_version"] = event_data.model_version
    if event_data.usage_metadata:
      usage_dict, _ = _recursive_smart_truncate(
          event_data.usage_metadata, self.config.max_content_length
      )
      if isinstance(usage_dict, dict):
        attrs["usage_metadata"] = usage_dict
      else:
        attrs["usage_metadata"] = event_data.usage_metadata

    if self.config.log_session_metadata:
      try:
        session = callback_context._invocation_context.session
        session_meta = {
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
        }
        # Include session state if non-empty (contains user-set metadata
        # like gchat thread-id, customer_id, etc.)
        if session.state:
          truncated_state, _ = _recursive_smart_truncate(
              dict(session.state),
              self.config.max_content_length,
          )
          session_meta["state"] = truncated_state
        attrs["session_metadata"] = session_meta
      except Exception:
        pass

    if self.config.custom_tags:
      attrs["custom_tags"] = self.config.custom_tags

    return attrs

  async def _log_event(
      self,
      event_type: str,
      callback_context: CallbackContext,
      raw_content: Any = None,
      is_truncated: bool = False,
      event_data: Optional[EventData] = None,
  ) -> None:
    """Logs an event to BigQuery.

    Args:
        event_type: The type of event (e.g., 'LLM_REQUEST').
        callback_context: The callback context.
        raw_content: The raw content to log.
        is_truncated: Whether the content is already truncated.
        event_data: Typed container for structured fields and extra
            attributes. Defaults to ``EventData()`` when not provided.
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

    if event_data is None:
      event_data = EventData()

    timestamp = datetime.now(timezone.utc)
    if self.config.content_formatter:
      try:
        raw_content = self.config.content_formatter(raw_content, event_type)
      except Exception as e:
        logger.warning("Content formatter failed: %s", e)

    trace_id, span_id, parent_span_id = self._resolve_ids(
        event_data, callback_context
    )

    if not self.parser:
      logger.warning("Parser not initialized; skipping event %s.", event_type)
      return

    # Update parser's trace/span IDs for GCS pathing (reuse instance)
    self.parser.trace_id = trace_id or "no_trace"
    self.parser.span_id = span_id or "no_span"
    content_json, content_parts, parser_truncated = await self.parser.parse(
        raw_content
    )
    is_truncated = is_truncated or parser_truncated

    latency_json = self._extract_latency(event_data)
    attributes = self._enrich_attributes(event_data, callback_context)

    # Serialize attributes to JSON string
    try:
      attributes_json = json.dumps(attributes)
    except (TypeError, ValueError):
      attributes_json = json.dumps(attributes, default=str)

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
        "latency_ms": latency_json,
        "status": event_data.status,
        "error_message": event_data.error_message,
        "is_truncated": is_truncated,
    }

    state = await self._get_loop_state()
    await state.batch_processor.append(row)

  # --- UPDATED CALLBACKS FOR V1 PARITY ---

  @_safe_callback
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> None:
    """Parity with V1: Logs USER_MESSAGE_RECEIVED event.

    Also detects HITL completion responses (user-sent
    ``FunctionResponse`` parts with ``adk_request_*`` names) and emits
    dedicated ``HITL_*_COMPLETED`` events.

    Args:
        invocation_context: The context of the current invocation.
        user_message: The message content received from the user.
    """
    callback_ctx = CallbackContext(invocation_context)
    TraceManager.ensure_invocation_span(callback_ctx)
    await self._log_event(
        "USER_MESSAGE_RECEIVED",
        callback_ctx,
        raw_content=user_message,
    )

    # Detect HITL completion responses in the user message.
    if user_message and user_message.parts:
      for part in user_message.parts:
        if part.function_response:
          hitl_event = _HITL_EVENT_MAP.get(part.function_response.name)
          if hitl_event:
            resp_truncated, is_truncated = _recursive_smart_truncate(
                part.function_response.response or {},
                self.config.max_content_length,
            )
            content_dict = {
                "tool": part.function_response.name,
                "result": resp_truncated,
            }
            await self._log_event(
                hitl_event + "_COMPLETED",
                callback_ctx,
                raw_content=content_dict,
                is_truncated=is_truncated,
            )

  @_safe_callback
  async def on_event_callback(
      self,
      *,
      invocation_context: InvocationContext,
      event: "Event",
  ) -> None:
    """Logs state changes, HITL events, and A2A interactions.

    - Checks each event for a non-empty state_delta and logs it as a
      STATE_DELTA event.
    - Detects synthetic ``adk_request_*`` function calls (HITL pause
      events) and their corresponding function responses (HITL
      completions) and emits dedicated HITL event types.
    - Detects events carrying A2A interaction metadata
      (``a2a:request`` / ``a2a:response`` in ``custom_metadata``)
      and logs them as ``A2A_INTERACTION`` events so the remote
      agent's response and cross-reference IDs (``a2a:task_id``,
      ``a2a:context_id``) are visible in BigQuery.

    The HITL detection must happen here (not in tool callbacks) because
    ``adk_request_credential``, ``adk_request_confirmation``, and
    ``adk_request_input`` are synthetic function calls injected by the
    framework — they never go through ``before_tool_callback`` /
    ``after_tool_callback``.

    Args:
        invocation_context: The context for the current invocation.
        event: The event raised by the runner.
    """
    callback_ctx = CallbackContext(invocation_context)

    # --- State delta logging ---
    if event.actions and event.actions.state_delta:
      await self._log_event(
          "STATE_DELTA",
          callback_ctx,
          event_data=EventData(
              extra_attributes={"state_delta": dict(event.actions.state_delta)}
          ),
      )

    # --- HITL event logging ---
    if event.content and event.content.parts:
      for part in event.content.parts:
        # Detect HITL function calls (request events).
        if part.function_call:
          hitl_event = _HITL_EVENT_MAP.get(part.function_call.name)
          if hitl_event:
            args_truncated, is_truncated = _recursive_smart_truncate(
                part.function_call.args or {},
                self.config.max_content_length,
            )
            content_dict = {
                "tool": part.function_call.name,
                "args": args_truncated,
            }
            await self._log_event(
                hitl_event,
                callback_ctx,
                raw_content=content_dict,
                is_truncated=is_truncated,
            )
        # Detect HITL function responses (completion events).
        if part.function_response:
          hitl_event = _HITL_EVENT_MAP.get(part.function_response.name)
          if hitl_event:
            resp_truncated, is_truncated = _recursive_smart_truncate(
                part.function_response.response or {},
                self.config.max_content_length,
            )
            content_dict = {
                "tool": part.function_response.name,
                "result": resp_truncated,
            }
            await self._log_event(
                hitl_event + "_COMPLETED",
                callback_ctx,
                raw_content=content_dict,
                is_truncated=is_truncated,
            )

    # --- A2A interaction logging ---
    # RemoteA2aAgent attaches cross-reference metadata to events:
    #   a2a:task_id, a2a:context_id  — correlation keys
    #   a2a:request, a2a:response    — full interaction payload
    # Log an A2A_INTERACTION event when meaningful payload is present
    # so the supervisor's BQ trace contains the remote agent's
    # response and cross-reference IDs for JOINs.
    meta = getattr(event, "custom_metadata", None)
    if meta and (
        meta.get("a2a:request") is not None
        or meta.get("a2a:response") is not None
    ):
      a2a_keys = {k: v for k, v in meta.items() if k.startswith("a2a:")}
      a2a_truncated, is_truncated = _recursive_smart_truncate(
          a2a_keys, self.config.max_content_length
      )
      # Use the a2a:response as the event content when available,
      # so the remote agent's answer is visible in the content
      # column.
      response_payload = a2a_keys.get("a2a:response")
      content_dict = None
      content_truncated = False
      if response_payload is not None:
        content_dict, content_truncated = _recursive_smart_truncate(
            response_payload,
            self.config.max_content_length,
        )
      await self._log_event(
          "A2A_INTERACTION",
          callback_ctx,
          raw_content=content_dict,
          is_truncated=is_truncated or content_truncated,
          event_data=EventData(
              extra_attributes={
                  "a2a_metadata": a2a_truncated,
              },
          ),
      )

    return None

  async def on_state_change_callback(
      self,
      *,
      callback_context: CallbackContext,
      state_delta: dict[str, Any],
  ) -> None:
    """Deprecated: use on_event_callback instead.

    This method is retained for API compatibility but is never invoked
    by the framework (not in BasePlugin, PluginManager, or Runner).
    State deltas are now captured via on_event_callback.
    """
    logger.warning(
        "on_state_change_callback is deprecated and never called by"
        " the framework. State deltas are captured via"
        " on_event_callback."
    )

  @_safe_callback
  async def before_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    """Callback before the agent run starts.

    Args:
        invocation_context: The context of the current invocation.
    """
    await self._ensure_started()
    callback_ctx = CallbackContext(invocation_context)
    TraceManager.ensure_invocation_span(callback_ctx)
    await self._log_event(
        "INVOCATION_STARTING",
        callback_ctx,
    )

  @_safe_callback
  async def after_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    """Callback after the agent run completes.

    Args:
        invocation_context: The context of the current invocation.
    """
    try:
      # Capture trace_id BEFORE popping the invocation-root span so
      # that INVOCATION_COMPLETED shares the same trace_id as all
      # earlier events in this invocation (fixes #4645).
      callback_ctx = CallbackContext(invocation_context)
      trace_id = TraceManager.get_trace_id(callback_ctx)

      # Pop the invocation-root span pushed by ensure_invocation_span.
      span_id, duration = TraceManager.pop_span()
      parent_span_id = TraceManager.get_current_span_id()

      await self._log_event(
          "INVOCATION_COMPLETED",
          callback_ctx,
          event_data=EventData(
              trace_id_override=trace_id,
              latency_ms=duration,
              span_id_override=span_id,
              parent_span_id_override=parent_span_id,
          ),
      )
    finally:
      # Cleanup must run even if _log_event raises, otherwise
      # stale invocation metadata leaks into the next invocation.
      TraceManager.clear_stack()
      _active_invocation_id_ctx.set(None)
      _root_agent_name_ctx.set(None)
      # Ensure all logs are flushed before the agent returns.
      await self.flush()

  @_safe_callback
  async def before_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext
  ) -> None:
    """Callback before an agent starts processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    TraceManager.init_trace(callback_context)
    TraceManager.push_span(callback_context, "agent")
    await self._log_event(
        "AGENT_STARTING",
        callback_context,
        raw_content=getattr(agent, "instruction", ""),
    )

  @_safe_callback
  async def after_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext
  ) -> None:
    """Callback after an agent completes processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "AGENT_COMPLETED",
        callback_context,
        event_data=EventData(
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )

  @_safe_callback
  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
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
          "presence_penalty",
          "frequency_penalty",
          "response_mime_type",
          "response_schema",
          "seed",
          "response_logprobs",
          "logprobs",
      ]:
        val = getattr(llm_request.config, field_name, None)
        if val is not None:
          config_dict[field_name] = val

      if config_dict:
        attributes["llm_config"] = config_dict

      if labels := getattr(llm_request.config, "labels", None):
        attributes["labels"] = labels

    if hasattr(llm_request, "tools_dict") and llm_request.tools_dict:
      attributes["tools"] = list(llm_request.tools_dict.keys())

    TraceManager.push_span(callback_context, "llm_request")
    await self._log_event(
        "LLM_REQUEST",
        callback_context,
        raw_content=llm_request,
        event_data=EventData(
            model=llm_request.model,
            extra_attributes=attributes,
        ),
    )

  @_safe_callback
  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: "LlmResponse",
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

    await self._log_event(
        "LLM_RESPONSE",
        callback_context,
        raw_content=content_str,
        is_truncated=is_truncated,
        event_data=EventData(
            latency_ms=duration,
            time_to_first_token_ms=tfft,
            model_version=llm_response.model_version,
            usage_metadata=llm_response.usage_metadata,
            span_id_override=span_id if is_popped else None,
            parent_span_id_override=(parent_span_id if is_popped else None),
        ),
    )

  @_safe_callback
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> None:
    """Callback on LLM error.

    Args:
        callback_context: The callback context.
        llm_request: The request that was sent to the model.
        error: The exception that occurred.
    """
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "LLM_ERROR",
        callback_context,
        event_data=EventData(
            status="ERROR",
            error_message=str(error),
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )

  @_safe_callback
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
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
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "args": args_truncated,
        "tool_origin": tool_origin,
    }
    TraceManager.push_span(tool_context, "tool")
    await self._log_event(
        "TOOL_STARTING",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
    )

  @_safe_callback
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
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
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "result": resp_truncated,
        "tool_origin": tool_origin,
    }
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    event_data = EventData(
        latency_ms=duration,
        span_id_override=span_id,
        parent_span_id_override=parent_span_id,
    )
    await self._log_event(
        "TOOL_COMPLETED",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
        event_data=event_data,
    )

  @_safe_callback
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Callback on tool error.

    Args:
        tool: The tool that failed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
        error: The exception that occurred.
    """
    args_truncated, is_truncated = _recursive_smart_truncate(
        tool_args, self.config.max_content_length
    )
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "args": args_truncated,
        "tool_origin": tool_origin,
    }
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "TOOL_ERROR",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
        event_data=EventData(
            status="ERROR",
            error_message=str(error),
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )
