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

# NOTE:
#
#    We expect that the underlying GenAI SDK will provide a certain
#    level of tracing and logging telemetry aligned with Open Telemetry
#    Semantic Conventions (such as logging prompts, responses,
#    request properties, etc.) and so the information that is recorded by the
#    Agent Development Kit should be focused on the higher-level
#    constructs of the framework that are not observable by the SDK.

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextlib import contextmanager
from contextlib import ExitStack
import logging
import os
import re
from typing import Final
from typing import TYPE_CHECKING

from google.genai import types
from google.genai.models import Models
from opentelemetry import _logs
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry._logs import LogRecord
from opentelemetry._logs import SeverityNumber
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_DESCRIPTION
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_REQUEST_MODEL
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_CALL_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_DESCRIPTION
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_TYPE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_PROTOCOL_VERSION
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_SESSION_ID
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_REQUEST_METHOD
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS
from opentelemetry.semconv.attributes.server_attributes import SERVER_PORT
from opentelemetry.semconv.attributes.url_attributes import URL_FULL
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import Span
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue
from typing_extensions import deprecated

from .. import version
from ..utils.env_utils import is_enterprise_mode_enabled
from ..utils.model_name_utils import extract_model_name
from ..utils.model_name_utils import is_gemini_model
from ._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT
from ._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT
from ._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT
from ._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED
from ._experimental_semconv import maybe_log_completion_details
from ._experimental_semconv import set_operation_details_attributes_from_request
from ._experimental_semconv import set_operation_details_attributes_from_response
from ._experimental_semconv import set_operation_details_common_attributes
from ._finish_reason import is_reported_finish_reason
from ._serialization import safe_json_serialize
from ._stable_semconv import choice_body
from ._stable_semconv import GEN_AI_CHOICE_EVENT
from ._stable_semconv import GEN_AI_SYSTEM_MESSAGE_EVENT
from ._stable_semconv import GEN_AI_USER_MESSAGE_EVENT
from ._stable_semconv import OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
from ._stable_semconv import system_message_body
from ._stable_semconv import USER_CONTENT_ELIDED
from ._stable_semconv import user_message_body
from ._token_usage import TokenUsage
from .context import _TRUTHY_ENV_VALUES
from .context import TelemetryConfig

# By default some ADK spans include attributes with potential PII data.
# This env, when set to false, allows to disable populating those attributes.
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"

# Used to associate a span with a destination resource for AppHub. Tools with
# this key in their BaseTool.custom_metadata will have the mapping added as a
# span attribute
GCP_MCP_SERVER_DESTINATION_ID = "gcp.mcp.server.destination.id"

# Event name for the log record of one HTTP exchange with an MCP server. OTel's
# MCP conventions define attributes but no event for the transport hop, so this
# one is ADK's own, and lives in the `adk.experimental.*` namespace: emitted
# only under `ADK_EXPERIMENTAL_TELEMETRY`, and free to be renamed or dropped
# the moment a standard covers it -- either a generic HTTP client
# response-end event, or body capture in
# `opentelemetry-instrumentation-httpx`. The attributes on it are the
# semconv-defined ones.
_ADK_EXPERIMENTAL_MCP_HTTP_RESPONSE_END_EVENT: Final[str] = (
    "adk.experimental.mcp.http.client.response.end"
)

# Payload attribute names from the HTTP body conventions
# (open-telemetry/semantic-conventions#3521), merged but not in any released
# `opentelemetry-semconv`. Named here rather than imported until one has them.
_HTTP_REQUEST_BODY_CONTENT: Final[str] = "http.request.body.content"
_HTTP_RESPONSE_BODY_CONTENT: Final[str] = "http.response.body.content"
_HTTP_REQUEST_HEADER_TEMPLATE: Final[str] = "http.request.header.{}"
_HTTP_RESPONSE_HEADER_TEMPLATE: Final[str] = "http.response.header.{}"

# Opt-in for the payload on the record above. That semconv PR requires body
# capture to be off by default and leaves the switch to the instrumentation, so
# this is ADK's own and defaults off. Deliberately not the GenAI content knob:
# agreeing to record prompt content is not the same decision as agreeing to
# export raw HTTP payloads.
_ADK_CAPTURE_MCP_HTTP_BODIES: Final[str] = "ADK_CAPTURE_MCP_HTTP_BODIES"

# Which headers to record. Nothing is recorded that these do not name: the
# semconv default is to capture no header, because any of them can carry a
# credential the redaction list has not heard of yet. They are the same env
# vars `opentelemetry-instrumentation-httpx` reads, so one setting configures
# the HTTP client span and this record alike. Comma-separated names or regexes,
# matched case-insensitively against the whole header name.
_OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST: Final[str] = (
    "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST"
)
_OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE: Final[str] = (
    "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE"
)

# Silence unused warnings, but keep the public interface the same.
_ = OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT

# Needed to avoid circular imports
if TYPE_CHECKING:

  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ..models.cache_metadata import CacheMetadata
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool
  from ..workflow._base_node import BaseNode

tracer = trace.get_tracer(
    instrumenting_module_name="gcp.vertex.agent",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

otel_logger = _logs.get_logger(
    instrumenting_module_name="gcp.vertex.agent",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

logger = logging.getLogger("google_adk." + __name__)


def resolve_error_type(error: BaseException) -> str:
  """Derives a higher-resolution ``error.type`` label for a failure.

  Prefers, in order: a pre-classified ``error_type`` carried by ADK errors; the
  HTTP status code for ``google.genai`` ``APIError``s (e.g. ``429``, since the
  SDK collapses every 4xx into ``ClientError`` and every 5xx into
  ``ServerError``); finally the class name.
  """
  from google.genai import errors as genai_errors

  custom_error_type = getattr(error, "error_type", None)
  if custom_error_type is not None:
    return str(custom_error_type)
  if isinstance(error, genai_errors.APIError):
    return str(error.code)
  return type(error).__name__


def trace_agent_invocation(
    span: trace.Span, agent: BaseAgent, ctx: InvocationContext
) -> None:
  """Sets span attributes immediately available on agent invocation according to OTEL semconv version 1.37.

  Args:
    span: Span on which attributes are set.
    agent: Agent from which attributes are gathered.
    ctx: InvocationContext from which attributes are gathered.

  Inference related fields are not set, because the OpenTelemetry semantic
    conventions plan to remove them from the invoke_agent span.

  `gen_ai.agent.id` is not set because currently it's unclear what attributes
    this field should have, specifically:
  - In which scope should it be unique (globally, given project, given agentic
    flow, given deployment).
  - Should it be unchanging between deployments, and how this should this be
    achieved.

  `gen_ai.data_source.id` is not set because it's not available.
  Closest type which could contain this information is types.GroundingMetadata,
    which does not have an ID.

  `server.*` attributes are not set pending confirmation from aabmass.
  """

  # Required
  span.set_attribute(GEN_AI_OPERATION_NAME, "invoke_agent")

  # Conditionally Required
  span.set_attribute(GEN_AI_AGENT_DESCRIPTION, agent.description)

  span.set_attribute(GEN_AI_AGENT_NAME, agent.name)
  span.set_attribute(GEN_AI_CONVERSATION_ID, ctx.session.id)


def trace_tool_call(
    tool: BaseTool,
    args: dict[str, object],
    function_response_event: Event | None,
    error: Exception | None = None,
    span: Span | None = None,
    error_type: str | None = None,
    invocation_context: InvocationContext | None = None,
):
  """Traces tool call.

  Args:
    tool: The tool that was called.
    args: The arguments to the tool call.
    function_response_event: The event with the function response details.
    error: The exception raised during tool execution, if any.
    span: The span to record attributes on. If None, uses current span.
    error_type: An error type string detected from the tool's response dict
      (e.g., "HTTP_ERROR", "MCP_TOOL_ERROR"). Used when the tool returned an
      error as a dict rather than raising an exception. Ignored if `error` is
      also set (exception takes precedence).
    invocation_context: Optional invocation context. Forwarded so its
      ``run_config.telemetry`` overrides the env-var content toggle.
  """
  span = span or trace.get_current_span()
  if not span.is_recording():
    return

  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )

  span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")

  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, tool.description)
  span.set_attribute(GEN_AI_TOOL_NAME, tool.name)

  # e.g. FunctionTool
  span.set_attribute(GEN_AI_TOOL_TYPE, tool.__class__.__name__)

  if (
      invocation_context is not None
      and (agent := invocation_context.agent) is not None
  ):
    span.set_attribute(GEN_AI_AGENT_NAME, agent.name)

  failure_type: str | None = None
  if error is not None:
    failure_type = resolve_error_type(error)
    span.record_exception(error)
  elif error_type is not None:
    failure_type = error_type
  if failure_type is not None:
    span.set_attribute(ERROR_TYPE, failure_type)
    # Without an explicit error status the span renders as successful, which
    # hides tools that reported a failure as a response dict instead of
    # raising. The description repeats the type rather than the error message
    # so no tool content lands in an attribute the content toggle cannot gate.
    span.set_status(Status(StatusCode.ERROR, failure_type))

  # Special case for client side association with a remote tool call
  if (
      tool.custom_metadata
      and GCP_MCP_SERVER_DESTINATION_ID in tool.custom_metadata
  ):
    destination_id = tool.custom_metadata[GCP_MCP_SERVER_DESTINATION_ID]
    span.set_attribute(GCP_MCP_SERVER_DESTINATION_ID, destination_id)

  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute("gcp.vertex.agent.llm_request", "{}")
  span.set_attribute("gcp.vertex.agent.llm_response", "{}")

  if telemetry_config.should_add_content_to_legacy_spans:
    span.set_attribute(
        "gcp.vertex.agent.tool_call_args",
        safe_json_serialize(args),
    )
  else:
    span.set_attribute("gcp.vertex.agent.tool_call_args", "{}")

  # Tracing tool response
  tool_call_id = "<not specified>"
  tool_response = "<not specified>"
  if (
      function_response_event is not None
      and function_response_event.content is not None
      and function_response_event.content.parts
  ):
    response_parts = function_response_event.content.parts
    function_response = response_parts[0].function_response
    if function_response is not None:
      if function_response.id is not None:
        tool_call_id = function_response.id
      if function_response.response is not None:
        tool_response = function_response.response

  span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)

  if not isinstance(tool_response, dict):
    tool_response = {"result": tool_response}
  if function_response_event is not None:
    span.set_attribute("gcp.vertex.agent.event_id", function_response_event.id)
  if telemetry_config.should_add_content_to_legacy_spans:
    span.set_attribute(
        "gcp.vertex.agent.tool_response",
        safe_json_serialize(tool_response),
    )
  else:
    span.set_attribute("gcp.vertex.agent.tool_response", "{}")


def _should_report_mcp_http_exchanges() -> bool:
  """Whether MCP HTTP exchanges are reported to OTel. Off unless asked for.

  The record is experimental (`adk.experimental.*`), so it rides on the
  experimental telemetry opt-in. Resolved from the env rather than from
  `RunConfig.telemetry`, because the httpx response hook that reports an
  exchange has no invocation context to read a per-request override from.
  """
  return TelemetryConfig().should_emit_experimental_telemetry


def _should_capture_mcp_http_bodies() -> bool:
  """Whether MCP HTTP payloads may be recorded. Off unless asked for."""
  return (
      os.getenv(_ADK_CAPTURE_MCP_HTTP_BODIES, "").strip().lower()
      in _TRUTHY_ENV_VALUES
  )


def _header_patterns(env_var: str) -> list[re.Pattern[str]]:
  """Compiles the header allowlist `env_var` holds, one regex per entry."""
  patterns = []
  for entry in os.getenv(env_var, "").split(","):
    entry = entry.strip()
    if not entry:
      continue
    try:
      patterns.append(re.compile(entry, re.IGNORECASE))
    except re.error:
      logger.warning(
          "Ignoring malformed header pattern %r in %s.", entry, env_var
      )
  return patterns


def _captured_headers(
    headers: Mapping[str, str], template: str, env_var: str
) -> dict[str, AttributeValue]:
  """Allowlisted headers, in the semconv `http.*.header.<key>` shape.

  Values arrive already redacted, so allowlisting a credential header yields
  the redaction marker rather than the secret.

  Args:
    headers: Headers of one side of the exchange, already redacted.
    template: Attribute name to format the lowercased header name into.
    env_var: Env var naming the headers to record.

  Returns:
    One `string[]` attribute per header the env var names.
  """
  allowlist = _header_patterns(env_var)
  captured: dict[str, AttributeValue] = {}
  for name, value in headers.items():
    lowered = name.lower()
    if any(pattern.fullmatch(lowered) for pattern in allowlist):
      captured[template.format(lowered)] = [value]
  return captured


def _trace_mcp_http_exchange(
    *,
    method: str,
    url: str,
    server_address: str | None,
    server_port: int | None,
    status_code: int,
    mcp_session_id: str | None,
    mcp_protocol_version: str | None,
    request_headers: Mapping[str, str],
    request_body: str | None,
    response_headers: Mapping[str, str],
    response_body: str | None,
) -> None:
  """Emits one DEBUG log record for an MCP server HTTP request/response pair.

  The record picks up the trace context active on the calling task: the
  enclosing `execute_tool` span for an inline tool call, or the session-creating
  invocation for exchanges the MCP client drives from a background task (notably
  the SSE message POST).

  Callers own the opt-in check (`_should_report_mcp_http_exchanges`), redaction
  and truncation. Headers named in the OTel capture env vars become semconv
  attributes; the bodies go in the record body and appear only when
  `ADK_CAPTURE_MCP_HTTP_BODIES` asks for them, because MCP payloads routinely
  carry user data.

  Args:
    method: HTTP method of the request.
    url: Request URL, already sanitized.
    server_address: Host the request was sent to, if known.
    server_port: Port the request was sent to, if known.
    status_code: HTTP status code of the response.
    mcp_session_id: MCP session this exchange belongs to, if known.
    mcp_protocol_version: MCP protocol version the exchange ran under, if known.
    request_headers: Request headers, already redacted.
    request_body: Request body, already truncated, or None if unread.
    response_headers: Response headers, already redacted.
    response_body: Response body, already truncated, `<SSE stream>` for a body
      that must not be read, or None if unread.
  """
  attributes: dict[str, AttributeValue] = {
      HTTP_REQUEST_METHOD: method,
      URL_FULL: url,
      HTTP_RESPONSE_STATUS_CODE: status_code,
  }
  if server_address:
    attributes[SERVER_ADDRESS] = server_address
  if server_port:
    attributes[SERVER_PORT] = server_port
  # The two facts a reader of an MCP exchange always wants are recorded as the
  # semconv attributes for them, so that reading them costs no header capture.
  if mcp_session_id:
    attributes[MCP_SESSION_ID] = mcp_session_id
  if mcp_protocol_version:
    attributes[MCP_PROTOCOL_VERSION] = mcp_protocol_version
  # HTTP semconv records `error.type` as the status code string.
  if status_code >= 400:
    attributes[ERROR_TYPE] = str(status_code)
  attributes.update(
      _captured_headers(
          request_headers,
          _HTTP_REQUEST_HEADER_TEMPLATE,
          _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST,
      )
  )
  attributes.update(
      _captured_headers(
          response_headers,
          _HTTP_RESPONSE_HEADER_TEMPLATE,
          _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE,
      )
  )

  if _should_capture_mcp_http_bodies():
    body = {
        _HTTP_REQUEST_BODY_CONTENT: request_body,
        _HTTP_RESPONSE_BODY_CONTENT: response_body,
    }
  else:
    body = {
        _HTTP_REQUEST_BODY_CONTENT: USER_CONTENT_ELIDED,
        _HTTP_RESPONSE_BODY_CONTENT: USER_CONTENT_ELIDED,
    }

  otel_logger.emit(
      LogRecord(
          event_name=_ADK_EXPERIMENTAL_MCP_HTTP_RESPONSE_END_EVENT,
          severity_number=SeverityNumber.DEBUG,
          body=body,
          attributes=attributes,
      )
  )


def trace_merged_tool_calls(
    response_event_id: str,
    function_response_event: Event,
    invocation_context: InvocationContext | None = None,
):
  """Traces merged tool call events.

  Calling this function is not needed for telemetry purposes. This is provided
  for preventing /debug/trace requests (typically sent by web UI).

  Args:
    response_event_id: The ID of the response event.
    function_response_event: The merged response event.
    invocation_context: Optional invocation context. Forwarded so its
      ``run_config.telemetry`` overrides the env-var content toggle.
  """
  span = trace.get_current_span()
  if not span.is_recording():
    return

  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )

  span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
  span.set_attribute(GEN_AI_TOOL_NAME, "(merged tools)")
  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, "(merged tools)")
  span.set_attribute(GEN_AI_TOOL_CALL_ID, response_event_id)

  # Pending cleanup: drop these placeholder attributes once no downstream
  # consumer reads them.
  span.set_attribute("gcp.vertex.agent.tool_call_args", "N/A")
  span.set_attribute("gcp.vertex.agent.event_id", response_event_id)
  if telemetry_config.should_add_content_to_legacy_spans:
    try:
      function_response_event_json = function_response_event.model_dump_json(
          exclude_none=True
      )
    except Exception:  # pylint: disable=broad-exception-caught
      function_response_event_json = "<not serializable>"

    span.set_attribute(
        "gcp.vertex.agent.tool_response",
        function_response_event_json,
    )
  else:
    span.set_attribute("gcp.vertex.agent.tool_response", "{}")
  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute("gcp.vertex.agent.llm_request", "{}")
  span.set_attribute(
      "gcp.vertex.agent.llm_response",
      "{}",
  )


def _set_usage_metadata_attributes(
    span: Span,
    usage_metadata: types.GenerateContentResponseUsageMetadata | None,
) -> None:
  """Records usage metadata attributes on the given span."""
  if usage_metadata is None:
    return
  span.set_attributes(TokenUsage(usage_metadata).to_attributes())


def _set_context_cache_attributes(
    span: Span,
    cache_metadata: CacheMetadata | None,
    telemetry_config: TelemetryConfig,
) -> None:
  """Records context cache state on the given span."""
  if cache_metadata is None:
    return
  # The fingerprint is a content hash, so these attributes stay behind the
  # experimental opt-in rather than landing on every span by default.
  if not telemetry_config.should_emit_experimental_telemetry:
    return
  attributes: dict[str, AttributeValue] = {
      ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT: cache_metadata.cache_name is not None,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT: cache_metadata.fingerprint,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT: (
          cache_metadata.contents_count
      ),
  }
  if cache_metadata.invocations_used is not None:
    attributes[ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED] = (
        cache_metadata.invocations_used
    )
  span.set_attributes(attributes)


def trace_call_llm(
    invocation_context: InvocationContext,
    event_id: str,
    llm_request: LlmRequest,
    llm_response: LlmResponse,
    span: Span | None = None,
):
  """Traces a call to the LLM.

  This function records details about the LLM request and response as
  attributes on the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    llm_request: The LLM request object.
    llm_response: The LLM response object.
  """
  if span is None:
    span = trace.get_current_span()
  if not span.is_recording():
    return

  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  # Special standard Open Telemetry GenaI attributes that indicate
  # that this is a span related to a Generative AI system.
  span.set_attribute("gen_ai.system", "gcp.vertex.agent")
  span.set_attribute("gen_ai.request.model", llm_request.model)
  span.set_attribute(
      "gcp.vertex.agent.invocation_id", invocation_context.invocation_id
  )
  span.set_attribute(
      "gcp.vertex.agent.session_id", invocation_context.session.id
  )
  span.set_attribute("gcp.vertex.agent.event_id", event_id)
  # Consider removing once GenAI SDK provides a way to record this info.
  if telemetry_config.should_add_content_to_legacy_spans:
    span.set_attribute(
        "gcp.vertex.agent.llm_request",
        safe_json_serialize(_build_llm_request_for_trace(llm_request)),
    )
  else:
    span.set_attribute("gcp.vertex.agent.llm_request", "{}")
  # Consider removing once GenAI SDK provides a way to record this info.
  if llm_request.config:
    if llm_request.config.top_p:
      span.set_attribute(
          "gen_ai.request.top_p",
          llm_request.config.top_p,
      )
    if llm_request.config.max_output_tokens:
      span.set_attribute(
          "gen_ai.request.max_tokens",
          llm_request.config.max_output_tokens,
      )
    try:
      if (
          llm_request.config.thinking_config
          and llm_request.config.thinking_config.thinking_budget is not None
      ):
        span.set_attribute(
            "gen_ai.usage.experimental.reasoning_tokens_limit",
            llm_request.config.thinking_config.thinking_budget,
        )
    except AttributeError:
      pass

  if telemetry_config.should_add_content_to_legacy_spans:
    try:
      response_for_trace = llm_response
      if llm_response.content is not None:
        response_for_trace = llm_response.model_copy(
            update={"content": _summarize_inline_data(llm_response.content)}
        )
      llm_response_json = response_for_trace.model_dump_json(exclude_none=True)
    except Exception:  # pylint: disable=broad-exception-caught
      llm_response_json = "<not serializable>"

    span.set_attribute(
        "gcp.vertex.agent.llm_response",
        llm_response_json,
    )
  else:
    span.set_attribute("gcp.vertex.agent.llm_response", "{}")

  _set_usage_metadata_attributes(span, llm_response.usage_metadata)
  _set_context_cache_attributes(
      span, getattr(llm_response, "cache_metadata", None), telemetry_config
  )
  if is_reported_finish_reason(finish_reason := llm_response.finish_reason):
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])


def _without_thought_signature(part: types.Part) -> types.Part:
  """Returns ``part`` with any thought signature removed.

  A thought signature is opaque bytes that the model round-trips through the
  conversation, so it stays in history and is replayed on every later request.
  Serializing a part in JSON mode base64-encodes it onto a span attribute,
  which grows with the history on each call and carries nothing a reader of
  the trace can act on.

  Args:
    part: The part to copy.

  Returns:
    ``part`` itself when it carries no signature, otherwise a copy without one.
  """
  if part.thought_signature is None:
    return part
  return part.model_copy(update={"thought_signature": None})


def _summarize_inline_data(content: types.Content) -> types.Content:
  """Returns ``content`` with inline binary parts reduced to a description.

  Serializing a part in JSON mode base64-encodes its ``inline_data``, so a
  live session's audio chunks would otherwise be copied wholesale onto a span
  attribute. Only the mime type and byte count are kept.

  Args:
    content: The content to summarize.

  Returns:
    A copy of ``content`` whose inline binary parts carry a text description
    instead of the bytes.
  """
  parts = []
  for part in content.parts or []:
    blob = part.inline_data
    if blob is None:
      parts.append(_without_thought_signature(part))
      continue
    parts.append(
        types.Part(
            text=(
                f"<inline_data: {blob.mime_type or 'unknown'},"
                f" {len(blob.data or b'')} bytes>"
            )
        )
    )
  return types.Content(role=content.role, parts=parts)


def trace_send_data(
    invocation_context: InvocationContext,
    event_id: str,
    data: list[types.Content],
):
  """Traces the sending of data to the agent.

  This function records details about the data sent to the agent as
  attributes on the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    data: A list of content objects.
  """
  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  span = trace.get_current_span()
  span.set_attribute(
      "gcp.vertex.agent.invocation_id", invocation_context.invocation_id
  )
  span.set_attribute("gcp.vertex.agent.event_id", event_id)
  # Once instrumentation is added to the GenAI SDK, consider whether this
  # information still needs to be recorded by the Agent Development Kit.
  if telemetry_config.should_add_content_to_legacy_spans:
    span.set_attribute(
        "gcp.vertex.agent.data",
        safe_json_serialize([
            _summarize_inline_data(content).model_dump(
                exclude_none=True, mode="json"
            )
            for content in data
        ]),
    )
  else:
    span.set_attribute("gcp.vertex.agent.data", "{}")


def _build_compaction_attributes(
    *,
    session_id: str,
    trigger: str,
    summarizer_type: str,
    event_count: int,
    token_threshold: int | None = None,
    event_retention_size: int | None = None,
    compaction_interval: int | None = None,
    overlap_size: int | None = None,
) -> dict[str, AttributeValue]:
  """Builds span attributes for event compaction tracing."""
  attributes: dict[str, AttributeValue] = {
      GEN_AI_SYSTEM: _guess_gemini_system_name(),
      GEN_AI_OPERATION_NAME: "compact_events",
      GEN_AI_CONVERSATION_ID: session_id,
      "gen_ai.compaction.trigger": trigger,
      "gen_ai.compaction.summarizer_type": summarizer_type,
      "gen_ai.compaction.event_count": event_count,
  }
  if token_threshold is not None:
    attributes["gen_ai.compaction.token_threshold"] = token_threshold
  if event_retention_size is not None:
    attributes["gen_ai.compaction.event_retention_size"] = event_retention_size
  if compaction_interval is not None:
    attributes["gen_ai.compaction.compaction_interval"] = compaction_interval
  if overlap_size is not None:
    attributes["gen_ai.compaction.overlap_size"] = overlap_size
  return attributes


def _build_compaction_result_attributes(
    compacted_event: Event | None,
) -> dict[str, AttributeValue]:
  """Builds span attributes for compaction result."""
  if (
      compacted_event is None
      or compacted_event.actions is None
      or compacted_event.actions.compaction is None
  ):
    return {}

  attributes: dict[str, AttributeValue] = {}
  compaction = compacted_event.actions.compaction
  attributes["gen_ai.compaction.result_event_id"] = compacted_event.id
  if compaction.start_timestamp is not None:
    attributes["gen_ai.compaction.start_timestamp"] = compaction.start_timestamp
  if compaction.end_timestamp is not None:
    attributes["gen_ai.compaction.end_timestamp"] = compaction.end_timestamp
  return attributes


def _build_llm_request_for_trace(llm_request: LlmRequest) -> dict[str, object]:
  """Builds a dictionary representation of the LLM request for tracing.

  This function prepares a dictionary representation of the LlmRequest
  object, suitable for inclusion in a trace. It excludes fields that cannot
  be serialized (e.g., function pointers) and avoids sending bytes data.

  Args:
    llm_request: The LlmRequest object.

  Returns:
    A dictionary representation of the LLM request.
  """
  # Some fields in LlmRequest are function pointers and cannot be serialized.
  result = {
      "model": llm_request.model,
      "config": llm_request.config.model_dump(
          exclude_none=True,
          exclude={
              "response_schema": True,
              # `http_options` carries caller-supplied credentials: `headers`
              # commonly holds an Authorization bearer token, and
              # `extra_body` / `*client_args` are free-form passthroughs that
              # can hold auth material too. None of it may reach an exported
              # span attribute. The client fields are also unserializable.
              "http_options": {
                  "httpx_client": True,
                  "httpx_async_client": True,
                  "aiohttp_client": True,
                  "headers": True,
                  "extra_body": True,
                  "client_args": True,
                  "async_client_args": True,
              },
          },
          mode="json",
      ),
      "contents": [],
  }
  # We do not want to send bytes data to the trace.
  for content in llm_request.contents:
    parts = [
        _without_thought_signature(part)
        for part in content.parts
        if not part.inline_data
    ]
    result["contents"].append(
        types.Content(role=content.role, parts=parts).model_dump(
            exclude_none=True, mode="json"
        )
    )
  return result


def _telemetry_config_from_invocation_context(
    invocation_context: InvocationContext | None,
) -> TelemetryConfig:
  """Returns ``invocation_context.run_config.telemetry`` if reachable, else ``None``."""
  if invocation_context is None:
    return TelemetryConfig()
  try:
    if (run_config := invocation_context.run_config) is None:
      return TelemetryConfig()
    return run_config.telemetry or TelemetryConfig()
  except AttributeError:
    logger.warning(
        "Failed to access run_config from invocation_context: type is %s",
        type(invocation_context).__name__,
    )
    return TelemetryConfig()


@deprecated("Replaced by use_inference_span to support experimental semconv.")
@contextmanager
def use_generate_content_span(
    llm_request: LlmRequest,
    invocation_context: InvocationContext,
    model_response_event: Event,
) -> Iterator[Span | None]:
  """Context manager encompassing `generate_content {model.name}` span.

  When an external library for inference instrumentation is installed (e.g.
  opentelemetry-instrumentation-google-genai),
  span creation is delegated to said library.
  """

  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  common_attributes = {
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      "gcp.vertex.agent.event_id": model_response_event.id,
      "gcp.vertex.agent.invocation_id": invocation_context.invocation_id,
  }
  log_only_common_attributes = {}
  if invocation_context.session.user_id is not None:
    log_only_common_attributes[USER_ID] = invocation_context.session.user_id
  if _should_emit_native_telemetry(invocation_context.agent):
    with _use_native_generate_content_span_stable_semconv(
        llm_request=llm_request,
        common_attributes=common_attributes,
        log_only_common_attributes=log_only_common_attributes,
        telemetry_config=telemetry_config,
    ) as span:
      yield span.span
  else:
    with _use_extra_generate_content_attributes(
        common_attributes,
        log_only_extra_attributes=log_only_common_attributes,
    ):
      yield


@asynccontextmanager
async def use_inference_span(
    llm_request: LlmRequest,
    invocation_context: InvocationContext,
    model_response_event: Event,
) -> AsyncIterator[GenerateContentSpan | None]:
  """Context manager encompassing `generate_content {model.name}` span.

  When an external library for inference instrumentation is installed (e.g.
  opentelemetry-instrumentation-google-genai),
  span creation is delegated to said library.
  """

  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  common_attributes = {
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      "gcp.vertex.agent.event_id": model_response_event.id,
      "gcp.vertex.agent.invocation_id": invocation_context.invocation_id,
  }
  log_only_common_attributes = {}
  if invocation_context.session.user_id is not None:
    log_only_common_attributes[USER_ID] = invocation_context.session.user_id
  if _should_emit_native_telemetry(invocation_context.agent):
    with ExitStack() as stack:
      gc_span = stack.enter_context(
          _use_native_generate_content_span(
              llm_request=llm_request,
              common_attributes=common_attributes,
              log_only_common_attributes=log_only_common_attributes,
              telemetry_config=telemetry_config,
          )
      )
      gc_span._exit_stack = stack  # pylint: disable=protected-access
      if telemetry_config.should_use_experimental_genai_semconv:
        set_operation_details_common_attributes(
            gc_span.operation_details_common_attributes,
            telemetry_config,
            common_attributes,
            log_only_attributes=log_only_common_attributes,
        )
      # Registered last, so it unwinds first: while the span is still open.
      _ = gc_span._exit_stack.callback(
          lambda: maybe_log_completion_details(
              gc_span.span,
              otel_logger,
              gc_span.operation_details_attributes,
              gc_span.operation_details_common_attributes,
              telemetry_config,
          )
      )  # pylint: disable=protected-access
      yield gc_span
  else:
    with _use_extra_generate_content_attributes(
        common_attributes,
        log_only_extra_attributes=log_only_common_attributes,
    ):
      yield


def _instrumented_with_opentelemetry_instrumentation_google_genai() -> bool:
  maybe_wrapped_function = Models.generate_content
  while wrapped := getattr(maybe_wrapped_function, "__wrapped__", None):
    if (
        "opentelemetry/instrumentation/google_genai"
        in maybe_wrapped_function.__code__.co_filename.replace("\\", "/")
    ):
      return True
    maybe_wrapped_function = wrapped  # pyright: ignore[reportAny]

  return False


def _should_emit_native_telemetry(agent: BaseAgent) -> bool:
  """If the google-genai instrumentation lib is active AND this is a Gemini agent, then the lib already emits inference metrics."""
  if (
      _instrumented_with_opentelemetry_instrumentation_google_genai()
      and _is_gemini_agent(agent)
  ):
    return False

  return True


@contextmanager
def _use_extra_generate_content_attributes(
    extra_attributes: Mapping[str, AttributeValue],
    log_only_extra_attributes: Mapping[str, AttributeValue] | None = None,
):
  try:
    from opentelemetry.instrumentation.google_genai import GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
  except (ImportError, AttributeError):
    logger.warning(
        "opentelemetry-instrumentor-google-genai is installed but has"
        " insufficient version,"
        + " so some tracing dependent features may not work properly."
        + " Please upgrade to version to 0.6b0 or above."
    )
    yield

    return

  ctx = otel_context.set_value(
      GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY, extra_attributes
  )
  if log_only_extra_attributes:
    try:
      from opentelemetry.instrumentation.google_genai import GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY

      ctx = otel_context.set_value(
          GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY,
          log_only_extra_attributes,
          context=ctx,
      )
    except (ImportError, AttributeError):
      pass

  tok = otel_context.attach(ctx)
  try:
    yield
  finally:
    otel_context.detach(tok)


def _is_gemini_agent(agent: BaseAgent) -> bool:
  return is_gemini_model(_agent_model_name(agent))


def _set_common_generate_content_attributes(
    span: Span,
    llm_request: LlmRequest,
    common_attributes: Mapping[str, AttributeValue],
):
  span.set_attribute(GEN_AI_OPERATION_NAME, "generate_content")
  span.set_attribute(GEN_AI_REQUEST_MODEL, llm_request.model or "")
  span.set_attributes(common_attributes)


@contextmanager
def _use_native_generate_content_span_stable_semconv(
    llm_request: LlmRequest,
    common_attributes: Mapping[str, AttributeValue],
    log_only_common_attributes: Mapping[str, AttributeValue] | None = None,
    telemetry_config: TelemetryConfig | None = None,
) -> Iterator[GenerateContentSpan]:
  telemetry_config = telemetry_config or TelemetryConfig()
  system_name = _resolve_gen_ai_system_name(llm_request.model)
  with tracer.start_as_current_span(
      f"generate_content {llm_request.model or ''}".strip()
  ) as span:
    span.set_attribute(GEN_AI_SYSTEM, system_name)
    _set_common_generate_content_attributes(
        span, llm_request, common_attributes
    )
    gc_span = GenerateContentSpan(span)

    otel_logger.emit(
        LogRecord(
            event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
            body=system_message_body(llm_request, telemetry_config),
            attributes={GEN_AI_SYSTEM: system_name},
        )
    )
    user_message_attributes = {GEN_AI_SYSTEM: system_name}
    if (
        telemetry_config.should_add_content_to_logs
        and log_only_common_attributes
    ):
      user_id = log_only_common_attributes.get(USER_ID)
      if user_id is not None:
        user_message_attributes[USER_ID] = user_id

    for content in llm_request.contents:
      otel_logger.emit(
          LogRecord(
              event_name=GEN_AI_USER_MESSAGE_EVENT,
              body=user_message_body(content, telemetry_config),
              attributes=user_message_attributes,
          )
      )

    yield gc_span


@contextmanager
def _use_native_generate_content_span(
    llm_request: LlmRequest,
    common_attributes: Mapping[str, AttributeValue],
    telemetry_config: TelemetryConfig,
    log_only_common_attributes: Mapping[str, AttributeValue] | None = None,
) -> Iterator[GenerateContentSpan]:
  if not telemetry_config.should_use_experimental_genai_semconv:
    with _use_native_generate_content_span_stable_semconv(
        llm_request,
        common_attributes,
        log_only_common_attributes=log_only_common_attributes,
        telemetry_config=telemetry_config,
    ) as gc_span:
      yield gc_span
    return

  with tracer.start_as_current_span(
      f"generate_content {llm_request.model or ''}".strip()
  ) as span:
    _set_common_generate_content_attributes(
        span, llm_request, common_attributes
    )
    gc_span = GenerateContentSpan(span)

    set_operation_details_attributes_from_request(
        gc_span.operation_details_attributes,
        llm_request,
    )
    yield gc_span


class GenerateContentSpan:
  """Manages tracing within a `generate_content` OpenTelemetry span.

  This class provides attributes for the experimental semantic convention.
  """

  def __init__(self, span: Span):
    self.span: Final = span
    self.operation_details_attributes: dict[str, AttributeValue] = {}
    self.operation_details_common_attributes: dict[str, AttributeValue] = {}
    # Ends underlying span and records completion details log record.
    # Used over contextmanager to end the underlying span as soon as the
    # inference is done, instead of when the caller is done with the response.
    # Matches opentelemetry-instrumentation-google-genai behavior.
    self._exit_stack: ExitStack | None = None


@deprecated(
    "Replaced by trace_inference_result to support experimental semconv."
)
def trace_generate_content_result(span: Span | None, llm_response: LlmResponse):
  """Trace result of the inference in generate_content span."""

  if span is None:
    return

  # This deprecated path has no notion of an inference ending, so it keeps
  # skipping partials rather than reporting one choice per chunk.
  if llm_response.partial:
    return

  if is_reported_finish_reason(finish_reason := llm_response.finish_reason):
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])
  _set_usage_metadata_attributes(span, llm_response.usage_metadata)

  otel_logger.emit(
      LogRecord(
          event_name=GEN_AI_CHOICE_EVENT,
          body=choice_body(llm_response, TelemetryConfig()),
          attributes={
              GEN_AI_SYSTEM: _inference_system_name(None, llm_response)
          },
          context=trace.set_span_in_context(span),
      )
  )


def trace_inference_result(
    invocation_context: InvocationContext | None,
    span: Span | None | GenerateContentSpan,
    llm_response: LlmResponse,
):
  """Trace result of the inference in generate_content span."""
  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  gc_span = None
  if isinstance(span, GenerateContentSpan):
    gc_span = span
    span = gc_span.span

  if span is None:
    return

  # No `partial` check: a streamed chunk differs from a whole answer only by
  # the finish reason it does not carry. The caller stops recording once one
  # arrives, so the response an aggregator assembles from the chunks it already
  # reported does not reach here.
  if is_reported_finish_reason(finish_reason := llm_response.finish_reason):
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])
  _set_usage_metadata_attributes(span, llm_response.usage_metadata)
  # Callers outside adk pass their own response objects here, which are only
  # required to carry the fields this function already read.
  _set_context_cache_attributes(
      span, getattr(llm_response, "cache_metadata", None), telemetry_config
  )

  if telemetry_config.should_use_experimental_genai_semconv and isinstance(
      gc_span, GenerateContentSpan
  ):
    set_operation_details_attributes_from_response(
        llm_response,
        gc_span.operation_details_attributes,
        gc_span.operation_details_common_attributes,
    )

  else:
    otel_logger.emit(
        LogRecord(
            event_name=GEN_AI_CHOICE_EVENT,
            body=choice_body(
                llm_response, telemetry_config or TelemetryConfig()
            ),
            attributes={
                GEN_AI_SYSTEM: _inference_system_name(
                    invocation_context, llm_response
                )
            },
            context=trace.set_span_in_context(span),
        )
    )


def _guess_gemini_system_name() -> str:
  return (
      GenAiSystemValues.VERTEX_AI.name.lower()
      if is_enterprise_mode_enabled()
      else GenAiSystemValues.GEMINI.name.lower()
  )


# Anthropic models reach ADK either as a bare `claude-*` id (the built-in
# Anthropic backend, and the Vertex `publishers/anthropic/models/...` path once
# normalized) or behind a LiteLLM `anthropic/...` prefix, which the generic
# prefix rule below already covers.
_ANTHROPIC_MODEL_PATTERN: Final = re.compile(r"^claude[-.]", re.IGNORECASE)

# Leading segments of a resource-path model id, e.g. a Model Garden path like
# `projects/<p>/locations/<l>/publishers/<pub>/models/<m>` or a tuned-model id
# like `tunedModels/<id>`. They name a resource collection, never a provider,
# so the provider-prefix rule must not read one as one.
_RESOURCE_COLLECTION_SEGMENTS: Final = frozenset({
    "endpoints",
    "locations",
    "models",
    "projects",
    "publishers",
    "tunedmodels",
})


def _resolve_gen_ai_system_name(model: str | None) -> str:
  """Returns the `gen_ai.system` / `gen_ai.provider.name` value for a model.

  The name has to follow the model actually being served, otherwise every
  provider is reported as Gemini. A LiteLLM-style `<provider>/<model>` id
  carries the provider in its prefix, which semantic conventions allow as a
  lowercased name outside their well-known set. The prefix is read off the bare
  model name so that a resource path, whose leading segments describe where the
  model lives rather than who serves it, is not mistaken for one. When no model
  id is available, or the id names no provider, the deployment-derived
  Gemini/Vertex name is used, since Gemini is the backend ADK talks to
  natively.

  Args:
    model: The model id the request is being served by, if known.
  """
  if not model or is_gemini_model(model):
    return _guess_gemini_system_name()

  model_name = extract_model_name(model)
  if _ANTHROPIC_MODEL_PATTERN.match(model_name):
    return GenAiSystemValues.ANTHROPIC.name.lower()

  provider, separator, _ = model_name.partition("/")
  provider = provider.lower()
  if separator and provider and provider not in _RESOURCE_COLLECTION_SEGMENTS:
    return provider

  return _guess_gemini_system_name()


def _agent_model_name(agent: BaseAgent | BaseNode) -> str | None:
  """Returns the model id configured on an agent, if it has one."""
  from ..agents.llm_agent import LlmAgent

  if not isinstance(agent, LlmAgent):
    return None

  model = agent.model if agent.model != "" else agent._default_model
  return model if isinstance(model, str) else model.model


def _inference_system_name(
    invocation_context: InvocationContext | None,
    llm_response: LlmResponse,
) -> str:
  """Returns the system name of the model that produced an inference result."""
  model = llm_response.model_version
  if not model and invocation_context is not None:
    agent = invocation_context.agent
    if agent is not None:
      model = _agent_model_name(agent)
  return _resolve_gen_ai_system_name(model)
