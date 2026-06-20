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
import logging
from typing import Final
from typing import TYPE_CHECKING

from google.genai import types
from google.genai.models import Models
from opentelemetry import _logs
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry._logs import LogRecord
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
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from typing_extensions import deprecated

from .. import version
from ..utils.env_utils import is_enterprise_mode_enabled
from ..utils.model_name_utils import is_gemini_model
from ._experimental_semconv import maybe_log_completion_details
from ._experimental_semconv import set_operation_details_attributes_from_request
from ._experimental_semconv import set_operation_details_attributes_from_response
from ._experimental_semconv import set_operation_details_common_attributes
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
from .context import TelemetryConfig

# By default some ADK spans include attributes with potential PII data.
# This env, when set to false, allows to disable populating those attributes.
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"

# Used to associate a span with a destination resource for AppHub. Tools with
# this key in their BaseTool.custom_metadata will have the mapping added as a
# span attribute
GCP_MCP_SERVER_DESTINATION_ID = "gcp.mcp.server.destination.id"

# Silence unused warnings, but keep the public interface the same.
_ = OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
_ = USER_CONTENT_ELIDED

# Needed to avoid circular imports
if TYPE_CHECKING:

  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool

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


def trace_agent_invocation(
    span: trace.Span, agent: BaseAgent, ctx: InvocationContext
) -> None:
  """Sets span attributes immediately available on agent invocation according to OTEL semconv version 1.37.

  Args:
    span: Span on which attributes are set.
    agent: Agent from which attributes are gathered.
    ctx: InvocationContext from which attributes are gathered.

  Inference related fields are not set, due to their planned removal from
    invoke_agent span:
  https://github.com/open-telemetry/semantic-conventions/issues/2632

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
  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  span = span or trace.get_current_span()

  span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")

  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, tool.description)
  span.set_attribute(GEN_AI_TOOL_NAME, tool.name)

  # e.g. FunctionTool
  span.set_attribute(GEN_AI_TOOL_TYPE, tool.__class__.__name__)

  if error is not None:
    if hasattr(error, "error_type") and error.error_type is not None:
      span.set_attribute(ERROR_TYPE, str(error.error_type))
    else:
      span.set_attribute(ERROR_TYPE, type(error).__name__)
  elif error_type is not None:
    span.set_attribute(ERROR_TYPE, error_type)

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
  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  span = trace.get_current_span()

  span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
  span.set_attribute(GEN_AI_TOOL_NAME, "(merged tools)")
  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, "(merged tools)")
  span.set_attribute(GEN_AI_TOOL_CALL_ID, response_event_id)

  # TODO(b/441461932): See if these are still necessary
  span.set_attribute("gcp.vertex.agent.tool_call_args", "N/A")
  span.set_attribute("gcp.vertex.agent.event_id", response_event_id)
  try:
    function_response_event_json = function_response_event.model_dumps_json(
        exclude_none=True
    )
  except Exception:  # pylint: disable=broad-exception-caught
    function_response_event_json = "<not serializable>"

  if telemetry_config.should_add_content_to_legacy_spans:
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
  telemetry_config = _telemetry_config_from_invocation_context(
      invocation_context
  )
  span = span or trace.get_current_span()
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
      llm_response_json = llm_response.model_dump_json(exclude_none=True)
    except Exception:  # pylint: disable=broad-exception-caught
      llm_response_json = "<not serializable>"

    span.set_attribute(
        "gcp.vertex.agent.llm_response",
        llm_response_json,
    )
  else:
    span.set_attribute("gcp.vertex.agent.llm_response", "{}")

  _set_usage_metadata_attributes(span, llm_response.usage_metadata)
  if llm_response.finish_reason:
    try:
      finish_reason_str = llm_response.finish_reason.value.lower()
    except AttributeError:
      finish_reason_str = str(llm_response.finish_reason).lower()
    span.set_attribute(
        "gen_ai.response.finish_reasons",
        [finish_reason_str],
    )


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
            types.Content(role=content.role, parts=content.parts).model_dump(
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
              "http_options": {
                  "httpx_client": True,
                  "httpx_async_client": True,
                  "aiohttp_client": True,
              },
          },
          mode="json",
      ),
      "contents": [],
  }
  # We do not want to send bytes data to the trace.
  for content in llm_request.contents:
    parts = [part for part in content.parts if not part.inline_data]
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
  if (run_config := invocation_context.run_config) is None:
    return TelemetryConfig()
  return run_config.telemetry or TelemetryConfig()


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
    async with _use_native_generate_content_span(
        llm_request=llm_request,
        common_attributes=common_attributes,
        log_only_common_attributes=log_only_common_attributes,
        telemetry_config=telemetry_config,
    ) as gc_span:
      if telemetry_config.should_use_experimental_genai_semconv:
        set_operation_details_common_attributes(
            gc_span.operation_details_common_attributes,
            telemetry_config,
            common_attributes,
            log_only_attributes=log_only_common_attributes,
        )
      try:
        yield gc_span
      finally:
        maybe_log_completion_details(
            gc_span.span,
            otel_logger,
            gc_span.operation_details_attributes,
            gc_span.operation_details_common_attributes,
            telemetry_config,
        )
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
        in maybe_wrapped_function.__code__.co_filename
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
  from ..agents.llm_agent import LlmAgent

  if not isinstance(agent, LlmAgent):
    return False

  model = agent.model if agent.model != "" else agent._default_model
  model_name = model if isinstance(model, str) else model.model
  return is_gemini_model(model_name)


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
  with tracer.start_as_current_span(
      f"generate_content {llm_request.model or ''}"
  ) as span:
    span.set_attribute(GEN_AI_SYSTEM, _guess_gemini_system_name())
    _set_common_generate_content_attributes(
        span, llm_request, common_attributes
    )
    gc_span = GenerateContentSpan(span)

    otel_logger.emit(
        LogRecord(
            event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
            body=system_message_body(llm_request, telemetry_config),
            attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
        )
    )
    user_message_attributes = {GEN_AI_SYSTEM: _guess_gemini_system_name()}
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


@asynccontextmanager
async def _use_native_generate_content_span(
    llm_request: LlmRequest,
    common_attributes: Mapping[str, AttributeValue],
    telemetry_config: TelemetryConfig,
    log_only_common_attributes: Mapping[str, AttributeValue] | None = None,
) -> AsyncIterator[GenerateContentSpan]:
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
      f"generate_content {llm_request.model or ''}"
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


@deprecated(
    "Replaced by trace_inference_result to support experimental semconv."
)
def trace_generate_content_result(span: Span | None, llm_response: LlmResponse):
  """Trace result of the inference in generate_content span."""

  if span is None:
    return

  if llm_response.partial:
    return

  if finish_reason := llm_response.finish_reason:
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])
  _set_usage_metadata_attributes(span, llm_response.usage_metadata)

  otel_logger.emit(
      LogRecord(
          event_name=GEN_AI_CHOICE_EVENT,
          body=choice_body(llm_response, TelemetryConfig()),
          attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
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

  if llm_response.partial:
    return

  if finish_reason := llm_response.finish_reason:
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])
  _set_usage_metadata_attributes(span, llm_response.usage_metadata)

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
            attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
        )
    )


def _guess_gemini_system_name() -> str:
  return (
      GenAiSystemValues.VERTEX_AI.name.lower()
      if is_enterprise_mode_enabled()
      else GenAiSystemValues.GEMINI.name.lower()
  )
