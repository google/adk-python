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

from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
import json
import logging
import os
from typing import Any
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
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import Span
from opentelemetry.util.types import AnyValue
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from .. import version
from ..utils.model_name_utils import is_gemini_model

# By default some ADK spans include attributes with potential PII data.
# This env, when set to false, allows to disable populating those attributes.
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = 'ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS'

# Standard OTEL env variable to enable logging of prompt/response content.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
)

USER_CONTENT_ELIDED = '<elided>'

# Needed to avoid circular imports
if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool

tracer = trace.get_tracer(
    instrumenting_module_name='gcp.vertex.agent',
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

otel_logger = _logs.get_logger(
    instrumenting_module_name='gcp.vertex.agent',
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

logger = logging.getLogger('google_adk.' + __name__)


def _safe_json_serialize(obj) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or <non-serializable> if the object cannot be serialized.
  """

  try:
    # Try direct JSON serialization first
    return json.dumps(
        obj, ensure_ascii=False, default=lambda o: '<not serializable>'
    )
  except (TypeError, OverflowError):
    return '<not serializable>'


def trace_agent_invocation(
    span: trace.Span, agent: BaseAgent, ctx: InvocationContext
) -> None:
  """Sets span attributes immediately available on agent invocation according to OTEL semconv version 1.37.

  Args:
    span: Span on which attributes are set.
    agent: Agent from which attributes are gathered.
    ctx: InvocationContext from which attributes are gathered.

  Inference related fields are not set, due to their planned removal from invoke_agent span:
  https://github.com/open-telemetry/semantic-conventions/issues/2632

  `gen_ai.agent.id` is not set because currently it's unclear what attributes this field should have, specifically:
  - In which scope should it be unique (globally, given project, given agentic flow, given deployment).
  - Should it be unchanging between deployments, and how this should this be achieved.

  `gen_ai.data_source.id` is not set because it's not available.
  Closest type which could contain this information is types.GroundingMetadata, which does not have an ID.

  `server.*` attributes are not set pending confirmation from aabmass.
  """

  # Required
  span.set_attribute(GEN_AI_OPERATION_NAME, 'invoke_agent')

  # Conditionally Required
  span.set_attribute(GEN_AI_AGENT_DESCRIPTION, agent.description)

  span.set_attribute(GEN_AI_AGENT_NAME, agent.name)
  span.set_attribute(GEN_AI_CONVERSATION_ID, ctx.session.id)


def trace_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    function_response_event: Event | None,
):
  """Traces tool call.

  Args:
    tool: The tool that was called.
    args: The arguments to the tool call.
    function_response_event: The event with the function response details.
  """
  span = trace.get_current_span()

  span.set_attribute(GEN_AI_OPERATION_NAME, 'execute_tool')

  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, tool.description)
  span.set_attribute(GEN_AI_TOOL_NAME, tool.name)

  # e.g. FunctionTool
  span.set_attribute(GEN_AI_TOOL_TYPE, tool.__class__.__name__)

  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute('gcp.vertex.agent.llm_request', '{}')
  span.set_attribute('gcp.vertex.agent.llm_response', '{}')

  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.tool_call_args',
        _safe_json_serialize(args),
    )
  else:
    span.set_attribute('gcp.vertex.agent.tool_call_args', '{}')

  # Tracing tool response
  tool_call_id = '<not specified>'
  tool_response = '<not specified>'
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
    tool_response = {'result': tool_response}
  if function_response_event is not None:
    span.set_attribute('gcp.vertex.agent.event_id', function_response_event.id)
  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.tool_response',
        _safe_json_serialize(tool_response),
    )
  else:
    span.set_attribute('gcp.vertex.agent.tool_response', '{}')


def trace_merged_tool_calls(
    response_event_id: str,
    function_response_event: Event,
):
  """Traces merged tool call events.

  Calling this function is not needed for telemetry purposes. This is provided
  for preventing /debug/trace requests (typically sent by web UI).

  Args:
    response_event_id: The ID of the response event.
    function_response_event: The merged response event.
  """

  span = trace.get_current_span()

  span.set_attribute(GEN_AI_OPERATION_NAME, 'execute_tool')
  span.set_attribute(GEN_AI_TOOL_NAME, '(merged tools)')
  span.set_attribute(GEN_AI_TOOL_DESCRIPTION, '(merged tools)')
  span.set_attribute(GEN_AI_TOOL_CALL_ID, response_event_id)

  # TODO(b/441461932): See if these are still necessary
  span.set_attribute('gcp.vertex.agent.tool_call_args', 'N/A')
  span.set_attribute('gcp.vertex.agent.event_id', response_event_id)
  try:
    function_response_event_json = function_response_event.model_dumps_json(
        exclude_none=True
    )
  except Exception:  # pylint: disable=broad-exception-caught
    function_response_event_json = '<not serializable>'

  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.tool_response',
        function_response_event_json,
    )
  else:
    span.set_attribute('gcp.vertex.agent.tool_response', '{}')
  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute('gcp.vertex.agent.llm_request', '{}')
  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      '{}',
  )


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
  span = span or trace.get_current_span()
  # Special standard Open Telemetry GenaI attributes that indicate
  # that this is a span related to a Generative AI system.
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute('gen_ai.request.model', llm_request.model)
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute(
      'gcp.vertex.agent.session_id', invocation_context.session.id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  # Consider removing once GenAI SDK provides a way to record this info.
  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.llm_request',
        _safe_json_serialize(_build_llm_request_for_trace(llm_request)),
    )
  else:
    span.set_attribute('gcp.vertex.agent.llm_request', '{}')
  # Consider removing once GenAI SDK provides a way to record this info.
  if llm_request.config:
    if llm_request.config.top_p:
      span.set_attribute(
          'gen_ai.request.top_p',
          llm_request.config.top_p,
      )
    if llm_request.config.max_output_tokens:
      span.set_attribute(
          'gen_ai.request.max_tokens',
          llm_request.config.max_output_tokens,
      )

  try:
    llm_response_json = llm_response.model_dump_json(exclude_none=True)
  except Exception:  # pylint: disable=broad-exception-caught
    llm_response_json = '<not serializable>'

  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.llm_response',
        llm_response_json,
    )
  else:
    span.set_attribute('gcp.vertex.agent.llm_response', '{}')

  if llm_response.usage_metadata is not None:
    if llm_response.usage_metadata.prompt_token_count is not None:
      span.set_attribute(
          'gen_ai.usage.input_tokens',
          llm_response.usage_metadata.prompt_token_count,
      )
    if llm_response.usage_metadata.candidates_token_count is not None:
      span.set_attribute(
          'gen_ai.usage.output_tokens',
          llm_response.usage_metadata.candidates_token_count,
      )
  if llm_response.finish_reason:
    try:
      finish_reason_str = llm_response.finish_reason.value.lower()
    except AttributeError:
      finish_reason_str = str(llm_response.finish_reason).lower()
    span.set_attribute(
        'gen_ai.response.finish_reasons',
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
  span = trace.get_current_span()
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  # Once instrumentation is added to the GenAI SDK, consider whether this
  # information still needs to be recorded by the Agent Development Kit.
  if _should_add_request_response_to_spans():
    span.set_attribute(
        'gcp.vertex.agent.data',
        _safe_json_serialize([
            types.Content(role=content.role, parts=content.parts).model_dump(
                exclude_none=True, mode='json'
            )
            for content in data
        ]),
    )
  else:
    span.set_attribute('gcp.vertex.agent.data', '{}')


def _build_llm_request_for_trace(llm_request: LlmRequest) -> dict[str, Any]:
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
      'model': llm_request.model,
      'config': llm_request.config.model_dump(
          exclude_none=True, exclude='response_schema', mode='json'
      ),
      'contents': [],
  }
  # We do not want to send bytes data to the trace.
  for content in llm_request.contents:
    parts = [part for part in content.parts if not part.inline_data]
    result['contents'].append(
        types.Content(role=content.role, parts=parts).model_dump(
            exclude_none=True, mode='json'
        )
    )
  return result


# Defaults to true for now to preserve backward compatibility.
# Once prompt and response logging is well established in ADK, we might start
# a deprecation of request/response content in spans by switching the default
# to false.
def _should_add_request_response_to_spans() -> bool:
  disabled_via_env_var = os.getenv(
      ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'true'
  ).lower() in ('false', '0')
  return not disabled_via_env_var


@contextmanager
def use_generate_content_span(
    llm_request: LlmRequest,
    invocation_context: InvocationContext,
    model_response_event: Event,
) -> Iterator[Span | None]:
  """Context manager encompassing `generate_content {model.name}` span.

  When an external library for inference instrumentation is installed (e.g. opentelemetry-instrumentation-google-genai),
  span creation is delegated to said library.
  """

  common_attributes = {
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      USER_ID: invocation_context.session.user_id,
      'gcp.vertex.agent.event_id': model_response_event.id,
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  }
  if (
      _is_gemini_agent(invocation_context.agent)
      and _instrumented_with_opentelemetry_instrumentation_google_genai()
  ):
    with _use_extra_generate_content_attributes(common_attributes):
      yield
  else:
    with _use_native_generate_content_span(
        llm_request=llm_request,
        common_attributes=common_attributes,
    ) as span:
      yield span


def _should_log_prompt_response_content() -> bool:
  return os.getenv(
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ''
  ).lower() in ('1', 'true')


def _serialize_content(content: types.ContentUnion) -> AnyValue:
  if isinstance(content, BaseModel):
    return content.model_dump()
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return [_serialize_content(part) for part in content]
  return _safe_json_serialize(content)


def _serialize_content_with_elision(
    content: types.ContentUnion | None,
) -> AnyValue:
  if not _should_log_prompt_response_content():
    return USER_CONTENT_ELIDED
  if content is None:
    return None
  return _serialize_content(content)


def _instrumented_with_opentelemetry_instrumentation_google_genai() -> bool:
  maybe_wrapped_function = Models.generate_content
  while wrapped := getattr(maybe_wrapped_function, '__wrapped__', None):
    if (
        'opentelemetry/instrumentation/google_genai'
        in maybe_wrapped_function.__code__.co_filename
    ):
      return True
    maybe_wrapped_function = wrapped  # pyright: ignore[reportAny]

  return False


@contextmanager
def _use_extra_generate_content_attributes(
    extra_attributes: Mapping[str, AttributeValue],
):
  try:
    from opentelemetry.instrumentation.google_genai import GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
  except (ImportError, AttributeError):
    logger.warning(
        'opentelemetry-instrumentor-google-genai is installed but has'
        ' insufficient version,'
        + ' so some tracing dependent features may not work properly.'
        + ' Please upgrade to version to 0.6b0 or above.'
    )
    yield
    return

  tok = otel_context.attach(
      otel_context.set_value(
          GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY, extra_attributes
      )
  )
  try:
    yield
  finally:
    otel_context.detach(tok)


def _is_gemini_agent(agent: BaseAgent) -> bool:
  from ..agents.llm_agent import LlmAgent

  if not isinstance(agent, LlmAgent):
    return False

  if isinstance(agent.model, str):
    return is_gemini_model(agent.model)

  from ..models.google_llm import Gemini

  return isinstance(agent.model, Gemini)


@contextmanager
def _use_native_generate_content_span(
    llm_request: LlmRequest,
    common_attributes: Mapping[str, AttributeValue],
) -> Iterator[Span]:
  with tracer.start_as_current_span(
      f"generate_content {llm_request.model or ''}"
  ) as span:
    span.set_attribute(GEN_AI_SYSTEM, _guess_gemini_system_name())
    span.set_attribute(GEN_AI_OPERATION_NAME, 'generate_content')
    span.set_attribute(GEN_AI_REQUEST_MODEL, llm_request.model or '')
    span.set_attributes(common_attributes)

    otel_logger.emit(
        LogRecord(
            event_name='gen_ai.system.message',
            body={
                'content': _serialize_content_with_elision(
                    llm_request.config.system_instruction
                )
            },
            attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
        )
    )

    for content in llm_request.contents:
      otel_logger.emit(
          LogRecord(
              event_name='gen_ai.user.message',
              body={'content': _serialize_content_with_elision(content)},
              attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
          )
      )

    yield span


def trace_generate_content_result(span: Span | None, llm_response: LlmResponse):
  """Trace result of the inference in generate_content span."""

  if span is None:
    return

  if llm_response.partial:
    return

  if finish_reason := llm_response.finish_reason:
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason.lower()])
  if usage_metadata := llm_response.usage_metadata:
    if usage_metadata.prompt_token_count is not None:
      span.set_attribute(
          GEN_AI_USAGE_INPUT_TOKENS, usage_metadata.prompt_token_count
      )
    if usage_metadata.candidates_token_count is not None:
      span.set_attribute(
          GEN_AI_USAGE_OUTPUT_TOKENS, usage_metadata.candidates_token_count
      )

  otel_logger.emit(
      LogRecord(
          event_name='gen_ai.choice',
          body={
              'content': _serialize_content_with_elision(llm_response.content),
              'index': 0,  # ADK always returns a single candidate
          }
          | {'finish_reason': llm_response.finish_reason.value}
          if llm_response.finish_reason is not None
          else {},
          attributes={GEN_AI_SYSTEM: _guess_gemini_system_name()},
      )
  )


def _guess_gemini_system_name() -> str:
  return (
      GenAiSystemValues.VERTEX_AI.name.lower()
      if os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').lower() in ('true', '1')
      else GenAiSystemValues.GEMINI.name.lower()
  )
