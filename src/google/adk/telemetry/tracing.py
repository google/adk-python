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

# NOTE:
#
#    We expect that the underlying GenAI SDK will provide a certain
#    level of tracing and logging telemetry aligned with Open Telemetry
#    Semantic Conventions (such as logging prompts, responses,
#    request properties, etc.) and so the information that is recorded by the
#    Agent Development Kit should be focused on the higher-level
#    constructs of the framework that are not observable by the SDK.

from __future__ import annotations

import json
import os
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry import trace

from .. import version
from ..events.event import Event

# By default some ADK spans include attributes with potential PII data.
# This env, when set to false, allows to disable populating those attributes.
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS = 'ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS'
# TODO: Replace with constant from opentelemetry.semconv when it reaches version 1.37 in g3.
GEN_AI_AGENT_DESCRIPTION = 'gen_ai.agent.description'
GEN_AI_AGENT_NAME = 'gen_ai.agent.name'
GEN_AI_CONVERSATION_ID = 'gen_ai.conversation.id'
GEN_AI_OPERATION_NAME = 'gen_ai.operation.name'
GEN_AI_TOOL_CALL_ID = 'gen_ai.tool.call.id'
GEN_AI_TOOL_DESCRIPTION = 'gen_ai.tool.description'
GEN_AI_TOOL_NAME = 'gen_ai.tool.name'
GEN_AI_TOOL_TYPE = 'gen_ai.tool.type'

# Needed to avoid circular imports
if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool

tracer = trace.get_tracer(
    instrumenting_module_name='gcp.vertex.agent',
    instrumenting_library_version=version.__version__,
    # TODO: Replace with constant from opentelemetry.semconv when it reaches version 1.37 in g3.
    schema_url='https://opentelemetry.io/schemas/1.37.0',
)


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
    function_response_event: Optional[Event],
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
    span.set_attribute('gcp.vertex.agent.tool_call_args', {})

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
    span.set_attribute('gcp.vertex.agent.tool_response', {})


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
    span.set_attribute('gcp.vertex.agent.tool_response', {})
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
  span = trace.get_current_span()
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
    span.set_attribute('gcp.vertex.agent.llm_request', {})
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
    span.set_attribute('gcp.vertex.agent.llm_response', {})

  if llm_response.usage_metadata is not None:
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
                exclude_none=True
            )
            for content in data
        ]),
    )
  else:
    span.set_attribute('gcp.vertex.agent.data', {})


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
          exclude_none=True, exclude='response_schema'
      ),
      'contents': [],
  }
  # We do not want to send bytes data to the trace.
  for content in llm_request.contents:
    parts = [part for part in content.parts if not part.inline_data]
    result['contents'].append(
        types.Content(role=content.role, parts=parts).model_dump(
            exclude_none=True
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
