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
from typing import Any

from google.genai import types
from opentelemetry import trace

from .agents.invocation_context import InvocationContext
from .events.event import Event
from .models.llm_request import LlmRequest
from .models.llm_response import LlmResponse
from .tools.base_tool import BaseTool

tracer = trace.get_tracer('gcp.vertex.agent')


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


def trace_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    function_response_event: Event,
):
  """Traces tool call.

  Args:
    tool: The tool that was called.
    args: The arguments to the tool call.
    function_response_event: The event with the function response details.
  """
  span = trace.get_current_span()

  # Standard OpenTelemetry GenAI attributes as of OTel SemConv v1.36.0 for Agents and Frameworks
  span.set_attribute('gen_ai.system', 'gcp.vertex_ai')
  span.set_attribute('gen_ai.operation.name', 'execute_tool')
  span.set_attribute('gen_ai.tool.name', tool.name)
  span.set_attribute('gen_ai.tool.description', tool.description)

  tool_call_id = '<not specified>'
  tool_response = '<not specified>'
  if function_response_event.content.parts:
    function_response = function_response_event.content.parts[
        0
    ].function_response
    if function_response is not None:
      tool_call_id = function_response.id
      tool_response = function_response.response

  span.set_attribute('gen_ai.tool.call.id', tool_call_id)

  # Vendor-specific attributes (moved from gen_ai.* to gcp.vertex.agent.*)
  if not isinstance(tool_response, dict):
    tool_response = {'result': tool_response}
  span.set_attribute(
      'gcp.vertex.agent.tool_call_args',
      _safe_json_serialize(args),
  )
  span.set_attribute('gcp.vertex.agent.event_id', function_response_event.id)
  span.set_attribute(
      'gcp.vertex.agent.tool_response',
      _safe_json_serialize(tool_response),
  )
  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute('gcp.vertex.agent.llm_request', '{}')
  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      '{}',
  )


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

  # Standard OpenTelemetry GenAI attributes
  span.set_attribute('gen_ai.system', 'gcp.vertex_ai')
  span.set_attribute('gen_ai.operation.name', 'execute_tool')
  span.set_attribute('gen_ai.tool.name', '(merged tools)')
  span.set_attribute('gen_ai.tool.description', '(merged tools)')
  span.set_attribute('gen_ai.tool.call.id', response_event_id)

  # Vendor-specific attributes
  span.set_attribute('gcp.vertex.agent.tool_call_args', 'N/A')
  span.set_attribute('gcp.vertex.agent.event_id', response_event_id)
  try:
    function_response_event_json = function_response_event.model_dumps_json(
        exclude_none=True
    )
  except Exception:  # pylint: disable=broad-exception-caught
    function_response_event_json = '<not serializable>'

  span.set_attribute(
      'gcp.vertex.agent.tool_response',
      function_response_event_json,
  )
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

  # Standard OpenTelemetry GenAI attributes
  span.set_attribute('gen_ai.system', 'gcp.vertex_ai')
  span.set_attribute('gen_ai.operation.name', 'chat')
  span.set_attribute('gen_ai.request.model', llm_request.model)

  if hasattr(llm_response, 'id') and llm_response.id:
    span.set_attribute('gen_ai.response.id', llm_response.id)

  # Set response model if different from request model
  if (
      hasattr(llm_response, 'model')
      and llm_response.model
      and llm_response.model != llm_request.model
  ):
    span.set_attribute('gen_ai.response.model', llm_response.model)

  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute(
      'gcp.vertex.agent.session_id', invocation_context.session.id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)

  # Consider removing once GenAI SDK provides a way to record this info.
  span.set_attribute(
      'gcp.vertex.agent.llm_request',
      _safe_json_serialize(_build_llm_request_for_trace(llm_request)),
  )

  # Standard GenAI request attributes
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
    if (
        hasattr(llm_request.config, 'temperature')
        and llm_request.config.temperature is not None
    ):
      span.set_attribute(
          'gen_ai.request.temperature',
          llm_request.config.temperature,
      )

  try:
    llm_response_json = llm_response.model_dump_json(exclude_none=True)
  except Exception:  # pylint: disable=broad-exception-caught
    llm_response_json = '<not serializable>'

  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      llm_response_json,
  )

  # Standard GenAI usage and response attributes
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
    span.set_attribute(
        'gen_ai.response.finish_reasons',
        [llm_response.finish_reason.value.lower()],
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
  span.set_attribute(
      'gcp.vertex.agent.data',
      _safe_json_serialize([
          types.Content(role=content.role, parts=content.parts).model_dump(
              exclude_none=True
          )
          for content in data
      ]),
  )


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
  # Some fields in LlmRequest are function pointers and can not be serialized.
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


def _create_span_name(operation_name: str, model_name: str) -> str:
  """Creates a span name following OpenTelemetry GenAI conventions.

  Args:
    operation_name: The GenAI operation name (e.g., 'generate_content', 'execute_tool').
    model_name: The model name being used.

  Returns:
    A span name in the format '{operation_name} {model_name}'.
  """
  return f'{operation_name} {model_name}'


def add_genai_prompt_event(span: trace.Span, prompt_content: str):
  """Adds a GenAI prompt event to the span following OpenTelemetry conventions.

  Args:
    span: The OpenTelemetry span to add the event to.
    prompt_content: The prompt content as a JSON string.
  """
  span.add_event(
      name='gen_ai.content.prompt', attributes={'gen_ai.prompt': prompt_content}
  )


def add_genai_completion_event(span: trace.Span, completion_content: str):
  """Adds a GenAI completion event to the span following OpenTelemetry conventions.

  Args:
    span: The OpenTelemetry span to add the event to.
    completion_content: The completion content as a JSON string.
  """
  span.add_event(
      name='gen_ai.content.completion',
      attributes={'gen_ai.completion': completion_content},
  )
