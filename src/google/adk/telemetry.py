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
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
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
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute('gen_ai.operation.name', 'execute_tool')
  span.set_attribute('gen_ai.tool.name', '(merged tools)')
  span.set_attribute('gen_ai.tool.description', '(merged tools)')
  span.set_attribute('gen_ai.tool.call.id', response_event_id)

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
  span.set_attribute(
      'gcp.vertex.agent.llm_request',
      _safe_json_serialize(_build_llm_request_for_trace(llm_request)),
  )
  # Consider removing once GenAI SDK provides a way to record this info.

  try:
    llm_response_json = llm_response.model_dump_json(exclude_none=True)
  except Exception:  # pylint: disable=broad-exception-caught
    llm_response_json = '<not serializable>'

  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      llm_response_json,
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
