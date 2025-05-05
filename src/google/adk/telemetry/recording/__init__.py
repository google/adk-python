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

import json
from typing import Any

from google.genai import types
from opentelemetry import trace

from .agents.invocation_context import InvocationContext
from .events.event import Event
from .models.llm_request import LlmRequest
from .models.llm_response import LlmResponse


tracer = trace.get_tracer('gcp.vertex.agent')


def trace_tool_call(
    args: dict[str, Any],
):
  """Traces tool call.

  Args:
    args: The arguments to the tool call.
  """
  span = trace.get_current_span()
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute('gcp.vertex.agent.tool_call_args', json.dumps(args))


def trace_tool_response(
    invocation_context: InvocationContext,
    event_id: str,
    function_response_event: Event,
):
  """Traces tool response event.

  This function records details about the tool response event as attributes on
  the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    function_response_event: The function response event which can be either
      merged function response for parallel function calls or individual
      function response for sequential function calls.
  """
  span = trace.get_current_span()
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  span.set_attribute(
      'gcp.vertex.agent.tool_response',
      function_response_event.model_dump_json(exclude_none=True),
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
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  # Consider removing once GenAI SDK provides a way to record this info.
  span.set_attribute(
      'gcp.vertex.agent.llm_request',
      json.dumps(_build_llm_request_for_trace(llm_request)),
  )
  # Consider removing once GenAI SDK provides a way to record this info.
  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      llm_response.model_dump_json(exclude_none=True),
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
      json.dumps([
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
