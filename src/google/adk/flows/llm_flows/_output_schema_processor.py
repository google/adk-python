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

"""Handles output schema when tools are also present."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.set_model_response_tool import SetModelResponseTool
from ...utils._schema_utils import is_basemodel_schema
from ...utils.output_schema_utils import can_use_output_schema_with_tools
from ._base_llm_processor import BaseLlmRequestProcessor

logger = logging.getLogger('google_adk.' + __name__)

# Max tool rounds before forcing set_model_response (N-1) or terminating (N).
_MAX_TOOL_ROUNDS = 25


class _OutputSchemaRequestProcessor(BaseLlmRequestProcessor):
  """Processor that handles output schema for agents with tools."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent

    # Check if we need the processor: output_schema + tools + cannot use output
    # schema with tools
    if (
        not agent.output_schema
        or not agent.tools
        or can_use_output_schema_with_tools(agent.canonical_model)
    ):
      return

    # Count how many tool rounds have occurred in this invocation.
    tool_rounds = sum(
        1
        for e in invocation_context._get_events(
            current_invocation=True, current_branch=True
        )
        if e.get_function_responses()
    )

    # Terminate the invocation if the model never calls set_model_response.
    if tool_rounds >= _MAX_TOOL_ROUNDS:
      logger.error(
          'Tool execution reached %d rounds without producing structured'
          ' output via set_model_response. Breaking loop to prevent'
          ' runaway API costs.',
          tool_rounds,
      )
      invocation_context.end_invocation = True
      return

    # Add the set_model_response tool to handle structured output
    set_response_tool = SetModelResponseTool(agent.output_schema)
    llm_request.append_tools([set_response_tool])

    # Primitive types (str, int, etc.) produce a trivial tool signature
    # that flash models tend to ignore use a stronger instruction.
    if is_basemodel_schema(agent.output_schema):
      instruction = (
          'After completing any needed tool calls, provide your final'
          ' response by calling set_model_response with the required'
          ' fields.'
      )
    else:
      instruction = (
          'IMPORTANT: After using any needed tools, you MUST call'
          ' set_model_response to provide your final answer.'
          ' This is required to complete the task.'
      )
    llm_request.append_instructions([instruction])

    # On round N-1, restrict the model to only call set_model_response.
    if tool_rounds >= _MAX_TOOL_ROUNDS - 1:
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tool_config = types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(
              mode=types.FunctionCallingConfigMode.ANY,
              allowed_function_names=['set_model_response'],
          )
      )

    return
    yield  # Generator requires yield statement in function body.


def create_final_model_response_event(
    invocation_context: InvocationContext, json_response: str
) -> Event:
  """Create a final model response event from set_model_response JSON.

  Args:
    invocation_context: The invocation context.
    json_response: The JSON response from set_model_response tool.

  Returns:
    A new Event that looks like a normal model response.
  """
  from google.genai import types

  # Create a proper model response event
  final_event = Event(
      author=invocation_context.agent.name,
      invocation_id=invocation_context.invocation_id,
      branch=invocation_context.branch,
  )
  final_event.content = types.Content(
      role='model', parts=[types.Part(text=json_response)]
  )
  return final_event


def get_structured_model_response(function_response_event: Event) -> str | None:
  """Check if function response contains set_model_response and extract JSON.

  Args:
    function_response_event: The function response event to check.

  Returns:
    JSON response string if set_model_response was called, None otherwise.
  """
  if (
      not function_response_event
      or not function_response_event.get_function_responses()
  ):
    return None

  for func_response in function_response_event.get_function_responses():
    if func_response.name == 'set_model_response':
      # Extract the actual result from the wrapped response.
      # Tool results are wrapped as {'result': ...} when not already a dict.
      response = func_response.response
      if isinstance(response, dict) and 'result' in response:
        response = response['result']
      return json.dumps(response, ensure_ascii=False)

  return None


# Export the processors
request_processor = _OutputSchemaRequestProcessor()
