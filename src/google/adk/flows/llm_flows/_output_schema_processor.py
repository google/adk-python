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

"""Handles output schema when tools are also present."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.set_model_response_tool import SetModelResponseTool
from ._base_llm_processor import BaseLlmRequestProcessor


class _OutputSchemaRequestProcessor(BaseLlmRequestProcessor):
  """Processor that handles output schema for agents with tools."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    # Check if we need the processor: output_schema + tools
    if not agent.output_schema or not agent.tools:
      return

    # Add the set_model_response tool to handle structured output
    set_response_tool = SetModelResponseTool(agent.output_schema)
    llm_request.append_tools([set_response_tool])

    # Add instruction about using the set_model_response tool
    instruction = (
        'IMPORTANT: You have access to other tools, but you must provide '
        'your final response using the set_model_response tool with the '
        'required structured format. After using any other tools needed '
        'to complete the task, always call set_model_response with your '
        'final answer in the specified schema format.'
    )
    llm_request.append_instructions([instruction])

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
  final_event = Event(author=invocation_context.agent.name)
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
      # Convert dict to JSON string
      return json.dumps(func_response.response)

  return None


# Export the processors
request_processor = _OutputSchemaRequestProcessor()
