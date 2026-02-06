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
import concurrent.futures
import json
import re
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.tools.agent_simulator.strategies.base import MockStrategy
from google.adk.tools.agent_simulator.tool_connection_map import ToolConnectionMap
from google.adk.tools.base_tool import BaseTool
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types

_TOOL_SPEC_MOCK_PROMPT_TEMPLATE = """
  You are a stateful tool simulator. Your task is to generate a
  realistic JSON response for a tool call, maintaining consistency based
  on a shared state.

  Here is the map of how tools connect via stateful parameters:
  {tool_connection_map_json}

  Here is the current state of all stateful parameters:
  {state_store_json}

  You are now simulating the following tool call:
  Tool Name: {tool_name}
  Tool Description: {tool_description}
  Tool Schema: {tool_schema_json}
  Tool Arguments: {tool_arguments_json}

  Your instructions:
  1.  Analyze the tool call. Is it a "creating" or "consuming" tool
      based on the connection map?
  2.  If it's a "consuming" tool, check the provided arguments against
      the state store. If an ID is provided that does not exist in the
      state, return a realistic error (e.g., a 404 Not Found error).
      Otherwise, use the data from the state to generate the response.
  3.  If it's a "creating" tool, generate a new, unique ID for the
      stateful parameter (e.g., a random string for a ticket_id). Include
      this new ID in your response. I will then update the state with it.
  4.  Generate a convincing, valid JSON object that mocks the tool's
      response. The response must be only the JSON object, without any
      additional text or formatting.
  5.  The response must start with '{{' and end with '}}'.
  """


class ToolSpecMockStrategy(MockStrategy):
  """Mocks a tool response based on the tool's specification."""

  def __init__(
      self, llm_name: str, llm_config: genai_types.GenerateContentConfig
  ):
    self._llm_name = llm_name
    self._llm_config = llm_config
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(self._llm_name)
    self._llm = llm_class(model=self._llm_name)

  async def mock(
      self,
      tool: BaseTool,
      args: Dict[str, Any],
      tool_context: Any,
      tool_connection_map: Optional[ToolConnectionMap],
      state_store: Dict[str, Any],
  ) -> Dict[str, Any]:
    declaration = tool._get_declaration()
    if not declaration:
      return {
          "status": "error",
          "error_message": "Could not get tool declaration.",
      }

    tool_connection_map_json = (
        json.dumps(tool_connection_map.model_dump(exclude_none=True), indent=2)
        if tool_connection_map
        else "''"
    )
    state_store_json = json.dumps(state_store, indent=2)
    tool_schema_json = json.dumps(declaration.model_dump(), indent=2)
    tool_arguments_json = json.dumps(args, indent=2)

    prompt = _TOOL_SPEC_MOCK_PROMPT_TEMPLATE.format(
        tool_connection_map_json=tool_connection_map_json,
        state_store_json=state_store_json,
        tool_name=tool.name,
        tool_description=tool.description,
        tool_schema_json=tool_schema_json,
        tool_arguments_json=tool_arguments_json,
    )

    request_contents = [
        genai_types.Content(parts=[genai_types.Part(text=prompt)], role="user")
    ]
    request = LlmRequest(
        contents=request_contents,
        model=self._llm_name,
        config=self._llm_config,
        generation_config=genai_types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    response_text = ""
    async with Aclosing(self._llm.generate_content_async(request)) as agen:
      async for llm_response in agen:
        generated_content: genai_types.Content = llm_response.content
        if generated_content.parts:
          for part in generated_content.parts:
            if part.text:
              response_text += part.text

    try:
      clean_json_text = re.sub(r"^```[a-zA-Z]*\n", "", response_text)
      clean_json_text = re.sub(r"\n```$", "", clean_json_text)
      mock_response = json.loads(clean_json_text.strip())
      # After getting the response, update the state if this was a creating tool.
      if tool_connection_map:
        for param_info in tool_connection_map.stateful_parameters:
          param_name = param_info.parameter_name
          if tool.name in param_info.creating_tools:
            if param_name in mock_response:
              param_value = mock_response[param_name]
              if param_name not in state_store:
                state_store[param_name] = {}
              state_store[param_name][param_value] = mock_response
      return mock_response
    except json.JSONDecodeError:
      return {
          "status": "error",
          "error_message": "Failed to generate valid JSON mock response.",
          "llm_output": response_text,
      }
