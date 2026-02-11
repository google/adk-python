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
import logging
import re
from typing import Any
from typing import Dict
from typing import List

from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.tools.agent_simulator.tool_connection_map import ToolConnectionMap
from google.adk.tools.base_tool import BaseTool
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types

_TOOL_CONNECTION_ANALYSIS_PROMPT_TEMPLATE = """
  You are an expert software architect analyzing a set of tools to understand
  stateful dependencies. Your task is to identify parameters that act as
  stateful identifiers (like IDs) and classify the tools that interact with
  them.

  **Definitions:**
  - A **"creating tool"** is a tool that creates a new resource or makes a
    significant state change to an existing one (e.g., creating, updating,
    canceling, or deleting). Tool names like `create_account`, `cancel_order`,
    or `update_price` are strong indicators. These tools are responsible for
    generating or modifying the state associated with an ID.
  - A **"consuming tool"** is a tool that uses a resource's ID to retrieve
    information without changing its state. Tool names like `get_user`,
    `list_events`, or `find_order` are strong indicators.

  **Your Goal:**
  Analyze the following tool schemas and identify the shared, stateful
  parameters (like `user_id`, `order_id`, etc.).

  For each stateful parameter you identify, classify the tools into
  `creating_tools` and `consuming_tools` based on the definitions above.

  **Example:** A `create_ticket` tool would be a `creating_tool` for
  `ticket_id`. A `get_ticket` tool would be a `consuming_tool` for
  `ticket_id`. A `list_tickets` tool that takes a `user_id` as input is a
  `consuming_tool` for `user_id`.

  **Analyze the following tool schemas:**
  {tool_schemas_json}

  **Output Format:**
  Generate a JSON object with a single key, "stateful_parameters", which is a
  list. Each item in the list must have these keys:
  - "parameter_name": The name of the shared parameter (e.g., "ticket_id").
  - "creating_tools": A list of tools that create or modify this parameter's
    state.
  - "consuming_tools": A list of tools that use this parameter as input for
    read-only operations.

  ONLY return the raw JSON object.
  Your response must start with '{{' and end with '}}'.
  """


class ToolConnectionAnalyzer:
  """
  Uses an LLM to analyze stateful connections between tools. For example,
  get_ticket will consume a ticket_id created by create_ticket, the analyzer
  will create a list of such connections.
  """

  def __init__(
      self, llm_name: str, llm_config: genai_types.GenerateContentConfig
  ):
    self._llm_name = llm_name
    self._llm_config = llm_config
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(self._llm_name)
    self._llm = llm_class(model=self._llm_name)

  async def analyze(self, tools: List[BaseTool]) -> ToolConnectionMap:
    """
    Analyzes a list of tools and returns a map of their connections.
    """
    tool_schemas = [
        tool._get_declaration().model_dump(exclude_none=True)
        for tool in tools
        if tool._get_declaration()
    ]
    tool_schemas_json = json.dumps(tool_schemas, indent=2)
    prompt = _TOOL_CONNECTION_ANALYSIS_PROMPT_TEMPLATE.format(
        tool_schemas_json=tool_schemas_json
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
        if not generated_content.parts:
          continue
        for part in generated_content.parts:
          if part.text:
            response_text += part.text

    try:
      clean_json_text = re.sub(r"^```[a-zA-Z]*\n", "", response_text)
      clean_json_text = re.sub(r"\n```$", "", clean_json_text)
      response_json = json.loads(clean_json_text.strip())
    except json.JSONDecodeError:
      logging.warning(
          "Failed to parse tool connection analysis from LLM. Proceeding"
          " without connection map. Error: %s\nLLM Output:\n%s",
          e,
          response_text,
      )
      return ToolConnectionMap(stateful_parameters=[])
    return ToolConnectionMap.model_validate(response_json)
