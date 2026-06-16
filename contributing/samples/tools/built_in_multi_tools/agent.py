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

import os
import random

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

load_dotenv(override=True)

VERTEXAI_DATASTORE_ID = os.getenv("VERTEXAI_DATASTORE_ID")
if not VERTEXAI_DATASTORE_ID:
  raise ValueError("VERTEXAI_DATASTORE_ID environment variable not set")


def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  result = random.randint(1, sides)
  if "rolls" not in tool_context.state:
    tool_context.state["rolls"] = []

  tool_context.state["rolls"] = tool_context.state["rolls"] + [result]
  return result


root_agent = Agent(
    model="gemini-2.5-pro",
    name="hello_world_agent",
    description="A hello world agent with multiple tools.",
    instruction="""
      You are a helpful assistant which can help user to roll dice and search for information.
      - Use `roll_die` tool to roll dice.
      - Use `VertexAISearchTool` to search for Google Agent Development Kit (ADK) information in the datastore.
      - Use `google_search` to search for general information.
    """,
    tools=[
        roll_die,
        VertexAiSearchTool(
            data_store_id=VERTEXAI_DATASTORE_ID, bypass_multi_tools_limit=True
        ),
        GoogleSearchTool(bypass_multi_tools_limit=True),
    ],
)
