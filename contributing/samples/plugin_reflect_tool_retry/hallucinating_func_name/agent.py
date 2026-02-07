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

import random

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools.tool_context import ToolContext

APP_NAME = "hallucinating_func_name"
USER_ID = "test_user"

hallucinated = False  # Whether the tool name is hallucinated


def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  result = random.randint(1, sides)
  if not "rolls" in tool_context.state:
    tool_context.state["rolls"] = []

  tool_context.state["rolls"] = tool_context.state["rolls"] + [result]
  return result


def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
):
  """After model callback to produce one hallucinating tool call."""
  global hallucinated

  if hallucinated:
    return None

  if (
      llm_response.content
      and llm_response.content.parts
      and llm_response.content.parts[0].function_call
      and llm_response.content.parts[0].function_call.name == "roll_die"
  ):
    llm_response.content.parts[0].function_call.name = "roll_die_wrong_name"
    hallucinated = True
  return None


root_agent = LlmAgent(
    name="hello_world",
    description="Helpful agent",
    instruction="""Use guess_number_tool to guess a number.""",
    model="gemini-2.5-flash",
    tools=[roll_die],
    after_model_callback=after_model_callback,
)


app = App(
    name=APP_NAME,
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
