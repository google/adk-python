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

import random

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.
    tool_context: The tool context to use for the die roll.

  Returns:
    An integer of the result of rolling the die.
    The result is also stored in the tool context for future use.
  """
  result = random.randint(1, sides)
  if 'rolls' not in tool_context.state:
    tool_context.state['rolls'] = []

  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result


async def before_model_callback(callback_context, llm_request):
  print('@before_model_callback')
  print(f'Beginning model choice: {llm_request.model}')
  callback_context.state['beginning_model_choice'] = llm_request.model
  return None


async def after_model_callback(callback_context, llm_response):
  print('@after_model_callback')
  print(f'Final model choice: {llm_response.model_version}')
  callback_context.state['final_model_choice'] = llm_response.model_version
  return None


root_agent = Agent(
    model=LiteLlm(
        model='gemini/gemini-2.5-pro',
        fallbacks=[
            'anthropic/claude-sonnet-4-5-20250929',
            'openai/gpt-4o',
        ],
    ),
    name='resilient_agent',
    description=(
        'hello world agent that can roll a dice of given number of sides.'
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
      You can roll dice of different sizes.
      It is ok to discuss previous dice roles, and comment on the dice rolls.
      When you are asked to roll a die, you must call the roll_die tool with the number of sides. Be sure to pass in an integer. Do not pass in a string.
      You should never roll a die on your own.
    """,
    tools=[
        roll_die,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
