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

from datetime import datetime
import random
from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.planners import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types


async def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  print('@roll_die is starting', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
  import asyncio
  await asyncio.sleep(3)  # Use async sleep for non-blocking delay
  result = random.randint(1, sides)
  if not 'rolls' in tool_context.state:
    tool_context.state['rolls'] = []

  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result

def before_tool_cb(tool, args, tool_context):
  print('@before_tool_cb1', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

def after_tool_cb(tool, args, tool_context, tool_response):
  print('@after_tool_cb1', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))


root_agent = Agent(
    model='gemini-2.0-flash',
    name='data_processing_agent',
    description=(
        'hello world agent that can roll a dice of 8 sides and check prime'
        ' numbers.'
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
      You can roll dice of different sizes.
      You can use multiple tools in parallel by calling functions in parallel (in one request and in one round).
      It is ok to discuss previous dice roles, and comment on the dice rolls.
      When you are asked to roll a die, you must call the roll_die tool with the number of sides. Be sure to pass in an integer. Do not pass in a string.
      
      IMPORTANT: When you are asked to roll multiple dice (e.g., "roll a die four times"), you MUST make ALL the roll_die function calls in parallel (simultaneously in one turn) to provide a faster response. Do NOT wait between dice rolls.
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
    before_tool_callback=[before_tool_cb],
    after_tool_callback=[after_tool_cb],
)
