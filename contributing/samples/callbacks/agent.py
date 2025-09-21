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
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  result = random.randint(1, sides)
  if not 'rolls' in tool_context.state:
    tool_context.state['rolls'] = []

  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result


async def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime.

  Args:
    nums: The list of numbers to check.

  Returns:
    A str indicating which number is prime.
  """
  primes = set()
  for number in nums:
    number = int(number)
    if number <= 1:
      continue
    is_prime = True
    for i in range(2, int(number**0.5) + 1):
      if number % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.add(number)
  return (
      'No prime numbers found.'
      if not primes
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


async def before_agent_callback(callback_context):
  print('@before_agent_callback')
  return None


async def after_agent_callback(callback_context):
  print('@after_agent_callback')
  return None


async def before_model_callback(callback_context, llm_request):
  print('@before_model_callback')
  return None


async def after_model_callback(callback_context, llm_response):
  print('@after_model_callback')
  return None


def after_agent_cb1(callback_context):
  print('@after_agent_cb1')


def after_agent_cb2(callback_context):
  print('@after_agent_cb2')
  # ModelContent (or Content with role set to 'model') must be returned.
  # Otherwise, the event will be excluded from the context in the next turn.
  return types.ModelContent(
      parts=[
          types.Part(
              text='(stopped) after_agent_cb2',
          ),
      ],
  )


def after_agent_cb3(callback_context):
  print('@after_agent_cb3')


def before_agent_cb1(callback_context):
  print('@before_agent_cb1')


def before_agent_cb2(callback_context):
  print('@before_agent_cb2')


def before_agent_cb3(callback_context):
  print('@before_agent_cb3')


def before_tool_cb1(tool, args, tool_context):
  print('@before_tool_cb1')


def before_tool_cb2(tool, args, tool_context):
  print('@before_tool_cb2')


def before_tool_cb3(tool, args, tool_context):
  print('@before_tool_cb3')


def after_tool_cb1(tool, args, tool_context, tool_response):
  print('@after_tool_cb1')


def after_tool_cb2(tool, args, tool_context, tool_response):
  print('@after_tool_cb2')
  return {'test': 'after_tool_cb2', 'response': tool_response}


def after_tool_cb3(tool, args, tool_context, tool_response):
  print('@after_tool_cb3')


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
      You can use multiple tools in parallel by calling functions in parallel(in one request and in one round).
      It is ok to discuss previous dice roles, and comment on the dice rolls.
      When you are asked to roll a die, you must call the roll_die tool with the number of sides. Be sure to pass in an integer. Do not pass in a string.
      You should never roll a die on your own.
      When checking prime numbers, call the check_prime tool with a list of integers. Be sure to pass in a list of integers. You should never pass in a string.
      You should not check prime numbers before calling the tool.
      When you are asked to roll a die and check prime numbers, you should always make the following two function calls:
      1. You should first call the roll_die tool to get a roll. Wait for the function response before calling the check_prime tool.
      2. After you get the function response from roll_die tool, you should call the check_prime tool with the roll_die result.
        2.1 If user asks you to check primes based on previous rolls, make sure you include the previous rolls in the list.
      3. When you respond, you must include the roll_die result from step 1.
      You should always perform the previous 3 steps when asking for a roll and checking prime numbers.
      You should not rely on the previous history on prime results.
    """,
    tools=[
        roll_die,
        check_prime,
    ],
    # planner=BuiltInPlanner(
    #     thinking_config=types.ThinkingConfig(
    #         include_thoughts=True,
    #     ),
    # ),
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
    before_agent_callback=[
        before_agent_cb1,
        before_agent_cb2,
        before_agent_cb3,
    ],
    after_agent_callback=[after_agent_cb1, after_agent_cb2, after_agent_cb3],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=[before_tool_cb1, before_tool_cb2, before_tool_cb3],
    after_tool_callback=[after_tool_cb1, after_tool_cb2, after_tool_cb3],
)
