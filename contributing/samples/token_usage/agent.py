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
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.anthropic_llm import Claude
from google.adk.models.lite_llm import LiteLlm
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
  if 'rolls' not in tool_context.state:
    tool_context.state['rolls'] = []

  tool_context.state['rolls'] = tool_context.state['rolls'] + [result]
  return result


roll_agent_with_openai = LlmAgent(
    model=LiteLlm(model='openai/gpt-4o'),
    description='Handles rolling dice of different sizes.',
    name='roll_agent_with_openai',
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
)

roll_agent_with_claude = LlmAgent(
    model=Claude(model='claude-3-7-sonnet@20250219'),
    description='Handles rolling dice of different sizes.',
    name='roll_agent_with_claude',
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
)

roll_agent_with_litellm_claude = LlmAgent(
    model=LiteLlm(model='vertex_ai/claude-3-7-sonnet'),
    description='Handles rolling dice of different sizes.',
    name='roll_agent_with_litellm_claude',
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
)

roll_agent_with_gemini = LlmAgent(
    model='gemini-2.0-flash',
    description='Handles rolling dice of different sizes.',
    name='roll_agent_with_gemini',
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
)

root_agent = SequentialAgent(
    name='code_pipeline_agent',
    sub_agents=[
        roll_agent_with_openai,
        roll_agent_with_claude,
        roll_agent_with_litellm_claude,
        roll_agent_with_gemini,
    ],
)
