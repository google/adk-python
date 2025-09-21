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

from typing import List
from typing import Union

from google.adk import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext


def update_fc(
    data_one: str,
    data_two: Union[int, float, str],
    data_three: list[str],
    data_four: List[Union[int, float, str]],
    tool_context: ToolContext,
) -> str:
  """Simply ask to update these variables in the context"""
  tool_context.actions.update_state('data_one', data_one)
  tool_context.actions.update_state('data_two', data_two)
  tool_context.actions.update_state('data_three', data_three)
  tool_context.actions.update_state('data_four', data_four)
  return 'The function `update_fc` executed successfully'


def echo_info(customer_id: str) -> str:
  """Echo the context variable"""
  return customer_id


def build_global_instruction(invocation_context: InvocationContext) -> str:
  return (
      'This is the gloabl agent instruction for invocation:'
      f' {invocation_context.invocation_id}.'
  )


def build_sub_agent_instruction(invocation_context: InvocationContext) -> str:
  return 'This is the plain text sub agent instruction.'


context_variable_echo_agent = Agent(
    model='gemini-1.5-flash',
    name='context_variable_echo_agent',
    instruction=(
        'Use the echo_info tool to echo {customerId}, {customerInt},'
        ' {customerFloat}, and {customerJson}. Ask for it if you need to.'
    ),
    flow='auto',
    tools=[echo_info],
)

context_variable_with_complicated_format_agent = Agent(
    model='gemini-1.5-flash',
    name='context_variable_echo_agent',
    instruction=(
        'Use the echo_info tool to echo { customerId }, {{customer_int  }, { '
        " non-identifier-float}}, {artifact.fileName}, {'key1': 'value1'} and"
        " {{'key2': 'value2'}}. Ask for it if you need to."
    ),
    flow='auto',
    tools=[echo_info],
)

context_variable_with_nl_planner_agent = Agent(
    model='gemini-1.5-flash',
    name='context_variable_with_nl_planner_agent',
    instruction=(
        'Use the echo_info tool to echo {customerId}. Ask for it if you'
        ' need to.'
    ),
    flow='auto',
    planner=PlanReActPlanner(),
    tools=[echo_info],
)

context_variable_with_function_instruction_agent = Agent(
    model='gemini-1.5-flash',
    name='context_variable_with_function_instruction_agent',
    instruction=build_sub_agent_instruction,
    flow='auto',
)

context_variable_update_agent = Agent(
    model='gemini-1.5-flash',
    name='context_variable_update_agent',
    instruction='Call tools',
    flow='auto',
    tools=[update_fc],
)

root_agent = Agent(
    model='gemini-1.5-flash',
    name='root_agent',
    description='The root agent.',
    flow='auto',
    global_instruction=build_global_instruction,
    sub_agents=[
        context_variable_with_nl_planner_agent,
        context_variable_update_agent,
    ],
)
