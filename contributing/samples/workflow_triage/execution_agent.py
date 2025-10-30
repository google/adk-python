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


from typing import Optional

from google.adk.agents import Agent
from google.adk.agents import ParallelAgent
from google.adk.agents.base_agent import BeforeAgentCallback
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.genai import types


def before_agent_callback_check_relevance(
    agent_name: str,
) -> BeforeAgentCallback:
  """Callback to check if the state is relevant before executing the agent."""

  def callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """Check if the state is relevant."""
    if agent_name not in callback_context.state["execution_agents"]:
      return types.Content(
          parts=[
              types.Part(
                  text=(
                      f"Skipping execution agent {agent_name} as it is not"
                      " relevant to the current state."
                  )
              )
          ]
      )

  return callback


code_agent = Agent(
    model="gemini-2.5-flash",
    name="code_agent",
    instruction="""\
You are the Code Agent, responsible for generating code.

NOTE: You should only generate code and ignore other askings from the user.
""",
    before_agent_callback=before_agent_callback_check_relevance("code_agent"),
    output_key="code_agent_output",
)

math_agent = Agent(
    model="gemini-2.5-flash",
    name="math_agent",
    instruction="""\
You are the Math Agent, responsible for performing mathematical calculations.

NOTE: You should only perform mathematical calculations and ignore other askings from the user.
""",
    before_agent_callback=before_agent_callback_check_relevance("math_agent"),
    output_key="math_agent_output",
)


worker_parallel_agent = ParallelAgent(
    name="worker_parallel_agent",
    sub_agents=[
        code_agent,
        math_agent,
    ],
)


def instruction_provider_for_execution_summary_agent(
    readonly_context: ReadonlyContext,
) -> str:
  """Provides the instruction for the execution agent."""
  activated_agents = readonly_context.state["execution_agents"]
  prompt = f"""\
You are the Execution Summary Agent, responsible for summarizing the execution of the plan in the current invocation.

In this invocation, the following agents were involved: {', '.join(activated_agents)}.

Below are their outputs:
"""
  for agent_name in activated_agents:
    output = readonly_context.state.get(f"{agent_name}_output", "")
    prompt += f"\n\n{agent_name} output:\n{output}"

  prompt += (
      "\n\nPlease summarize the execution of the plan based on the above"
      " outputs."
  )
  return prompt.strip()


execution_summary_agent = Agent(
    model="gemini-2.5-flash",
    name="execution_summary_agent",
    instruction=instruction_provider_for_execution_summary_agent,
    include_contents="none",
)

plan_execution_agent = SequentialAgent(
    name="plan_execution_agent",
    sub_agents=[
        worker_parallel_agent,
        execution_summary_agent,
    ],
)
