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


from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext

from . import execution_agent


def update_execution_plan(
    execution_agents: list[str], tool_context: ToolContext
) -> str:
  """Updates the execution plan for the agents to run."""

  tool_context.state["execution_agents"] = execution_agents
  return "execution_agents updated."


root_agent = Agent(
    model="gemini-2.5-flash",
    name="execution_manager_agent",
    instruction="""\
You are the Execution Manager Agent, responsible for setting up execution plan and delegate to plan_execution_agent for the actual plan execution.

You ONLY have the following worker agents: `code_agent`, `math_agent`.

You should do the following:

1. Analyze the user input and decide any worker agents that are relevant;
2. If none of the worker agents are relevant, you should explain to user that no relevant agents are available and ask for something else;
2. Update the execution plan with the relevant worker agents using `update_execution_plan` tool.
3. Transfer control to the plan_execution_agent for the actual plan execution.

When calling the `update_execution_plan` tool, you should pass the list of worker agents that are relevant to user's input.

NOTE:

* If you are not clear about user's intent, you should ask for clarification first;
* Only after you're clear about user's intent, you can proceed to step #2.
""",
    sub_agents=[
        execution_agent.plan_execution_agent,
    ],
    tools=[update_execution_plan],
)
