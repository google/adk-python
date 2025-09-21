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

import pytest

# Skip until fixed.
pytest.skip(allow_module_level=True)

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.genai import types

from .fixture import context_variable_agent
from .utils import TestRunner

nl_planner_si = """
You are an intelligent tool use agent built upon the Gemini large language model. When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under /*PLANNING*/. (2) The tool code snippets should be under /*ACTION*/, and the reasoning parts should be under /*REASONING*/. (3) The final answer part should be under /*FINAL_ANSWER*/.


Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be be under /*REPLANNING*/. Then use tools to follow the new plan.

Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.



Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.



Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- Use "print" to output execution results for the next step or final answer that you need for responding to the user. Never generate ```tool_outputs yourself.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.



VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.

You should ONLY generate code snippets prefixed with "```tool_code" if you need to use the tools to answer the question.

If you are asked to write code by user specifically,
- you should ALWAYS use "```python" to format the code.
- you should NEVER put "tool_code" to format the code.
- Good example:
```python
print('hello')
```
- Bad example:
```tool_code
print('hello')
```
"""


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": context_variable_agent.agent.state_variable_echo_agent}],
    indirect=True,
)
def test_context_variable(agent_runner: TestRunner):
  session = Session(
      context={
          "customerId": "1234567890",
          "customerInt": 30,
          "customerFloat": 12.34,
          "customerJson": {"name": "John Doe", "age": 30, "count": 11.1},
      }
  )
  si = UnitFlow()._build_system_instruction(
      InvocationContext(
          invocation_id="1234567890", agent=agent_runner.agent, session=session
      )
  )

  assert (
      "Use the echo_info tool to echo 1234567890, 30, 12.34, and {'name': 'John"
      " Doe', 'age': 30, 'count': 11.1}. Ask for it if you need to."
      in si
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{
        "agent": (
            context_variable_agent.agent.state_variable_with_complicated_format_agent
        )
    }],
    indirect=True,
)
def test_context_variable_with_complicated_format(agent_runner: TestRunner):
  session = Session(
      context={"customerId": "1234567890", "customer_int": 30},
      artifacts={"fileName": [types.Part(text="test artifact")]},
  )
  si = _context_formatter.populate_context_and_artifact_variable_values(
      agent_runner.agent.instruction,
      session.get_state(),
      session.get_artifact_dict(),
  )

  assert (
      si
      == "Use the echo_info tool to echo 1234567890, 30, { "
      " non-identifier-float}}, test artifact, {'key1': 'value1'} and"
      " {{'key2': 'value2'}}. Ask for it if you need to."
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{
        "agent": (
            context_variable_agent.agent.state_variable_with_nl_planner_agent
        )
    }],
    indirect=True,
)
def test_nl_planner(agent_runner: TestRunner):
  session = Session(context={"customerId": "1234567890"})
  si = UnitFlow()._build_system_instruction(
      InvocationContext(
          invocation_id="1234567890",
          agent=agent_runner.agent,
          session=session,
      )
  )

  for line in nl_planner_si.splitlines():
    assert line in si


@pytest.mark.parametrize(
    "agent_runner",
    [{
        "agent": (
            context_variable_agent.agent.state_variable_with_function_instruction_agent
        )
    }],
    indirect=True,
)
def test_function_instruction(agent_runner: TestRunner):
  session = Session(context={"customerId": "1234567890"})
  si = UnitFlow()._build_system_instruction(
      InvocationContext(
          invocation_id="1234567890", agent=agent_runner.agent, session=session
      )
  )

  assert "This is the plain text sub agent instruction." in si
