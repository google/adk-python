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
"""Tests for edge cases of resuming invocations."""

import copy

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai.types import FunctionResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name="transfer_to_agent", args={"agent_name": agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name="transfer_to_agent", response={"result": None}
)


def test_tool():
  """A test tool; returns None to simulate a pending long-running operation."""
  return None


@pytest.mark.xfail(
    reason=(
        "Tests implementation details that are different in V2 and will be"
        " deprecated."
    )
)
@pytest.mark.asyncio
async def test_resume_invocation_from_sub_agent():
  """A test case for an edge case, where an invocation-to-resume starts from a sub-agent.

  For example:
    invocation1: root_agent -> sub_agent (sub_agent completes normally)
    invocation2: sub_agent calls long_running_tool -> pauses
    resume invocation2: sub_agent gets function response -> responds
  """
  # Step 1: Setup
  long_running_test_tool = LongRunningFunctionTool(func=test_tool)
  sub_agent = LlmAgent(
      name="sub_agent",
      model=testing_utils.MockModel.create(
          responses=[
              "first response from sub_agent",
              Part.from_function_call(name="test_tool", args={}),
              "response from sub_agent after resume",
          ]
      ),
      tools=[long_running_test_tool],
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[transfer_call_part(sub_agent.name)]
      ),
      sub_agents=[sub_agent],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation
  # root_agent transfers to sub_agent, sub_agent responds normally.
  invocation_1_events = await runner.run_async("test user query")
  inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv1_behavioral == [
      (
          root_agent.name,
          transfer_call_part(sub_agent.name),
      ),
      (
          root_agent.name,
          TRANSFER_RESPONSE_PART,
      ),
      (
          root_agent.name,
          testing_utils.END_OF_AGENT,
      ),
      (
          sub_agent.name,
          "first response from sub_agent",
      ),
      (
          sub_agent.name,
          testing_utils.END_OF_AGENT,
      ),
  ]

  # Step 3: Run the second invocation
  # sub_agent is now active. It calls long_running_tool, which pauses.
  invocation_2_events = await runner.run_async("test user query 2")
  inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv2_behavioral == [
      # execute_tools yields the interrupt event with long_running_tool_ids.
      (
          sub_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Find the function_call_id for resume.
  invocation_2_function_call_id = None
  for ev in invocation_2_events:
    if (
        ev.content
        and ev.content.parts
        and ev.content.parts[0].function_call
        and ev.content.parts[0].function_call.name == "test_tool"
    ):
      invocation_2_function_call_id = ev.content.parts[0].function_call.id
      break
  assert invocation_2_function_call_id is not None

  # Step 4: Resume the second invocation with function response.
  resumed_invocation_2_events = await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id,
      new_message=testing_utils.UserContent(
          Part(
              function_response=FunctionResponse(
                  id=invocation_2_function_call_id,
                  name="test_tool",
                  response={"result": "test tool update"},
              )
          ),
      ),
  )
  resumed_inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(resumed_invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert resumed_inv2_behavioral == [
      # execute_tools yields the function response from resume.
      (
          sub_agent.name,
          Part.from_function_response(
              name="test_tool",
              response={"result": "test tool update"},
          ),
      ),
      (
          sub_agent.name,
          "response from sub_agent after resume",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.skip(
    reason=(
        "Cross-invocation resume (resuming a non-latest invocation) is not"
        " supported by the Mesh-based LlmAgent. The Mesh's output aggregation"
        " in node_output_utils.py collects events from multiple invocations,"
        " causing CallLlmResult to be wrapped in a list."
    )
)
@pytest.mark.asyncio
async def test_resume_any_invocation():
  """A test case for resuming a previous invocation instead of the last one."""
  # Step 1: Setup
  long_running_test_tool = LongRunningFunctionTool(
      func=test_tool,
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[
              Part.from_function_call(name="test_tool", args={}),
              "llm response in invocation 2",
              Part.from_function_call(name="test_tool", args={}),
              "llm response after resuming invocation 1",
          ]
      ),
      tools=[long_running_test_tool],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Step 2: Run the first invocation, which pauses on the long running function.
  invocation_1_events = await runner.run_async("test user query")
  inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv1_behavioral == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Find the function_call_id for resume.
  invocation_1_function_call_id = None
  for ev in invocation_1_events:
    if (
        ev.content
        and ev.content.parts
        and ev.content.parts[0].function_call
        and ev.content.parts[0].function_call.name == "test_tool"
    ):
      invocation_1_function_call_id = ev.content.parts[0].function_call.id
      break
  assert invocation_1_function_call_id is not None

  # Step 3: Run the second invocation, expect it to finish normally.
  invocation_2_events = await runner.run_async(
      "test user query 2",
  )
  inv2_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_2_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv2_behavioral == [
      (
          root_agent.name,
          "llm response in invocation 2",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]

  # Step 4: Run the third invocation, which also pauses on the long running
  # function.
  invocation_3_events = await runner.run_async(
      "test user query 3",
  )
  inv3_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(invocation_3_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert inv3_behavioral == [
      (
          root_agent.name,
          Part.from_function_call(name="test_tool", args={}),
      ),
  ]

  # Step 5: Resume the first invocation with long running function response.
  resumed_invocation_1_events = await runner.run_async(
      invocation_id=invocation_1_events[0].invocation_id,
      new_message=testing_utils.UserContent(
          Part(
              function_response=FunctionResponse(
                  id=invocation_1_function_call_id,
                  name="test_tool",
                  response={"result": "test tool update"},
              )
          ),
      ),
  )
  resumed_inv1_behavioral = [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(resumed_invocation_1_events)
      )
      if not isinstance(e[1], dict)
  ]
  assert resumed_inv1_behavioral == [
      (
          root_agent.name,
          "llm response after resuming invocation 1",
      ),
      (root_agent.name, testing_utils.END_OF_AGENT),
  ]
