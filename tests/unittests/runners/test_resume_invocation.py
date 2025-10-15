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
"""Tests for edge cases of resuming invocations."""

import copy

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
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


@pytest.mark.asyncio
async def test_resume_invocation_from_sub_agent():
  """A test case for an edge case, where an invocation-to-resume starts from a sub-agent.

  For example:
    invocation1: root_agent -> sub_agent
    invocation2: sub_agent [paused][resume]
  """
  # Step 1: Setup
  # root_agent -> sub_agent
  sub_agent = LlmAgent(
      name="sub_agent",
      model=testing_utils.MockModel.create(
          responses=[
              "first response from sub_agent",
              "second response from sub_agent",
              "third response from sub_agent",
          ]
      ),
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
  # Expect the invocation to start from root_agent and transferred to sub_agent.
  invocation_1_events = runner.run("test user query")
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_1_events)
  ) == [
      (
          root_agent.name,
          transfer_call_part(sub_agent.name),
      ),
      (
          root_agent.name,
          TRANSFER_RESPONSE_PART,
      ),
      (
          sub_agent.name,
          "first response from sub_agent",
      ),
      (
          sub_agent.name,
          testing_utils.END_OF_AGENT,
      ),
      (
          root_agent.name,
          testing_utils.END_OF_AGENT,
      ),
  ]

  # Step 3: Run the second invocation
  # Expect the invocation to directly start from sub_agent.
  invocation_2_events = runner.run(
      "test user query 2",
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(invocation_2_events)
  ) == [
      (
          sub_agent.name,
          "second response from sub_agent",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]
  # Asserts the invocation will be a no-op if the current agent in context is
  # already final.
  assert not await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id
  )

  # Step 4: Copy all session.events[:-1] to a new session
  # This is to simulate the case where we pause on the second invocation.
  session_id = runner.session_id
  session = await runner.runner.session_service.get_session(
      app_name="test_app", user_id="test_user", session_id=session_id
  )
  new_session = await runner.runner.session_service.create_session(
      app_name=session.app_name, user_id=session.user_id
  )
  for event in session.events[:-1]:
    await runner.runner.session_service.append_event(new_session, event)
  runner.session_id = new_session.id

  # Step 5: Resume the second invocation
  resumed_invocation_2_events = await runner.run_async(
      invocation_id=invocation_2_events[0].invocation_id
  )
  assert testing_utils.simplify_resumable_app_events(
      copy.deepcopy(resumed_invocation_2_events)
  ) == [
      (
          sub_agent.name,
          "third response from sub_agent",
      ),
      (sub_agent.name, testing_utils.END_OF_AGENT),
  ]
