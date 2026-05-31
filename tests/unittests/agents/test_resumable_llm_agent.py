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

"""Tests for resumable LlmAgent scenarios.

Verifies that the Mesh-based LlmAgent correctly resumes from various
states: after transfers, tool calls, tool responses, and with
sub-agent tool calls.
"""

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

END_OF_AGENT = testing_utils.END_OF_AGENT


def some_tool():
  return {"result": "ok"}


def _behavioral_events(events):
  """Extract behavioral events (non-state) from resumable app events."""
  return [
      e
      for e in testing_utils.simplify_resumable_app_events(
          copy.deepcopy(events)
      )
      if not isinstance(e[1], dict)
  ]


@pytest.mark.asyncio
async def test_resume_from_transfer():
  """Tests that the agent resumes from the correct sub-agent after a transfer.

  invocation1: root_agent transfers to sub_agent_1
  invocation2: sub_agent_1 responds (resuming from the transfer)
  """
  sub_agent_1 = LlmAgent(
      name="sub_agent_1",
      model=testing_utils.MockModel.create(
          responses=[
              "response from sub_agent_1",
              "second response from sub_agent_1",
          ]
      ),
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[transfer_call_part("sub_agent_1")]
      ),
      sub_agents=[sub_agent_1],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Invocation 1: root transfers to sub_agent_1.
  inv1_events = await runner.run_async("test query")
  inv1_behavioral = _behavioral_events(inv1_events)
  assert inv1_behavioral == [
      ("root_agent", transfer_call_part("sub_agent_1")),
      ("root_agent", TRANSFER_RESPONSE_PART),
      ("root_agent", END_OF_AGENT),
      ("sub_agent_1", "response from sub_agent_1"),
      ("sub_agent_1", END_OF_AGENT),
  ]

  # Invocation 2: sub_agent_1 is now active, should respond directly.
  inv2_events = await runner.run_async("follow up query")
  inv2_behavioral = _behavioral_events(inv2_events)
  assert inv2_behavioral == [
      ("sub_agent_1", "second response from sub_agent_1"),
      ("sub_agent_1", END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resume_from_model_response():
  """Tests that the root agent resumes when there has been no transfer."""
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[
              "first response from root",
              "second response from root",
          ]
      ),
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Invocation 1: root responds normally.
  inv1_events = await runner.run_async("test query")
  inv1_behavioral = _behavioral_events(inv1_events)
  assert inv1_behavioral == [
      ("root_agent", "first response from root"),
      ("root_agent", END_OF_AGENT),
  ]

  # Invocation 2: root should respond again (no transfer happened).
  inv2_events = await runner.run_async("follow up")
  inv2_behavioral = _behavioral_events(inv2_events)
  assert inv2_behavioral == [
      ("root_agent", "second response from root"),
      ("root_agent", END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resume_from_tool_call():
  """Tests that the agent resumes from a tool call.

  invocation1: root_agent calls some_tool, gets response, then responds
  invocation2: root_agent responds again (tool was non-long-running)
  """
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[
              Part.from_function_call(name="some_tool", args={}),
              "response after tool call",
              "second response",
          ]
      ),
      tools=[some_tool],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Invocation 1: root calls tool, gets response, responds.
  inv1_events = await runner.run_async("test query")
  inv1_behavioral = _behavioral_events(inv1_events)
  assert inv1_behavioral == [
      ("root_agent", Part.from_function_call(name="some_tool", args={})),
      (
          "root_agent",
          Part.from_function_response(
              name="some_tool", response={"result": "ok"}
          ),
      ),
      ("root_agent", "response after tool call"),
      ("root_agent", END_OF_AGENT),
  ]

  # Invocation 2: root resumes normally.
  inv2_events = await runner.run_async("follow up")
  inv2_behavioral = _behavioral_events(inv2_events)
  assert inv2_behavioral == [
      ("root_agent", "second response"),
      ("root_agent", END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resume_subagent_after_transfer_and_tool_call():
  """Tests resuming a sub-agent that called a tool after being transferred to.

  invocation1: root_agent transfers to sub_agent_1, sub_agent_1 calls tool
               and responds
  invocation2: sub_agent_1 is still active, responds directly
  """

  def sub_agent_tool():
    return {"result": "ok"}

  sub_agent_1 = LlmAgent(
      name="sub_agent_1",
      model=testing_utils.MockModel.create(
          responses=[
              Part.from_function_call(name="sub_agent_tool", args={}),
              "response from sub_agent_1 after tool",
              "second response from sub_agent_1",
          ]
      ),
      tools=[sub_agent_tool],
  )
  root_agent = LlmAgent(
      name="root_agent",
      model=testing_utils.MockModel.create(
          responses=[transfer_call_part("sub_agent_1")]
      ),
      sub_agents=[sub_agent_1],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=root_agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      )
  )

  # Invocation 1: root transfers, sub_agent calls tool and responds.
  inv1_events = await runner.run_async("test query")
  inv1_behavioral = _behavioral_events(inv1_events)
  assert inv1_behavioral == [
      ("root_agent", transfer_call_part("sub_agent_1")),
      ("root_agent", TRANSFER_RESPONSE_PART),
      ("root_agent", END_OF_AGENT),
      (
          "sub_agent_1",
          Part.from_function_call(name="sub_agent_tool", args={}),
      ),
      (
          "sub_agent_1",
          Part.from_function_response(
              name="sub_agent_tool", response={"result": "ok"}
          ),
      ),
      ("sub_agent_1", "response from sub_agent_1 after tool"),
      ("sub_agent_1", END_OF_AGENT),
  ]

  # Invocation 2: sub_agent_1 is still active, responds directly.
  inv2_events = await runner.run_async("follow up")
  inv2_behavioral = _behavioral_events(inv2_events)
  assert inv2_behavioral == [
      ("sub_agent_1", "second response from sub_agent_1"),
      ("sub_agent_1", END_OF_AGENT),
  ]
