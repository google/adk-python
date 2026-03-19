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

from typing import Union

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content
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


def tool_call_part(tool_name: str) -> Part:
  part = Part.from_function_call(name=tool_name, args={})
  part.function_call.id = f"{tool_name}_id"
  return part


def tool_response_part(tool_name: str) -> Part:
  part = Part.from_function_response(name=tool_name, response={"result": "ok"})
  part.function_response.id = f"{tool_name}_id"
  return part


def tool_response_part_no_id(tool_name: str) -> Part:
  part = Part.from_function_response(name=tool_name, response={"result": "ok"})
  return part


END_OF_AGENT = testing_utils.END_OF_AGENT


def some_tool():
  return {"result": "ok"}


async def _create_resumable_invocation_context(
    invocation_id: str, agent: BaseAgent, events: list[Event]
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  for event in events:
    await session_service.append_event(session, event)
  return InvocationContext(
      invocation_id=invocation_id,
      agent=agent,
      session=session,
      session_service=session_service,
      resumability_config=ResumabilityConfig(is_resumable=True),
      run_config=RunConfig(),
  )


async def _resume_and_get_events(
    agent: BaseAgent, invocation_context: InvocationContext
) -> list[(str, Union[Part, str])]:
  events = []
  async for event in agent.run_async(invocation_context):
    await invocation_context.session_service.append_event(
        invocation_context.session, event
    )
    events.append(event)
  return testing_utils.simplify_resumable_app_events(events)


class TestResumableLlmAgent:
  """Test suite for resumable LlmAgent."""

  @pytest.fixture
  async def resumable_invocation_context(self):
    """Creates an invocation context for the specified agent."""

    async def factory(agent: BaseAgent, events: list[Event]):
      return await _create_resumable_invocation_context(
          invocation_id="test_invocation", agent=agent, events=events
      )

    return factory

  @pytest.fixture
  def mock_model(self):
    """Provides a mock model for the test."""

    def factory(responses: list[Part]):
      return testing_utils.MockModel.create(responses=responses)

    return factory

  @pytest.mark.asyncio
  async def test_resume_from_transfer_call(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent resumes from the correct sub-agent after a transfer."""
    sub_agent_1 = LlmAgent(
        name="sub_agent_1",
        model=mock_model([
            "response from sub_agent_1",
        ]),
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model(["response from root"]),
        sub_agents=[sub_agent_1],
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(
                parts=[
                    transfer_call_part("sub_agent_1"),
                ]
            ),
        )
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("root_agent", TRANSFER_RESPONSE_PART),
        ("sub_agent_1", "response from sub_agent_1"),
        ("sub_agent_1", END_OF_AGENT),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_from_transfer_response(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent resumes from the correct sub-agent after a transfer."""
    sub_agent_1 = LlmAgent(
        name="sub_agent_1",
        model=mock_model([
            "response from sub_agent_1",
        ]),
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model(["response from root"]),
        sub_agents=[sub_agent_1],
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(
                parts=[
                    TRANSFER_RESPONSE_PART,
                ]
            ),
            actions=EventActions(transfer_to_agent="sub_agent_1"),
        )
    ]
    ctx: InvocationContext = await resumable_invocation_context(
        root_agent, past_events
    )
    # Initialize the agent state for the root agent.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("sub_agent_1", "response from sub_agent_1"),
        ("sub_agent_1", END_OF_AGENT),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_from_model_response(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that no sub-agent is resumed when there has been no transfer."""
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model([
            "second response from root",
        ]),
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[Part(text="initial response from root")]),
        )
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("root_agent", "second response from root"),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_from_tool_call(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent resumes from a tool call successfully."""
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model(["response after tool call"]),
        tools=[some_tool],
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[tool_call_part("some_tool")]),
        ),
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("root_agent", tool_response_part_no_id("some_tool")),
        ("root_agent", "response after tool call"),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_after_tool_response(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent does not resume a sub-agent when the user responds to the current agent."""
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model([
            "response after tool call",
        ]),
        tools=[some_tool],
    )

    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[tool_call_part("some_tool")]),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[tool_response_part("some_tool")]),
        ),
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("root_agent", "response after tool call"),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_root_agent_on_user_provided_function_response(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent resumes the correct sub-agent after a user responds to its tool call."""

    def sub_agent_tool():
      return {"result": "ok"}

    sub_agent_1 = LlmAgent(
        name="sub_agent_1",
        model=mock_model([
            "response from sub_agent_1 after tool call",
        ]),
        tools=[sub_agent_tool],
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model(["response from root after tool call"]),
        sub_agents=[sub_agent_1],
        tools=[some_tool],
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            actions=EventActions(transfer_to_agent="sub_agent_1"),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[transfer_call_part("sub_agent_1")]),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[TRANSFER_RESPONSE_PART]),
            actions=EventActions(transfer_to_agent="sub_agent_1"),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[tool_call_part("some_tool")]),
        ),
        Event(
            author="sub_agent_1",
            invocation_id="test_invocation",
            content=Content(parts=[tool_call_part("sub_agent_tool")]),
        ),
        Event(
            author="user",
            invocation_id="test_invocation",
            content=Content(parts=[tool_response_part("some_tool")]),
        ),
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent and sub_agent_1.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")
    ctx.agent_states[sub_agent_1.name] = BaseAgentState().model_dump(
        mode="json"
    )

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("root_agent", "response from root after tool call"),
        ("root_agent", END_OF_AGENT),
    ]

  @pytest.mark.asyncio
  async def test_resume_subagent_on_user_provided_function_response(
      self, resumable_invocation_context, mock_model
  ):
    """Tests that the agent resumes the correct sub-agent after a user responds to its tool call."""

    def sub_agent_tool():
      return {"result": "ok"}

    sub_agent_1 = LlmAgent(
        name="sub_agent_1",
        model=mock_model([
            "response from sub_agent_1 after tool call",
        ]),
        tools=[sub_agent_tool],
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=mock_model(["response from root after tool call"]),
        sub_agents=[sub_agent_1],
    )
    past_events = [
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            actions=EventActions(transfer_to_agent="sub_agent_1"),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[transfer_call_part("sub_agent_1")]),
        ),
        Event(
            author="root_agent",
            invocation_id="test_invocation",
            content=Content(parts=[TRANSFER_RESPONSE_PART]),
            actions=EventActions(transfer_to_agent="sub_agent_1"),
        ),
        Event(
            author="sub_agent_1",
            invocation_id="test_invocation",
            content=Content(parts=[tool_call_part("sub_agent_tool")]),
        ),
        Event(
            author="user",
            invocation_id="test_invocation",
            content=Content(parts=[tool_response_part("sub_agent_tool")]),
        ),
    ]
    ctx = await resumable_invocation_context(root_agent, past_events)
    # Initialize the agent state for the root agent and sub_agent_1.
    ctx.agent_states[root_agent.name] = BaseAgentState().model_dump(mode="json")
    ctx.agent_states[sub_agent_1.name] = BaseAgentState().model_dump(
        mode="json"
    )

    assert await _resume_and_get_events(root_agent, ctx) == [
        ("sub_agent_1", "response from sub_agent_1 after tool call"),
        ("sub_agent_1", END_OF_AGENT),
        ("root_agent", END_OF_AGENT),
    ]
