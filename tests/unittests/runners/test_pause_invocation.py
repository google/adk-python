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

"""Tests for the resumption flow with different agent structures."""

import asyncio
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.loop_agent import LoopAgentState
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai.types import Part
import pytest

from .. import testing_utils


def _transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name="transfer_to_agent", args={"agent_name": agent_name}
  )


def test_tool() -> str:
  return "result"


class _TestingAgent(BaseAgent):
  """A testing agent that generates an event after a delay."""

  delay: float = 0
  """The delay before the agent generates an event."""

  def event(self, ctx: InvocationContext):
    return Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=testing_utils.ModelContent(
            parts=[Part.from_text(text="Delayed message")]
        ),
    )

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    await asyncio.sleep(self.delay)
    yield self.event(ctx)


_TRANSFER_RESPONSE_PART = Part.from_function_response(
    name="transfer_to_agent", response={"result": None}
)
END_OF_AGENT = testing_utils.END_OF_AGENT


class BasePauseInvocationTest:
  """Base class for pausing invocation tests with common fixtures."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    return BaseAgent(name="test_agent")

  @pytest.fixture
  def app(self, agent: BaseAgent) -> App:
    """Provides an App for the test."""
    return App(
        name="test_app",
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

  @pytest.fixture
  def runner(self, app: App) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    return testing_utils.InMemoryRunner(app=app)

  @staticmethod
  def mock_model(responses: list[Part]) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel.create(responses=responses)


class TestPauseInvocationWithSingleLlmAgent(BasePauseInvocationTest):
  """Tests the resumption flow with a single LlmAgent."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    return LlmAgent(
        name="root_agent",
        model=self.mock_model(
            responses=[Part.from_function_call(name="test_tool", args={})]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a single LlmAgent pauses on long running function call."""
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        ("root_agent", Part.from_function_call(name="test_tool", args={})),
        (
            "root_agent",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]


class TestPauseInvocationWithSequentialAgent(BasePauseInvocationTest):
  """Tests pausing invocation with a SequentialAgent."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    sub_agent1 = LlmAgent(
        name="sub_agent_1",
        model=self.mock_model(
            responses=[Part.from_function_call(name="test_tool", args={})]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    sub_agent2 = LlmAgent(
        name="sub_agent_2",
        model=self.mock_model(
            responses=[Part.from_function_call(name="test_tool", args={})]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    return SequentialAgent(
        name="root_agent",
        sub_agents=[sub_agent1, sub_agent2],
    )

  @pytest.mark.asyncio
  def test_pause_first_agent_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a SequentialAgent pauses on the first sub-agent."""
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        (
            "root_agent",
            SequentialAgentState(current_sub_agent="sub_agent_1").model_dump(
                mode="json"
            ),
        ),
        ("sub_agent_1", Part.from_function_call(name="test_tool", args={})),
        (
            "sub_agent_1",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]

  @pytest.mark.asyncio
  def test_pause_second_agent_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a single LlmAgent pauses on long running function call."""
    # Change the base sequential agent, so that the first agent does not pause.
    runner.root_agent.sub_agents[0].tools = [FunctionTool(func=test_tool)]
    runner.root_agent.sub_agents[0].model = self.mock_model(
        responses=[
            Part.from_function_call(name="test_tool", args={}),
            Part.from_text(text="model response after tool call"),
        ]
    )
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        (
            "root_agent",
            SequentialAgentState(current_sub_agent="sub_agent_1").model_dump(
                mode="json"
            ),
        ),
        ("sub_agent_1", Part.from_function_call(name="test_tool", args={})),
        (
            "sub_agent_1",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
        ("sub_agent_1", "model response after tool call"),
        ("sub_agent_1", END_OF_AGENT),
        (
            "root_agent",
            SequentialAgentState(current_sub_agent="sub_agent_2").model_dump(
                mode="json"
            ),
        ),
        ("sub_agent_2", Part.from_function_call(name="test_tool", args={})),
        (
            "sub_agent_2",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]


class TestPauseInvocationWithParallelAgent(BasePauseInvocationTest):
  """Tests pausing invocation with a ParallelAgent."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    sub_agent1 = LlmAgent(
        name="sub_agent_1",
        model=self.mock_model(
            responses=[Part.from_function_call(name="test_tool", args={})]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    sub_agent2 = _TestingAgent(
        name="sub_agent_2",
        delay=0.5,
    )
    return ParallelAgent(
        name="root_agent",
        sub_agents=[sub_agent1, sub_agent2],
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a ParallelAgent pauses on long running function call."""
    simplified_event_parts = testing_utils.simplify_resumable_app_events(
        runner.run("test")
    )
    assert (
        "sub_agent_1",
        Part.from_function_call(name="test_tool", args={}),
    ) in simplified_event_parts
    assert ("sub_agent_2", "Delayed message") in simplified_event_parts


class TestPauseInvocationWithNestedParallelAgent(BasePauseInvocationTest):
  """Tests pausing invocation with a nested ParallelAgent."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    nested_sub_agent_1 = LlmAgent(
        name="nested_sub_agent_1",
        model=self.mock_model(
            responses=[Part.from_function_call(name="test_tool", args={})]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    nested_sub_agent_2 = _TestingAgent(
        name="nested_sub_agent_2",
        delay=0.5,
    )
    nested_parallel_agent = ParallelAgent(
        name="nested_parallel_agent",
        sub_agents=[nested_sub_agent_1, nested_sub_agent_2],
    )
    sub_agent_1 = _TestingAgent(
        name="sub_agent_1",
        delay=0.5,
    )
    return ParallelAgent(
        name="root_agent",
        sub_agents=[sub_agent_1, nested_parallel_agent],
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a nested ParallelAgent pauses on long running function call."""
    simplified_event_parts = testing_utils.simplify_resumable_app_events(
        runner.run("test")
    )
    assert (
        "nested_sub_agent_1",
        Part.from_function_call(name="test_tool", args={}),
    ) in simplified_event_parts
    assert ("sub_agent_1", "Delayed message") in simplified_event_parts
    assert ("nested_sub_agent_2", "Delayed message") in simplified_event_parts

  @pytest.mark.asyncio
  def test_pause_on_multiple_long_running_function_calls(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a ParallelAgent pauses on long running function calls."""
    runner.root_agent.sub_agents[0] = LlmAgent(
        name="sub_agent_1",
        model=self.mock_model(
            responses=[
                Part.from_function_call(name="test_tool", args={}),
            ]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    simplified_events = testing_utils.simplify_resumable_app_events(
        runner.run("test")
    )
    assert (
        "sub_agent_1",
        Part.from_function_call(name="test_tool", args={}),
    ) in simplified_events
    assert ("sub_agent_1", END_OF_AGENT) not in simplified_events
    assert (
        "nested_sub_agent_1",
        Part.from_function_call(name="test_tool", args={}),
    ) in simplified_events
    assert ("nested_sub_agent_1", END_OF_AGENT) not in simplified_events


class TestPauseInvocationWithLoopAgent(BasePauseInvocationTest):
  """Tests pausing invocation with a LoopAgent."""

  @pytest.fixture
  def agent(self) -> BaseAgent:
    """Provides a BaseAgent for the test."""
    sub_agent_1 = LlmAgent(
        name="sub_agent_1",
        model=self.mock_model(
            responses=[
                Part.from_text(text="sub agent 1 response"),
            ]
        ),
    )
    sub_agent_2 = LlmAgent(
        name="sub_agent_2",
        model=self.mock_model(
            responses=[
                Part.from_function_call(name="test_tool", args={}),
            ]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    sub_agent_3 = LlmAgent(
        name="sub_agent_3",
        model=self.mock_model(
            responses=[
                Part.from_function_call(name="exit_loop", args={}),
            ]
        ),
        tools=[exit_loop],
    )
    return LoopAgent(
        name="root_agent",
        sub_agents=[sub_agent_1, sub_agent_2, sub_agent_3],
        max_iterations=2,
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a LoopAgent pauses on long running function call."""
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        (
            "root_agent",
            LoopAgentState(current_sub_agent="sub_agent_1").model_dump(
                mode="json"
            ),
        ),
        ("sub_agent_1", "sub agent 1 response"),
        ("sub_agent_1", END_OF_AGENT),
        (
            "root_agent",
            LoopAgentState(current_sub_agent="sub_agent_2").model_dump(
                mode="json"
            ),
        ),
        ("sub_agent_2", Part.from_function_call(name="test_tool", args={})),
        (
            "sub_agent_2",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]


class TestPauseInvocationWithLlmAgentTree(BasePauseInvocationTest):
  """Tests the pausing invocation with a tree of LlmAgents."""

  @pytest.fixture
  def agent(self) -> LlmAgent:
    """Provides an LlmAgent with sub-agents for the test."""
    sub_llm_agent_1 = LlmAgent(
        name="sub_llm_agent_1",
        model=self.mock_model(
            responses=[
                _transfer_call_part("sub_llm_agent_2"),
                "llm response not used",
            ]
        ),
    )
    sub_llm_agent_2 = LlmAgent(
        name="sub_llm_agent_2",
        model=self.mock_model(
            responses=[
                Part.from_function_call(name="test_tool", args={}),
                "llm response not used",
            ]
        ),
        tools=[LongRunningFunctionTool(func=test_tool)],
    )
    return LlmAgent(
        name="root_agent",
        model=self.mock_model(
            responses=[
                _transfer_call_part("sub_llm_agent_1"),
                "llm response not used",
            ]
        ),
        sub_agents=[sub_llm_agent_1, sub_llm_agent_2],
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a tree of resumable LlmAgents yields checkpoint events."""
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        ("root_agent", _transfer_call_part("sub_llm_agent_1")),
        ("root_agent", _TRANSFER_RESPONSE_PART),
        ("sub_llm_agent_1", _transfer_call_part("sub_llm_agent_2")),
        ("sub_llm_agent_1", _TRANSFER_RESPONSE_PART),
        ("sub_llm_agent_2", Part.from_function_call(name="test_tool", args={})),
        (
            "sub_llm_agent_2",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]


class TestPauseInvocationWithWithTransferLoop(BasePauseInvocationTest):
  """Tests pausing the invocation when the agent transfer forms a loop."""

  @pytest.fixture
  def agent(self) -> LlmAgent:
    """Provides an LlmAgent with sub-agents for the test."""
    sub_llm_agent_1 = LlmAgent(
        name="sub_llm_agent_1",
        model=self.mock_model(
            responses=[
                _transfer_call_part("sub_llm_agent_2"),
                "llm response not used",
            ]
        ),
    )
    sub_llm_agent_2 = LlmAgent(
        name="sub_llm_agent_2",
        model=self.mock_model(
            responses=[
                _transfer_call_part("root_agent"),
                "llm response not used",
            ]
        ),
    )
    return LlmAgent(
        name="root_agent",
        model=self.mock_model(
            responses=[
                _transfer_call_part("sub_llm_agent_1"),
                Part.from_function_call(name="test_tool", args={}),
                "llm response not used",
            ]
        ),
        sub_agents=[sub_llm_agent_1, sub_llm_agent_2],
        tools=[LongRunningFunctionTool(func=test_tool)],
    )

  @pytest.mark.asyncio
  def test_pause_on_long_running_function_call(
      self,
      runner: testing_utils.InMemoryRunner,
  ):
    """Tests that a tree of resumable LlmAgents yields checkpoint events."""
    assert testing_utils.simplify_resumable_app_events(runner.run("test")) == [
        ("root_agent", _transfer_call_part("sub_llm_agent_1")),
        ("root_agent", _TRANSFER_RESPONSE_PART),
        ("sub_llm_agent_1", _transfer_call_part("sub_llm_agent_2")),
        ("sub_llm_agent_1", _TRANSFER_RESPONSE_PART),
        ("sub_llm_agent_2", _transfer_call_part("root_agent")),
        ("sub_llm_agent_2", _TRANSFER_RESPONSE_PART),
        ("root_agent", Part.from_function_call(name="test_tool", args={})),
        (
            "root_agent",
            Part.from_function_response(
                name="test_tool", response={"result": "result"}
            ),
        ),
    ]
