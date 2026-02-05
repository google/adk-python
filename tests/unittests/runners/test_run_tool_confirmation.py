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

"""Tests for HITL flows with different agent structures."""

import copy
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentResponse
from google.genai.types import Part
import pytest

from .. import testing_utils

HINT_TEXT = (
    "Please approve or reject the tool call _test_function() by"
    " responding with a FunctionResponse with an"
    " expected ToolConfirmation payload."
)

TOOL_CALL_ERROR_RESPONSE = {
    "error": "This tool call requires confirmation, please approve or reject."
}


def _create_llm_response_from_tools(
    tools: list[FunctionTool],
) -> GenerateContentResponse:
  """Creates a mock LLM response containing a function call."""
  parts = [
      Part(function_call=FunctionCall(name=tool.name, args={}))
      for tool in tools
  ]
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=parts)
  )


def _create_llm_response_from_text(text: str) -> GenerateContentResponse:
  """Creates a mock LLM response containing text."""
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=[Part(text=text)])
  )


def _test_function(
    tool_context: ToolContext,
) -> dict[str, str]:
  return {"result": f"confirmed={tool_context.tool_confirmation.confirmed}"}


def _test_request_confirmation_function_with_custom_schema(
    tool_context: ToolContext,
) -> dict[str, str]:
  """A test tool function that requests confirmation, but with a custom payload schema."""
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(
        hint="test hint for request_confirmation with custom payload schema",
        payload={
            "test_custom_payload": {
                "int_field": 0,
                "str_field": "",
                "bool_field": False,
            }
        },
    )
    return TOOL_CALL_ERROR_RESPONSE
  return {
      "result": f"confirmed={tool_context.tool_confirmation.confirmed}",
      "custom_payload": tool_context.tool_confirmation.payload,
  }


class BaseHITLTest:
  """Base class for HITL tests with common fixtures."""

  @pytest.fixture
  def runner(self, agent: BaseAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    return testing_utils.InMemoryRunner(root_agent=agent)


class TestHITLConfirmationFlowWithSingleAgent(BaseHITLTest):
  """Tests the HITL confirmation flow with a single LlmAgent."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)
    tools = agent.tools

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    ask_for_confirmation_function_call_id = (
        events[1].content.parts[0].function_call.id
    )
    invocation_id = events[1].invocation_id
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": tool_call_confirmed},
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response={"result": f"confirmed={tool_call_confirmed}"}
                    if tool_call_confirmed
                    else {"error": "This tool call is rejected."},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]
    for event in events:
      assert event.invocation_id != invocation_id
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithCustomPayloadSchema(BaseHITLTest):
  """Tests the HITL confirmation flow with a single agent, for custom confirmation payload schema."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [
        FunctionTool(
            func=_test_request_confirmation_function_with_custom_schema
        )
    ]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text(
            "test llm response after final tool call"
        ),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow with custom payload schema."""
    tools = agent.tools
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": (
                                "test hint for request_confirmation with"
                                " custom payload schema"
                            ),
                            "confirmed": False,
                            "payload": {
                                "test_custom_payload": {
                                    "int_field": 0,
                                    "str_field": "",
                                    "bool_field": False,
                                }
                            },
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    ask_for_confirmation_function_call_id = (
        events[1].content.parts[0].function_call.id
    )
    invocation_id = events[1].invocation_id
    custom_payload = {
        "test_custom_payload": {
            "int_field": 123,
            "str_field": "test_str",
            "bool_field": True,
        }
    }
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={
                    "confirmed": tool_call_confirmed,
                    "payload": custom_payload,
                },
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_response = {
        "result": f"confirmed={tool_call_confirmed}",
        "custom_payload": custom_payload,
    }
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response=expected_response,
                )
            ),
        ),
        (agent.name, "test llm response after final tool call"),
    ]
    for event in events:
      assert event.invocation_id != invocation_id
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithResumableApp:
  """Tests the HITL confirmation flow with a resumable app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.fixture
  def runner(self, agent: LlmAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = runner.run("test user query")

    # Verify that the invocation is paused when tool confirmation is requested.
    # The tool call returns error response, and summarization was skipped.
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    ) == [
        (
            agent.name,
            Part(function_call=FunctionCall(name=agent.tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": agent.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]
    ask_for_confirmation_function_call_id = (
        events[1].content.parts[0].function_call.id
    )
    invocation_id = events[1].invocation_id
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events = await runner.run_async(
        user_confirmation, invocation_id=invocation_id
    )
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
    for event in events:
      assert event.invocation_id == invocation_id
    assert (
        testing_utils.simplify_resumable_app_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithSequentialAgentAndResumableApp:
  """Tests the HITL confirmation flow with a resumable sequential agent app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text("test llm response from second agent"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> SequentialAgent:
    """Provides a single LlmAgent for the test."""
    return SequentialAgent(
        name="root_agent",
        sub_agents=[
            LlmAgent(name="agent1", model=mock_model, tools=tools),
            LlmAgent(name="agent2", model=mock_model, tools=[]),
        ],
    )

  @pytest.fixture
  def runner(self, agent: SequentialAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: SequentialAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""

    # Test setup:
    # - root_agent is a SequentialAgent with two sub-agents: sub_agent1 and
    #   sub_agent2.
    #   - sub_agent1 has a tool call that asks for HITL confirmation.
    #   - sub_agent2 does not have any tool calls.
    # - The test will:
    #   - Run the query and verify that the invocation is paused when tool
    #     confirmation is requested, at sub_agent1.
    #   - Resume the invocation and execute the tool call from sub_agent1.
    #   - Verify that root_agent continues to run sub_agent2.

    events = runner.run("test user query")
    sub_agent1 = agent.sub_agents[0]
    sub_agent2 = agent.sub_agents[1]

    # Step 1:
    # Verify that the invocation is paused when tool confirmation is requested.
    # So that no intermediate llm response is generated.
    # And the second sub agent is not started.
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    ) == [
        (
            agent.name,
            SequentialAgentState(current_sub_agent=sub_agent1.name).model_dump(
                mode="json"
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=sub_agent1.tools[0].name, args={}
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": sub_agent1.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response=TOOL_CALL_ERROR_RESPONSE,
                )
            ),
        ),
    ]
    ask_for_confirmation_function_call_id = (
        events[2].content.parts[0].function_call.id
    )
    invocation_id = events[2].invocation_id

    # Step 2:
    # Resume the invocation and confirm the tool call from sub_agent1, and
    # sub_agent2 will continue.
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events = await runner.run_async(
        user_confirmation, invocation_id=invocation_id
    )
    expected_parts_final = [
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (sub_agent1.name, "test llm response after tool call"),
        (sub_agent1.name, testing_utils.END_OF_AGENT),
        (
            agent.name,
            SequentialAgentState(current_sub_agent=sub_agent2.name).model_dump(
                mode="json"
            ),
        ),
        (sub_agent2.name, "test llm response from second agent"),
        (sub_agent2.name, testing_utils.END_OF_AGENT),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
    for event in events:
      assert event.invocation_id == invocation_id
    assert (
        testing_utils.simplify_resumable_app_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithParallelAgentAndResumableApp:
  """Tests the HITL confirmation flow with a resumable sequential agent app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def agent(
      self,
      tools: list[FunctionTool],
      llm_responses: list[GenerateContentResponse],
  ) -> ParallelAgent:
    """Provides a single ParallelAgent for the test."""
    return ParallelAgent(
        name="root_agent",
        sub_agents=[
            LlmAgent(
                name="agent1",
                model=testing_utils.MockModel(responses=llm_responses),
                tools=tools,
            ),
            LlmAgent(
                name="agent2",
                model=testing_utils.MockModel(responses=llm_responses),
                tools=tools,
            ),
        ],
    )

  @pytest.fixture
  def runner(self, agent: ParallelAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: ParallelAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = runner.run("test user query")

    # Test setup:
    # - root_agent is a ParallelAgent with two sub-agents: sub_agent1 and
    #   sub_agent2.
    # - Both sub_agents have a tool call that asks for HITL confirmation.
    # - The test will:
    #   - Run the query and verify that each branch is paused when tool
    #     confirmation is requested.
    #   - Resume the invocation and execute the tool call of each branch.

    sub_agent1 = agent.sub_agents[0]
    sub_agent2 = agent.sub_agents[1]

    # Verify that each branch is paused after the long running tool call.
    # So that no intermediate llm response is generated.
    root_agent_events = [event for event in events if event.branch is None]
    sub_agent1_branch_events = [
        event
        for event in events
        if event.branch == f"{agent.name}.{sub_agent1.name}"
    ]
    sub_agent2_branch_events = [
        event
        for event in events
        if event.branch == f"{agent.name}.{sub_agent2.name}"
    ]
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(root_agent_events)
    ) == [
        (
            agent.name,
            BaseAgentState().model_dump(mode="json"),
        ),
    ]
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(sub_agent1_branch_events)
    ) == [
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=sub_agent1.tools[0].name, args={}
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": sub_agent1.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response=TOOL_CALL_ERROR_RESPONSE,
                )
            ),
        ),
    ]
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(sub_agent2_branch_events)
    ) == [
        (
            sub_agent2.name,
            Part(
                function_call=FunctionCall(
                    name=sub_agent2.tools[0].name, args={}
                )
            ),
        ),
        (
            sub_agent2.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": sub_agent2.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            sub_agent2.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent2.tools[0].name,
                    response=TOOL_CALL_ERROR_RESPONSE,
                )
            ),
        ),
    ]

    ask_for_confirmation_function_call_ids = [
        sub_agent1_branch_events[1].content.parts[0].function_call.id,
        sub_agent2_branch_events[1].content.parts[0].function_call.id,
    ]
    assert (
        sub_agent1_branch_events[1].invocation_id
        == sub_agent2_branch_events[1].invocation_id
    )
    invocation_id = sub_agent1_branch_events[1].invocation_id

    # Resume the invocation and confirm the tool call from sub_agent1.
    user_confirmations = [
        testing_utils.UserContent(
            Part(
                function_response=FunctionResponse(
                    id=id,
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    response={"confirmed": True},
                )
            )
        )
        for id in ask_for_confirmation_function_call_ids
    ]

    events = await runner.run_async(
        user_confirmations[0], invocation_id=invocation_id
    )
    for event in events:
      assert event.invocation_id == invocation_id

    root_agent_events = [event for event in events if event.branch is None]
    sub_agent1_branch_events = [
        event
        for event in events
        if event.branch == f"{agent.name}.{sub_agent1.name}"
    ]
    sub_agent2_branch_events = [
        event
        for event in events
        if event.branch == f"{agent.name}.{sub_agent2.name}"
    ]

    # Verify that sub_agent1 is resumed and final; sub_agent2 is still paused;
    # root_agent is not final.
    assert not root_agent_events
    assert not sub_agent2_branch_events
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(sub_agent1_branch_events)
    ) == [
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (sub_agent1.name, "test llm response after tool call"),
        (sub_agent1.name, testing_utils.END_OF_AGENT),
    ]

    # Resume the invocation again and confirm the tool call from sub_agent2.
    events = await runner.run_async(
        user_confirmations[1], invocation_id=invocation_id
    )
    for event in events:
      assert event.invocation_id == invocation_id

    # Verify that sub_agent2 is resumed and final; root_agent is final.
    assert testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    ) == [
        (
            sub_agent2.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent2.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (sub_agent2.name, "test llm response after tool call"),
        (sub_agent2.name, testing_utils.END_OF_AGENT),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
