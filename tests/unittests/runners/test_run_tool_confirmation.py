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

import asyncio
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
from google.adk.events.event import Event
from google.adk.events.ui_widget import UiWidget
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.context_utils import Aclosing
from google.genai import types
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
      assert event.invocation_id == invocation_id
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )

  @pytest.mark.asyncio
  @pytest.mark.parametrize("concurrent", [False, True])
  async def test_repeated_confirmation_is_executed_once(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      concurrent: bool,
  ):
    """The public Runner must consume one confirmation across invocations."""
    executions = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def counted_tool(tool_context):
      nonlocal executions
      executions += 1
      started.set()
      if concurrent:
        await release.wait()
      return {"executions": executions}

    agent.tools[0].func = counted_tool
    # Duplicate submissions still traverse the runner's normal model loop
    # before the confirmation claim filters the tool call.
    agent.model.responses.extend(
        [_create_llm_response_from_text("done") for _ in range(2)]
    )
    initial_events = await runner.run_async(testing_utils.UserContent("test"))
    confirmation_id = initial_events[1].content.parts[0].function_call.id
    original_function_call_id = (
        initial_events[0].content.parts[0].function_call.id
    )
    confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=confirmation_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )

    if concurrent:
      first = asyncio.create_task(runner.run_async(confirmation))
      await started.wait()
      second = asyncio.create_task(runner.run_async(confirmation))
      await asyncio.sleep(0)
      release.set()
      results = await asyncio.gather(first, second)
      resumed_events = [event for result in results for event in result]
    else:
      first_events = await runner.run_async(confirmation)
      duplicate_events = await runner.run_async(confirmation)
      resumed_events = first_events + duplicate_events

    assert executions == 1
    assert (
        sum(
            response.id == original_function_call_id
            for event in resumed_events
            for response in event.get_function_responses()
        )
        == 1
    )
    assert not any(
        call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
        for event in resumed_events
        for call in event.get_function_calls()
    )

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      ("second_namespace", "second_call_id"),
      [
          (("test_app", "test_user", "session_b"), "shared_call"),
          (("test_app", "other_user", "session_a"), "shared_call"),
          (("other_app", "test_user", "session_a"), "shared_call"),
          (("test_app", "other_user", "session_a"), "other_call"),
      ],
      ids=[
          "different-session",
          "different-user",
          "different-app",
          "different-user-and-call",
      ],
  )
  async def test_independent_session_confirmations_do_not_collide(
      self,
      second_namespace: tuple[str, str, str],
      second_call_id: str,
  ):
    """A claim in one session must not consume another session's approval."""
    service = InMemorySessionService()
    namespaces = [
        ("test_app", "test_user", "session_a"),
        second_namespace,
    ]
    call_ids = ["shared_call", second_call_id]
    for app_name, user_id, session_id in namespaces:
      await service.create_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    executions = [0, 0]
    first_tool_started = asyncio.Event()
    release_first_tool = asyncio.Event()
    runners = []
    approvals = []

    def make_runner(index: int) -> Runner:
      async def local_counter(tool_context: ToolContext) -> dict[str, int]:
        del tool_context  # This tool deliberately does not inspect the verdict.
        executions[index] += 1
        if index == 0:
          first_tool_started.set()
          await release_first_tool.wait()
        return {"executions": executions[index]}

      tool = FunctionTool(func=local_counter, require_confirmation=True)
      tool_call_response = LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  Part(
                      function_call=FunctionCall(
                          name=tool.name, id=call_ids[index], args={}
                      )
                  )
              ],
          )
      )
      model = testing_utils.MockModel(
          responses=[
              tool_call_response,
              _create_llm_response_from_text("done"),
          ]
      )
      app_name = namespaces[index][0]
      agent = LlmAgent(name="root_agent", model=model, tools=[tool])
      return Runner(app_name=app_name, agent=agent, session_service=service)

    async def invoke(index: int, message: types.Content) -> list[Event]:
      app_name, user_id, session_id = namespaces[index]
      return [
          event
          async for event in runners[index].run_async(
              user_id=user_id, session_id=session_id, new_message=message
          )
      ]

    pending_tasks = []
    try:
      for index in range(2):
        runners.append(make_runner(index))
        initial_events = await invoke(
            index, testing_utils.UserContent("request")
        )
        confirmation_ids = [
            call.id
            for event in initial_events
            for call in event.get_function_calls()
            if call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
        ]
        assert len(confirmation_ids) == 1
        approvals.append(
            testing_utils.UserContent(
                Part(
                    function_response=FunctionResponse(
                        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                        id=confirmation_ids[0],
                        response={"confirmed": True},
                    )
                )
            )
        )

      first_approval = asyncio.create_task(invoke(0, approvals[0]))
      pending_tasks.append(first_approval)
      await asyncio.wait_for(first_tool_started.wait(), timeout=5)
      second_approval_events = await asyncio.wait_for(
          invoke(1, approvals[1]), timeout=5
      )
      assert executions == [1, 1]

      release_first_tool.set()
      first_approval_events = await asyncio.wait_for(first_approval, timeout=5)
      assert executions == [1, 1]

      for index, events in enumerate(
          [first_approval_events, second_approval_events]
      ):
        assert (
            sum(
                response.id == call_ids[index]
                for event in events
                for response in event.get_function_responses()
            )
            == 1
        )
        assert not any(
            call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
            for event in events
            for call in event.get_function_calls()
        )

      # A replay after the result is persisted is a no-op, not another prompt.
      replay_events = await invoke(1, approvals[1])
      assert executions == [1, 1]
      assert not any(
          call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
          for event in replay_events
          for call in event.get_function_calls()
      )
    finally:
      release_first_tool.set()
      for task in pending_tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*pending_tasks, return_exceptions=True)
      for runner in runners:
        await runner.close()


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
      assert event.invocation_id == invocation_id
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

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation_without_invocation_id(
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
    events = await runner.run_async(user_confirmation)
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


class TestHITLConfirmationWithUngatedParallelSibling:
  """Tests a gated call issued in parallel with an ungated sibling call."""

  @pytest.mark.parametrize("confirmed", [True, False])
  @pytest.mark.asyncio
  async def test_sibling_result_survives_caller_stopping_at_pause(
      self, confirmed: bool
  ):
    """The sibling runs once and its result reaches the model on resume."""
    sibling_calls = []
    widget = UiWidget(id="w1", provider="mcp")

    def _gated_tool() -> dict[str, str]:
      return {"result": "gated ran"}

    def _sibling_tool(tool_context: ToolContext) -> dict[str, str]:
      sibling_calls.append(tool_context.function_call_id)
      tool_context.state["sibling_ran"] = True
      tool_context.actions.render_ui_widgets = [widget]
      return {"result": "sibling ran"}

    gated_tool = FunctionTool(func=_gated_tool, require_confirmation=True)
    sibling_tool = FunctionTool(func=_sibling_tool)
    mock_model = testing_utils.MockModel(
        responses=[
            _create_llm_response_from_tools([gated_tool, sibling_tool]),
            _create_llm_response_from_text("final response"),
        ]
    )
    agent = LlmAgent(
        name="root_agent",
        model=mock_model,
        tools=[gated_tool, sibling_tool],
    )
    runner = testing_utils.InMemoryRunner(root_agent=agent)
    session = runner.session

    pause_event = None
    async with Aclosing(
        runner.runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=testing_utils.UserContent("test user query"),
        )
    ) as agen:
      async for event in agen:
        if event.is_final_response():
          pause_event = event
          break

    assert pause_event is not None
    confirmation_calls = pause_event.get_function_calls()
    assert [fc.name for fc in confirmation_calls] == [
        REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    ]
    assert len(sibling_calls) == 1
    assert runner.session.state["sibling_ran"] is True
    settled_event = runner.session.events[-2]
    assert settled_event.actions.render_ui_widgets == [widget]
    assert settled_event.timestamp <= pause_event.timestamp

    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=confirmation_calls[0].id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": confirmed},
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    assert len(sibling_calls) == 1
    assert testing_utils.simplify_events(copy.deepcopy(events))[-1] == (
        agent.name,
        "final response",
    )
    responses_sent_to_model = [
        {fr.name: fr.response for fr in content_responses}
        for content in mock_model.requests[-1].contents
        if (
            content_responses := [
                part.function_response
                for part in content.parts or []
                if part.function_response
            ]
        )
    ]
    expected_gated_response = (
        {"result": "gated ran"}
        if confirmed
        else {"error": "This tool call is rejected."}
    )
    assert responses_sent_to_model == [{
        gated_tool.name: expected_gated_response,
        sibling_tool.name: {"result": "sibling ran"},
    }]
