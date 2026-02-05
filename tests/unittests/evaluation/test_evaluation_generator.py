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

from __future__ import annotations

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.evaluation.simulation.user_simulator import NextUserMessage
from google.adk.evaluation.simulation.user_simulator import Status as UserSimulatorStatus
from google.adk.evaluation.simulation.user_simulator import UserSimulator
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


def _build_event(
    author: str, parts: list[types.Part], invocation_id: str
) -> Event:
  """Builds an Event object with specified parts."""

  return Event(
      author=author,
      content=types.Content(parts=parts),
      invocation_id=invocation_id,
  )


class TestConvertEventsToEvalInvocation:
  """Test cases for EvaluationGenerator.convert_events_to_eval_invocations method."""

  def test_convert_events_to_eval_invocations_empty(
      self,
  ):
    """Tests conversion with an empty list of events."""
    invocations = EvaluationGenerator.convert_events_to_eval_invocations([])
    assert invocations == []

  def test_convert_single_turn_text_only(
      self,
  ):
    """Tests a single turn with a text response."""
    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent", [types.Part(text="Hi there!")], "inv1"),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.invocation_id == "inv1"
    assert invocation.user_content.parts[0].text == "Hello"
    assert invocation.final_response.parts[0].text == "Hi there!"
    assert len(invocation.intermediate_data.invocation_events) == 0

  def test_convert_single_turn_tool_call(
      self,
  ):
    """Tests a single turn with a tool call."""
    events = [
        _build_event("user", [types.Part(text="what is the weather?")], "inv1"),
        _build_event(
            "agent",
            [
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather", args={}
                    )
                )
            ],
            "inv1",
        ),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.user_content.parts[0].text == "what is the weather?"
    assert invocation.final_response is None
    events = invocation.intermediate_data.invocation_events
    assert len(events) == 1
    assert events[0].author == "agent"
    assert events[0].content.parts[0].function_call.name == "get_weather"

  def test_convert_single_turn_tool_and_text_response(
      self,
  ):
    """Tests a single turn with a tool call and a final text response."""
    events = [
        _build_event("user", [types.Part(text="what is the weather?")], "inv1"),
        _build_event(
            "agent",
            [
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather", args={}
                    )
                )
            ],
            "inv1",
        ),
        _build_event("agent", [types.Part(text="It is sunny in SF.")], "inv1"),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.final_response.parts[0].text == "It is sunny in SF."
    events = invocation.intermediate_data.invocation_events
    assert len(events) == 1
    assert events[0].content.parts[0].function_call.name == "get_weather"

  def test_multi_turn(
      self,
  ):
    """Tests a conversation with multiple turns."""
    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent", [types.Part(text="Hi there!")], "inv1"),
        _build_event("user", [types.Part(text="How are you?")], "inv2"),
        _build_event("agent", [types.Part(text="I am fine.")], "inv2"),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 2
    assert invocations[0].user_content.parts[0].text == "Hello"
    assert invocations[0].final_response.parts[0].text == "Hi there!"
    assert invocations[1].user_content.parts[0].text == "How are you?"
    assert invocations[1].final_response.parts[0].text == "I am fine."

  def test_multi_agent(
      self,
  ):
    """Tests a multi-agent scenario creating multiple steps."""
    events = [
        _build_event("user", [types.Part(text="Do something")], "inv1"),
        _build_event(
            "root_agent",
            [
                types.Part(
                    function_call=types.FunctionCall(name="tool1", args={})
                )
            ],
            "inv1",
        ),
        _build_event(
            "sub_agent_1",
            [
                types.Part(
                    function_call=types.FunctionCall(name="tool2", args={})
                )
            ],
            "inv1",
        ),
        _build_event(
            "sub_agent_1",
            [
                types.Part(
                    function_call=types.FunctionCall(name="tool3", args={})
                ),
                types.Part(text="intermediate response"),
            ],
            "inv1",
        ),
        _build_event(
            "sub_agent_2",
            [
                types.Part(
                    function_call=types.FunctionCall(name="tool4", args={})
                )
            ],
            "inv1",
        ),
        _build_event("root_agent", [types.Part(text="All done.")], "inv1"),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.final_response.parts[0].text == "All done."
    events = invocation.intermediate_data.invocation_events

    assert len(events) == 4
    assert events[0].author == "root_agent"
    assert events[1].author == "sub_agent_1"
    assert events[2].author == "sub_agent_1"
    assert events[3].author == "sub_agent_2"


class TestGetAppDetailsByInvocationId:
  """Test cases for EvaluationGenerator._get_app_details_by_invocation_id method."""

  def test_get_app_details_by_invocation_id_empty(self, mocker):
    """Tests with an empty list of events."""
    mock_request_intercepter = mocker.MagicMock(spec=_RequestIntercepterPlugin)
    app_details = EvaluationGenerator._get_app_details_by_invocation_id(
        [], mock_request_intercepter
    )
    assert app_details == {}

  def test_get_app_details_by_invocation_id_no_model_requests(self, mocker):
    """Tests when request_intercepter returns no model requests."""
    mock_request_intercepter = mocker.MagicMock(spec=_RequestIntercepterPlugin)
    mock_request_intercepter.get_model_request.return_value = None
    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent", [types.Part(text="Hi there!")], "inv1"),
    ]
    app_details = EvaluationGenerator._get_app_details_by_invocation_id(
        events, mock_request_intercepter
    )
    assert app_details == {"inv1": AppDetails(agent_details={})}
    mock_request_intercepter.get_model_request.assert_called_once_with(
        events[1]
    )

  def test_get_app_details_single_invocation_single_agent(self, mocker):
    """Tests a single invocation with one agent."""
    mock_request_intercepter = mocker.MagicMock(spec=_RequestIntercepterPlugin)
    mock_llm_request = LlmRequest(model="test")
    mock_llm_request.config.system_instruction = "instruction1"
    mock_llm_request.config.tools = [types.Tool()]
    mock_request_intercepter.get_model_request.return_value = mock_llm_request

    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent", [types.Part(text="Hi there!")], "inv1"),
    ]
    app_details = EvaluationGenerator._get_app_details_by_invocation_id(
        events, mock_request_intercepter
    )

    expected_app_details = {
        "inv1": AppDetails(
            agent_details={
                "agent": AgentDetails(
                    name="agent",
                    instructions="instruction1",
                    tool_declarations=[types.Tool()],
                )
            }
        )
    }
    assert app_details == expected_app_details
    mock_request_intercepter.get_model_request.assert_called_once_with(
        events[1]
    )

  def test_get_app_details_multiple_invocations_multiple_agents(self, mocker):
    """Tests multiple invocations with multiple agents."""
    mock_request_intercepter = mocker.MagicMock(spec=_RequestIntercepterPlugin)

    def get_model_request_side_effect(event):
      mock_llm_request = LlmRequest(model="test")
      if event.invocation_id == "inv1" and event.author == "agent1":
        mock_llm_request.config.system_instruction = "instruction1"
        mock_llm_request.config.tools = [
            types.Tool(
                function_declarations=[types.FunctionDeclaration(name="tool1")]
            )
        ]
        return mock_llm_request
      if event.invocation_id == "inv2" and event.author == "agent2":
        mock_llm_request.config.system_instruction = "instruction2"
        return mock_llm_request
      return None

    mock_request_intercepter.get_model_request.side_effect = (
        get_model_request_side_effect
    )

    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent1", [types.Part(text="Hi there!")], "inv1"),
        _build_event("user", [types.Part(text="Hello again")], "inv2"),
        _build_event("agent2", [types.Part(text="Hi again!")], "inv2"),
        _build_event(
            "agent1", [types.Part(text="Hi again from agent1")], "inv2"
        ),  # no request
    ]
    app_details = EvaluationGenerator._get_app_details_by_invocation_id(
        events, mock_request_intercepter
    )

    expected_app_details = {
        "inv1": AppDetails(
            agent_details={
                "agent1": AgentDetails(
                    name="agent1",
                    instructions="instruction1",
                    tool_declarations=[
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(name="tool1")
                            ]
                        )
                    ],
                )
            }
        ),
        "inv2": AppDetails(
            agent_details={
                "agent2": AgentDetails(
                    name="agent2",
                    instructions="instruction2",
                    tool_declarations=[],
                )
            }
        ),
    }
    assert app_details == expected_app_details
    assert mock_request_intercepter.get_model_request.call_count == 3


class TestGenerateInferencesForSingleUserInvocation:
  """Test cases for EvaluationGenerator._generate_inferences_for_single_user_invocation method."""

  @pytest.mark.asyncio
  async def test_generate_inferences_with_mock_runner(self, mocker):
    """Tests inference generation with a mocked runner."""
    runner = mocker.MagicMock()

    agent_parts = [types.Part(text="Agent response")]

    async def mock_run_async(*args, **kwargs):
      yield _build_event(
          author="agent",
          parts=agent_parts,
          invocation_id="inv1",
      )

    runner.run_async.return_value = mock_run_async()

    user_content = types.Content(parts=[types.Part(text="User query")])

    events = [
        event
        async for event in EvaluationGenerator._generate_inferences_for_single_user_invocation(
            runner, "test_user", "test_session", user_content
        )
    ]

    assert len(events) == 2
    assert events[0].author == "user"
    assert events[0].content == user_content
    assert events[0].invocation_id == "inv1"
    assert events[1].author == "agent"
    assert events[1].content.parts == agent_parts

    runner.run_async.assert_called_once_with(
        user_id="test_user",
        session_id="test_session",
        new_message=user_content,
    )


@pytest.fixture
def mock_runner(mocker):
  """Provides a mock Runner for testing."""
  mock_runner_cls = mocker.patch(
      "google.adk.evaluation.evaluation_generator.Runner"
  )
  mock_runner_instance = mocker.AsyncMock()
  mock_runner_instance.__aenter__.return_value = mock_runner_instance
  mock_runner_cls.return_value = mock_runner_instance
  yield mock_runner_instance


@pytest.fixture
def mock_session_service(mocker):
  """Provides a mock InMemorySessionService for testing."""
  mock_session_service_cls = mocker.patch(
      "google.adk.evaluation.evaluation_generator.InMemorySessionService"
  )
  mock_session_service_instance = mocker.MagicMock()
  mock_session_service_instance.create_session = mocker.AsyncMock()
  mock_session_service_cls.return_value = mock_session_service_instance
  yield mock_session_service_instance


class TestGenerateInferencesFromRootAgent:
  """Test cases for EvaluationGenerator._generate_inferences_from_root_agent method."""

  @pytest.mark.asyncio
  async def test_generates_inferences_with_user_simulator(
      self, mocker, mock_runner, mock_session_service
  ):
    """Tests that inferences are generated by interacting with a user simulator."""
    mock_agent = mocker.MagicMock()
    mock_user_sim = mocker.MagicMock(spec=UserSimulator)

    # Mock user simulator will produce one message, then stop.
    async def get_next_user_message_side_effect(*args, **kwargs):
      if mock_user_sim.get_next_user_message.call_count == 1:
        return NextUserMessage(
            status=UserSimulatorStatus.SUCCESS,
            user_message=types.Content(parts=[types.Part(text="message 1")]),
        )
      return NextUserMessage(status=UserSimulatorStatus.STOP_SIGNAL_DETECTED)

    mock_user_sim.get_next_user_message = mocker.AsyncMock(
        side_effect=get_next_user_message_side_effect
    )

    mock_generate_inferences = mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator._generate_inferences_for_single_user_invocation"
    )
    mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator._get_app_details_by_invocation_id"
    )
    mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator.convert_events_to_eval_invocations"
    )

    # Each call to _generate_inferences_for_single_user_invocation will
    # yield one user and one agent event.
    async def mock_generate_inferences_side_effect(
        runner, user_id, session_id, user_content
    ):
      yield _build_event("user", user_content.parts, "inv1")
      yield _build_event("agent", [types.Part(text="agent_response")], "inv1")

    mock_generate_inferences.side_effect = mock_generate_inferences_side_effect

    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=mock_agent,
        user_simulator=mock_user_sim,
    )

    # Check that user simulator was called until it stopped.
    assert mock_user_sim.get_next_user_message.call_count == 2

    # Check that we generated inferences for each user message.
    assert mock_generate_inferences.call_count == 1

    # Check the content of the user messages passed to inference generation
    mock_generate_inferences.assert_called_once()
    called_with_content = mock_generate_inferences.call_args.args[3]
    assert called_with_content.parts[0].text == "message 1"
