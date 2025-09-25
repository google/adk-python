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

from __future__ import annotations

from unittest import mock

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.genai import types


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

  def test_get_app_details_by_invocation_id_empty(self):
    """Tests with an empty list of events."""
    mock_request_intercepter = mock.MagicMock(spec=_RequestIntercepterPlugin)
    app_details = EvaluationGenerator._get_app_details_by_invocation_id(
        [], mock_request_intercepter
    )
    assert app_details == {}

  def test_get_app_details_by_invocation_id_no_model_requests(self):
    """Tests when request_intercepter returns no model requests."""
    mock_request_intercepter = mock.MagicMock(spec=_RequestIntercepterPlugin)
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

  def test_get_app_details_single_invocation_single_agent(self):
    """Tests a single invocation with one agent."""
    mock_request_intercepter = mock.MagicMock(spec=_RequestIntercepterPlugin)
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

  def test_get_app_details_multiple_invocations_multiple_agents(self):
    """Tests multiple invocations with multiple agents."""
    mock_request_intercepter = mock.MagicMock(spec=_RequestIntercepterPlugin)

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
