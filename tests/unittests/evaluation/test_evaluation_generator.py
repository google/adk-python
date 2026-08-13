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

import asyncio
import builtins

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.evaluation import evaluation_generator as evaluation_generator_module
from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import SessionInput
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import _LiveSession
from google.adk.evaluation.evaluation_generator import _send_audio_to_live
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.user_simulator import NextUserMessage
from google.adk.evaluation.simulation.user_simulator import Status as UserSimulatorStatus
from google.adk.evaluation.simulation.user_simulator import UserSimulator
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
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


def _build_transcription_event(
    author: str,
    text: str,
    invocation_id: str,
    *,
    partial: bool,
    is_input: bool = False,
) -> Event:
  """Builds a transcription-bearing Event (text in *_transcription, no content)."""

  transcription = types.Transcription(text=text)
  return Event(
      author=author,
      invocation_id=invocation_id,
      input_transcription=transcription if is_input else None,
      output_transcription=None if is_input else transcription,
      partial=partial,
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

  def test_convert_keeps_text_response_over_trailing_audio(
      self,
  ):
    """A text response is kept over trailing audio-only events of the turn."""
    events = [
        _build_event("user", [types.Part(text="Hi")], "inv1"),
        _build_event("agent", [types.Part(text="Hello there.")], "inv1"),
        _build_event(
            "agent",
            [
                types.Part(
                    inline_data=types.Blob(
                        mime_type="audio/pcm", data=b"fake-audio"
                    )
                )
            ],
            "inv1",
        ),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.final_response.parts[0].text == "Hello there."
    intermediate = invocation.intermediate_data.invocation_events
    assert len(intermediate) == 1
    assert intermediate[0].content.parts[0].inline_data.data == b"fake-audio"

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

  def test_convert_multi_agent_final_responses(
      self,
  ):
    """Tests that only the last final response is excluded from intermediate data."""
    events = [
        _build_event("user", [types.Part(text="Hello")], "inv1"),
        _build_event("agent1", [types.Part(text="First response")], "inv1"),
        _build_event("agent2", [types.Part(text="Second response")], "inv1"),
    ]

    invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.final_response.parts[0].text == "Second response"

    intermediate_events = invocation.intermediate_data.invocation_events
    # agent1 is included because it is not the final_event (which is agent2)
    assert len(intermediate_events) == 1
    assert intermediate_events[0].author == "agent1"
    assert intermediate_events[0].content.parts[0].text == "First response"


class TestNormalizeLiveTranscriptions:
  """Test cases for EvaluationGenerator._normalize_live_transcriptions method."""

  def test_output_transcription_becomes_model_content(self):
    """A consolidated output transcription is folded into model content."""
    events = [
        _build_transcription_event(
            "agent", "Hello there.", "inv1", partial=False
        )
    ]

    normalized = EvaluationGenerator._normalize_live_transcriptions(events)

    assert len(normalized) == 1
    assert normalized[0].content.role == "model"
    assert normalized[0].content.parts[0].text == "Hello there."
    assert normalized[0].output_transcription is None

  def test_input_transcription_becomes_user_content(self):
    """A consolidated input transcription is folded into user content."""
    events = [
        _build_transcription_event(
            "user", "Kick off a task.", "inv1", partial=False, is_input=True
        )
    ]

    normalized = EvaluationGenerator._normalize_live_transcriptions(events)

    assert len(normalized) == 1
    assert normalized[0].content.role == "user"
    assert normalized[0].content.parts[0].text == "Kick off a task."
    assert normalized[0].input_transcription is None

  def test_partial_transcriptions_are_passed_through(self):
    """Partial fragments are left as-is to avoid duplicating the turn text."""
    partial = _build_transcription_event("agent", "Hel", "inv1", partial=True)

    normalized = EvaluationGenerator._normalize_live_transcriptions([partial])

    assert normalized[0].content is None
    assert normalized[0].output_transcription.text == "Hel"

  def test_content_events_are_passed_through_untouched(self):
    """Events that already carry content (text or audio) are not rewritten."""
    audio = _build_event(
        "agent",
        [types.Part(inline_data=types.Blob(mime_type="audio/pcm", data=b"a"))],
        "inv1",
    )

    normalized = EvaluationGenerator._normalize_live_transcriptions([audio])

    assert normalized[0] is audio


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
        async for event in (
            EvaluationGenerator._generate_inferences_for_single_user_invocation(
                runner, "test_user", "test_session", user_content
            )
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


class TestGenerateInferencesForSingleUserInvocationLive:
  """Test cases for EvaluationGenerator._generate_inferences_for_single_user_invocation_live method."""

  @pytest.mark.asyncio
  async def test_generate_inferences_live(self, mocker):
    """Tests live inference generation."""
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    user_content = types.Content(parts=[types.Part(text="User query")])
    invocation_id = "inv1"

    agent_event = _build_event(
        "agent", [types.Part(text="Agent response")], invocation_id
    )
    other_event = _build_event(
        "agent", [types.Part(text="Other response")], "inv2"
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    # First yield should be the user message
    first_event = await gen.__anext__()
    assert first_event.author == "user"
    assert first_event.content == user_content
    assert first_event.invocation_id == invocation_id

    # Mock turn_complete_event.wait to avoid blocking
    turn_complete_event.wait = mocker.AsyncMock()

    # Put events in queue BEFORE advancing
    await event_queue.put(agent_event)
    await event_queue.put(other_event)

    # Now advance to get the next event
    second_event = await gen.__anext__()

    assert mock_live_request_queue.send_content.called
    mock_live_request_queue.send_content.assert_called_once_with(user_content)

    assert second_event == agent_event

    # The generator should be exhausted now because other_event doesn't match invocation_id
    with pytest.raises(StopAsyncIteration):
      await gen.__anext__()

  @pytest.mark.asyncio
  async def test_generate_inferences_live_yields_transcription_events_as_is(
      self, mocker
  ):
    """The live loop yields transcription events unchanged, adding no synthetic events.

    Transcription-to-text consolidation happens later in
    `convert_events_to_eval_invocations`, so the inference loop must forward the
    raw events (partial and consolidated) without emitting extra events.
    """
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    user_content = types.Content(parts=[types.Part(text="User query")])
    invocation_id = "inv1"

    partial_event = Event(
        author="agent",
        content=types.Content(parts=[]),
        invocation_id=invocation_id,
        output_transcription=types.Transcription(text="Hello "),
        partial=True,
    )
    final_transcription = Event(
        author="agent",
        content=types.Content(parts=[]),
        invocation_id=invocation_id,
        output_transcription=types.Transcription(text="Hello there."),
        partial=False,
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    # First yield should be the user message.
    first_event = await gen.__anext__()
    assert first_event.author == "user"
    assert first_event.content == user_content
    assert first_event.invocation_id == invocation_id

    # Mock turn_complete_event.wait to avoid blocking.
    turn_complete_event.wait = mocker.AsyncMock()

    await event_queue.put(partial_event)
    await event_queue.put(final_transcription)

    # Both transcription events are yielded unchanged, and nothing else follows.
    assert await gen.__anext__() == partial_event
    assert await gen.__anext__() == final_transcription
    with pytest.raises(StopAsyncIteration):
      await gen.__anext__()

  @pytest.mark.asyncio
  async def test_generate_inferences_live_streams_audio_as_realtime(
      self, mocker
  ):
    """An audio message is streamed to the agent as bracketed realtime input.

    The agent should receive the audio as realtime input (not a content turn),
    bracketed by activity markers, while the full Content (text + audio) is
    preserved in the yielded user Event for trajectory logging and autorater
    evaluation.
    """
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    text_part = types.Part(text="User query")
    audio_part = types.Part(
        inline_data=types.Blob(mime_type="audio/pcm", data=b"fake-audio")
    )
    user_content = types.Content(role="user", parts=[text_part, audio_part])
    invocation_id = "inv1"

    agent_event = _build_event(
        "agent", [types.Part(text="Agent response")], invocation_id
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    # The yielded user event preserves the full content (text + audio).
    first_event = await gen.__anext__()
    assert first_event.author == "user"
    assert first_event.content == user_content
    assert len(first_event.content.parts) == 2

    # Mock turn_complete_event.wait to avoid blocking
    turn_complete_event.wait = mocker.AsyncMock()
    await event_queue.put(agent_event)

    second_event = await gen.__anext__()
    assert second_event == agent_event

    # The agent receives the audio as realtime input, not a content turn.
    mock_live_request_queue.send_content.assert_not_called()
    mock_live_request_queue.send_activity_start.assert_called_once()
    mock_live_request_queue.send_activity_end.assert_called_once()
    sent_blob = mock_live_request_queue.send_realtime.call_args.args[0]
    assert sent_blob.data == b"fake-audio"

  @pytest.mark.asyncio
  async def test_generate_inferences_live_audio_only_message(self, mocker):
    """Audio-only messages are streamed to the agent as realtime input."""
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    audio_part = types.Part(
        inline_data=types.Blob(mime_type="audio/pcm", data=b"fake-audio")
    )
    user_content = types.Content(role="user", parts=[audio_part])
    invocation_id = "inv1"

    agent_event = _build_event(
        "agent", [types.Part(text="Agent response")], invocation_id
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    first_event = await gen.__anext__()
    assert first_event.content == user_content

    # Mock turn_complete_event.wait to avoid blocking
    turn_complete_event.wait = mocker.AsyncMock()
    await event_queue.put(agent_event)
    await gen.__anext__()

    mock_live_request_queue.send_content.assert_not_called()
    sent_blob = mock_live_request_queue.send_realtime.call_args.args[0]
    assert sent_blob.data == b"fake-audio"

  @pytest.mark.asyncio
  async def test_generate_inferences_live_text_only_message_unchanged(
      self, mocker
  ):
    """Text-only messages are forwarded to the agent unchanged."""
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    user_content = types.Content(
        role="user", parts=[types.Part(text="User query")]
    )
    invocation_id = "inv1"

    agent_event = _build_event(
        "agent", [types.Part(text="Agent response")], invocation_id
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    await gen.__anext__()
    turn_complete_event.wait = mocker.AsyncMock()
    await event_queue.put(agent_event)
    await gen.__anext__()

    # No audio present, so the original content object is sent as-is.
    mock_live_request_queue.send_content.assert_called_once_with(user_content)

  @pytest.mark.asyncio
  async def test_generate_inferences_live_audio_with_text_streams_audio(
      self, mocker
  ):
    """A part carrying both text and audio still streams its audio as realtime.

    A single part may carry both a transcript and inline audio; its audio is
    streamed as realtime input so the agent can decode it.
    """
    mock_live_request_queue = mocker.MagicMock()
    event_queue = asyncio.Queue()
    turn_complete_event = asyncio.Event()

    combined_part = types.Part(
        text="User query",
        inline_data=types.Blob(mime_type="audio/pcm", data=b"fake-audio"),
    )
    user_content = types.Content(role="user", parts=[combined_part])
    invocation_id = "inv1"

    agent_event = _build_event(
        "agent", [types.Part(text="Agent response")], invocation_id
    )

    gen = EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
        live_request_queue=mock_live_request_queue,
        event_queue=event_queue,
        user_message=user_content,
        current_invocation_id=invocation_id,
        turn_complete_event=turn_complete_event,
        live_timeout_seconds=300,
    )

    first_event = await gen.__anext__()
    assert first_event.content == user_content

    # Mock turn_complete_event.wait to avoid blocking
    turn_complete_event.wait = mocker.AsyncMock()
    await event_queue.put(agent_event)
    await gen.__anext__()

    mock_live_request_queue.send_content.assert_not_called()
    sent_blob = mock_live_request_queue.send_realtime.call_args.args[0]
    assert sent_blob.data == b"fake-audio"


class TestSendAudioToLive:
  """Test cases for _send_audio_to_live."""

  def test_brackets_audio_with_activity_markers(self, mocker):
    """Audio is streamed between an activity start and end marker."""
    queue = mocker.MagicMock()
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                inline_data=types.Blob(mime_type="audio/pcm", data=b"1234")
            )
        ],
    )

    _send_audio_to_live(queue, content)

    queue.send_activity_start.assert_called_once()
    queue.send_activity_end.assert_called_once()
    queue.send_realtime.assert_called_once()

  def test_splits_audio_into_chunks(self, mocker):
    """Audio larger than the chunk size is streamed in multiple realtime sends."""
    queue = mocker.MagicMock()
    # 2.5 chunks worth of audio -> 3 realtime sends.
    data = b"\x00" * (16000 * 2 + 8000)
    content = types.Content(
        role="user",
        parts=[
            types.Part(inline_data=types.Blob(mime_type="audio/pcm", data=data))
        ],
    )

    _send_audio_to_live(queue, content)

    assert queue.send_realtime.call_count == 3
    sent = b"".join(
        call.args[0].data for call in queue.send_realtime.call_args_list
    )
    assert sent == data

  def test_preserves_blob_mime_type(self, mocker):
    """The source blob's mime type is carried on each realtime send."""
    queue = mocker.MagicMock()
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                inline_data=types.Blob(
                    mime_type="audio/pcm;rate=16000", data=b"12"
                )
            )
        ],
    )

    _send_audio_to_live(queue, content)

    sent_blob = queue.send_realtime.call_args.args[0]
    assert sent_blob.mime_type == "audio/pcm;rate=16000"


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

  @pytest.mark.asyncio
  async def test_pinned_session_id_reused_across_runs_no_collision(
      self, mocker, mock_runner
  ):
    """A reused (pinned) session_id does not collide on a rerun."""
    session_service = InMemorySessionService()
    mock_user_sim = mocker.MagicMock(spec=UserSimulator)
    mock_user_sim.get_next_user_message = mocker.AsyncMock(
        return_value=NextUserMessage(
            status=UserSimulatorStatus.STOP_SIGNAL_DETECTED
        )
    )

    # Two runs share one session_service, mirroring num_runs=2.
    for _ in range(2):
      await EvaluationGenerator._generate_inferences_from_root_agent(
          root_agent=mocker.MagicMock(),
          user_simulator=mock_user_sim,
          initial_session=SessionInput(
              app_name="test_app", user_id="u", session_id="fixed"
          ),
          session_service=session_service,
      )

    assert (
        await session_service.get_session(
            app_name="test_app", user_id="u", session_id="fixed"
        )
        is not None
    )

  @pytest.mark.asyncio
  async def test_pinned_session_id_preserves_existing_session(
      self, mocker, mock_runner
  ):
    """A session the caller prepared keeps its events and state."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test_app",
        user_id="u",
        session_id="fixed",
        state={"prepared_by": "caller"},
    )
    await session_service.append_event(
        session, _build_event("user", [types.Part(text="earlier turn")], "inv0")
    )
    mock_user_sim = mocker.MagicMock(spec=UserSimulator)
    mock_user_sim.get_next_user_message = mocker.AsyncMock(
        return_value=NextUserMessage(
            status=UserSimulatorStatus.STOP_SIGNAL_DETECTED
        )
    )

    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=mocker.MagicMock(),
        user_simulator=mock_user_sim,
        initial_session=SessionInput(
            app_name="test_app", user_id="u", session_id="fixed", state={}
        ),
        session_service=session_service,
    )

    reloaded = await session_service.get_session(
        app_name="test_app", user_id="u", session_id="fixed"
    )
    assert reloaded.state["prepared_by"] == "caller"
    assert [e.content.parts[0].text for e in reloaded.events] == [
        "earlier turn"
    ]

  @pytest.mark.asyncio
  async def test_generates_inferences_with_user_simulator_live(
      self, mocker, mock_runner, mock_session_service
  ):
    """Tests that inferences are generated by interacting with a user simulator in live mode."""
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

    mock_generate_inferences_live = mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator._generate_inferences_for_single_user_invocation_live"
    )
    mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator._get_app_details_by_invocation_id"
    )
    mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator.convert_events_to_eval_invocations"
    )

    # Mock _LiveSession context manager
    mock_live_session = mocker.MagicMock()
    mock_live_session.__aenter__ = mocker.AsyncMock(
        return_value=mock_live_session
    )
    mock_live_session.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_live_session.live_request_queue = mocker.MagicMock()
    mock_live_session.event_queue = asyncio.Queue()
    mock_live_session.turn_complete_event = asyncio.Event()
    mock_live_session.live_finished = asyncio.Event()

    mock_live_session_cls = mocker.patch(
        "google.adk.evaluation.evaluation_generator._LiveSession",
        return_value=mock_live_session,
    )

    # Each call to _generate_inferences_for_single_user_invocation_live will
    # yield one user and one agent event.
    async def mock_generate_inferences_live_side_effect(*args, **kwargs):
      yield _build_event("user", [types.Part(text="message 1")], "inv1")
      yield _build_event("agent", [types.Part(text="agent_response")], "inv1")

    mock_generate_inferences_live.side_effect = (
        mock_generate_inferences_live_side_effect
    )

    await EvaluationGenerator._generate_inferences_from_root_agent_live(
        root_agent=mock_agent,
        user_simulator=mock_user_sim,
        live_timeout_seconds=600,
    )

    # Check that user simulator was called until it stopped.
    assert mock_user_sim.get_next_user_message.call_count == 2

    # Check that we generated inferences for each user message.
    mock_generate_inferences_live.assert_called_once()
    called_with_content = mock_generate_inferences_live.call_args.kwargs[
        "user_message"
    ]
    assert called_with_content.parts[0].text == "message 1"
    assert (
        mock_generate_inferences_live.call_args.kwargs["live_timeout_seconds"]
        == 600
    )

    # Verify that the agent response was collected
    mock_convert = EvaluationGenerator.convert_events_to_eval_invocations
    mock_convert.assert_called_once()
    events_passed = mock_convert.call_args.args[0]

    agent_events = [e for e in events_passed if e.author == "agent"]
    assert len(agent_events) == 1
    assert agent_events[0].content.parts[0].text == "agent_response"

    # Verify that the _LiveSession constructor was called
    mock_live_session_cls.assert_called_once()


class TestGenerateResponses:
  """Test cases for EvaluationGenerator.generate_responses method."""

  @pytest.mark.asyncio
  async def test_generate_responses_passes_config_to_simulator_instance(
      self, mocker
  ):
    """Tests that user_simulator_config reaches the actual UserSimulator instance when UserSimulatorProvider is not mocked."""
    mock_process_query = mocker.patch(
        "google.adk.evaluation.evaluation_generator.EvaluationGenerator._process_query",
        new_callable=mocker.AsyncMock,
        return_value=[],
    )

    user_simulator_config = LlmBackedUserSimulatorConfig(
        model="gemini-2.5-flash",
        max_allowed_invocations=5,
        custom_instructions=(
            "custom {{ stop_signal }} {{ conversation_plan }} {{"
            " conversation_history }}"
        ),
    )
    eval_set = EvalSet(
        eval_set_id="test_set",
        eval_cases=[
            EvalCase(
                eval_id="case_0",
                conversation_scenario=ConversationScenario(
                    starting_prompt="hello",
                    conversation_plan="test plan",
                ),
            )
        ],
    )

    await EvaluationGenerator.generate_responses(
        eval_set=eval_set,
        agent_module_path="some.agent.module",
        repeat_num=1,
        user_simulator_config=user_simulator_config,
    )

    mock_process_query.assert_called_once()
    user_simulator = mock_process_query.call_args.args[1]
    assert isinstance(user_simulator, LlmBackedUserSimulator)
    assert user_simulator._config.model == "gemini-2.5-flash"
    assert user_simulator._config.max_allowed_invocations == 5
    assert (
        user_simulator._config.custom_instructions
        == "custom {{ stop_signal }} {{ conversation_plan }} {{"
        " conversation_history }}"
    )


class TestLiveSessionCallbacks:
  """Unit tests verifying that _LiveSession manually triggers callbacks."""

  @pytest.mark.asyncio
  async def test_live_session_manually_triggers_callbacks(self, mocker):
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.llm_agent import Agent
    from google.adk.models.llm_request import LlmRequest

    # 1. Setup mock runner, agent, and session
    mock_runner = mocker.MagicMock()
    mock_runner.session_service.append_event = mocker.AsyncMock()
    mock_session = mocker.MagicMock()
    mock_agent = mocker.MagicMock(spec=Agent)
    mock_runner.agent = mock_agent
    mock_runner._find_agent_to_run.return_value = mock_agent

    mock_agent.name = "test_agent"

    # Mock _llm_flow._preprocess_async to set dummy instruction
    async def mock_preprocess_async(invocation_context, llm_request):
      llm_request.config.system_instruction = "mock instruction"
      return
      yield  # make it an async generator

    mock_flow = mocker.MagicMock()
    mock_flow._preprocess_async = mock_preprocess_async
    mock_agent._llm_flow = mock_flow

    # Mock run_live stream yielding one event
    mock_event = Event(
        author="agent",
        content=types.Content(parts=[types.Part(text="Hello")]),
        invocation_id="test_invocation_id",
    )

    async def mock_run_live(*args, **kwargs):
      yield mock_event

    mock_agent.run_live.return_value = mock_run_live()

    # Mock plugin_manager on invocation context
    mock_plugin_manager = mocker.MagicMock()
    mock_plugin_manager.run_before_model_callback = mocker.AsyncMock()
    mock_plugin_manager.run_after_model_callback = mocker.AsyncMock()
    mock_runner._new_invocation_context_for_live.return_value.plugin_manager = (
        mock_plugin_manager
    )
    mock_runner._new_invocation_context_for_live.return_value.agent = mock_agent

    # 2. Instantiate and enter _LiveSession
    live_session = _LiveSession(
        runner=mock_runner,
        session=mock_session,
        user_id="test_user",
        session_id="test_session",
    )

    # Directly run _consume_events as a coroutine for synchronous-style testing
    await live_session._consume_events()

    # 3. Assertions
    mock_plugin_manager.run_before_model_callback.assert_called_once()
    called_before_args = mock_plugin_manager.run_before_model_callback.call_args
    assert isinstance(
        called_before_args.kwargs["callback_context"], CallbackContext
    )
    assert isinstance(called_before_args.kwargs["llm_request"], LlmRequest)
    assert (
        called_before_args.kwargs["llm_request"].config.system_instruction
        == "mock instruction"
    )

    mock_plugin_manager.run_after_model_callback.assert_called_once()
    called_after_args = mock_plugin_manager.run_after_model_callback.call_args
    assert isinstance(
        called_after_args.kwargs["callback_context"], CallbackContext
    )
    assert isinstance(called_after_args.kwargs["llm_response"], Event)
    assert called_after_args.kwargs["llm_response"] == mock_event

  @pytest.mark.asyncio
  async def test_live_session_manually_triggers_callbacks_with_tools(
      self, mocker
  ):
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.llm_agent import Agent
    from google.adk.models.llm_request import LlmRequest

    # 1. Setup mock runner, agent, and session
    mock_runner = mocker.MagicMock()
    mock_runner.session_service.append_event = mocker.AsyncMock()
    mock_session = mocker.MagicMock()
    mock_agent = mocker.MagicMock(spec=Agent)
    mock_runner.agent = mock_agent
    mock_runner._find_agent_to_run.return_value = mock_agent

    mock_agent.name = "test_agent"

    # Set up a mock tool
    mock_tool = mocker.MagicMock()
    mock_tool.name = "get_weather"
    mock_decl = types.FunctionDeclaration(
        name="get_weather",
        description="Get weather details",
    )
    mock_tool._get_declaration.return_value = mock_decl

    # Mock _llm_flow._preprocess_async to set instruction and append tool
    async def mock_preprocess_async(invocation_context, llm_request):
      llm_request.config.system_instruction = "mock instruction"
      llm_request.append_tools([mock_tool])
      return
      yield  # make it an async generator

    mock_flow = mocker.MagicMock()
    mock_flow._preprocess_async = mock_preprocess_async
    mock_agent._llm_flow = mock_flow

    # Mock run_live stream yielding one event
    mock_event = Event(
        author="agent",
        content=types.Content(parts=[types.Part(text="Hello")]),
        invocation_id="test_invocation_id",
    )

    async def mock_run_live(*args, **kwargs):
      yield mock_event

    mock_agent.run_live.return_value = mock_run_live()

    # Mock plugin_manager on invocation context
    mock_plugin_manager = mocker.MagicMock()
    mock_plugin_manager.run_before_model_callback = mocker.AsyncMock()
    mock_plugin_manager.run_after_model_callback = mocker.AsyncMock()
    mock_runner._new_invocation_context_for_live.return_value.plugin_manager = (
        mock_plugin_manager
    )
    mock_runner._new_invocation_context_for_live.return_value.agent = mock_agent

    # 2. Instantiate and enter _LiveSession
    live_session = _LiveSession(
        runner=mock_runner,
        session=mock_session,
        user_id="test_user",
        session_id="test_session",
    )

    # Directly run _consume_events as a coroutine
    await live_session._consume_events()

    # 3. Assertions
    mock_plugin_manager.run_before_model_callback.assert_called_once()
    called_before_args = mock_plugin_manager.run_before_model_callback.call_args
    assert isinstance(
        called_before_args.kwargs["callback_context"], CallbackContext
    )

    llm_request = called_before_args.kwargs["llm_request"]
    assert isinstance(llm_request, LlmRequest)
    assert llm_request.config.system_instruction == "mock instruction"

    # Assert that tool was correctly wrapped under types.Tool format
    assert len(llm_request.config.tools) == 1
    wrapped_tool = llm_request.config.tools[0]
    assert isinstance(wrapped_tool, types.Tool)
    assert len(wrapped_tool.function_declarations) == 1
    assert wrapped_tool.function_declarations[0].name == "get_weather"

    mock_plugin_manager.run_after_model_callback.assert_called_once()
    called_after_args = mock_plugin_manager.run_after_model_callback.call_args
    assert isinstance(
        called_after_args.kwargs["callback_context"], CallbackContext
    )
    assert isinstance(called_after_args.kwargs["llm_response"], Event)
    assert called_after_args.kwargs["llm_response"] == mock_event


class TestLiveSessionNodeRouting:
  """Verifies non-Agent BaseNode roots are driven via `Runner.run_live`."""

  @pytest.mark.asyncio
  async def test_workflow_root_uses_runner_run_live(self, mocker):
    from google.adk.workflow import Workflow

    # A Workflow has no `_llm_flow`/`run_live`; the session must fall back to
    # `Runner.run_live`, which handles node scheduling and function calls.
    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow

    mock_event = Event(
        author="agent",
        content=types.Content(parts=[types.Part(text="Hi")]),
        invocation_id="unused",
        turn_complete=True,
    )

    async def mock_run_live(*args, **kwargs):
      yield mock_event

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="test_user",
        session_id="test_session",
    )

    await live_session._consume_events()

    # The Agent-only driver must not be touched for a Workflow root.
    mock_runner._new_invocation_context_for_live.assert_not_called()
    mock_runner.run_live.assert_called_once()
    call_kwargs = mock_runner.run_live.call_args.kwargs
    assert call_kwargs["user_id"] == "test_user"
    assert call_kwargs["session_id"] == "test_session"

    # The event is re-stamped with the turn's invocation id and enqueued, and
    # the agent's turn-complete releases the waiter.
    queued_event = await live_session.event_queue.get()
    assert queued_event.invocation_id == live_session.current_invocation_id
    assert live_session.turn_complete_event.is_set()
    assert live_session.live_finished.is_set()

  @pytest.mark.asyncio
  async def test_workflow_root_swallows_tool_call_turn_complete(self, mocker):
    from google.adk.workflow import Workflow

    # The intermediate `turn_complete` that accompanies a tool call must not
    # release the waiter; only the true end-of-turn after the model continues.
    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow

    # A tool call and its `turn_complete` arrive as separate events.
    fc_event = Event(
        author="dob_verifier_agent",
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="validate_date_of_birth",
                        args={"dob": "1985-07-12"},
                    )
                )
            ]
        ),
        invocation_id="unused",
    )
    intermediate_turn_complete = Event(
        author="dob_verifier_agent",
        invocation_id="unused",
        turn_complete=True,
    )
    # The real end-of-turn after the model continues.
    final_event = Event(
        author="dob_verifier_agent",
        content=types.Content(
            parts=[types.Part(text="Your identity is verified.")]
        ),
        invocation_id="unused",
        turn_complete=True,
    )

    async def mock_run_live(*args, **kwargs):
      yield fc_event
      assert not live_session.turn_complete_event.is_set()
      yield intermediate_turn_complete
      assert not live_session.turn_complete_event.is_set()
      yield final_event

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="test_user",
        session_id="test_session",
    )
    mocker.patch.object(
        live_session,
        "_record_node_app_details",
        new=mocker.AsyncMock(return_value={}),
    )

    await live_session._consume_node_events()

    # The waiter is released only on the true end-of-turn.
    assert live_session.turn_complete_event.is_set()

  @pytest.mark.asyncio
  async def test_workflow_root_finish_task_does_not_swallow_next_turn(
      self, mocker
  ):
    from google.adk.workflow import Workflow

    # `finish_task` hands off to the next agent, so its following `turn_complete`
    # (the next agent's question) is a real end-of-turn and must not be swallowed
    # -- otherwise the harness hangs.
    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow

    finish_task_event = Event(
        author="greeter_agent",
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="finish_task", args={"result": "John Doe"}
                    )
                )
            ]
        ),
        invocation_id="unused",
    )
    # The next agent takes over and asks its first question, ending the turn.
    next_agent_turn = Event(
        author="dob_verifier_agent",
        content=types.Content(
            parts=[types.Part(text="What is your date of birth?")]
        ),
        invocation_id="unused",
        turn_complete=True,
    )

    async def mock_run_live(*args, **kwargs):
      yield finish_task_event
      yield next_agent_turn

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="test_user",
        session_id="test_session",
    )
    mocker.patch.object(
        live_session,
        "_record_node_app_details",
        new=mocker.AsyncMock(return_value={}),
    )

    await live_session._consume_node_events()

    # The next agent's turn_complete must release the waiter.
    assert live_session.turn_complete_event.is_set()

  @pytest.mark.asyncio
  async def test_workflow_root_tolerates_normal_ws_closure(self, mocker):
    from google.adk.workflow import Workflow
    from google.genai import errors

    # The node runner unwraps the end-of-session `1000` closure and re-raises it
    # from `run_live`; the session must treat it as a normal end of stream
    # rather than aborting the eval case.
    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow

    async def mock_run_live(*args, **kwargs):
      raise errors.APIError(1000, {}, None)
      yield  # pragma: no cover - makes this an async generator

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="test_user",
        session_id="test_session",
    )

    # Must not raise; the normal closure is swallowed.
    await live_session._consume_events()

    # The `finally` still flags the stream finished and unblocks waiters.
    assert live_session.live_finished.is_set()
    assert live_session.turn_complete_event.is_set()

  @pytest.mark.asyncio
  async def test_workflow_root_reraises_abnormal_ws_closure(self, mocker):
    from google.adk.workflow import Workflow
    from google.genai import errors

    # A non-1000 closure is a real failure and must propagate.
    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow

    async def mock_run_live(*args, **kwargs):
      raise errors.APIError(1011, {}, None)
      yield  # pragma: no cover - makes this an async generator

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="test_user",
        session_id="test_session",
    )

    with pytest.raises(errors.APIError):
      await live_session._consume_events()

  @pytest.mark.asyncio
  async def test_workflow_fires_after_model_callback_by_author(self, mocker):
    from google.adk.workflow import Workflow

    mock_workflow = mocker.MagicMock(spec=Workflow)
    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow
    mock_runner.plugin_manager.run_after_model_callback = mocker.AsyncMock()

    greeter_event = Event(author="greeter", invocation_id="x")
    other_event = Event(author="unrecorded_agent", invocation_id="x")

    async def mock_run_live(*args, **kwargs):
      yield greeter_event
      yield other_event

    mock_runner.run_live.return_value = mock_run_live()

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="u",
        session_id="s",
    )
    greeter_context = mocker.MagicMock()
    mocker.patch.object(
        live_session,
        "_record_node_app_details",
        new=mocker.AsyncMock(return_value={"greeter": greeter_context}),
    )

    await live_session._consume_node_events()

    # Only the recorded author's events replay after_model_callback, and with
    # that author's callback context.
    mock_runner.plugin_manager.run_after_model_callback.assert_called_once_with(
        callback_context=greeter_context, llm_response=greeter_event
    )


class TestLiveSessionNodeAppDetails:
  """Verifies the stop-gap that records autorater app details for node roots.

  ``_record_app_details_for_agent`` (the per-agent preprocess + callback firing)
  is exercised by ``TestLiveSessionCallbacks``; here we focus on the node graph
  enumeration/mapping/skip logic in ``_record_node_app_details``.
  """

  def _make_agent_node(self, mocker, name: str):
    from google.adk.agents.llm_agent import Agent

    agent = mocker.MagicMock(spec=Agent)
    agent.name = name
    return agent

  def _make_runner(self, mocker, nodes):
    mock_workflow = mocker.MagicMock()
    mock_workflow.graph.nodes = nodes

    mock_runner = mocker.MagicMock()
    mock_runner.agent = mock_workflow
    # `model_copy` carries the target agent onto the per-agent context so
    # `_record_app_details_for_agent` sees the right agent.
    base_ic = mock_runner._new_invocation_context_for_live.return_value
    base_ic.model_copy.side_effect = lambda update: mocker.MagicMock(
        agent=update["agent"]
    )
    return mock_runner

  @pytest.mark.asyncio
  async def test_record_node_app_details_maps_each_agent(self, mocker):
    greeter = self._make_agent_node(mocker, "greeter")
    verifier = self._make_agent_node(mocker, "verifier")
    mock_runner = self._make_runner(mocker, [greeter, verifier])

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="u",
        session_id="s",
    )

    # Return a distinct callback context per recorded agent.
    greeter_context = mocker.sentinel.greeter_context
    verifier_context = mocker.sentinel.verifier_context
    mocker.patch.object(
        live_session,
        "_record_app_details_for_agent",
        new=mocker.AsyncMock(side_effect=[greeter_context, verifier_context]),
    )

    callback_context_by_author = await live_session._record_node_app_details()

    assert callback_context_by_author == {
        "greeter": greeter_context,
        "verifier": verifier_context,
    }

  @pytest.mark.asyncio
  async def test_record_node_app_details_no_graph_returns_empty(self, mocker):
    mock_runner = mocker.MagicMock()
    mock_runner.agent = object()  # no `graph` attribute

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="u",
        session_id="s",
    )

    assert await live_session._record_node_app_details() == {}

  @pytest.mark.asyncio
  async def test_record_node_app_details_skips_failed_agent(self, mocker):
    good = self._make_agent_node(mocker, "good")
    bad = self._make_agent_node(mocker, "bad")
    mock_runner = self._make_runner(mocker, [bad, good])

    live_session = _LiveSession(
        runner=mock_runner,
        session=mocker.MagicMock(),
        user_id="u",
        session_id="s",
    )

    good_context = mocker.sentinel.good_context

    async def record(ic):
      if ic.agent.name == "bad":
        raise RuntimeError("boom")
      return good_context

    mocker.patch.object(
        live_session,
        "_record_app_details_for_agent",
        new=mocker.AsyncMock(side_effect=record),
    )

    # The failing agent is skipped; the healthy one is still recorded.
    callback_context_by_author = await live_session._record_node_app_details()
    assert callback_context_by_author == {"good": good_context}


def test_convert_events_preserves_tool_calls_when_skip_summarization():
  """Regression test for tool calls dropped from invocation_events.

  When an event has skip_summarization=True, is_final_response() returns True
  even if the event contains function calls.  Previously such an event was
  treated as final_event and excluded from invocation_events, causing
  get_all_tool_calls() to return an empty list and tool_trajectory_avg_score
  to always be 0.0 despite matching tool calls.
  """
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.Content(
              parts=[types.Part(text="run a query")], role="user"
          ),
          timestamp=1000.0,
      ),
      Event(
          invocation_id="inv1",
          author="agent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id="call_01",
                          name="execute_sql",
                          args={"project_id": "my-proj", "query": "SELECT 1"},
                      )
                  )
              ]
          ),
          actions=EventActions(skip_summarization=True),
      ),
  ]

  invocations = EvaluationGenerator.convert_events_to_eval_invocations(events)
  assert len(invocations) == 1

  tool_calls = get_all_tool_calls(invocations[0].intermediate_data)
  assert len(tool_calls) == 1
  assert tool_calls[0].name == "execute_sql"
  assert tool_calls[0].args == {"project_id": "my-proj", "query": "SELECT 1"}


class _SpyPlugin(BasePlugin):
  """A user-defined plugin used to assert merge behavior."""

  pass


class TestGenerateInferencesFromRootAgentWithApp:
  """Tests that App.plugins / configs are honored when an App is provided."""

  @pytest.fixture
  def runner_cls(self, mocker):
    """Patches Runner and returns the patched class for kwargs inspection."""
    mock_runner_cls = mocker.patch(
        "google.adk.evaluation.evaluation_generator.Runner"
    )
    mock_runner_instance = mocker.AsyncMock()
    mock_runner_instance.__aenter__.return_value = mock_runner_instance
    mock_runner_cls.return_value = mock_runner_instance
    yield mock_runner_cls

  @pytest.fixture
  def stop_immediately_simulator(self, mocker):
    """Returns a UserSimulator that stops on first call (no inference work)."""
    sim = mocker.MagicMock(spec=UserSimulator)
    sim.get_next_user_message = mocker.AsyncMock(
        return_value=NextUserMessage(
            status=UserSimulatorStatus.STOP_SIGNAL_DETECTED
        )
    )
    return sim

  @pytest.mark.asyncio
  async def test_runner_built_from_app_when_provided(
      self, runner_cls, mock_session_service, stop_immediately_simulator
  ):
    """When `app` is passed, Runner is built with `app=` (merged) instead of `agent=`."""
    root_agent = BaseAgent(name="root_agent")
    user_plugin = _SpyPlugin(name="user_plugin")
    app = App(name="my_app", root_agent=root_agent, plugins=[user_plugin])

    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=root_agent,
        user_simulator=stop_immediately_simulator,
        app=app,
    )

    runner_cls.assert_called_once()
    kwargs = runner_cls.call_args.kwargs
    assert "agent" not in kwargs, (
        "Runner must not receive `agent=` when `app=` is provided "
        "(would raise ValueError)."
    )
    assert "plugins" not in kwargs, (
        "Runner must not receive `plugins=` when `app=` is provided "
        "(would raise ValueError)."
    )
    runner_app = kwargs["app"]
    assert isinstance(runner_app, App)
    plugin_names = [p.name for p in runner_app.plugins]
    assert (
        "user_plugin" in plugin_names
    ), "User plugin must be preserved in the merged App passed to Runner."
    assert "request_intercepter_plugin" in plugin_names
    assert "ensure_retry_options" in plugin_names

  @pytest.mark.asyncio
  async def test_user_app_is_not_mutated(
      self, runner_cls, mock_session_service, stop_immediately_simulator
  ):
    """The user's App instance must not be mutated across eval runs."""
    root_agent = BaseAgent(name="root_agent")
    user_plugin = _SpyPlugin(name="user_plugin")
    app = App(name="my_app", root_agent=root_agent, plugins=[user_plugin])
    original_plugins_id = id(app.plugins)

    for _ in range(3):
      await EvaluationGenerator._generate_inferences_from_root_agent(
          root_agent=root_agent,
          user_simulator=stop_immediately_simulator,
          app=app,
      )

    # The user's App instance must still hold exactly its original plugin set,
    # regardless of how many eval runs reused it.
    assert app.plugins == [user_plugin]
    assert id(app.plugins) == original_plugins_id

  @pytest.mark.asyncio
  async def test_runner_falls_back_to_bare_agent_when_no_app(
      self, runner_cls, mock_session_service, stop_immediately_simulator
  ):
    """When `app` is None, Runner is built with the legacy `agent=`/`plugins=` shape."""
    root_agent = BaseAgent(name="root_agent")

    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=root_agent,
        user_simulator=stop_immediately_simulator,
    )

    runner_cls.assert_called_once()
    kwargs = runner_cls.call_args.kwargs
    assert "app" not in kwargs
    assert kwargs["agent"] is root_agent
    plugin_names = [p.name for p in kwargs["plugins"]]
    assert plugin_names == [
        "request_intercepter_plugin",
        "ensure_retry_options",
    ]

  @pytest.mark.asyncio
  async def test_root_agent_override_propagates_to_merged_app(
      self, runner_cls, mock_session_service, stop_immediately_simulator
  ):
    """If a sub-agent is passed as root_agent, the merged App reflects that."""
    full_root = BaseAgent(name="full_root")
    sub_agent = BaseAgent(name="sub_agent")
    app = App(name="my_app", root_agent=full_root)

    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=sub_agent,
        user_simulator=stop_immediately_simulator,
        app=app,
    )

    runner_app = runner_cls.call_args.kwargs["app"]
    assert runner_app.root_agent is sub_agent
    # User's App must be untouched.
    assert app.root_agent is full_root


# -----------------------------------------------------------------------------
# `generate_responses_from_session` -- replays a recorded session file instead of
# invoking an agent, annotating each eval row with what the session actually did.
# -----------------------------------------------------------------------------


def _write_session_file(tmp_path, events: list[Event]) -> str:
  session = Session(
      id="recorded_session",
      app_name="test_app",
      user_id="test_user",
      events=events,
  )
  session_file = tmp_path / "session.json"
  session_file.write_text(session.model_dump_json())
  return str(session_file)


def _recorded_events() -> list[Event]:
  return [
      _build_event("user", [types.Part(text="Roll a 6 sided dice")], "inv1"),
      _build_event(
          "agent",
          [
              types.Part(
                  function_call=types.FunctionCall(
                      name="roll_die", args={"sides": 6}
                  )
              )
          ],
          "inv1",
      ),
      _build_event("agent", [types.Part(text="I rolled a 4.")], "inv1"),
      _build_event("user", [types.Part(text="Thanks")], "inv2"),
      _build_event("agent", [types.Part(text="You are welcome.")], "inv2"),
  ]


def test_generate_responses_from_session_annotates_rows_from_session(tmp_path):
  """Each eval row gains the tool calls and final text of its invocation."""
  session_path = _write_session_file(tmp_path, _recorded_events())
  eval_dataset = [[
      {"query": "Roll a 6 sided dice"},
      {"query": "Thanks"},
  ]]

  results = EvaluationGenerator.generate_responses_from_session(
      session_path, eval_dataset
  )

  # One result per entry in the eval dataset.
  assert len(results) == 1
  first, second = results[0]
  assert first["actual_tool_use"] == [
      {"tool_name": "roll_die", "tool_input": {"sides": 6}}
  ]
  assert first["response"] == "I rolled a 4."
  # The second invocation used no tools.
  assert second["actual_tool_use"] == []
  assert second["response"] == "You are welcome."


def test_generate_responses_from_session_query_absent_from_session(tmp_path):
  """A query the session never saw yields no tool calls and no response."""
  session_path = _write_session_file(tmp_path, _recorded_events())

  results = EvaluationGenerator.generate_responses_from_session(
      session_path, [[{"query": "Roll a 20 sided dice"}]]
  )

  assert results[0][0]["actual_tool_use"] == []
  assert results[0][0]["response"] is None


def test_generate_responses_from_session_scopes_by_invocation_id(tmp_path):
  """Only events sharing the matched user event's invocation id are used."""
  events = [
      _build_event("user", [types.Part(text="Roll a 6 sided dice")], "inv1"),
      _build_event("agent", [types.Part(text="I rolled a 4.")], "inv1"),
      # A different invocation whose tool call must not leak into inv1.
      _build_event("user", [types.Part(text="Book a flight")], "inv2"),
      _build_event(
          "agent",
          [
              types.Part(
                  function_call=types.FunctionCall(
                      name="book_flight", args={"to": "LAX"}
                  )
              )
          ],
          "inv2",
      ),
  ]
  session_path = _write_session_file(tmp_path, events)

  results = EvaluationGenerator.generate_responses_from_session(
      session_path, [[{"query": "Roll a 6 sided dice"}]]
  )

  assert results[0][0]["actual_tool_use"] == []
  assert results[0][0]["response"] == "I rolled a 4."


_real_open = builtins.open


def _non_utf8_default_open(file, mode="r", *args, **kwargs):
  """Emulates a platform whose default text encoding is not UTF-8.

  Falls back to ASCII when a text-mode open does not specify an encoding, so a
  missing `encoding="utf-8"` argument raises instead of silently depending on
  the host locale (for example cp1252 on Windows).
  """
  if "b" not in mode and "encoding" not in kwargs:
    kwargs["encoding"] = "ascii"
  return _real_open(file, mode, *args, **kwargs)


def test_generate_responses_from_session_reads_non_ascii_with_non_utf8_default(
    tmp_path, mocker
):
  """The session file must be read as UTF-8 regardless of platform locale.

  Session files serialized via `model_dump_json` contain raw (unescaped)
  non-ASCII characters, so reading them without an explicit UTF-8 encoding
  fails on platforms whose default encoding is not UTF-8.
  """
  non_ascii_text = "😀 你好 café"
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              author="user",
              invocation_id="inv1",
              content=types.Content(
                  role="user", parts=[types.Part(text=non_ascii_text)]
              ),
          ),
          Event(
              author="agent",
              invocation_id="inv1",
              content=types.Content(
                  role="model",
                  parts=[types.Part(text="response " + non_ascii_text)],
              ),
          ),
      ],
  )
  session_path = tmp_path / "session.json"
  session_path.write_text(session.model_dump_json(), encoding="utf-8")

  mocker.patch.object(
      evaluation_generator_module, "open", _non_utf8_default_open, create=True
  )

  results = EvaluationGenerator.generate_responses_from_session(
      str(session_path), [[{"query": non_ascii_text}]]
  )

  assert results[0][0]["query"] == non_ascii_text
  assert results[0][0]["response"] == "response " + non_ascii_text
