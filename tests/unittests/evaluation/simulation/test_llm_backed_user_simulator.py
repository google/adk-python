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

from google.adk.evaluation import conversation_scenarios
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.user_simulator import Status
from google.adk.events.event import Event
from google.genai import types
from pydantic import ValidationError
import pytest

_INPUT_EVENTS = [
    Event(
        author="user",
        content=types.Content(
            parts=[types.Part(text="Can you help me?")], role="user"
        ),
        invocation_id="inv1",
    ),
    Event(
        author="helpful_assistant",
        content=types.Content(
            parts=[
                types.Part(
                    text="I'll get the user's name and greet them first.",
                    thought=True,
                ),
                types.Part(
                    function_call=types.FunctionCall(name="get_user_name")
                ),
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_user_name",
                        response={"name": "John Doe"},
                    )
                ),
                types.Part(text="Hi John, what can I do for you?"),
            ],
            role="model",
        ),
        invocation_id="inv1",
    ),
]

_INPUT_EVENTS_LONG = _INPUT_EVENTS + [
    Event(
        author="user",
        content=types.Content(
            parts=[types.Part(text="I need to book a flight.")], role="user"
        ),
        invocation_id="inv2",
    ),
    Event(
        author="helpful_assistant",
        content=types.Content(
            parts=[
                types.Part(
                    text="Sure, what is your departure date and destination?",
                ),
            ],
            role="model",
        ),
        invocation_id="inv2",
    ),
]

_EXPECTED_REWRITTEN_DIALOGUE = """user: Can you help me?

helpful_assistant: Hi John, what can I do for you?"""

_EXPECTED_REWRITTEN_DIALOGUE_LONG = _EXPECTED_REWRITTEN_DIALOGUE + """

user: I need to book a flight.

helpful_assistant: Sure, what is your departure date and destination?"""


def test_llm_backed_user_simulator_config_validation():
  """Tests for LlmBackedUserSimulatorConfig."""
  config = LlmBackedUserSimulatorConfig(custom_instructions=None)
  assert config.custom_instructions is None
  valid_instructions = (
      "{stop_signal} {conversation_plan} {conversation_history}"
  )
  config = LlmBackedUserSimulatorConfig(custom_instructions=valid_instructions)
  assert config.custom_instructions == valid_instructions
  invalid_instructions = "Instructions with missing formatting placeholders"
  with pytest.raises(ValidationError):
    LlmBackedUserSimulatorConfig(custom_instructions=invalid_instructions)


class TestHelperMethods:
  """Test cases for LlmBackedUserSimulator helper methods."""

  def test_convert_conversation_to_user_sim_pov(self):
    """Tests _convert_conversation_to_user_sim_pov method."""
    rewritten_dialogue = LlmBackedUserSimulator._summarize_conversation(
        _INPUT_EVENTS
    )
    assert rewritten_dialogue == _EXPECTED_REWRITTEN_DIALOGUE
    rewritten_dialogue = LlmBackedUserSimulator._summarize_conversation(
        _INPUT_EVENTS_LONG
    )
    assert rewritten_dialogue == _EXPECTED_REWRITTEN_DIALOGUE_LONG


async def to_async_iter(items):
  for item in items:
    yield item


@pytest.fixture
def mock_llm_agent(mocker):
  """Provides a mock LLM agent."""
  mock_llm_registry_cls = mocker.patch(
      "google.adk.evaluation.simulation.llm_backed_user_simulator.LLMRegistry"
  )
  mock_llm_registry = mocker.MagicMock()
  mock_llm_registry_cls.return_value = mock_llm_registry
  mock_agent = mocker.MagicMock()
  mock_llm_registry.resolve.return_value.return_value = mock_agent
  return mock_agent


@pytest.fixture
def conversation_scenario():
  """Provides a test conversation scenario."""
  return conversation_scenarios.ConversationScenario(
      starting_prompt="Hello", conversation_plan="test plan"
  )


@pytest.fixture
def simulator(mock_llm_agent, conversation_scenario):
  """Provides an LlmBackedUserSimulator instance for testing."""
  config = LlmBackedUserSimulatorConfig(
      model="test-model",
      model_configuration=types.GenerateContentConfig(),
  )
  sim = LlmBackedUserSimulator(
      config=config, conversation_scenario=conversation_scenario
  )
  sim._invocation_count = 1  # Bypass starting prompt by default for tests
  return sim


class TestLlmBackedUserSimulator:
  """Test cases for LlmBackedUserSimulator main methods."""

  @pytest.mark.asyncio
  async def test_get_llm_response_return_value(
      self, simulator, mock_llm_agent, mocker
  ):
    """Tests that _get_llm_response returns the full response correctly."""
    mock_llm_response = mocker.MagicMock()
    mock_llm_response.content = types.Content(
        parts=[
            types.Part(text="some thought", thought=True),
            types.Part(text="Hello world!"),
        ]
    )
    mock_llm_agent.generate_content_async.return_value = to_async_iter(
        [mock_llm_response]
    )
    response = await simulator._get_llm_response(rewritten_dialogue="")
    assert response == "Hello world!"

  @pytest.mark.asyncio
  async def test_get_next_user_message_first_invocation(
      self, simulator, mock_llm_agent, conversation_scenario
  ):
    """Tests that the first invocation returns the starting prompt."""
    simulator._invocation_count = 0  # override testing default
    next_user_message = await simulator.get_next_user_message(events=[])

    expected_user_message = types.Content(
        parts=[types.Part(text=conversation_scenario.starting_prompt)],
        role="user",
    )
    assert next_user_message.status == Status.SUCCESS
    assert next_user_message.user_message == expected_user_message
    mock_llm_agent.generate_content_async.assert_not_called()

  @pytest.mark.asyncio
  async def test_turn_limit_reached(self, conversation_scenario):
    """Tests get_next_user_message when the turn limit is reached."""
    config = LlmBackedUserSimulatorConfig(
        max_allowed_invocations=1,
    )
    simulator = LlmBackedUserSimulator(
        config=config, conversation_scenario=conversation_scenario
    )
    simulator._invocation_count = 1

    next_user_message = await simulator.get_next_user_message(
        events=_INPUT_EVENTS
    )

    assert next_user_message.status == Status.TURN_LIMIT_REACHED
    assert next_user_message.user_message is None

  @pytest.mark.asyncio
  async def test_stop_signal_detected(self, simulator, mock_llm_agent, mocker):
    """Tests get_next_user_message when the stop signal is detected."""
    mock_llm_response = mocker.MagicMock()
    mock_llm_response.content = types.Content(
        parts=[types.Part(text="Thanks! Bye!</finished>")]
    )
    mock_llm_agent.generate_content_async.return_value = to_async_iter(
        [mock_llm_response]
    )

    next_user_message = await simulator.get_next_user_message(
        events=_INPUT_EVENTS
    )

    assert next_user_message.status == Status.STOP_SIGNAL_DETECTED
    assert next_user_message.user_message is None

  @pytest.mark.asyncio
  async def test_no_message_generated(self, simulator, mock_llm_agent):
    """Tests get_next_user_message when no message is generated."""
    mock_llm_agent.generate_content_async.return_value = to_async_iter([])

    with pytest.raises(RuntimeError, match="Failed to generate a user message"):
      await simulator.get_next_user_message(events=_INPUT_EVENTS)

  @pytest.mark.asyncio
  async def test_get_next_user_message_success(
      self, simulator, mock_llm_agent, mocker
  ):
    """Tests get_next_user_message when the user message is generated successfully."""
    mock_llm_response = mocker.MagicMock()
    mock_llm_response.content = types.Content(
        parts=[types.Part(text="I need to book a flight.")]
    )
    mock_llm_agent.generate_content_async.return_value = to_async_iter(
        [mock_llm_response]
    )

    next_user_message = await simulator.get_next_user_message(
        events=_INPUT_EVENTS
    )

    expected_user_message = types.Content(
        parts=[types.Part(text="I need to book a flight.")], role="user"
    )

    assert next_user_message.status == Status.SUCCESS
    assert next_user_message.user_message == expected_user_message
