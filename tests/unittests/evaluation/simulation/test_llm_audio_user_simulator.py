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
from google.adk.evaluation.simulation._llm_audio_user_simulator import _LlmAudioUserSimulator
from google.adk.evaluation.simulation._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.simulation.user_simulator import NextUserMessage
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
]


async def to_async_iter(items):
  for item in items:
    yield item


def _text_message(text: str) -> NextUserMessage:
  """Builds a SUCCESS NextUserMessage carrying a single text part."""
  return NextUserMessage(
      status=Status.SUCCESS,
      user_message=types.Content(parts=[types.Part(text=text)], role="user"),
  )


def _audio_response(mocker, *, data=b"AUDIO_BYTES", mime_type="audio/pcm"):
  """Builds a mock audio LLM response carrying an inline_data audio part."""
  response = mocker.create_autospec(
      types.GenerateContentResponse, instance=True
  )
  response.error_code = None
  response.content = types.Content(
      parts=[
          types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))
      ],
      role="user",
  )
  return response


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
  """A default config carries the expected discriminator and audio defaults."""
  config = LlmAudioUserSimulatorConfig()
  assert config.type == "llm_audio"
  assert config.model == "gemini-2.5-flash"
  assert config.audio_model == "cloud_tts"
  assert config.include_text_with_audio is True


def test_config_custom_instructions_validation():
  """custom_instructions must contain the required Jinja placeholders."""
  config = LlmAudioUserSimulatorConfig(custom_instructions=None)
  assert config.custom_instructions is None

  valid_instructions = (
      "{{ stop_signal }} {{ conversation_plan }} {{ conversation_history }}"
  )
  config = LlmAudioUserSimulatorConfig(custom_instructions=valid_instructions)
  assert config.custom_instructions == valid_instructions

  with pytest.raises(ValidationError):
    LlmAudioUserSimulatorConfig(
        custom_instructions="missing formatting placeholders"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conversation_scenario():
  """Provides a test conversation scenario."""
  return conversation_scenarios.ConversationScenario(
      starting_prompt="Hello", conversation_plan="test plan"
  )


@pytest.fixture
def mock_llms(mocker):
  """Mocks LLMRegistry in both the audio and the (composed) text modules.

  Returns a tuple of (mock_text_llm, mock_audio_llm).
  """
  # Text-generation LLM used by the internal LlmBackedUserSimulator.
  mock_text_registry_cls = mocker.patch(
      "google.adk.evaluation.simulation.llm_backed_user_simulator.LLMRegistry",
      autospec=True,
  )
  mock_text_llm = mocker.MagicMock()
  mock_text_registry_cls.return_value.resolve.return_value.return_value = (
      mock_text_llm
  )

  # Audio-generation LLM resolved directly by _LlmAudioUserSimulator.
  mock_audio_registry_cls = mocker.patch(
      "google.adk.evaluation.simulation._llm_audio_user_simulator.LLMRegistry",
      autospec=True,
  )
  mock_audio_llm = mocker.MagicMock()
  mock_audio_registry_cls.return_value.resolve.return_value.return_value = (
      mock_audio_llm
  )
  return mock_text_llm, mock_audio_llm


@pytest.fixture
def simulator(mock_llms, conversation_scenario):
  """Provides an _LlmAudioUserSimulator wrapping a text simulator."""
  config = LlmAudioUserSimulatorConfig(
      model="test-model", audio_model="test-audio-model"
  )
  text_simulator = LlmBackedUserSimulator(
      config=LlmBackedUserSimulatorConfig(model="test-model"),
      conversation_scenario=conversation_scenario,
  )
  return _LlmAudioUserSimulator(config=config, text_simulator=text_simulator)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_composes_text_simulator_and_audio_llm(simulator, mock_llms):
  """__init__ stores the injected text simulator and resolves the audio LLM."""
  _, mock_audio_llm = mock_llms
  assert isinstance(simulator._text_simulator, LlmBackedUserSimulator)
  assert simulator._audio_llm is mock_audio_llm


def test_init_with_static_user_simulator(mock_llms):
  """The audio decorator can wrap a StaticUserSimulator."""
  static_conversation = [
      types.Content(parts=[types.Part(text="Hello!")], role="user"),
  ]
  config = LlmAudioUserSimulatorConfig(audio_model="test-audio-model")
  text_simulator = StaticUserSimulator(static_conversation=static_conversation)
  simulator = _LlmAudioUserSimulator(
      config=config, text_simulator=text_simulator
  )
  assert isinstance(simulator._text_simulator, StaticUserSimulator)
  assert simulator._text_simulator.static_conversation == static_conversation


# ---------------------------------------------------------------------------
# get_next_user_message
# ---------------------------------------------------------------------------


class TestGetNextUserMessage:
  """Test cases for _LlmAudioUserSimulator.get_next_user_message."""

  @pytest.mark.asyncio
  async def test_success_with_text_and_audio(self, simulator, mocker):
    """A successful turn returns both a text part and an audio part."""
    simulator._text_simulator.get_next_user_message = mocker.AsyncMock(
        return_value=_text_message("Book me a flight.")
    )
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [_audio_response(mocker, data=b"WAV", mime_type="audio/pcm")]
    )

    result = await simulator.get_next_user_message(events=_INPUT_EVENTS)

    assert result.status == Status.SUCCESS
    assert result.user_message.role == "user"
    assert len(result.user_message.parts) == 2
    assert result.user_message.parts[0].text == "Book me a flight."
    assert result.user_message.parts[1].inline_data.data == b"WAV"
    assert result.user_message.parts[1].inline_data.mime_type == "audio/pcm"

  @pytest.mark.asyncio
  async def test_success_audio_only(
      self, mock_llms, conversation_scenario, mocker
  ):
    """With include_text_with_audio=False, only the audio part is returned."""
    config = LlmAudioUserSimulatorConfig(
        model="test-model",
        audio_model="test-audio-model",
        include_text_with_audio=False,
    )
    text_simulator = LlmBackedUserSimulator(
        config=LlmBackedUserSimulatorConfig(model="test-model"),
        conversation_scenario=conversation_scenario,
    )
    simulator = _LlmAudioUserSimulator(
        config=config, text_simulator=text_simulator
    )
    simulator._text_simulator.get_next_user_message = mocker.AsyncMock(
        return_value=_text_message("Book me a flight.")
    )
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [_audio_response(mocker, data=b"WAV")]
    )

    result = await simulator.get_next_user_message(events=_INPUT_EVENTS)

    assert result.status == Status.SUCCESS
    assert len(result.user_message.parts) == 1
    assert result.user_message.parts[0].inline_data.data == b"WAV"

  @pytest.mark.asyncio
  async def test_passthrough_non_success_status(self, simulator, mocker):
    """A non-SUCCESS text result is returned unchanged; audio is not called."""
    text_result = NextUserMessage(status=Status.TURN_LIMIT_REACHED)
    simulator._text_simulator.get_next_user_message = mocker.AsyncMock(
        return_value=text_result
    )

    result = await simulator.get_next_user_message(events=_INPUT_EVENTS)

    assert result is text_result
    simulator._audio_llm.generate_content_async.assert_not_called()

  @pytest.mark.asyncio
  async def test_passthrough_empty_text(self, simulator, mocker):
    """A SUCCESS result with no text is returned unchanged; audio not called."""
    text_result = NextUserMessage(
        status=Status.SUCCESS,
        user_message=types.Content(parts=[], role="user"),
    )
    simulator._text_simulator.get_next_user_message = mocker.AsyncMock(
        return_value=text_result
    )

    result = await simulator.get_next_user_message(events=_INPUT_EVENTS)

    assert result is text_result
    simulator._audio_llm.generate_content_async.assert_not_called()


# ---------------------------------------------------------------------------
# _generate_audio
# ---------------------------------------------------------------------------


class TestGenerateAudio:
  """Test cases for _LlmAudioUserSimulator._generate_audio."""

  @pytest.mark.asyncio
  async def test_returns_bytes_and_mime_type(self, simulator, mocker):
    """Audio bytes and mime type are aggregated from the response stream."""
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [_audio_response(mocker, data=b"HELLO", mime_type="audio/wav")]
    )

    audio_bytes, mime_type = await simulator._generate_audio("hello")

    assert audio_bytes == b"HELLO"
    assert mime_type == "audio/wav"

  @pytest.mark.asyncio
  async def test_error_code_raises(self, simulator, mocker):
    """An error_code in the response surfaces as a RuntimeError."""
    response = mocker.create_autospec(
        types.GenerateContentResponse, instance=True
    )
    response.error_code = "SAFETY"
    response.error_message = "blocked"
    response.content = None
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [response]
    )

    with pytest.raises(RuntimeError, match="Audio generation failed: SAFETY"):
      await simulator._generate_audio("hello")

  @pytest.mark.asyncio
  async def test_no_audio_data_raises(self, simulator, mocker):
    """A response with no inline audio data raises a RuntimeError."""
    response = mocker.create_autospec(
        types.GenerateContentResponse, instance=True
    )
    response.error_code = None
    response.content = types.Content(
        parts=[types.Part(text="not audio")], role="user"
    )
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [response]
    )

    with pytest.raises(
        RuntimeError, match="Audio model returned no audio data"
    ):
      await simulator._generate_audio("hello")


# ---------------------------------------------------------------------------
# to_audio_content
# ---------------------------------------------------------------------------


class TestToAudioContent:
  """Test cases for _LlmAudioUserSimulator.to_audio_content."""

  @pytest.mark.asyncio
  async def test_to_audio_content(self, simulator, mocker):
    """to_audio_content converts text into a text+audio user Content."""
    simulator._audio_llm.generate_content_async.return_value = to_async_iter(
        [_audio_response(mocker, data=b"WAV", mime_type="audio/pcm")]
    )

    content = await simulator.to_audio_content("Hello there")

    assert content.role == "user"
    assert len(content.parts) == 2
    assert content.parts[0].text == "Hello there"
    assert content.parts[1].inline_data.data == b"WAV"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_get_simulation_evaluator_not_implemented(simulator):
  """get_simulation_evaluator is not implemented for this simulator."""
  with pytest.raises(NotImplementedError):
    simulator.get_simulation_evaluator()
