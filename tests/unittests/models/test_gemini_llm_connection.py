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

from unittest import mock

from google.adk.models.gemini_llm_connection import GeminiLlmConnection
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
import pytest

MODEL_VERSION = 'gemini-2.5-pro'


@pytest.fixture
def mock_gemini_session():
  """Mock Gemini session for testing."""
  mock_session = mock.AsyncMock()
  mock_session.session_id = 'test-session-id'
  return mock_session


@pytest.fixture
def gemini_connection(mock_gemini_session):
  """GeminiLlmConnection instance with mocked session."""
  return GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.VERTEX_AI,
      model_version=MODEL_VERSION,
  )


@pytest.fixture
def gemini_api_connection(mock_gemini_session):
  """GeminiLlmConnection instance with mocked session for Gemini API."""
  return GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.GEMINI_API,
      model_version=MODEL_VERSION,
  )


@pytest.fixture
def test_blob():
  """Test blob for audio data."""
  return types.Blob(data=b'\x00\xFF\x00\xFF', mime_type='audio/pcm')


@pytest.mark.asyncio
async def test_send_realtime_default_behavior(
    gemini_connection, mock_gemini_session, test_blob
):
  """Test send_realtime with default automatic_activity_detection value (True)."""
  await gemini_connection.send_realtime(test_blob)

  # Should call send once
  mock_gemini_session.send_realtime_input.assert_called_once_with(
      media=test_blob
  )
  # Should not call .send function
  mock_gemini_session.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_realtime_audio_uses_audio_channel_for_live_translate(
    mock_gemini_session, test_blob
):
  """Live Translate models stream audio via the dedicated `audio=` channel."""
  connection = GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.GEMINI_API,
      model_version='gemini-3.5-live-translate-preview',
  )

  await connection.send_realtime(test_blob)

  mock_gemini_session.send_realtime_input.assert_called_once_with(
      audio=test_blob
  )


@pytest.mark.asyncio
async def test_send_history(gemini_connection, mock_gemini_session):
  """Test send_history method."""
  history = [
      types.Content(role='user', parts=[types.Part.from_text(text='Hello')]),
      types.Content(
          role='model', parts=[types.Part.from_text(text='Hi there!')]
      ),
  ]

  await gemini_connection.send_history(history)

  mock_gemini_session.send_client_content.assert_called_once()
  call_args = mock_gemini_session.send_client_content.call_args[1]
  assert 'turns' in call_args
  assert call_args['turns'] == history
  assert call_args['turn_complete'] is False  # Last message is from model


@pytest.mark.asyncio
async def test_send_content_text(gemini_connection, mock_gemini_session):
  """Test send_content with text content."""
  content = types.Content(
      role='user', parts=[types.Part.from_text(text='Hello')]
  )

  await gemini_connection.send_content(content)

  mock_gemini_session.send.assert_called_once()
  call_args = mock_gemini_session.send.call_args[1]
  assert 'input' in call_args
  assert call_args['input'].turns == [content]
  assert call_args['input'].turn_complete is True


@pytest.mark.asyncio
async def test_send_content_function_response(
    gemini_connection, mock_gemini_session
):
  """Test send_content with function response."""
  function_response = types.FunctionResponse(
      name='test_function', response={'result': 'success'}
  )
  content = types.Content(
      role='user', parts=[types.Part(function_response=function_response)]
  )

  await gemini_connection.send_content(content)

  mock_gemini_session.send_tool_response.assert_called_once()
  call_args = mock_gemini_session.send_tool_response.call_args[1]
  assert 'function_responses' in call_args
  assert call_args['function_responses'] == [function_response]


@pytest.mark.asyncio
async def test_close(gemini_connection, mock_gemini_session):
  """Test close method."""
  await gemini_connection.close()

  mock_gemini_session.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('tx_direction', ['input', 'output'])
async def test_receive_transcript_finished(
    gemini_connection, mock_gemini_session, tx_direction
):
  """Test receive_transcript_finished for input and output transcription."""

  finished_tx = types.Transcription(finished=True)

  msg = mock.Mock()
  msg.tool_call = None
  msg.usage_metadata = None
  msg.session_resumption_update = None
  msg.go_away = None
  msg.server_content.model_turn = None
  msg.server_content.interrupted = False
  msg.server_content.turn_complete = False
  msg.server_content.input_transcription = (
      finished_tx if tx_direction == 'input' else None
  )
  msg.server_content.output_transcription = (
      finished_tx if tx_direction == 'output' else None
  )
  msg.server_content.grounding_metadata = None

  async def gen():
    yield msg

  mock_gemini_session.receive = mock.Mock(return_value=gen())

  responses = []
  async for r in gemini_connection.receive():
    responses.append(r)

  attr_name = f'{tx_direction}_transcription'
  tx_resps = [r for r in responses if getattr(r, attr_name)]
  assert tx_resps, f'Expected {tx_direction} transcription response'

  transcription = getattr(tx_resps[0], attr_name)
  assert transcription.finished is True
  assert not transcription.text


async def test_receive_usage_metadata_and_server_content(
    gemini_connection, mock_gemini_session
):
  """Test receive with usage metadata and server content in one message."""
  usage_metadata = types.UsageMetadata(
      prompt_token_count=10,
      cached_content_token_count=5,
      response_token_count=20,
      total_token_count=35,
      thoughts_token_count=2,
      prompt_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=10)
      ],
      cache_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=5)
      ],
      response_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=20)
      ],
  )
  mock_content = types.Content(
      role='model', parts=[types.Part.from_text(text='response text')]
  )
  mock_server_content = mock.Mock()
  mock_server_content.model_turn = mock_content
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.turn_complete = False
  mock_server_content.grounding_metadata = None

  mock_message = mock.AsyncMock()
  mock_message.usage_metadata = usage_metadata
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert responses

  usage_response = next((r for r in responses if r.usage_metadata), None)
  assert usage_response is not None
  assert usage_response.model_version == MODEL_VERSION
  content_response = next((r for r in responses if r.content), None)
  assert content_response is not None

  # The live API's `response_token_count`/`response_tokens_details` are remapped
  # to `candidates_token_count`/`candidates_tokens_details`.
  expected_usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=10,
      cached_content_token_count=5,
      candidates_token_count=20,
      total_token_count=35,
      thoughts_token_count=2,
      prompt_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=10)
      ],
      cache_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=5)
      ],
      candidates_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=20)
      ],
  )
  assert usage_response.usage_metadata == expected_usage
  assert content_response.content == mock_content


async def test_receive_usage_metadata_remaps_output_tokens(
    gemini_connection, mock_gemini_session
):
  """Test that live API output tokens are remapped to candidates_token_count."""
  usage_metadata = types.UsageMetadata(
      prompt_token_count=10,
      cached_content_token_count=5,
      response_token_count=20,
      total_token_count=35,
      thoughts_token_count=2,
      tool_use_prompt_token_count=3,
      prompt_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=10)
      ],
      cache_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=5)
      ],
      response_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=20)
      ],
  )

  mock_message = mock.AsyncMock()
  mock_message.usage_metadata = usage_metadata
  mock_message.server_content = None
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  usage_response = next((r for r in responses if r.usage_metadata), None)
  assert usage_response is not None
  result = usage_response.usage_metadata
  assert isinstance(result, types.GenerateContentResponseUsageMetadata)
  # Output tokens are remapped from response_* to candidates_*.
  assert result.candidates_token_count == 20
  assert result.candidates_tokens_details == [
      types.ModalityTokenCount(modality='text', token_count=20)
  ]
  # Shared fields are carried over unchanged.
  assert result.prompt_token_count == 10
  assert result.cached_content_token_count == 5
  assert result.total_token_count == 35
  assert result.thoughts_token_count == 2
  assert result.tool_use_prompt_token_count == 3
  assert result.prompt_tokens_details == [
      types.ModalityTokenCount(modality='text', token_count=10)
  ]
  assert result.cache_tokens_details == [
      types.ModalityTokenCount(modality='text', token_count=5)
  ]


async def test_receive_populates_live_session_id(
    gemini_connection, mock_gemini_session
):
  """Test that receive populates live_session_id in LlmResponse."""
  mock_message = mock.AsyncMock()
  mock_message.usage_metadata = None
  mock_message.server_content = None
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  mock_server_content = mock.Mock()
  mock_server_content.model_turn = types.Content(
      role='model', parts=[types.Part.from_text(text='text')]
  )
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.turn_complete = False
  mock_server_content.grounding_metadata = None

  mock_message.server_content = mock_server_content

  async def mock_receive_generator():
    yield mock_message

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in gemini_connection.receive()]

  assert responses
  for resp in responses:
    assert resp.live_session_id == 'test-session-id'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'conn_fixture',
    ['gemini_api_connection', 'gemini_connection'],
)
async def test_receive_transcript_finished_on_interrupt(
    conn_fixture,
    mock_gemini_session,
    request,
):
  """Test receive finishes transcription on interrupt signal."""
  connection = request.getfixturevalue(conn_fixture)

  message1 = mock.Mock()
  message1.usage_metadata = None
  message1.server_content = mock.Mock()
  message1.server_content.model_turn = None
  message1.server_content.interrupted = False
  message1.server_content.input_transcription = types.Transcription(
      text='Hello', finished=False
  )
  message1.server_content.output_transcription = None
  message1.server_content.turn_complete = False
  message1.server_content.generation_complete = False
  message1.server_content.grounding_metadata = None
  message1.tool_call = None
  message1.session_resumption_update = None
  message1.go_away = None

  message2 = mock.Mock()
  message2.usage_metadata = None
  message2.server_content = mock.Mock()
  message2.server_content.model_turn = None
  message2.server_content.interrupted = False
  message2.server_content.input_transcription = None
  message2.server_content.output_transcription = types.Transcription(
      text='How can', finished=False
  )
  message2.server_content.turn_complete = False
  message2.server_content.generation_complete = False
  message2.server_content.grounding_metadata = None
  message2.tool_call = None
  message2.session_resumption_update = None
  message2.go_away = None

  message3 = mock.Mock()
  message3.usage_metadata = None
  message3.server_content = mock.Mock()
  message3.server_content.model_turn = None
  message3.server_content.interrupted = True
  message3.server_content.input_transcription = None
  message3.server_content.output_transcription = None
  message3.server_content.turn_complete = False
  message3.server_content.generation_complete = False
  message3.server_content.grounding_metadata = None
  message3.tool_call = None
  message3.session_resumption_update = None
  message3.go_away = None

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in connection.receive()]

  assert len(responses) == 5
  assert responses[4].interrupted is True

  assert responses[0].input_transcription.text == 'Hello'
  assert responses[0].input_transcription.finished is False
  assert responses[0].partial is True
  assert responses[1].output_transcription.text == 'How can'
  assert responses[1].output_transcription.finished is False
  assert responses[1].partial is True
  assert responses[2].input_transcription.text == 'Hello'
  assert responses[2].input_transcription.finished is True
  assert responses[2].partial is False
  assert responses[3].output_transcription.text == 'How can'
  assert responses[3].output_transcription.finished is True
  assert responses[3].partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'conn_fixture',
    ['gemini_api_connection', 'gemini_connection'],
)
async def test_receive_transcript_finished_on_generation_complete(
    conn_fixture,
    mock_gemini_session,
    request,
):
  """Test receive finishes transcription on generation_complete signal."""
  connection = request.getfixturevalue(conn_fixture)

  message1 = mock.Mock()
  message1.usage_metadata = None
  message1.server_content = mock.Mock()
  message1.server_content.model_turn = None
  message1.server_content.interrupted = False
  message1.server_content.input_transcription = types.Transcription(
      text='Hello', finished=False
  )
  message1.server_content.output_transcription = None
  message1.server_content.turn_complete = False
  message1.server_content.generation_complete = False
  message1.server_content.grounding_metadata = None
  message1.tool_call = None
  message1.session_resumption_update = None
  message1.go_away = None

  message2 = mock.Mock()
  message2.usage_metadata = None
  message2.server_content = mock.Mock()
  message2.server_content.model_turn = None
  message2.server_content.interrupted = False
  message2.server_content.input_transcription = None
  message2.server_content.output_transcription = types.Transcription(
      text='How can', finished=False
  )
  message2.server_content.turn_complete = False
  message2.server_content.generation_complete = False
  message2.server_content.grounding_metadata = None
  message2.tool_call = None
  message2.session_resumption_update = None
  message2.go_away = None

  message3 = mock.Mock()
  message3.usage_metadata = None
  message3.server_content = mock.Mock()
  message3.server_content.model_turn = None
  message3.server_content.interrupted = False
  message3.server_content.input_transcription = None
  message3.server_content.output_transcription = None
  message3.server_content.turn_complete = False
  message3.server_content.generation_complete = True
  message3.server_content.grounding_metadata = None
  message3.tool_call = None
  message3.session_resumption_update = None
  message3.go_away = None

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in connection.receive()]

  assert len(responses) == 4

  assert responses[0].input_transcription.text == 'Hello'
  assert responses[0].input_transcription.finished is False
  assert responses[0].partial is True
  assert responses[1].output_transcription.text == 'How can'
  assert responses[1].output_transcription.finished is False
  assert responses[1].partial is True
  assert responses[2].input_transcription.text == 'Hello'
  assert responses[2].input_transcription.finished is True
  assert responses[2].partial is False
  assert responses[3].output_transcription.text == 'How can'
  assert responses[3].output_transcription.finished is True
  assert responses[3].partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'conn_fixture',
    ['gemini_api_connection', 'gemini_connection'],
)
async def test_receive_transcript_finished_on_turn_complete(
    conn_fixture,
    mock_gemini_session,
    request,
):
  """Test receive finishes transcription on interrupt or complete signals."""
  connection = request.getfixturevalue(conn_fixture)

  message1 = mock.Mock()
  message1.usage_metadata = None
  message1.server_content = mock.Mock()
  message1.server_content.model_turn = None
  message1.server_content.interrupted = False
  message1.server_content.input_transcription = types.Transcription(
      text='Hello', finished=False
  )
  message1.server_content.output_transcription = None
  message1.server_content.turn_complete = False
  message1.server_content.generation_complete = False
  message1.server_content.grounding_metadata = None
  message1.tool_call = None
  message1.session_resumption_update = None
  message1.go_away = None

  message2 = mock.Mock()
  message2.usage_metadata = None
  message2.server_content = mock.Mock()
  message2.server_content.model_turn = None
  message2.server_content.interrupted = False
  message2.server_content.input_transcription = None
  message2.server_content.output_transcription = types.Transcription(
      text='How can', finished=False
  )
  message2.server_content.turn_complete = False
  message2.server_content.generation_complete = False
  message2.server_content.grounding_metadata = None
  message2.tool_call = None
  message2.session_resumption_update = None
  message2.go_away = None

  message3 = mock.Mock()
  message3.usage_metadata = None
  message3.server_content = mock.Mock()
  message3.server_content.model_turn = None
  message3.server_content.interrupted = False
  message3.server_content.input_transcription = None
  message3.server_content.output_transcription = None
  message3.server_content.turn_complete = True
  message3.server_content.generation_complete = False
  message3.server_content.grounding_metadata = None
  message3.tool_call = None
  message3.session_resumption_update = None
  message3.go_away = None

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in connection.receive()]

  assert len(responses) == 5
  assert responses[4].turn_complete is True

  assert responses[0].input_transcription.text == 'Hello'
  assert responses[0].input_transcription.finished is False
  assert responses[0].partial is True
  assert responses[1].output_transcription.text == 'How can'
  assert responses[1].output_transcription.finished is False
  assert responses[1].partial is True
  assert responses[2].input_transcription.text == 'Hello'
  assert responses[2].input_transcription.finished is True
  assert responses[2].partial is False
  assert responses[3].output_transcription.text == 'How can'
  assert responses[3].output_transcription.finished is True
  assert responses[3].partial is False


@pytest.mark.asyncio
async def test_receive_handles_input_transcription_fragments(
    gemini_connection, mock_gemini_session
):
  """Test receive handles input transcription fragments correctly."""
  message1 = mock.Mock()
  message1.usage_metadata = None
  message1.server_content = mock.Mock()
  message1.server_content.model_turn = None
  message1.server_content.interrupted = False
  message1.server_content.input_transcription = types.Transcription(
      text='Hello', finished=False
  )
  message1.server_content.output_transcription = None
  message1.server_content.turn_complete = False
  message1.server_content.generation_complete = False
  message1.server_content.grounding_metadata = None
  message1.tool_call = None
  message1.session_resumption_update = None
  message1.go_away = None

  message2 = mock.Mock()
  message2.usage_metadata = None
  message2.server_content = mock.Mock()
  message2.server_content.model_turn = None
  message2.server_content.interrupted = False
  message2.server_content.input_transcription = types.Transcription(
      text=' world', finished=False
  )
  message2.server_content.output_transcription = None
  message2.server_content.turn_complete = False
  message2.server_content.generation_complete = False
  message2.server_content.grounding_metadata = None
  message2.tool_call = None
  message2.session_resumption_update = None
  message2.go_away = None

  message3 = mock.Mock()
  message3.usage_metadata = None
  message3.server_content = mock.Mock()
  message3.server_content.model_turn = None
  message3.server_content.interrupted = False
  message3.server_content.input_transcription = types.Transcription(
      text=None, finished=True
  )
  message3.server_content.output_transcription = None
  message3.server_content.turn_complete = False
  message3.server_content.generation_complete = False
  message3.server_content.grounding_metadata = None
  message3.tool_call = None
  message3.session_resumption_update = None
  message3.go_away = None

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 3
  assert responses[0].input_transcription.text == 'Hello'
  assert responses[0].input_transcription.finished is False
  assert responses[0].partial is True
  assert responses[1].input_transcription.text == ' world'
  assert responses[1].input_transcription.finished is False
  assert responses[1].partial is True
  assert responses[2].input_transcription.text == 'Hello world'
  assert responses[2].input_transcription.finished is True
  assert responses[2].partial is False


@pytest.mark.asyncio
async def test_receive_handles_output_transcription_fragments(
    gemini_connection, mock_gemini_session
):
  """Test receive handles output transcription fragments correctly."""
  message1 = mock.Mock()
  message1.usage_metadata = None
  message1.server_content = mock.Mock()
  message1.server_content.model_turn = None
  message1.server_content.interrupted = False
  message1.server_content.input_transcription = None
  message1.server_content.output_transcription = types.Transcription(
      text='How can', finished=False
  )
  message1.server_content.turn_complete = False
  message1.server_content.generation_complete = False
  message1.server_content.grounding_metadata = None
  message1.tool_call = None
  message1.session_resumption_update = None
  message1.go_away = None

  message2 = mock.Mock()
  message2.usage_metadata = None
  message2.server_content = mock.Mock()
  message2.server_content.model_turn = None
  message2.server_content.interrupted = False
  message2.server_content.input_transcription = None
  message2.server_content.output_transcription = types.Transcription(
      text=' I help?', finished=False
  )
  message2.server_content.turn_complete = False
  message2.server_content.generation_complete = False
  message2.server_content.grounding_metadata = None
  message2.tool_call = None
  message2.session_resumption_update = None
  message2.go_away = None

  message3 = mock.Mock()
  message3.usage_metadata = None
  message3.server_content = mock.Mock()
  message3.server_content.model_turn = None
  message3.server_content.interrupted = False
  message3.server_content.input_transcription = None
  message3.server_content.output_transcription = types.Transcription(
      text=None, finished=True
  )
  message3.server_content.turn_complete = False
  message3.server_content.generation_complete = False
  message3.server_content.grounding_metadata = None
  message3.tool_call = None
  message3.session_resumption_update = None
  message3.go_away = None

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 3
  assert responses[0].output_transcription.text == 'How can'
  assert responses[0].output_transcription.finished is False
  assert responses[0].partial is True
  assert responses[1].output_transcription.text == ' I help?'
  assert responses[1].output_transcription.finished is False
  assert responses[1].partial is True
  assert responses[2].output_transcription.text == 'How can I help?'
  assert responses[2].output_transcription.finished is True
  assert responses[2].partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'audio_part',
    [
        types.Part(
            inline_data=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
        ),
        types.Part(
            file_data=types.FileData(
                file_uri='artifact://app/user/session/_adk_live/audio.pcm#1',
                mime_type='audio/pcm',
            )
        ),
    ],
)
async def test_send_history_filters_audio(mock_gemini_session, audio_part):
  """Test that audio parts (inline or file_data) are filtered out."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )
  history = [
      types.Content(
          role='user',
          parts=[audio_part],
      ),
      types.Content(
          role='model', parts=[types.Part.from_text(text='I heard you')]
      ),
  ]

  await connection.send_history(history)

  mock_gemini_session.send_client_content.assert_called_once()
  call_args = mock_gemini_session.send_client_content.call_args[1]
  sent_contents = call_args['turns']
  # Only the model response should be sent (user audio filtered out)
  assert len(sent_contents) == 1
  assert sent_contents[0].role == 'model'
  assert sent_contents[0].parts == [types.Part.from_text(text='I heard you')]


@pytest.mark.asyncio
async def test_send_history_keeps_image_data(mock_gemini_session):
  """Test that image data is NOT filtered out."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )
  image_blob = types.Blob(data=b'\x89PNG\r\n', mime_type='image/png')
  history = [
      types.Content(
          role='user',
          parts=[types.Part(inline_data=image_blob)],
      ),
      types.Content(
          role='model', parts=[types.Part.from_text(text='Nice image!')]
      ),
  ]

  await connection.send_history(history)

  mock_gemini_session.send_client_content.assert_called_once()
  call_args = mock_gemini_session.send_client_content.call_args[1]
  sent_contents = call_args['turns']
  # Both contents should be sent (image is not filtered)
  assert len(sent_contents) == 2
  assert sent_contents[0].parts[0].inline_data == image_blob


@pytest.mark.asyncio
async def test_send_history_mixed_content_filters_only_audio(
    mock_gemini_session,
):
  """Test that mixed content keeps non-audio parts."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )
  history = [
      types.Content(
          role='user',
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      data=b'\x00\xFF', mime_type='audio/wav'
                  )
              ),
              types.Part.from_text(text='transcribed text'),
          ],
      ),
  ]

  await connection.send_history(history)

  mock_gemini_session.send_client_content.assert_called_once()
  call_args = mock_gemini_session.send_client_content.call_args[1]
  sent_contents = call_args['turns']
  # Content should be sent but only with the text part
  assert len(sent_contents) == 1
  assert len(sent_contents[0].parts) == 1
  assert sent_contents[0].parts[0].text == 'transcribed text'


@pytest.mark.asyncio
async def test_send_history_all_audio_content_not_sent(mock_gemini_session):
  """Test that content with only audio parts is completely removed."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )
  history = [
      types.Content(
          role='user',
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      data=b'\x00\xFF', mime_type='audio/pcm'
                  )
              ),
              types.Part(
                  file_data=types.FileData(
                      file_uri='artifact://audio.pcm#1',
                      mime_type='audio/wav',
                  )
              ),
          ],
      ),
  ]

  await connection.send_history(history)

  # No content should be sent since all parts are audio
  mock_gemini_session.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_history_empty_history_not_sent(mock_gemini_session):
  """Test that empty history does not call send."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )

  await connection.send_history([])

  mock_gemini_session.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'audio_mime_type',
    ['audio/pcm', 'audio/wav', 'audio/mp3', 'audio/ogg'],
)
async def test_send_history_filters_various_audio_mime_types(
    mock_gemini_session,
    audio_mime_type,
):
  """Test that various audio mime types are all filtered."""
  connection = GeminiLlmConnection(
      mock_gemini_session, api_backend=GoogleLLMVariant.VERTEX_AI
  )
  history = [
      types.Content(
          role='user',
          parts=[
              types.Part(
                  inline_data=types.Blob(data=b'', mime_type=audio_mime_type)
              )
          ],
      ),
  ]

  await connection.send_history(history)

  # No content should be sent since the only part is audio
  mock_gemini_session.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_history_gemini_31_turn_complete(mock_gemini_session):
  """Verify Gemini 3.1 Live history seeding explicitly appends turn_complete=True."""
  conn = GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.GEMINI_API,
      model_version='gemini-3.1-flash-live-preview',
  )
  mock_gemini_session.send_client_content = mock.AsyncMock()

  mock_contents = [
      types.Content(role='user', parts=[types.Part.from_text(text='hi')]),
      types.Content(role='model', parts=[types.Part.from_text(text='hello')]),
  ]
  await conn.send_history(mock_contents)

  mock_gemini_session.send_client_content.assert_called_once_with(
      turns=mock_contents,
      turn_complete=True,
  )


@pytest.mark.asyncio
async def test_send_history_collapse_vertex_ai(mock_gemini_session):
  """Verify history prompt collapse when seeding Gemini 3.1 Live on Vertex AI backend."""
  conn = GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.VERTEX_AI,
      model_version='gemini-3.1-flash-live-preview',
  )
  mock_gemini_session.send_client_content = mock.AsyncMock()

  mock_contents = [
      types.Content(role='user', parts=[types.Part.from_text(text='hi')]),
      types.Content(role='model', parts=[types.Part.from_text(text='hello')]),
  ]
  await conn.send_history(mock_contents)

  assert mock_gemini_session.send_client_content.call_count == 1
  called_turns = mock_gemini_session.send_client_content.call_args.kwargs[
      'turns'
  ]
  assert len(called_turns) == 1
  assert called_turns[0].role == 'user'
  assert 'Previous conversation history:' in called_turns[0].parts[0].text
  assert '[user]: hi' in called_turns[0].parts[0].text
  assert '[model]: hello' in called_turns[0].parts[0].text
  assert (
      mock_gemini_session.send_client_content.call_args.kwargs['turn_complete']
      is True
  )


@pytest.mark.asyncio
async def test_receive_grounding_metadata_standalone(
    gemini_connection, mock_gemini_session
):
  """Test receive handles standalone grounding metadata correctly."""
  grounding_metadata = types.GroundingMetadata(
      web_search_queries=['stock price of google'],
      search_entry_point=types.SearchEntryPoint(
          rendered_content='<p>Google</p>'
      ),
  )
  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = None
  mock_server_content.grounding_metadata = grounding_metadata
  mock_server_content.turn_complete = False
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False

  mock_message = mock.create_autospec(types.LiveServerMessage, instance=True)
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].grounding_metadata == grounding_metadata
  assert responses[0].content is None


@pytest.mark.asyncio
async def test_receive_grounding_metadata_with_content(
    gemini_connection, mock_gemini_session
):
  """Test receive handles grounding metadata attached to regular content."""
  grounding_metadata = types.GroundingMetadata(
      web_search_queries=['stock price of google'],
      search_entry_point=types.SearchEntryPoint(
          rendered_content='<p>Google</p>'
      ),
  )
  mock_content = types.Content(
      role='model', parts=[types.Part.from_text(text='response text')]
  )
  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = mock_content
  mock_server_content.grounding_metadata = grounding_metadata
  mock_server_content.turn_complete = False
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False

  mock_message = mock.create_autospec(types.LiveServerMessage, instance=True)
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].grounding_metadata == grounding_metadata
  assert responses[0].content == mock_content


@pytest.mark.asyncio
async def test_receive_tool_call_and_grounding_metadata_with_native_audio(
    mock_gemini_session,
):
  """Test receive handles tool call followed by grounding metadata."""
  connection = GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.VERTEX_AI,
      model_version='gemini-live-2.5-flash-native-audio',
  )

  # 1. Message with tool call (e.g., enterprise_web_search)
  mock_tool_call_msg = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_tool_call_msg.usage_metadata = None
  mock_tool_call_msg.server_content = None
  mock_tool_call_msg.session_resumption_update = None
  mock_tool_call_msg.go_away = None

  function_call = types.FunctionCall(
      name='enterprise_web_search',
      args={'query': 'Google stock price today'},
  )
  mock_tool_call = mock.create_autospec(types.LiveServerToolCall, instance=True)
  mock_tool_call.function_calls = [function_call]
  mock_tool_call_msg.tool_call = mock_tool_call

  # 2. Message with grounding metadata and audio content (native audio model)
  grounding_metadata = types.GroundingMetadata(
      web_search_queries=['Google stock price today'],
      search_entry_point=types.SearchEntryPoint(
          rendered_content='<p>Google</p>'
      ),
  )
  audio_blob = types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  mock_content = types.Content(
      role='model', parts=[types.Part(inline_data=audio_blob)]
  )

  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = mock_content
  mock_server_content.grounding_metadata = grounding_metadata
  mock_server_content.turn_complete = False
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False

  mock_metadata_msg = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_metadata_msg.usage_metadata = None
  mock_metadata_msg.server_content = mock_server_content
  mock_metadata_msg.tool_call = None
  mock_metadata_msg.session_resumption_update = None
  mock_metadata_msg.go_away = None

  # 3. Message with turn_complete
  mock_turn_complete_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_turn_complete_content.model_turn = None
  mock_turn_complete_content.grounding_metadata = None
  mock_turn_complete_content.turn_complete = True
  mock_turn_complete_content.interrupted = False
  mock_turn_complete_content.input_transcription = None
  mock_turn_complete_content.output_transcription = None
  mock_turn_complete_content.generation_complete = False

  mock_turn_complete_msg = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_turn_complete_msg.usage_metadata = None
  mock_turn_complete_msg.server_content = mock_turn_complete_content
  mock_turn_complete_msg.tool_call = None
  mock_turn_complete_msg.session_resumption_update = None
  mock_turn_complete_msg.go_away = None

  async def mock_receive_generator():
    yield mock_tool_call_msg
    yield mock_metadata_msg
    yield mock_turn_complete_msg

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in connection.receive()]

  assert len(responses) == 3

  # First response: the audio content and grounding metadata
  assert responses[0].grounding_metadata == grounding_metadata
  assert responses[0].content == mock_content
  assert responses[0].content is not None
  assert responses[0].content.parts is not None
  assert responses[0].content.parts[0].inline_data == audio_blob

  # Second response: the tool call, buffered until turn_complete
  assert responses[1].content is not None
  assert responses[1].content.parts is not None
  assert responses[1].content.parts[0].function_call is not None
  assert (
      responses[1].content.parts[0].function_call.name
      == 'enterprise_web_search'
  )
  assert responses[1].content.parts[0].function_call.args == {
      'query': 'Google stock price today'
  }
  assert responses[1].grounding_metadata is None

  # Third response: the turn_complete
  assert responses[2].turn_complete is True


@pytest.mark.asyncio
async def test_receive_multiple_tool_calls_buffered_until_turn_complete(
    gemini_connection, mock_gemini_session
):
  """Test receive buffers multiple tool call messages until turn complete."""
  # First tool call message
  mock_tool_call_msg1 = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_tool_call_msg1.usage_metadata = None
  mock_tool_call_msg1.server_content = None
  mock_tool_call_msg1.session_resumption_update = None
  mock_tool_call_msg1.go_away = None

  function_call1 = types.FunctionCall(
      name='tool_1',
      args={'arg': 'value1'},
  )
  mock_tool_call1 = mock.create_autospec(
      types.LiveServerToolCall, instance=True
  )
  mock_tool_call1.function_calls = [function_call1]
  mock_tool_call_msg1.tool_call = mock_tool_call1

  # Second tool call message
  mock_tool_call_msg2 = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_tool_call_msg2.usage_metadata = None
  mock_tool_call_msg2.server_content = None
  mock_tool_call_msg2.session_resumption_update = None
  mock_tool_call_msg2.go_away = None

  function_call2 = types.FunctionCall(
      name='tool_2',
      args={'arg': 'value2'},
  )
  mock_tool_call2 = mock.create_autospec(
      types.LiveServerToolCall, instance=True
  )
  mock_tool_call2.function_calls = [function_call2]
  mock_tool_call_msg2.tool_call = mock_tool_call2

  # Turn complete message
  mock_turn_complete_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_turn_complete_content.model_turn = None
  mock_turn_complete_content.grounding_metadata = None
  mock_turn_complete_content.turn_complete = True
  mock_turn_complete_content.interrupted = False
  mock_turn_complete_content.input_transcription = None
  mock_turn_complete_content.output_transcription = None

  mock_turn_complete_msg = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_turn_complete_msg.usage_metadata = None
  mock_turn_complete_msg.server_content = mock_turn_complete_content
  mock_turn_complete_msg.tool_call = None
  mock_turn_complete_msg.session_resumption_update = None
  mock_turn_complete_msg.go_away = None

  async def mock_receive_generator():
    yield mock_tool_call_msg1
    yield mock_tool_call_msg2
    yield mock_turn_complete_msg

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  # Expected: One LlmResponse with both tool calls, then one with turn_complete
  assert len(responses) == 2

  # First response: single LlmResponse carrying both function calls
  assert responses[0].content is not None
  parts = responses[0].content.parts
  assert len(parts) == 2
  assert parts[0].function_call.name == 'tool_1'
  assert parts[0].function_call.args == {'arg': 'value1'}
  assert parts[1].function_call.name == 'tool_2'
  assert parts[1].function_call.args == {'arg': 'value2'}

  # Second response: turn_complete True
  assert responses[1].turn_complete is True


@pytest.mark.asyncio
async def test_receive_tool_calls_yielded_immediately_for_gemini_3_1(
    mock_gemini_session,
):
  """Test that tool calls are yielded immediately for Gemini 3.1."""
  connection = GeminiLlmConnection(
      mock_gemini_session,
      api_backend=GoogleLLMVariant.VERTEX_AI,
      model_version='gemini-3.1-flash-live-preview',
  )

  mock_tool_call_msg = mock.create_autospec(
      types.LiveServerMessage, instance=True
  )
  mock_tool_call_msg.usage_metadata = None
  mock_tool_call_msg.server_content = None
  mock_tool_call_msg.session_resumption_update = None
  mock_tool_call_msg.go_away = None

  function_call = types.FunctionCall(
      name='test_tool',
      args={'arg': 'value'},
  )
  mock_tool_call = mock.create_autospec(types.LiveServerToolCall, instance=True)
  mock_tool_call.function_calls = [function_call]
  mock_tool_call_msg.tool_call = mock_tool_call

  async def mock_receive_generator():
    yield mock_tool_call_msg

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = []
  async for resp in connection.receive():
    responses.append(resp)
    break

  assert len(responses) == 1
  assert responses[0].content is not None
  assert responses[0].content.parts[0].function_call.name == 'test_tool'


@pytest.mark.asyncio
async def test_receive_go_away(gemini_connection, mock_gemini_session):
  """Test receive yields go_away message."""
  mock_go_away = types.LiveServerGoAway(timeLeft='10s')
  mock_msg = mock.MagicMock()
  mock_msg.usage_metadata = None
  mock_msg.server_content = None
  mock_msg.tool_call = None
  mock_msg.session_resumption_update = None
  mock_msg.go_away = mock_go_away

  async def mock_receive_generator():
    yield mock_msg

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].go_away == mock_go_away


@pytest.mark.asyncio
async def test_receive_aggregates_thoughts_separately(
    gemini_connection, mock_gemini_session
):
  """Test receive aggregates thoughts and regular text separately."""

  part1 = types.Part.from_text(text='thought 1')
  part1.thought = True
  message1 = types.LiveServerMessage(
      server_content=types.LiveServerContent(
          model_turn=types.Content(role='model', parts=[part1]),
      )
  )

  part2 = types.Part.from_text(text=' thought 2')
  part2.thought = True
  message2 = types.LiveServerMessage(
      server_content=types.LiveServerContent(
          model_turn=types.Content(role='model', parts=[part2]),
      )
  )

  part3 = types.Part.from_text(text='answer')
  part3.thought = False
  message3 = types.LiveServerMessage(
      server_content=types.LiveServerContent(
          model_turn=types.Content(role='model', parts=[part3]),
      )
  )

  message4 = types.LiveServerMessage(
      server_content=types.LiveServerContent(
          turn_complete=True,
      )
  )

  async def mock_receive_generator():
    yield message1
    yield message2
    yield message3
    yield message4

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  # Expected responses:
  # 1. Message 1 (partial thought)
  # 2. Message 2 (partial thought)
  # 3. Aggregated thought (full)
  # 4. Message 3 (partial answer)
  # 5. Aggregated answer (full)
  # 6. Turn complete message

  assert len(responses) == 6

  # Check partials
  assert responses[0].content.parts[0].text == 'thought 1'
  assert responses[0].partial is True
  assert responses[1].content.parts[0].text == ' thought 2'
  assert responses[1].partial is True

  # Check aggregated thought
  assert responses[2].content.parts[0].text == 'thought 1 thought 2'
  assert responses[2].content.parts[0].thought is True
  assert responses[2].partial is False

  # Check partial answer
  assert responses[3].content.parts[0].text == 'answer'
  assert responses[3].partial is True

  # Check aggregated answer
  assert responses[4].content.parts[0].text == 'answer'
  assert not getattr(responses[4].content.parts[0], 'thought', False)
  assert responses[4].partial is False

  # Check turn complete
  assert responses[5].turn_complete is True


@pytest.mark.asyncio
async def test_receive_video_content(gemini_connection, mock_gemini_session):
  """Test receive with video content."""
  mock_content = types.Content(
      role='model',
      parts=[
          types.Part(
              inline_data=types.Blob(data=b'video_data', mime_type='video/mp4')
          )
      ],
  )
  mock_server_content = mock.Mock()
  mock_server_content.model_turn = mock_content
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.turn_complete = False
  mock_server_content.grounding_metadata = None

  mock_message = mock.AsyncMock()
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert responses
  content_response = next((r for r in responses if r.content), None)
  assert content_response is not None
  assert content_response.content == mock_content


@pytest.mark.asyncio
async def test_receive_grounding_metadata_pending(
    gemini_connection, mock_gemini_session
):
  """Test that grounding metadata in partial chunks is pending and yielded on full text."""
  grounding_metadata = types.GroundingMetadata(
      web_search_queries=['stock price of google'],
  )

  def make_msg(text=None, g_meta=None, tc=False):
    msg = mock.Mock(
        usage_metadata=None,
        tool_call=None,
        session_resumption_update=None,
        go_away=None,
    )
    msg.server_content = mock.Mock(
        interrupted=False,
        input_transcription=None,
        output_transcription=None,
        generation_complete=False,
        turn_complete=tc,
        grounding_metadata=g_meta,
        model_turn=types.Content(
            role='model', parts=[types.Part.from_text(text=text)]
        )
        if text
        else None,
    )
    return msg

  msg1 = make_msg(text='hello', g_meta=grounding_metadata)
  msg2 = make_msg(text=' world')
  msg3 = make_msg(tc=True)

  async def gen():
    yield msg1
    yield msg2
    yield msg3

  mock_gemini_session.receive = mock.Mock(return_value=gen())

  responses = [resp async for resp in gemini_connection.receive()]

  # Expected responses:
  # 1. Msg 1 partial (hello) with grounding_metadata
  # 2. Msg 2 partial ( world) without grounding_metadata
  # 3. Full text response (hello world) with PENDING grounding_metadata
  # 4. Turn complete response without grounding_metadata (already cleared)
  assert len(responses) == 4

  assert responses[0].content.parts[0].text == 'hello'
  assert responses[0].partial is True
  assert responses[0].grounding_metadata == grounding_metadata

  assert responses[1].content.parts[0].text == ' world'
  assert responses[1].partial is True
  assert responses[1].grounding_metadata is None

  assert responses[2].content.parts[0].text == 'hello world'
  assert responses[2].partial is False
  assert responses[2].grounding_metadata == grounding_metadata

  assert responses[3].turn_complete is True
  assert responses[3].grounding_metadata is None


@pytest.mark.asyncio
async def test_receive_populates_turn_complete_reason(
    gemini_connection, mock_gemini_session
):
  """Test that receive populates turn_complete_reason in LlmResponse."""
  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = None
  mock_server_content.grounding_metadata = None
  mock_server_content.turn_complete = True
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False
  mock_server_content.turn_complete_reason = (
      types.TurnCompleteReason.RESPONSE_REJECTED
  )

  mock_message = mock.create_autospec(types.LiveServerMessage, instance=True)
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].turn_complete is True
  assert (
      responses[0].turn_complete_reason
      == types.TurnCompleteReason.RESPONSE_REJECTED
  )


@pytest.mark.asyncio
async def test_receive_populates_turn_complete_reason_standalone_grounding(
    gemini_connection, mock_gemini_session
):
  """Test that receive populates turn_complete_reason in LlmResponse for standalone grounding metadata."""
  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = None
  mock_server_content.grounding_metadata = mock.create_autospec(
      types.GroundingMetadata, instance=True
  )
  mock_server_content.turn_complete = False
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False
  mock_server_content.turn_complete_reason = (
      types.TurnCompleteReason.RESPONSE_REJECTED
  )

  mock_message = mock.create_autospec(types.LiveServerMessage, instance=True)
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].grounding_metadata is not None
  assert responses[0].turn_complete is None
  assert (
      responses[0].turn_complete_reason
      == types.TurnCompleteReason.RESPONSE_REJECTED
  )


@pytest.mark.asyncio
async def test_receive_populates_turn_complete_reason_with_content(
    gemini_connection, mock_gemini_session
):
  """Test that receive populates turn_complete_reason in LlmResponse when model turn has content parts."""
  mock_content = types.Content(
      role='model',
      parts=[types.Part.from_text(text='hello')],
  )
  mock_server_content = mock.create_autospec(
      types.LiveServerContent, instance=True
  )
  mock_server_content.model_turn = mock_content
  mock_server_content.grounding_metadata = None
  mock_server_content.turn_complete = False
  mock_server_content.interrupted = False
  mock_server_content.input_transcription = None
  mock_server_content.output_transcription = None
  mock_server_content.generation_complete = False
  mock_server_content.turn_complete_reason = (
      types.TurnCompleteReason.RESPONSE_REJECTED
  )

  mock_message = mock.create_autospec(types.LiveServerMessage, instance=True)
  mock_message.usage_metadata = None
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None
  mock_message.go_away = None

  async def mock_receive_generator():
    yield mock_message

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in gemini_connection.receive()]

  assert len(responses) == 1
  assert responses[0].content == mock_content
  assert (
      responses[0].turn_complete_reason
      == types.TurnCompleteReason.RESPONSE_REJECTED
  )


@pytest.mark.asyncio
async def test_receive_grounding_metadata_default_gemini_3_1(
    mock_gemini_session,
):
  """Verify grounding_metadata defaults to empty GroundingMetadata for Gemini 3.1."""
  conn = GeminiLlmConnection(
      mock_gemini_session,
      model_version='gemini-3.1-flash-live-preview',
  )

  def make_msg(text=None, tc=False, tool_call=None):
    msg = mock.create_autospec(types.LiveServerMessage, instance=True)
    msg.usage_metadata = None
    msg.tool_call = tool_call
    msg.session_resumption_update = None
    msg.go_away = None
    msg.server_content = mock.Mock()
    msg.server_content.interrupted = False
    msg.server_content.input_transcription = None
    msg.server_content.output_transcription = None
    msg.server_content.generation_complete = False
    msg.server_content.turn_complete = tc
    msg.server_content.grounding_metadata = None
    msg.server_content.model_turn = (
        types.Content(role='model', parts=[types.Part.from_text(text=text)])
        if text
        else None
    )
    return msg

  # 1. Content event
  msg1 = make_msg(text='hello')
  # 2. Tool call event (yields immediately for Gemini 3.1)
  function_call = types.FunctionCall(name='foo', args={})
  tool_call = mock.create_autospec(types.LiveServerToolCall, instance=True)
  tool_call.function_calls = [function_call]
  msg2 = make_msg(tool_call=tool_call)
  # 3. Turn complete event
  msg3 = make_msg(tc=True)

  async def mock_receive_generator():
    yield msg1
    yield msg2
    yield msg3

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in conn.receive()]

  # Expected:
  # responses[0] -> partial content response for msg1 (has no grounding_metadata)
  # responses[1] -> full text response for msg1 (has no grounding_metadata)
  # responses[2] -> tool call response for msg2 (has no grounding_metadata)
  # responses[3] -> turn_complete response for msg3 (has grounding_metadata)
  assert len(responses) == 4

  assert responses[0].content.parts[0].text == 'hello'
  assert responses[0].grounding_metadata is None
  assert responses[0].partial is True

  assert responses[1].content.parts[0].text == 'hello'
  assert responses[1].grounding_metadata is None
  assert responses[1].partial is False

  assert responses[2].content.parts[0].function_call.name == 'foo'
  assert responses[2].grounding_metadata is None

  assert responses[3].turn_complete is True
  assert isinstance(responses[3].grounding_metadata, types.GroundingMetadata)


@pytest.mark.asyncio
async def test_receive_grounding_metadata_default_non_gemini_3_1(
    mock_gemini_session,
):
  """Verify grounding_metadata stays None for non-Gemini 3.1 models."""
  conn = GeminiLlmConnection(
      mock_gemini_session,
      model_version='gemini-2.5-flash-live',
  )

  def make_msg(text=None, tc=False):
    msg = mock.create_autospec(types.LiveServerMessage, instance=True)
    msg.usage_metadata = None
    msg.tool_call = None
    msg.session_resumption_update = None
    msg.go_away = None
    msg.server_content = mock.Mock()
    msg.server_content.interrupted = False
    msg.server_content.input_transcription = None
    msg.server_content.output_transcription = None
    msg.server_content.generation_complete = False
    msg.server_content.turn_complete = tc
    msg.server_content.grounding_metadata = None
    msg.server_content.model_turn = (
        types.Content(role='model', parts=[types.Part.from_text(text=text)])
        if text
        else None
    )
    return msg

  msg1 = make_msg(text='hello')
  msg2 = make_msg(tc=True)

  async def mock_receive_generator():
    yield msg1
    yield msg2

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in conn.receive()]

  assert len(responses) == 3

  assert responses[0].content.parts[0].text == 'hello'
  assert responses[0].grounding_metadata is None
  assert responses[0].partial is True

  assert responses[1].content.parts[0].text == 'hello'
  assert responses[1].grounding_metadata is None
  assert responses[1].partial is False

  assert responses[2].turn_complete is True
  assert responses[2].grounding_metadata is None


@pytest.mark.asyncio
async def test_receive_input_transcription_gemini_3_1(
    mock_gemini_session,
):
  """Verify input_transcription yields finished=True immediately for Gemini 3.1."""
  conn = GeminiLlmConnection(
      mock_gemini_session,
      model_version='gemini-3.1-flash-live-preview',
  )

  def make_msg(
      input_text=None, output_text=None, output_finished=False, tc=False
  ):
    msg = mock.create_autospec(types.LiveServerMessage, instance=True)
    msg.usage_metadata = None
    msg.tool_call = None
    msg.session_resumption_update = None
    msg.go_away = None
    msg.server_content = mock.Mock()
    msg.server_content.interrupted = False
    msg.server_content.input_transcription = (
        types.Transcription(text=input_text, finished=False)
        if input_text
        else None
    )
    msg.server_content.output_transcription = (
        types.Transcription(text=output_text, finished=output_finished)
        if output_text
        else None
    )
    msg.server_content.generation_complete = False
    msg.server_content.turn_complete = tc
    msg.server_content.grounding_metadata = None
    msg.server_content.model_turn = None
    return msg

  msg1 = make_msg(input_text='Hello')
  msg2 = make_msg(output_text='Hi there!', output_finished=True)
  msg3 = make_msg(tc=True)

  async def mock_receive_generator():
    yield msg1
    yield msg2
    yield msg3

  mock_gemini_session.receive = mock.Mock(return_value=mock_receive_generator())

  responses = [resp async for resp in conn.receive()]

  assert len(responses) == 4

  assert responses[0].input_transcription.text == 'Hello'
  assert responses[0].input_transcription.finished is True
  assert responses[0].partial is False

  assert responses[1].output_transcription.text == 'Hi there!'
  assert responses[1].output_transcription.finished is False
  assert responses[1].partial is True

  assert responses[2].output_transcription.text == 'Hi there!'
  assert responses[2].output_transcription.finished is True
  assert responses[2].partial is False

  assert responses[3].turn_complete is True
