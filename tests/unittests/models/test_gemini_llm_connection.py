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

from unittest import mock

from google.adk.models.gemini_llm_connection import GeminiLlmConnection
from google.genai import types
import pytest


@pytest.fixture
def mock_gemini_session():
  """Mock Gemini session for testing."""
  return mock.AsyncMock()


@pytest.fixture
def gemini_connection(mock_gemini_session):
  """GeminiLlmConnection instance with mocked session."""
  return GeminiLlmConnection(mock_gemini_session)


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
async def test_send_history(gemini_connection, mock_gemini_session):
  """Test send_history method."""
  history = [
      types.Content(role='user', parts=[types.Part.from_text(text='Hello')]),
      types.Content(
          role='model', parts=[types.Part.from_text(text='Hi there!')]
      ),
  ]

  await gemini_connection.send_history(history)

  mock_gemini_session.send.assert_called_once()
  call_args = mock_gemini_session.send.call_args[1]
  assert 'input' in call_args
  assert call_args['input'].turns == history
  assert call_args['input'].turn_complete is False  # Last message is from model


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

  mock_gemini_session.send.assert_called_once()
  call_args = mock_gemini_session.send.call_args[1]
  assert 'input' in call_args
  assert call_args['input'].function_responses == [function_response]


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
  msg.server_content.model_turn = None
  msg.server_content.interrupted = False
  msg.server_content.turn_complete = False
  msg.server_content.input_transcription = (
      finished_tx if tx_direction == 'input' else None
  )
  msg.server_content.output_transcription = (
      finished_tx if tx_direction == 'output' else None
  )

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

  mock_message = mock.AsyncMock()
  mock_message.usage_metadata = usage_metadata
  mock_message.server_content = mock_server_content
  mock_message.tool_call = None
  mock_message.session_resumption_update = None

  async def mock_receive_generator():
    yield mock_message

  receive_mock = mock.Mock(return_value=mock_receive_generator())
  mock_gemini_session.receive = receive_mock

  responses = [resp async for resp in gemini_connection.receive()]

  assert responses

  usage_response = next((r for r in responses if r.usage_metadata), None)
  assert usage_response is not None
  content_response = next((r for r in responses if r.content), None)
  assert content_response is not None

  expected_usage = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=10,
      cached_content_token_count=5,
      candidates_token_count=None,
      total_token_count=35,
      thoughts_token_count=2,
      prompt_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=10)
      ],
      cache_tokens_details=[
          types.ModalityTokenCount(modality='text', token_count=5)
      ],
      candidates_tokens_details=None,
  )
  assert usage_response.usage_metadata == expected_usage
  assert content_response.content == mock_content


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
  message1.tool_call = None
  message1.session_resumption_update = None

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
  message2.tool_call = None
  message2.session_resumption_update = None

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
  message3.tool_call = None
  message3.session_resumption_update = None

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
  message1.tool_call = None
  message1.session_resumption_update = None

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
  message2.tool_call = None
  message2.session_resumption_update = None

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
  message3.tool_call = None
  message3.session_resumption_update = None

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
