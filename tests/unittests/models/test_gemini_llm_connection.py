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
  mock_gemini_session.send.assert_called_once_with(input=test_blob.model_dump())


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
