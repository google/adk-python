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

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


class TestBaseLlmFlow(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.fixture
def test_blob():
  """Test blob for audio data."""
  return types.Blob(data=b'\x00\xFF\x00\xFF', mime_type='audio/pcm')


@pytest.fixture
def mock_llm_connection():
  """Mock LLM connection for testing."""
  connection = mock.AsyncMock()
  connection.send_realtime = mock.AsyncMock()
  return connection


@pytest.mark.asyncio
async def test_send_to_model_with_disabled_vad(test_blob, mock_llm_connection):
  """Test _send_to_model with automatic_activity_detection.disabled=True."""
  # Create LlmRequest with disabled VAD
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=types.AutomaticActivityDetection(
          disabled=True
      )
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='',
      run_config=RunConfig(realtime_input_config=realtime_input_config),
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_enabled_vad(test_blob, mock_llm_connection):
  """Test _send_to_model with automatic_activity_detection.disabled=False.

  Custom VAD activity signal is not supported so we should still disable it.
  """
  # Create LlmRequest with enabled VAD
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=types.AutomaticActivityDetection(
          disabled=False
      )
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_without_realtime_config(
    test_blob, mock_llm_connection
):
  """Test _send_to_model without realtime_input_config (default behavior)."""
  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_none_automatic_activity_detection(
    test_blob, mock_llm_connection
):
  """Test _send_to_model with automatic_activity_detection=None."""
  # Create LlmRequest with None automatic_activity_detection
  realtime_input_config = types.RealtimeInputConfig(
      automatic_activity_detection=None
  )

  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='',
      run_config=RunConfig(realtime_input_config=realtime_input_config),
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send a blob to the queue
  live_request = LiveRequest(blob=test_blob)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection.send_realtime.assert_called_once_with(test_blob)


@pytest.mark.asyncio
async def test_send_to_model_with_text_content(mock_llm_connection):
  """Test _send_to_model with text content (not blob)."""
  # Create invocation context with live request queue
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  # Create flow and start _send_to_model task
  flow = TestBaseLlmFlow()

  # Send text content to the queue
  content = types.Content(
      role='user', parts=[types.Part.from_text(text='Hello')]
  )
  live_request = LiveRequest(content=content)
  invocation_context.live_request_queue.send(live_request)
  invocation_context.live_request_queue.close()

  # Run _send_to_model
  await flow._send_to_model(mock_llm_connection, invocation_context)

  # Verify send_content was called instead of send_realtime
  mock_llm_connection._send_content.assert_called_once_with(
      content, partial=False
  )
  mock_llm_connection.send_realtime.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_model_with_intermediate_text_content(
    mock_llm_connection,
):
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()
  invocation_context.session_service.append_event = mock.AsyncMock()

  flow = TestBaseLlmFlow()

  content = types.Content(
      role='user', parts=[types.Part.from_text(text='progress')]
  )
  invocation_context.live_request_queue.send(
      LiveRequest(content=content, partial=True)
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(mock_llm_connection, invocation_context)

  mock_llm_connection._send_content.assert_called_once_with(
      content, partial=True
  )
  invocation_context.session_service.append_event.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_model_applies_state_delta(mock_llm_connection):
  """Test _send_to_model applies state_delta as a state-delta event."""
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  flow = TestBaseLlmFlow()

  invocation_context.live_request_queue.send(
      LiveRequest(state_delta={'k': 'v'})
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(mock_llm_connection, invocation_context)

  assert invocation_context.session.state['k'] == 'v'
  mock_llm_connection._send_content.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_model_state_delta_with_content(mock_llm_connection):
  """Test _send_to_model applies state_delta and forwards content together."""
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  flow = TestBaseLlmFlow()

  content = types.Content(role='user', parts=[types.Part.from_text(text='hi')])
  invocation_context.live_request_queue.send(
      LiveRequest(content=content, state_delta={'k': 'v'})
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(mock_llm_connection, invocation_context)

  assert invocation_context.session.state['k'] == 'v'
  # The state delta rides on the single user content event (no extra event).
  events = invocation_context.session.events
  assert len(events) == 1
  assert events[0].content == content
  assert events[0].actions.state_delta == {'k': 'v'}
  mock_llm_connection._send_content.assert_called_once_with(
      content, partial=False
  )


@pytest.mark.asyncio
async def test_send_to_model_state_delta_with_partial_content(
    mock_llm_connection,
):
  """state_delta applies even when the partial turn skips the content event."""
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  flow = TestBaseLlmFlow()

  content = types.Content(
      role='user', parts=[types.Part.from_text(text='progress')]
  )
  invocation_context.live_request_queue.send(
      LiveRequest(content=content, state_delta={'k': 'v'}, partial=True)
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(mock_llm_connection, invocation_context)

  assert invocation_context.session.state['k'] == 'v'
  # The partial content does not create a user content event.
  assert all(e.content is None for e in invocation_context.session.events)
  mock_llm_connection._send_content.assert_called_once_with(
      content, partial=True
  )


@pytest.mark.asyncio
async def test_send_to_model_state_delta_with_function_response(
    mock_llm_connection,
):
  """state_delta applies even when the content is a function response."""
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  flow = TestBaseLlmFlow()

  content = types.Content(
      role='user',
      parts=[
          types.Part.from_function_response(
              name='tool', response={'result': 'ok'}
          )
      ],
  )
  invocation_context.live_request_queue.send(
      LiveRequest(content=content, state_delta={'k': 'v'})
  )
  invocation_context.live_request_queue.close()

  await flow._send_to_model(mock_llm_connection, invocation_context)

  assert invocation_context.session.state['k'] == 'v'
  # Function responses do not create a user content event.
  assert all(e.content is None for e in invocation_context.session.events)
  mock_llm_connection._send_content.assert_called_once_with(
      content, partial=False
  )


@pytest.mark.asyncio
async def test_send_to_model_state_delta_with_close(mock_llm_connection):
  """state_delta is flushed even when the request also closes the connection."""
  agent = Agent(name='test_agent', model='mock')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  flow = TestBaseLlmFlow()

  invocation_context.live_request_queue.send(
      LiveRequest(state_delta={'k': 'v'}, close=True)
  )

  await flow._send_to_model(mock_llm_connection, invocation_context)

  assert invocation_context.session.state['k'] == 'v'
  mock_llm_connection.close.assert_called_once()
