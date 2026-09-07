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

"""Unit tests for _live_llm_flow helper module and its BaseLlmFlow shims."""

from __future__ import annotations

import asyncio
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows import _live_llm_flow
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


class _TestBaseLlmFlow(BaseLlmFlow):
  """Subclass of BaseLlmFlow for unit testing."""

  pass


def _create_test_context(
    *,
    live_request_queue: LiveRequestQueue | None = None,
    run_config: RunConfig | None = None,
) -> InvocationContext:
  """Creates a minimal InvocationContext for testing."""
  agent = LlmAgent(name='test_agent', model='gemini-2.0-flash')
  session = Session(id='s1', app_name='test_app', user_id='u1', events=[])
  session_service = InMemorySessionService()
  context = InvocationContext(
      invocation_id='inv-1',
      agent=agent,
      session=session,
      session_service=session_service,
      live_request_queue=live_request_queue,
      run_config=run_config or RunConfig(),
  )
  return context


async def test_require_live_request_queue_returns_queue():
  """Returns the LiveRequestQueue when present on the invocation context."""
  queue = LiveRequestQueue()
  context = _create_test_context(live_request_queue=queue)

  result = _live_llm_flow.require_live_request_queue(context)

  assert result is queue


async def test_require_live_request_queue_raises_when_missing():
  """Raises a ValueError when live_request_queue is None."""
  context = _create_test_context(live_request_queue=None)

  with pytest.raises(
      ValueError, match='Live model execution requires a LiveRequestQueue.'
  ):
    _live_llm_flow.require_live_request_queue(context)


async def test_postprocess_live_flow_yields_session_resumption_update():
  """A session resumption update yields an event stamped with the new handle."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  update = types.LiveServerSessionResumptionUpdate(new_handle='handle-123')
  response = LlmResponse(live_session_resumption_update=update)
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author='model',
  )

  events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow, context, LlmRequest(), response, event
      )
  ]

  assert len(events) == 1
  assert events[0].live_session_resumption_update == update


async def test_postprocess_live_flow_yields_voice_activity():
  """A voice activity signal yields an event with the voice activity payload."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  vad = types.VoiceActivity(
      voice_activity_type=types.VoiceActivityType.ACTIVITY_START,
      audio_offset='0.5s',
  )
  response = LlmResponse(voice_activity=vad)
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author='model',
  )

  events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow, context, LlmRequest(), response, event
      )
  ]

  assert len(events) == 1
  assert events[0].voice_activity == vad


async def test_postprocess_live_flow_yields_input_and_output_transcriptions():
  """Input and output transcription updates yield events with partial flags preserved."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  input_transcription = types.Transcription(text='hello', finished=False)
  response = LlmResponse(input_transcription=input_transcription, partial=True)
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author='user',
  )

  events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow, context, LlmRequest(), response, event
      )
  ]

  assert len(events) == 1
  assert events[0].input_transcription == input_transcription
  assert events[0].partial is True


async def test_postprocess_live_flow_skips_empty_response():
  """An empty LLM response with no content or control signals produces no events."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  response = LlmResponse()
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author='model',
  )

  events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow, context, LlmRequest(), response, event
      )
  ]

  assert events == []


async def test_handle_control_event_flush_on_interrupted():
  """An interrupted response triggers a model-only cache flush."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  response = LlmResponse(interrupted=True)

  with mock.patch.object(
      flow.audio_cache_manager, 'flush_caches', new_callable=mock.AsyncMock
  ) as mock_flush:
    mock_flush.return_value = [Event(id='flushed-event')]
    events = await _live_llm_flow.handle_control_event_flush(
        flow, context, response
    )

  assert len(events) == 1
  mock_flush.assert_awaited_once_with(
      context, flush_user_audio=False, flush_model_audio=True
  )


async def test_handle_control_event_flush_on_turn_complete():
  """A turn_complete response triggers both user and model audio cache flushes."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  response = LlmResponse(turn_complete=True)

  with mock.patch.object(
      flow.audio_cache_manager, 'flush_caches', new_callable=mock.AsyncMock
  ) as mock_flush:
    mock_flush.return_value = [Event(id='flushed-event')]
    events = await _live_llm_flow.handle_control_event_flush(
        flow, context, response
    )

  assert len(events) == 1
  mock_flush.assert_awaited_once_with(
      context, flush_user_audio=True, flush_model_audio=True
  )


async def test_stop_background_tool_tasks_cancels_and_clears():
  """Cancels pending background tasks and clears active tool registries on the context."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context()

  async def _long_task():
    await asyncio.sleep(100)

  task1 = asyncio.create_task(_long_task(), name='test_bg_task')
  mock_active = mock.MagicMock(task=task1)
  context.active_streaming_tools = {'stream_tool': mock_active}
  context.active_non_blocking_tool_tasks = {'non_blocking_tool': task1}

  await _live_llm_flow.stop_background_tool_tasks(flow, context)

  assert task1.cancelled()
  assert context.active_streaming_tools == {}
  assert context.active_non_blocking_tool_tasks == {}


async def test_screen_live_user_content_returns_blocked_event():
  """A blocked before_model_callback returns a finalized event marked with turn_complete."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context()
  content = types.Content(parts=[types.Part.from_text(text='blocked text')])
  blocked_response = LlmResponse(
      content=types.Content(
          parts=[types.Part.from_text(text='Blocked content')]
      )
  )

  with mock.patch.object(
      flow, '_handle_before_model_callback', new_callable=mock.AsyncMock
  ) as mock_cb:
    mock_cb.return_value = blocked_response
    blocked_event = await _live_llm_flow.screen_live_user_content(
        flow, context, content, LlmRequest()
    )

  assert blocked_event is not None
  assert blocked_event.turn_complete is True
  assert blocked_event.content == blocked_response.content


async def test_base_llm_flow_forwarding_shims():
  """BaseLlmFlow shims delegate to _live_llm_flow while preserving caller interface."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  update = types.LiveServerSessionResumptionUpdate(new_handle='shim-handle')
  response = LlmResponse(live_session_resumption_update=update)
  event = Event(id='e-shim', invocation_id=context.invocation_id)

  events = [
      e
      async for e in flow._postprocess_live(
          context, LlmRequest(), response, event
      )
  ]

  assert len(events) == 1
  assert events[0].live_session_resumption_update == update


async def test_stop_background_tool_tasks_uses_base_llm_flow_timeout():
  """stop_background_tool_tasks uses _TOOL_SHUTDOWN_TIMEOUT_SECONDS from base_llm_flow."""
  from google.adk.flows.llm_flows import base_llm_flow

  flow = _TestBaseLlmFlow()
  context = _create_test_context()

  async def _dummy():
    await asyncio.sleep(10)

  task = asyncio.create_task(_dummy())
  context.active_non_blocking_tool_tasks = {'t': task}

  with (
      mock.patch.object(base_llm_flow, '_TOOL_SHUTDOWN_TIMEOUT_SECONDS', 0.01),
      mock.patch('asyncio.wait', wraps=asyncio.wait) as mock_wait,
  ):
    await _live_llm_flow.stop_background_tool_tasks(flow, context)

  assert mock_wait.call_args.kwargs['timeout'] == 0.01


async def test_handle_control_event_flush_logs_stats_when_enabled_on_base_llm_flow():
  """handle_control_event_flush queries DEFAULT_ENABLE_CACHE_STATISTICS on base_llm_flow."""
  from google.adk.flows.llm_flows import base_llm_flow

  flow = _TestBaseLlmFlow()
  context = _create_test_context()
  response = LlmResponse(turn_complete=True)

  with (
      mock.patch.object(base_llm_flow, 'DEFAULT_ENABLE_CACHE_STATISTICS', True),
      mock.patch.object(
          flow.audio_cache_manager, 'get_cache_stats'
      ) as mock_get_stats,
      mock.patch.object(
          flow.audio_cache_manager, 'flush_caches', return_value=[]
      ),
  ):
    await _live_llm_flow.handle_control_event_flush(flow, context, response)

  mock_get_stats.assert_called_once_with(context)


async def test_send_to_model_uses_flow_audio_cache_manager():
  """send_to_model accesses the audio cache manager directly from the flow instance."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  queue.send_realtime(types.Blob(mime_type='audio/pcm', data=b'audio_bytes'))
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=True)
  )
  mock_connection = mock.AsyncMock()

  with mock.patch.object(
      flow.audio_cache_manager, 'cache_audio'
  ) as mock_cache_audio:
    # Run send_to_model briefly and cancel it after processing the queued item
    send_task = asyncio.create_task(
        _live_llm_flow.send_to_model(
            flow, mock_connection, context, LlmRequest()
        )
    )
    await asyncio.sleep(0.01)
    send_task.cancel()
    try:
      await send_task
    except asyncio.CancelledError:
      pass

  mock_cache_audio.assert_called_once()
