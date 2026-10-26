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

"""Unit tests for the _live_llm_flow helper module."""

from __future__ import annotations

import asyncio
import logging
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.live import _live_llm_flow
from google.adk.live._active_background_tool import ActiveBackgroundTool
from google.adk.live._active_streaming_tool import ActiveStreamingTool
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils


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


@pytest.mark.parametrize('control', ['turn_complete', 'interrupted'])
async def test_postprocess_live_flow_yields_control_event_after_flushed_audio(
    control,
):
  """Flushing saved audio still yields the control event that triggered it."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(
      live_request_queue=LiveRequestQueue(),
      run_config=RunConfig(save_live_blob=True),
  )
  response = LlmResponse(**{control: True})
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author='model',
  )
  flushed = Event(id='flushed-event')

  with mock.patch.object(
      flow._realtime_cache_manager, 'flush_caches', new_callable=mock.AsyncMock
  ) as mock_flush:
    mock_flush.return_value = [flushed]
    events = [
        e
        async for e in _live_llm_flow.postprocess_live_flow(
            flow, context, LlmRequest(), response, event
        )
    ]

  assert len(events) == 2
  assert events[0] is flushed
  assert getattr(events[1], control) is True


async def test_handle_control_event_flush_on_interrupted():
  """An interrupted response triggers a model-only cache flush."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  response = LlmResponse(interrupted=True)

  with mock.patch.object(
      flow._realtime_cache_manager, 'flush_caches', new_callable=mock.AsyncMock
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
      flow._realtime_cache_manager, 'flush_caches', new_callable=mock.AsyncMock
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
  context = _create_test_context()

  async def _long_task():
    await asyncio.sleep(100)

  task1 = asyncio.create_task(_long_task(), name='test_bg_task')
  task2 = asyncio.create_task(_long_task(), name='test_bg_task_2')
  context._add_background_tool(
      ActiveBackgroundTool(
          tool_name='stream_tool',
          function_call_id='call_1',
          task=task1,
          is_generator=True,
      )
  )
  context._add_background_tool(
      ActiveBackgroundTool(
          tool_name='non_blocking_tool', function_call_id='call_2', task=task2
      )
  )

  await _live_llm_flow.stop_background_tool_tasks(context)

  assert task1.cancelled()
  assert task2.cancelled()
  assert not context._active_background_tools


async def test_stop_background_tool_tasks_skips_cleared_tasks():
  """An entry whose task was cleared is retired without being cancelled."""
  context = _create_test_context()

  async def _long_task():
    await asyncio.sleep(100)

  task = asyncio.create_task(_long_task())
  context._add_background_tool(
      ActiveBackgroundTool(
          tool_name='stream_tool',
          function_call_id='call_1',
          task=None,
          input_stream=LiveRequestQueue(),
          is_generator=True,
      )
  )
  context._add_background_tool(
      ActiveBackgroundTool(
          tool_name='non_blocking_tool', function_call_id='call_2', task=task
      )
  )

  await _live_llm_flow.stop_background_tool_tasks(context)

  assert task.cancelled()
  assert not context._active_background_tools


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


async def test_stop_background_tool_tasks_uses_timeout():
  """stop_background_tool_tasks uses _TOOL_SHUTDOWN_TIMEOUT_SECONDS."""
  from google.adk.live import _flow_utils

  context = _create_test_context()

  async def _dummy():
    await asyncio.sleep(10)

  task = asyncio.create_task(_dummy())
  context._add_background_tool(
      ActiveBackgroundTool(tool_name='t', function_call_id='call_1', task=task)
  )

  with (
      mock.patch.object(_flow_utils, '_TOOL_SHUTDOWN_TIMEOUT_SECONDS', 0.01),
      mock.patch('asyncio.wait', wraps=asyncio.wait) as mock_wait,
  ):
    await _live_llm_flow.stop_background_tool_tasks(context)

  assert mock_wait.call_args.kwargs['timeout'] == 0.01


async def test_handle_control_event_flush_logs_stats_when_enabled():
  """handle_control_event_flush queries DEFAULT_ENABLE_CACHE_STATISTICS."""
  from google.adk.live import _flow_utils

  flow = _TestBaseLlmFlow()
  context = _create_test_context()
  response = LlmResponse(turn_complete=True)

  with (
      mock.patch.object(_flow_utils, 'DEFAULT_ENABLE_CACHE_STATISTICS', True),
      mock.patch.object(
          flow._realtime_cache_manager, 'get_cache_stats'
      ) as mock_get_stats,
      mock.patch.object(
          flow._realtime_cache_manager, 'flush_caches', return_value=[]
      ),
  ):
    await _live_llm_flow.handle_control_event_flush(flow, context, response)

  mock_get_stats.assert_called_once_with(context)


async def test_send_to_model_uses_flow_realtime_cache_manager():
  """send_to_model accesses the realtime cache manager directly from the flow instance."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  audio_blob = types.Blob(mime_type='audio/pcm', data=b'audio_bytes')
  queue.send_realtime(audio_blob)
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=True)
  )
  mock_connection = mock.AsyncMock()

  with mock.patch.object(
      flow._realtime_cache_manager, 'cache_blob'
  ) as mock_cache_blob:
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

  mock_cache_blob.assert_called_once_with(
      context, audio_blob, cache_type='input'
  )


async def test_send_to_model_caches_only_audio_blobs():
  """Non-audio blobs such as video frames are sent but not cached."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  audio_blob = types.Blob(mime_type='audio/pcm', data=b'audio_bytes')
  video_blob = types.Blob(mime_type='image/jpeg', data=b'video_frame')
  queue.send_realtime(video_blob)
  queue.send_realtime(audio_blob)
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=True)
  )
  mock_connection = mock.AsyncMock()

  send_task = asyncio.create_task(
      _live_llm_flow.send_to_model(flow, mock_connection, context, LlmRequest())
  )
  await asyncio.sleep(0.01)
  send_task.cancel()
  try:
    await send_task
  except asyncio.CancelledError:
    pass

  assert [entry.data for entry in context.input_realtime_cache] == [audio_blob]
  assert mock_connection.send_realtime.await_args_list == [
      mock.call(video_blob),
      mock.call(audio_blob),
  ]


async def test_send_to_model_does_not_mutate_caller_content_role():
  """Defaulting role='user' must not mutate the caller's Content object."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  content = types.Content(parts=[types.Part.from_text(text='hello')])
  queue.send_content(content)
  queue.close()
  context = _create_test_context(live_request_queue=queue)
  await context.session_service.create_session(
      app_name='test_app', user_id='u1', session_id='s1'
  )
  mock_connection = mock.AsyncMock()

  await _live_llm_flow.send_to_model(
      flow, mock_connection, context, LlmRequest()
  )

  assert content.role is None
  sent_content = mock_connection._send_content.await_args.args[0]
  assert sent_content.role == 'user'


async def test_send_to_model_feeds_streams_set_through_deprecated_field():
  """A stream registered through active_streaming_tools receives requests."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  blob = types.Blob(mime_type='audio/pcm', data=b'audio_bytes')
  queue.send_realtime(blob)
  context = _create_test_context(live_request_queue=queue)
  tool_stream = LiveRequestQueue()
  with pytest.warns(DeprecationWarning):
    context.active_streaming_tools = {
        'stream_tool': ActiveStreamingTool(stream=tool_stream)
    }

  send_task = asyncio.create_task(
      _live_llm_flow.send_to_model(
          flow, mock.AsyncMock(), context, LlmRequest()
      )
  )
  try:
    forwarded = await asyncio.wait_for(tool_stream.get(), timeout=1)
  finally:
    send_task.cancel()
    try:
      await send_task
    except asyncio.CancelledError:
      pass

  assert forwarded.blob == blob


async def test_receive_from_model_caches_only_audio_blobs():
  """Model audio blobs are cached; other inline blobs are not."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(run_config=RunConfig(save_live_blob=True))
  calls = 0

  async def _receive():
    nonlocal calls
    calls += 1
    if calls == 1:
      yield LlmResponse(
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          mime_type='image/png', data=b'model_image'
                      )
                  ),
                  types.Part(
                      inline_data=types.Blob(
                          mime_type='audio/pcm', data=b'model_audio'
                      )
                  ),
              ],
          )
      )

  connection = mock.MagicMock(receive=_receive)
  events = [
      event
      async for event in _live_llm_flow.receive_from_model(
          flow, connection, context, LlmRequest()
      )
  ]

  assert len(events) == 1
  assert [entry.data.data for entry in context.output_realtime_cache] == [
      b'model_audio'
  ]


def test_audio_cache_manager_alias_points_to_realtime_cache_manager():
  """`flow.audio_cache_manager` is kept as an alias of the flow's manager."""
  flow = _TestBaseLlmFlow()

  assert flow.audio_cache_manager is flow._realtime_cache_manager


class _WeatherReport(BaseModel):
  city: str
  temperature_celsius: float


def _get_weather(city: str) -> dict[str, object]:
  """Returns the weather in a city.

  Args:
    city: The city name.
  """
  return {'city': city, 'temperature_celsius': 26}


def test_run_live_ignores_output_schema(caplog):
  """A live session gets no set_model_response tool and logs a warning."""
  mock_model = testing_utils.MockModel.create([LlmResponse(turn_complete=True)])
  agent = LlmAgent(
      name='weather_agent',
      model=mock_model,
      tools=[_get_weather],
      output_schema=_WeatherReport,
  )
  runner = testing_utils.InMemoryRunner(
      root_agent=agent, response_modalities=['AUDIO']
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )

  with caplog.at_level(logging.WARNING):
    runner.run_live(live_request_queue)

  llm_request = mock_model.requests[0]
  assert SetModelResponseTool.NAME not in llm_request.tools_dict
  assert SetModelResponseTool.NAME not in (
      llm_request.config.system_instruction or ''
  )
  assert 'not supported in live mode' in caplog.text


def test_run_live_does_not_warn_for_task_mode_output_schema(caplog):
  """A task-mode agent reports output via finish_task, so no warning."""
  mock_model = testing_utils.MockModel.create([LlmResponse(turn_complete=True)])
  agent = LlmAgent(
      name='weather_agent',
      model=mock_model,
      mode='task',
      tools=[_get_weather],
      output_schema=_WeatherReport,
  )
  runner = testing_utils.InMemoryRunner(
      root_agent=agent, response_modalities=['AUDIO']
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )

  with caplog.at_level(logging.WARNING):
    runner.run_live(live_request_queue)

  assert mock_model.requests
  assert 'not supported in live mode' not in caplog.text


async def test_postprocess_live_flow_does_not_synthesize_structured_response():
  """A set_model_response call yields only the call and its response."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  llm_request = LlmRequest()
  llm_request.append_tools([SetModelResponseTool(_WeatherReport)])
  response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='call-1',
                      name=SetModelResponseTool.NAME,
                      args={'city': 'Tokyo', 'temperature_celsius': 26},
                  )
              )
          ],
      )
  )
  event = Event(
      id='ev-1',
      invocation_id=context.invocation_id,
      author=context.agent.name,
  )

  events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow, context, llm_request, response, event
      )
  ]

  assert len(events) == 2
  assert events[0].get_function_calls()
  assert events[1].get_function_responses()
