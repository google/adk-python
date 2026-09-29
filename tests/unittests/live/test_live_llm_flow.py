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
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.live import _live_llm_flow
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
      flow.cache_manager, 'flush_caches', new_callable=mock.AsyncMock
  ) as mock_flush:
    mock_flush.return_value = [Event(id='flushed-event')]
    events = await _live_llm_flow.handle_control_event_flush(
        flow, context, response
    )

  assert len(events) == 1
  mock_flush.assert_awaited_once_with(
      context,
      flush_user_audio=False,
      flush_model_audio=True,
      flush_user_media=False,
      flush_model_media=True,
  )


async def test_handle_control_event_flush_on_turn_complete():
  """A turn_complete response triggers both user and model audio and media cache flushes."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(live_request_queue=LiveRequestQueue())
  response = LlmResponse(turn_complete=True)

  with mock.patch.object(
      flow.cache_manager, 'flush_caches', new_callable=mock.AsyncMock
  ) as mock_flush:
    mock_flush.return_value = [Event(id='flushed-event')]
    events = await _live_llm_flow.handle_control_event_flush(
        flow, context, response
    )

  assert len(events) == 1
  mock_flush.assert_awaited_once_with(
      context,
      flush_user_audio=True,
      flush_model_audio=True,
      flush_user_media=True,
      flush_model_media=True,
  )


async def test_stop_background_tool_tasks_cancels_and_clears():
  """Cancels pending background tasks and clears active tool registries on the context."""
  context = _create_test_context()

  async def _long_task():
    await asyncio.sleep(100)

  task1 = asyncio.create_task(_long_task(), name='test_bg_task')
  mock_active = mock.MagicMock(task=task1)
  context.active_streaming_tools = {'stream_tool': mock_active}
  context.active_non_blocking_tool_tasks = {'non_blocking_tool': task1}

  await _live_llm_flow.stop_background_tool_tasks(context)

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


async def test_stop_background_tool_tasks_uses_timeout():
  """stop_background_tool_tasks uses _TOOL_SHUTDOWN_TIMEOUT_SECONDS."""
  from google.adk.live import _flow_utils

  context = _create_test_context()

  async def _dummy():
    await asyncio.sleep(10)

  task = asyncio.create_task(_dummy())
  context.active_non_blocking_tool_tasks = {'t': task}

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
          flow.cache_manager, 'get_cache_stats'
      ) as mock_get_stats,
      mock.patch.object(flow.cache_manager, 'flush_caches', return_value=[]),
  ):
    await _live_llm_flow.handle_control_event_flush(flow, context, response)

  mock_get_stats.assert_called_once_with(context)


async def _run_send_to_model_once(flow, context):
  """Drains the queued requests deterministically by appending a close signal."""
  mock_connection = mock.AsyncMock()
  context.live_request_queue.close()
  await _live_llm_flow.send_to_model(
      flow, mock_connection, context, LlmRequest()
  )
  return mock_connection


async def test_send_to_model_routes_audio_to_the_audio_cache():
  """send_to_model caches user audio into input_realtime_cache and forwards to connection."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  audio_blob = types.Blob(mime_type='audio/pcm', data=b'audio_bytes')
  queue.send_realtime(audio_blob)
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=True)
  )

  mock_connection = await _run_send_to_model_once(flow, context)

  assert len(context.input_realtime_cache) == 1
  assert context.input_realtime_cache[0].data == audio_blob
  assert not context.input_media_realtime_cache
  mock_connection.send_realtime.assert_awaited_once_with(audio_blob)


async def test_send_to_model_routes_video_to_the_media_cache():
  """send_to_model routes image/* and video/* blobs into input_media_realtime_cache."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  video_blob = types.Blob(mime_type='image/jpeg', data=b'frame_bytes')
  queue.send_realtime(video_blob)
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=True)
  )

  mock_connection = await _run_send_to_model_once(flow, context)

  assert len(context.input_media_realtime_cache) == 1
  assert context.input_media_realtime_cache[0].data == video_blob
  assert not context.input_realtime_cache
  mock_connection.send_realtime.assert_awaited_once_with(video_blob)


async def test_send_to_model_caches_nothing_when_save_live_blob_is_off():
  """`save_live_blob` gates the whole feature; with it off no frame may be cached."""
  flow = _TestBaseLlmFlow()
  queue = LiveRequestQueue()
  video_blob = types.Blob(mime_type='image/jpeg', data=b'frame_bytes')
  queue.send_realtime(video_blob)
  context = _create_test_context(
      live_request_queue=queue, run_config=RunConfig(save_live_blob=False)
  )

  mock_connection = await _run_send_to_model_once(flow, context)

  assert not context.input_media_realtime_cache
  assert not context.input_realtime_cache
  mock_connection.send_realtime.assert_awaited_once_with(video_blob)


async def test_audio_cache_manager_alias_still_resolves_and_warns():
  """Reading or setting `flow.audio_cache_manager` forwards to `cache_manager` and warns."""
  from google.adk.live._cache_manager import CacheManager

  flow = _TestBaseLlmFlow()

  with pytest.warns(DeprecationWarning, match='use cache_manager instead'):
    alias = flow.audio_cache_manager
  assert alias is flow.cache_manager

  replacement = CacheManager()
  with pytest.warns(DeprecationWarning, match='use cache_manager instead'):
    flow.audio_cache_manager = replacement
  assert flow.cache_manager is replacement


class _FakeLiveConnection:
  """Replays a fixed script of responses once, then goes quiet."""

  def __init__(self, responses: list[LlmResponse]):
    self._responses = responses
    self.receive_calls = 0

  async def receive(self):
    self.receive_calls += 1
    if self.receive_calls > 1:
      return
    for response in self._responses:
      yield response


def _model_blob_response(mime_type: str, data: bytes) -> LlmResponse:
  """A model turn carrying a single inline blob."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))
          ],
      )
  )


async def _drain_receive_from_model(flow, connection, context) -> list[Event]:
  events = []
  async for event in _live_llm_flow.receive_from_model(
      flow, connection, context, LlmRequest()
  ):
    events.append(event)
  return events


async def test_receive_from_model_routes_video_to_the_media_cache():
  """Model-generated image/* and video/* inline_data blobs are routed to output_media_realtime_cache."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(run_config=RunConfig(save_live_blob=True))
  connection = _FakeLiveConnection(
      [_model_blob_response('image/jpeg', b'model_frame')]
  )

  await _drain_receive_from_model(flow, connection, context)

  assert len(context.output_media_realtime_cache) == 1
  assert context.output_media_realtime_cache[0].data.data == b'model_frame'
  assert not context.output_realtime_cache


async def test_receive_from_model_routes_audio_to_the_audio_cache():
  """Routing by MIME type preserves the output audio caching path."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(run_config=RunConfig(save_live_blob=True))
  connection = _FakeLiveConnection(
      [_model_blob_response('audio/pcm', b'model_audio')]
  )

  await _drain_receive_from_model(flow, connection, context)

  assert len(context.output_realtime_cache) == 1
  assert context.output_realtime_cache[0].data.data == b'model_audio'
  assert not context.output_media_realtime_cache


async def test_receive_from_model_caches_nothing_when_save_live_blob_is_off():
  """`save_live_blob` gates model output caching as well as user input caching."""
  flow = _TestBaseLlmFlow()
  context = _create_test_context(run_config=RunConfig())
  connection = _FakeLiveConnection(
      [_model_blob_response('image/jpeg', b'model_frame')]
  )

  await _drain_receive_from_model(flow, connection, context)

  assert not context.output_media_realtime_cache
  assert not context.output_realtime_cache


async def test_postprocess_live_flow_flushes_media_on_interrupted_and_turn_complete():
  """Interrupted responses flush only model media/audio; turn_complete flushes remaining user media/audio."""
  import io
  import zipfile

  from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

  flow = _TestBaseLlmFlow()
  context = _create_test_context(
      live_request_queue=LiveRequestQueue(),
      run_config=RunConfig(save_live_blob=True),
  )
  artifact_service = InMemoryArtifactService()
  context.artifact_service = artifact_service

  flow.cache_manager.cache_blob(
      context, types.Blob(mime_type='image/jpeg', data=b'user_frame_1'), 'input'
  )
  flow.cache_manager.cache_blob(
      context, types.Blob(mime_type='image/jpeg', data=b'user_frame_2'), 'input'
  )
  flow.cache_manager.cache_blob(
      context, types.Blob(mime_type='image/png', data=b'model_frame'), 'output'
  )

  # 1. Interrupted response flushes only model media cache (1 frame -> stored as image/png)
  interrupted_events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow,
          context,
          LlmRequest(),
          LlmResponse(interrupted=True),
          Event(
              id='ev-int', invocation_id=context.invocation_id, author='model'
          ),
      )
  ]
  assert len(interrupted_events) == 1
  assert interrupted_events[0].author == 'test_agent'
  assert (
      interrupted_events[0].content.parts[0].file_data.mime_type == 'image/png'
  )
  assert context.output_media_realtime_cache == []
  assert len(context.input_media_realtime_cache) == 2

  # 2. Turn complete response flushes remaining user media cache (2 frames -> stored as .zip)
  turn_complete_events = [
      e
      async for e in _live_llm_flow.postprocess_live_flow(
          flow,
          context,
          LlmRequest(),
          LlmResponse(turn_complete=True),
          Event(
              id='ev-done', invocation_id=context.invocation_id, author='model'
          ),
      )
  ]
  assert len(turn_complete_events) == 1
  assert turn_complete_events[0].author == 'user'
  assert context.input_media_realtime_cache == []

  user_uri = turn_complete_events[0].content.parts[0].file_data.file_uri
  user_filename = user_uri.rsplit('/', 1)[-1].split('#')[0]
  loaded = await artifact_service.load_artifact(
      app_name=context.app_name,
      user_id=context.user_id,
      session_id=context.session.id,
      filename=user_filename,
  )
  assert loaded is not None
  assert loaded.inline_data.mime_type == 'application/zip'
  with zipfile.ZipFile(io.BytesIO(loaded.inline_data.data), mode='r') as zf:
    assert zf.read('frames/frame_0000.jpeg') == b'user_frame_1'
    assert zf.read('frames/frame_0001.jpeg') == b'user_frame_2'
