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

from google.adk.agents import Agent
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.models import LlmResponse
from google.genai import types
import pytest

from .. import testing_utils


def test_streaming():
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.output_audio_transcription
      is not None
  )


def test_streaming_with_output_audio_transcription():
  """Test streaming with output audio transcription configuration."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with output audio transcription
  run_config = RunConfig(
      output_audio_transcription=types.AudioTranscriptionConfig()
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.output_audio_transcription
      is not None
  )


def test_streaming_with_input_audio_transcription():
  """Test streaming with input audio transcription configuration."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with input audio transcription
  run_config = RunConfig(
      input_audio_transcription=types.AudioTranscriptionConfig()
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.input_audio_transcription
      is not None
  )


def test_streaming_with_realtime_input_config():
  """Test streaming with realtime input configuration."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with realtime input config
  run_config = RunConfig(
      realtime_input_config=types.RealtimeInputConfig(
          automatic_activity_detection=types.AutomaticActivityDetection(
              disabled=True
          )
      )
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.realtime_input_config.automatic_activity_detection.disabled
      is True
  )


def test_streaming_with_realtime_input_config_vad_enabled():
  """Test streaming with realtime input configuration with VAD enabled."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with realtime input config with VAD enabled
  run_config = RunConfig(
      realtime_input_config=types.RealtimeInputConfig(
          automatic_activity_detection=types.AutomaticActivityDetection(
              disabled=False
          )
      )
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.realtime_input_config.automatic_activity_detection.disabled
      is False
  )


def test_streaming_with_enable_affective_dialog_true():
  """Test streaming with affective dialog enabled."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with affective dialog enabled
  run_config = RunConfig(enable_affective_dialog=True)

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.enable_affective_dialog
      is True
  )


def test_streaming_with_enable_affective_dialog_false():
  """Test streaming with affective dialog disabled."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with affective dialog disabled
  run_config = RunConfig(enable_affective_dialog=False)

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.enable_affective_dialog
      is False
  )


def test_streaming_with_proactivity_config():
  """Test streaming with proactivity configuration."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with proactivity config
  run_config = RunConfig(proactivity=types.ProactivityConfig())

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert llm_request_sent_to_mock.live_connect_config.proactivity is not None


def test_streaming_with_combined_audio_transcription_configs():
  """Test streaming with both input and output audio transcription configurations."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with both input and output audio transcription
  run_config = RunConfig(
      input_audio_transcription=types.AudioTranscriptionConfig(),
      output_audio_transcription=types.AudioTranscriptionConfig(),
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.input_audio_transcription
      is not None
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.output_audio_transcription
      is not None
  )


def test_streaming_with_all_configs_combined():
  """Test streaming with all the new configurations combined."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with all configurations
  run_config = RunConfig(
      output_audio_transcription=types.AudioTranscriptionConfig(),
      input_audio_transcription=types.AudioTranscriptionConfig(),
      realtime_input_config=types.RealtimeInputConfig(
          automatic_activity_detection=types.AutomaticActivityDetection(
              disabled=True
          )
      ),
      enable_affective_dialog=True,
      proactivity=types.ProactivityConfig(),
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.realtime_input_config
      is not None
  )
  assert llm_request_sent_to_mock.live_connect_config.proactivity is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.enable_affective_dialog
      is True
  )


def test_streaming_with_multiple_audio_configs():
  """Test streaming with multiple audio transcription configurations."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with multiple audio transcription configs
  run_config = RunConfig(
      input_audio_transcription=types.AudioTranscriptionConfig(),
      output_audio_transcription=types.AudioTranscriptionConfig(),
      enable_affective_dialog=True,
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.input_audio_transcription
      is not None
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.output_audio_transcription
      is not None
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.enable_affective_dialog
      is True
  )


def test_streaming_with_session_resumption_config():
  """Test streaming with multiple audio transcription configurations."""
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with multiple audio transcription configs
  run_config = RunConfig(
      session_resumption=types.SessionResumptionConfig(transparent=True),
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1
  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.session_resumption
      is not None
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.session_resumption.transparent
      is True
  )


def test_streaming_with_context_window_compression_config():
  """Test streaming with context window compression config."""
  response = LlmResponse(turn_complete=True)

  mock_model = testing_utils.MockModel.create([response])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )

  # Create run config with context window compression
  run_config = RunConfig(
      context_window_compression=types.ContextWindowCompressionConfig(
          trigger_tokens=1000,
          sliding_window=types.SlidingWindow(target_tokens=500),
      )
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )

  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1

  # Get the request that was captured
  llm_request_sent_to_mock = mock_model.requests[0]

  # Assert that the request contained the correct configuration
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.context_window_compression
      is not None
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.context_window_compression.trigger_tokens
      == 1000
  )
  assert (
      llm_request_sent_to_mock.live_connect_config.context_window_compression.sliding_window.target_tokens
      == 500
  )


def test_streaming_with_avatar_config():
  """Test avatar_config propagation and video content through run_live.

  Verifies:
    1. avatar_config from RunConfig is propagated to live_connect_config.
    2. Video inline_data from the model flows through events correctly.
  """
  # Mock model returns video content followed by turn_complete.
  video_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      data=b'video_data', mime_type='video/mp4'
                  )
              )
          ],
      ),
  )
  turn_complete_response = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create(
      [video_response, turn_complete_response]
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['VIDEO']
  )

  run_config = RunConfig(
      response_modalities=['VIDEO'],
      avatar_config=types.AvatarConfig(avatar_name='Kai'),
  )

  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue, run_config)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
  assert len(mock_model.requests) == 1

  # 1. Verify avatar_config was propagated to the live_connect_config.
  llm_request_sent_to_mock = mock_model.requests[0]
  assert llm_request_sent_to_mock.live_connect_config is not None
  assert llm_request_sent_to_mock.live_connect_config.avatar_config is not None
  assert (
      llm_request_sent_to_mock.live_connect_config.avatar_config.avatar_name
      == 'Kai'
  )

  # 2. Verify video content flows through events.
  video_events = [
      e
      for e in res_events
      if e.content
      and e.content.parts
      and any(
          p.inline_data
          and p.inline_data.mime_type
          and p.inline_data.mime_type.startswith('video/')
          for p in e.content.parts
      )
  ]
  assert video_events, 'Expected at least one event with video inline_data.'

  video_event = video_events[0]
  assert video_event.content.role == 'model'
  video_part = video_event.content.parts[0]
  assert video_part.inline_data is not None
  assert video_part.inline_data.data == b'video_data'
  assert video_part.inline_data.mime_type == 'video/mp4'


def test_streaming_default_model_when_not_specified(mocker):
  """Test streaming uses default model when not specified in live mode."""
  from google.adk.agents import LlmAgent
  from google.adk.models.registry import LLMRegistry

  response1 = LlmResponse(turn_complete=True)
  mock_model = testing_utils.MockModel.create([response1])

  mock_new_llm = mocker.patch.object(
      LLMRegistry, 'new_llm', return_value=mock_model
  )

  # Save original default
  original_default = LlmAgent._default_live_model

  try:
    LlmAgent.set_default_live_model('my-custom-live-model')

    root_agent = Agent(
        name='root_agent',
        tools=[],
    )

    import asyncio
    from contextlib import aclosing

    from google.adk.agents.run_config import RunConfig
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    runner = Runner(
        app_name='test_app',
        agent=root_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )

    live_request_queue = LiveRequestQueue()
    live_request_queue.send_realtime(
        blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
    )

    async def run_test():
      session = await runner.session_service.create_session(
          app_name='test_app', user_id='test_user'
      )
      run_config = RunConfig(response_modalities=['AUDIO'])
      async with aclosing(
          runner.run_live(
              user_id=session.user_id,
              session_id=session.id,
              live_request_queue=live_request_queue,
              run_config=run_config,
          )
      ) as agen:
        async for event in agen:
          # We just need to trigger the resolution
          break

    asyncio.run(run_test())

    mock_new_llm.assert_any_call('my-custom-live-model')

  finally:
    # Restore original default
    LlmAgent.set_default_live_model(original_default)
