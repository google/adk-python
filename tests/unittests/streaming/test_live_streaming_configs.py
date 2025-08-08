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
      is None
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
