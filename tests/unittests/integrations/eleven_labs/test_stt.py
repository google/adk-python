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

"""Tests for ElevenLabsSTT."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest import mock

from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs._stt import _sample_rate_of
from google.adk.live import PartialTranscript
from google.adk.live import UserSpeechStarted
from google.adk.live import UserTurnFinished
from google.genai import types
import pytest


def test_sample_rate_of_pcm():
  blob = types.Blob(data=b"\x00\x00", mime_type="audio/pcm;rate=24000")
  assert _sample_rate_of(blob) == 24000


def test_sample_rate_of_default():
  blob = types.Blob(data=b"\x00\x00", mime_type="")
  assert _sample_rate_of(blob) == 16000


def test_sample_rate_of_invalid_mime_raises():
  blob = types.Blob(data=b"\x00\x00", mime_type="audio/mp3")
  with pytest.raises(
      ValueError, match="Expected audio/pcm or audio/l16 format"
  ):
    _sample_rate_of(blob)


def test_init_defaults():
  stt = ElevenLabsSTT()
  assert stt._model_id == "scribe_v2_realtime"
  assert stt._vad_silence_threshold_secs == 0.8
  assert stt._vad_threshold == 0.4


@pytest.mark.asyncio
async def test_stt_transcription_events():
  fake_elevenlabs = mock.MagicMock()
  handlers = {}

  fake_connection = mock.MagicMock()
  fake_connection.send = mock.AsyncMock()
  fake_connection.commit = mock.AsyncMock()
  fake_connection.close = mock.AsyncMock()

  def fake_on(event_name, handler):
    handlers[event_name] = handler

  fake_connection.on.side_effect = fake_on

  fake_client = mock.MagicMock()
  fake_client.speech_to_text.realtime.connect = mock.AsyncMock(
      return_value=fake_connection
  )
  fake_elevenlabs.AsyncElevenLabs.return_value = fake_client

  stt = ElevenLabsSTT(api_key="test-key", flush_timeout_secs=0.5)

  async def audio_stream() -> AsyncIterator[types.Blob]:
    # 16000 rate * 2 bytes * 100 ms / 1000 = 3200 bytes per chunk
    yield types.Blob(data=b"\x00" * 3200, mime_type="audio/pcm;rate=16000")
    # Simulate recognizer callbacks
    handlers[fake_elevenlabs.RealtimeEvents.PARTIAL_TRANSCRIPT](
        {"text": "Hello"}
    )
    handlers[fake_elevenlabs.RealtimeEvents.COMMITTED_TRANSCRIPT](
        {"text": "Hello world"}
    )
    yield types.Blob(data=b"", mime_type="audio/pcm;rate=16000")

  with (
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.load_elevenlabs",
          return_value=fake_elevenlabs,
      ),
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.resolve_api_key",
          return_value="test-key",
      ),
  ):
    events = []
    async for event in stt(audio_stream()):
      events.append(event)

  fake_connection.send.assert_awaited()
  fake_connection.commit.assert_awaited_once()
  fake_connection.close.assert_awaited_once()

  assert len(events) == 3
  assert isinstance(events[0], UserSpeechStarted)
  assert isinstance(events[1], PartialTranscript)
  assert events[1].text == "Hello"
  assert isinstance(events[2], UserTurnFinished)
  assert events[2].text == "Hello world"


@pytest.mark.asyncio
async def test_stt_unsupported_sample_rate():
  stt = ElevenLabsSTT(api_key="test-key")

  async def audio_stream():
    yield types.Blob(data=b"\x00" * 100, mime_type="audio/pcm;rate=12345")

  with (
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.load_elevenlabs"
      ) as mock_load,
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.resolve_api_key",
          return_value="test-key",
      ),
  ):
    mock_load.return_value = mock.MagicMock()
    with pytest.raises(ValueError, match="does not accept 12345 Hz PCM"):
      async for _ in stt(audio_stream()):
        pass


@pytest.mark.asyncio
async def test_stt_error_event_raises():
  fake_elevenlabs = mock.MagicMock()
  handlers = {}

  fake_connection = mock.MagicMock()
  fake_connection.send = mock.AsyncMock()
  fake_connection.commit = mock.AsyncMock()
  fake_connection.close = mock.AsyncMock()

  def fake_on(event_name, handler):
    handlers[event_name] = handler

  fake_connection.on.side_effect = fake_on

  fake_client = mock.MagicMock()
  fake_client.speech_to_text.realtime.connect = mock.AsyncMock(
      return_value=fake_connection
  )
  fake_elevenlabs.AsyncElevenLabs.return_value = fake_client

  stt = ElevenLabsSTT(api_key="test-key")

  async def audio_stream():
    yield types.Blob(data=b"\x00" * 3200, mime_type="audio/pcm;rate=16000")
    handlers[fake_elevenlabs.RealtimeEvents.ERROR]({"error": "Auth failed"})
    yield types.Blob(data=b"", mime_type="audio/pcm;rate=16000")

  with (
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.load_elevenlabs",
          return_value=fake_elevenlabs,
      ),
      mock.patch(
          "google.adk.integrations.eleven_labs._stt.resolve_api_key",
          return_value="test-key",
      ),
  ):
    with pytest.raises(
        RuntimeError, match="ElevenLabs Scribe reported: Auth failed"
    ):
      async for _ in stt(audio_stream()):
        pass
