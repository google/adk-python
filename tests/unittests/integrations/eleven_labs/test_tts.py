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

"""Tests for ElevenLabsTTS."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import AsyncGenerator
from typing import AsyncIterator
from unittest import mock

from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs import ElevenLabsTTS
from google.adk.integrations.eleven_labs._tts import _take_sentence
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import CascadeLive
from google.adk.models.base_llm import BaseLlm
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest
from typing_extensions import override


def test_take_sentence_punctuation():
  sentence, rest = _take_sentence("Hello world! How are you?", 100)
  assert sentence == "Hello world! "
  assert rest == "How are you?"


def test_take_sentence_no_boundary():
  sentence, rest = _take_sentence("Hello world", 100)
  assert sentence is None
  assert rest == "Hello world"


def test_take_sentence_length_fallback():
  sentence, rest = _take_sentence("one two three four five six", 14)
  assert sentence == "one two three "
  assert rest == "four five six"


def test_init_unsupported_sample_rate():
  with pytest.raises(ValueError, match="ElevenLabs cannot emit 12345 Hz PCM"):
    ElevenLabsTTS(sample_rate=12345)


@pytest.mark.asyncio
async def test_tts_synthesis_flow():
  fake_elevenlabs = mock.MagicMock()
  fake_client = mock.MagicMock()

  async def fake_stream(*args, **kwargs):
    yield b"\x01\x02\x03\x04"

  fake_stream_obj = mock.MagicMock()
  fake_stream_obj.__aiter__.side_effect = fake_stream
  fake_stream_obj.aclose = mock.AsyncMock()

  fake_client.text_to_speech.stream.return_value = fake_stream_obj
  fake_elevenlabs.AsyncElevenLabs.return_value = fake_client

  tts = ElevenLabsTTS(api_key="test-key")

  async def text_stream() -> AsyncIterator[str]:
    yield "Hello there! "
    yield "How are you today?"

  cancel = asyncio.Event()

  with (
      mock.patch(
          "google.adk.integrations.eleven_labs._tts.load_elevenlabs",
          return_value=fake_elevenlabs,
      ),
      mock.patch(
          "google.adk.integrations.eleven_labs._tts.resolve_api_key",
          return_value="test-key",
      ),
  ):
    events = []
    async for event in tts(text_stream(), cancel=cancel):
      events.append(event)

  assert [
      call.kwargs["text"]
      for call in fake_client.text_to_speech.stream.call_args_list
  ] == ["Hello there!", "How are you today?"]
  audio_events = [e for e in events if isinstance(e, AudioChunk)]
  spoken_events = [e for e in events if isinstance(e, AgentSpokenOutput)]
  assert len(audio_events) == 2
  # Spoken text keeps its whitespace, so it concatenates back to the reply.
  assert [e.text for e in spoken_events] == [
      "Hello there! ",
      "How are you today?",
  ]


@pytest.mark.asyncio
async def test_tts_cancel_barge_in():
  fake_elevenlabs = mock.MagicMock()
  fake_client = mock.MagicMock()

  cancel = asyncio.Event()

  async def fake_stream(*args, **kwargs):
    yield b"\x01\x02"
    cancel.set()
    yield b"\x03\x04"

  fake_stream_obj = mock.MagicMock()
  fake_stream_obj.__aiter__.side_effect = fake_stream
  fake_stream_obj.aclose = mock.AsyncMock()

  fake_client.text_to_speech.stream.return_value = fake_stream_obj
  fake_elevenlabs.AsyncElevenLabs.return_value = fake_client

  tts = ElevenLabsTTS(api_key="test-key")

  async def text_stream():
    yield "A very long sentence that gets interrupted by the user."

  with (
      mock.patch(
          "google.adk.integrations.eleven_labs._tts.load_elevenlabs",
          return_value=fake_elevenlabs,
      ),
      mock.patch(
          "google.adk.integrations.eleven_labs._tts.resolve_api_key",
          return_value="test-key",
      ),
  ):
    events = []
    async for event in tts(text_stream(), cancel=cancel):
      events.append(event)

  # AgentSpokenOutput event should NOT be emitted because of cancellation
  spoken_events = [e for e in events if isinstance(e, AgentSpokenOutput)]
  assert not spoken_events


class _ScriptedLlm(BaseLlm):
  """Replays one response list per request and records the requests."""

  model: str = "scripted"
  turns: list[list[LlmResponse]] = []
  requests: list[LlmRequest] = []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request)
    for response in self.turns[len(self.requests) - 1]:
      yield response


def _model_text(text: str, *, partial: bool = False) -> LlmResponse:
  return LlmResponse(
      content=types.Content(role="model", parts=[types.Part(text=text)]),
      partial=partial or None,
  )


async def _final_transcription(connection: BaseLlmConnection) -> str:
  """Reads one turn and returns its final output transcription."""

  async def _read() -> str:
    final = ""
    async with aclosing(connection.receive()) as responses:
      async for response in responses:
        transcription = response.output_transcription
        if transcription and transcription.finished:
          final = transcription.text
        if response.turn_complete:
          return final
    return final

  return await asyncio.wait_for(_read(), timeout=5.0)


@pytest.mark.asyncio
async def test_cascade_keeps_the_spaces_between_sentences():
  """History and the final transcription keep the spaces between sentences.

  Regression: each sentence was stripped and `CascadeLive` joins them with
  '', so "Hello there. How are you?" lost the space after the period.
  """
  reply = "Hello there. How are you?"
  llm = _ScriptedLlm(
      turns=[
          [
              _model_text("Hello there. How ", partial=True),
              _model_text("are you?", partial=True),
              _model_text(reply),
          ],
          [_model_text("Bye.")],
      ]
  )

  async def fake_stream(*args, **kwargs):
    del args, kwargs
    yield b"\x01\x02"

  fake_stream_obj = mock.MagicMock()
  fake_stream_obj.__aiter__.side_effect = fake_stream
  fake_stream_obj.aclose = mock.AsyncMock()
  fake_client = mock.MagicMock()
  fake_client.text_to_speech.stream.return_value = fake_stream_obj
  fake_elevenlabs = mock.MagicMock()
  fake_elevenlabs.AsyncElevenLabs.return_value = fake_client

  # Typed text drives the turns, so the STT is never called.
  model = CascadeLive(
      model=llm,
      stt=ElevenLabsSTT(api_key="test-key"),
      tts=ElevenLabsTTS(api_key="test-key"),
  )
  with mock.patch(
      "google.adk.integrations.eleven_labs._tts.load_elevenlabs",
      return_value=fake_elevenlabs,
  ):
    async with model.connect(LlmRequest()) as connection:
      await connection.send_content(
          types.Content(role="user", parts=[types.Part(text="Hi")])
      )
      first = await _final_transcription(connection)
      await connection.send_content(
          types.Content(role="user", parts=[types.Part(text="Bye")])
      )
      await _final_transcription(connection)

  # Only the text sent for synthesis is stripped.
  assert [
      call.kwargs["text"]
      for call in fake_client.text_to_speech.stream.call_args_list
  ] == ["Hello there.", "How are you?", "Bye."]
  assert first == reply
  assert llm.requests[-1].contents[1] == types.Content(
      role="model", parts=[types.Part(text=reply)]
  )
