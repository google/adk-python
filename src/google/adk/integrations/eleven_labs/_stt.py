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

"""A `LiveIngress` backed by ElevenLabs Scribe v2 Realtime.

Docs:
  https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/server-side-streaming
  https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Dict
from typing import Optional

from google.genai import types
from pydantic import BaseModel

from ...features import experimental
from ...features import FeatureName
from ...live._cascade_live_events import IngressEvent
from ...live._cascade_live_events import PartialTranscript
from ...live._cascade_live_events import UserSpeechStarted
from ...live._cascade_live_events import UserTurnFinished
from ...live._transforms import LiveIngress
from ._client import load_elevenlabs
from ._client import resolve_api_key

logger = logging.getLogger("google_adk." + __name__)

# Supported PCM sample rates for Scribe Realtime.
_SUPPORTED_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

_DEFAULT_SAMPLE_RATE = 16000

# 16-bit mono, so two bytes per sample.
_BYTES_PER_SAMPLE = 2

# Pushed onto the event queue when the ingress has nothing further to report.
_DONE = object()


class _Failure(BaseModel):
  """An error message from the recognizer, routed through the event queue."""

  message: str


def _sample_rate_of(blob: types.Blob) -> int:
  """Returns the sample rate from a PCM blob's mime type, defaulting to 16000."""
  mime_type = (blob.mime_type or "").lower()
  if mime_type and not mime_type.startswith(("audio/pcm", "audio/l16")):
    raise ValueError(
        "Expected audio/pcm or audio/l16 format, but received"
        f" {blob.mime_type!r}."
    )

  for parameter in mime_type.split(";")[1:]:
    name, _, value = parameter.partition("=")
    if name.strip() == "rate":
      try:
        return int(value)
      except ValueError:
        break
  return _DEFAULT_SAMPLE_RATE


@experimental(FeatureName.ELEVEN_LABS)
class ElevenLabsSTT(LiveIngress):
  """LiveIngress adapter for ElevenLabs Scribe v2 Realtime STT."""

  def __init__(
      self,
      *,
      model_id: str = "scribe_v2_realtime",
      language_code: Optional[str] = None,
      vad_silence_threshold_secs: float = 0.8,
      vad_threshold: float = 0.4,
      min_speech_duration_ms: int = 100,
      min_silence_duration_ms: int = 100,
      min_chunk_ms: int = 100,
      flush_timeout_secs: float = 3.0,
      api_key: Optional[str] = None,
  ) -> None:
    """Configures the recognizer.

    Args:
      model_id: The Scribe model to transcribe with.
      language_code: Optional ISO-639-1 or ISO-639-3 language code.
      vad_silence_threshold_secs: Duration of silence in seconds before ending a
        turn (0.3 to 3.0).
      vad_threshold: Speech detection threshold between 0.1 and 0.9.
      min_speech_duration_ms: Minimum speech duration in milliseconds.
      min_silence_duration_ms: Minimum silence duration in milliseconds.
      min_chunk_ms: Minimum audio duration in milliseconds to buffer before
        sending.
      flush_timeout_secs: Seconds to wait for final transcript on stream close.
      api_key: Optional API key override. Defaults to ELEVENLABS_API_KEY env var.
    """
    self._model_id = model_id
    self._language_code = language_code
    self._vad_silence_threshold_secs = vad_silence_threshold_secs
    self._vad_threshold = vad_threshold
    self._min_speech_duration_ms = min_speech_duration_ms
    self._min_silence_duration_ms = min_silence_duration_ms
    self._min_chunk_ms = min_chunk_ms
    self._flush_timeout_secs = flush_timeout_secs
    self._api_key = api_key

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[IngressEvent, None]:
    """Transcribes `audio`, yielding ingress events as Scribe reports them."""
    elevenlabs = load_elevenlabs()
    client = elevenlabs.AsyncElevenLabs(api_key=resolve_api_key(self._api_key))

    events: asyncio.Queue[Any] = asyncio.Queue()
    committed = asyncio.Event()
    connection: Any = None
    speaking = False
    last_partial = ""

    def on_partial(data: Dict[str, Any]) -> None:
      nonlocal speaking, last_partial
      text = (data.get("text") or "").strip()
      if not text or text == last_partial:
        return
      last_partial = text
      if not speaking:
        speaking = True
        # Emit speech onset event.
        events.put_nowait(UserSpeechStarted())
      events.put_nowait(PartialTranscript(text=text))

    def on_committed(data: Dict[str, Any]) -> None:
      nonlocal speaking, last_partial
      committed.set()
      last_partial = ""
      speaking = False
      text = (data.get("text") or "").strip()
      if text:
        # Emit complete user turn event.
        events.put_nowait(UserTurnFinished(text=text))

    def on_error(data: Dict[str, Any]) -> None:
      events.put_nowait(
          _Failure(message=str(data.get("error") or data.get("message_type")))
      )

    async def connect(sample_rate: int) -> Any:
      if sample_rate not in _SUPPORTED_RATES:
        raise ValueError(
            f"Scribe Realtime does not accept {sample_rate} Hz PCM. Supported"
            f' rates: {", ".join(str(rate) for rate in _SUPPORTED_RATES)}.'
        )

      options: Dict[str, Any] = {
          "model_id": self._model_id,
          "audio_format": elevenlabs.AudioFormat(f"pcm_{sample_rate}"),
          "sample_rate": sample_rate,
          # Server-side VAD configuration.
          "commit_strategy": elevenlabs.CommitStrategy.VAD,
          "vad_silence_threshold_secs": self._vad_silence_threshold_secs,
          "vad_threshold": self._vad_threshold,
          "min_speech_duration_ms": self._min_speech_duration_ms,
          "min_silence_duration_ms": self._min_silence_duration_ms,
      }
      if self._language_code:
        options["language_code"] = self._language_code

      opened = await client.speech_to_text.realtime.connect(
          elevenlabs.RealtimeAudioOptions(**options)
      )
      opened.on(elevenlabs.RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
      opened.on(elevenlabs.RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
      opened.on(elevenlabs.RealtimeEvents.ERROR, on_error)
      return opened

    async def send(payload: bytes) -> None:
      await connection.send(
          {"audio_base_64": base64.b64encode(payload).decode("ascii")}
      )

    async def pump() -> None:
      """Sends audio chunks to the recognizer and flushes remaining audio on close."""
      nonlocal connection
      buffer = bytearray()
      min_chunk_bytes = 0
      try:
        async for blob in audio:
          if connection is None:
            sample_rate = _sample_rate_of(blob)
            min_chunk_bytes = (
                sample_rate * _BYTES_PER_SAMPLE * self._min_chunk_ms // 1000
            )
            connection = await connect(sample_rate)
          if not blob.data:
            continue
          buffer.extend(blob.data)
          if len(buffer) >= min_chunk_bytes:
            await send(bytes(buffer))
            buffer.clear()

        if connection is None:
          return

        # Flush and commit any remaining buffered audio.
        if buffer:
          await send(bytes(buffer))
        committed.clear()
        await connection.commit()
        try:
          await asyncio.wait_for(committed.wait(), self._flush_timeout_secs)
        except asyncio.TimeoutError:
          logger.warning(
              "Scribe did not return a transcript for the final audio segment"
              " within %.1fs; the tail of the last utterance was dropped.",
              self._flush_timeout_secs,
          )
      finally:
        events.put_nowait(_DONE)

    pump_task = asyncio.create_task(pump())
    try:
      while True:
        item = await events.get()
        if item is _DONE:
          await pump_task
          return
        if isinstance(item, _Failure):
          raise RuntimeError(f"ElevenLabs Scribe reported: {item.message}")
        yield item
    finally:
      if not pump_task.done():
        pump_task.cancel()
      if connection is not None:
        await connection.close()
