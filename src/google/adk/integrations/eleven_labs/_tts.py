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

"""A `LiveEgress` backed by the ElevenLabs streaming text-to-speech API.

Docs:
  https://elevenlabs.io/docs/api-reference/text-to-speech/stream
  https://elevenlabs.io/docs/overview/models
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Optional
from typing import Tuple

from google.genai import types

from ...features import experimental
from ...features import FeatureName
from ...live._cascade_live_events import AgentSpokenOutput
from ...live._cascade_live_events import AudioChunk
from ...live._cascade_live_events import EgressEvent
from ...live._transforms import LiveEgress
from ._client import load_elevenlabs
from ._client import resolve_api_key

# Default ElevenLabs voice ID ("George").
_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# Supported PCM sample rates for ElevenLabs TTS output.
_SUPPORTED_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# Matches sentence boundaries followed by whitespace or newlines.
_SENTENCE_END = re.compile(r'[.!?\u2026]["\'\u201d\u2019)\]]*\s+|\n+')


def _take_sentence(buffer: str, max_chars: int) -> Tuple[Optional[str], str]:
  """Splits a leading sentence from `buffer`.

  The sentence keeps its trailing whitespace, so the pieces concatenate back
  to the original text. Returns `(None, buffer)` if no boundary is found and
  the buffer is under `max_chars`. Otherwise splits at the last word boundary.
  """
  match = _SENTENCE_END.search(buffer)
  if match:
    return buffer[: match.end()], buffer[match.end() :]

  if len(buffer) >= max_chars:
    cut = buffer.rfind(" ", 0, max_chars)
    if cut > 0:
      return buffer[: cut + 1], buffer[cut + 1 :]

  return None, buffer


@experimental(FeatureName.ELEVEN_LABS)
class ElevenLabsTTS(LiveEgress):
  """LiveEgress adapter for ElevenLabs streaming text-to-speech."""

  def __init__(
      self,
      *,
      voice_id: str = _DEFAULT_VOICE_ID,
      model_id: str = "eleven_flash_v2_5",
      sample_rate: int = 24000,
      max_sentence_chars: int = 240,
      api_key: Optional[str] = None,
  ) -> None:
    """Configures the synthesizer.

    Args:
      voice_id: ElevenLabs voice ID to synthesize with.
      model_id: ElevenLabs model ID for synthesis (defaults to 'eleven_flash_v2_5').
      sample_rate: PCM sample rate in Hz.
      max_sentence_chars: Maximum characters to buffer before forcing a sentence
        boundary.
      api_key: Optional API key override. Defaults to ELEVENLABS_API_KEY env var.
    """
    if sample_rate not in _SUPPORTED_RATES:
      raise ValueError(
          f"ElevenLabs cannot emit {sample_rate} Hz PCM. Supported rates:"
          f' {", ".join(str(rate) for rate in _SUPPORTED_RATES)}.'
      )
    self._voice_id = voice_id
    self._model_id = model_id
    self._output_format = f"pcm_{sample_rate}"
    self._mime_type = f"audio/pcm;rate={sample_rate}"
    self._max_sentence_chars = max_sentence_chars
    self._api_key = api_key

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[EgressEvent, None]:
    """Speaks `text`, yielding audio and a record of what was said."""
    elevenlabs = load_elevenlabs()
    client = elevenlabs.AsyncElevenLabs(api_key=resolve_api_key(self._api_key))

    buffer = ""
    previous_text = ""

    async def speak(piece: str) -> AsyncIterator[EgressEvent]:
      nonlocal previous_text
      # Strip only the synthesized text; `piece` keeps its whitespace.
      sentence = piece.strip()
      if not sentence or cancel.is_set():
        return

      stream = client.text_to_speech.stream(
          voice_id=self._voice_id,
          text=sentence,
          model_id=self._model_id,
          output_format=self._output_format,
          # Pass preceding text to maintain prosody across sentences.
          previous_text=previous_text or None,
      )
      try:
        async for audio in stream:
          if cancel.is_set():
            # On cancellation (e.g. barge-in), discard remaining audio and omit
            # the AgentSpokenOutput event.
            return
          if audio:
            yield AudioChunk(
                blob=types.Blob(data=audio, mime_type=self._mime_type)
            )
      finally:
        await stream.aclose()

      if cancel.is_set():
        return
      previous_text = sentence
      yield AgentSpokenOutput(text=piece)

    async for delta in text:
      if cancel.is_set():
        return
      if not delta:
        continue
      buffer += delta
      while True:
        sentence, buffer = _take_sentence(buffer, self._max_sentence_chars)
        if sentence is None:
          break
        async for event in speak(sentence):
          yield event
        if cancel.is_set():
          return

    # Synthesize any remaining text in the buffer.
    async for event in speak(buffer):
      yield event
