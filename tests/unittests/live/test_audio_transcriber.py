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

"""Unit tests for AudioTranscriber in live package."""

from typing import Any
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.agents.transcription_entry import TranscriptionEntry
from google.adk.live._audio_transcriber import AudioTranscriber
from google.genai import types
import pytest

from .. import testing_utils


class _RecordingSpeechClient:
  """Stands in for speech.SpeechClient, recording what it was asked to do."""

  def __init__(self, transcripts: list[str]):
    self._transcripts = list(transcripts)
    self.audio_contents: list[Any] = []

  def recognize(self, config: Any, audio: Any) -> Any:
    self.audio_contents.append(audio.content)
    transcript = self._transcripts.pop(0)

    class _Alternative:
      pass

    class _Result:
      pass

    class _Response:
      pass

    alternative = _Alternative()
    alternative.transcript = transcript
    result = _Result()
    result.alternatives = [alternative]
    response = _Response()
    response.results = [result]
    return response


def _text_content(role: str, text: str) -> types.Content:
  return types.Content(role=role, parts=[types.Part(text=text)])


def _audio_entry(role: str, data: Optional[bytes]) -> TranscriptionEntry:
  return TranscriptionEntry(
      role=role, data=types.Blob(mime_type='audio/pcm', data=data)
  )


async def _context_with_cache(
    cache: list[TranscriptionEntry],
):
  agent = Agent(
      name='test_agent', model=testing_utils.MockModel.create(responses=[])
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.transcription_cache = cache
  return invocation_context


@pytest.mark.asyncio
async def test_transcribe_file_resets_the_transcription_cache():
  """Consumed entries are cleared so the next turn does not re-transcribe."""
  invocation_context = await _context_with_cache(
      [TranscriptionEntry(role='model', data=_text_content('model', 'hello'))]
  )

  AudioTranscriber().transcribe_file(invocation_context)

  assert invocation_context.transcription_cache == []


@pytest.mark.asyncio
async def test_transcribe_file_passes_text_content_through_in_order():
  """Entries that are already text are returned untouched, in cache order."""
  first = _text_content('user', 'first')
  second = _text_content('model', 'second')
  third = _text_content('user', 'third')
  invocation_context = await _context_with_cache([
      TranscriptionEntry(role='user', data=first),
      TranscriptionEntry(role='model', data=second),
      TranscriptionEntry(role='user', data=third),
  ])

  contents = AudioTranscriber().transcribe_file(invocation_context)

  assert contents == [first, second, third]


@pytest.mark.asyncio
async def test_transcribe_file_skips_blobs_with_no_audio_data():
  """An empty blob contributes nothing rather than an empty segment."""
  text = _text_content('model', 'hello')
  invocation_context = await _context_with_cache([
      _audio_entry('user', b''),
      TranscriptionEntry(role='model', data=text),
  ])

  contents = AudioTranscriber().transcribe_file(invocation_context)

  assert contents == [text]


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        'bundled audio is stored as raw bytes, so the Blob check in the'
        ' transcription step never matches and audio is never transcribed'
    ),
)
async def test_transcribe_file_transcribes_merged_same_speaker_audio():
  """Consecutive same-speaker blobs become one transcription, in order."""
  interleaved_text = _text_content('model', 'go on')
  invocation_context = await _context_with_cache([
      _audio_entry('user', b'aa'),
      _audio_entry('user', b'bb'),
      TranscriptionEntry(role='model', data=interleaved_text),
      _audio_entry('user', b'cc'),
  ])
  transcriber = AudioTranscriber()
  client = _RecordingSpeechClient(['first half', 'second half'])
  transcriber.client = client

  contents = transcriber.transcribe_file(invocation_context)

  # The two adjacent user blobs are sent as a single request; the blob after
  # the model turn is a separate one.
  assert client.audio_contents == [b'aabb', b'cc']
  assert contents == [
      _text_content('user', 'first half'),
      interleaved_text,
      _text_content('user', 'second half'),
  ]
