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
from __future__ import annotations

from typing import TYPE_CHECKING

from google.cloud import speech
from google.genai import types as genai_types

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext


class AudioTranscriber:
  """Transcribes audio using Google Cloud Speech-to-Text."""

  def __init__(self, init_client=False):
    if init_client:
      self.client = speech.SpeechClient()

  def transcribe_file(
      self, invocation_context: InvocationContext
  ) -> list[genai_types.Content]:
    """Transcribe audio, bundling consecutive segments from the same speaker.

    The ordering of speakers will be preserved. Audio blobs will be merged for
    the same speaker as much as we can do reduce the transcription latency.

    Args:
        invocation_context: The invocation context to access the transcription
          cache.

    Returns:
        A list of Content objects containing the transcribed text.
    """

    bundled_audio = []
    current_speaker = None
    current_audio_data = b''
    contents = []

    # Step1: merge audio blobs
    for transcription_entry in invocation_context.transcription_cache or []:
      speaker, audio_data = (
          transcription_entry.role,
          transcription_entry.data,
      )

      if isinstance(audio_data, genai_types.Content):
        if current_speaker is not None:
          bundled_audio.append((current_speaker, current_audio_data))
          current_speaker = None
          current_audio_data = b''
        bundled_audio.append((speaker, audio_data))
        continue

      if not audio_data.data:
        continue

      if speaker == current_speaker:
        current_audio_data += audio_data.data
      else:
        if current_speaker is not None:
          bundled_audio.append((current_speaker, current_audio_data))
        current_speaker = speaker
        current_audio_data = audio_data.data

    # Append the last audio segment if any
    if current_speaker is not None:
      bundled_audio.append((current_speaker, current_audio_data))

    # reset cache
    invocation_context.transcription_cache = []

    # Step2: transcription
    for speaker, data in bundled_audio:
      if isinstance(data, genai_types.Blob):
        audio = speech.RecognitionAudio(content=data)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US',
        )

        response = self.client.recognize(config=config, audio=audio)

        for result in response.results:
          transcript = result.alternatives[0].transcript

          parts = [genai_types.Part(text=transcript)]
          role = speaker.lower()
          content = genai_types.Content(role=role, parts=parts)
          contents.append(content)
      else:
        # don't need to transcribe model which are already text
        contents.append(data)

    return contents
