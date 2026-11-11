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

"""Typed events exchanged between live transforms and CascadeLiveConnection.

STT transforms (`LiveIngress`) emit `IngressEvent`s: interim transcripts, user
speech start and completed user turns. TTS transforms (`LiveEgress`) emit
`EgressEvent`s: synthesized audio and the text that was spoken.
"""

from __future__ import annotations

from typing import Union

from google.genai import types
from pydantic import BaseModel

from ..features import experimental
from ..features import FeatureName

# --------------------------------------------------------------------------
# Ingress events: events emitted during speech-to-text recognition.
# --------------------------------------------------------------------------


@experimental(FeatureName.CASCADE_LIVE)
class PartialTranscript(BaseModel):
  """Interim recognition result. Surfaces to the client as live captions."""

  text: str
  """The full hypothesis so far for the current utterance, not a delta.

  Each event replaces the previous one; STTs may revise earlier words.
  """


@experimental(FeatureName.CASCADE_LIVE)
class UserTurnFinished(BaseModel):
  """A completed user utterance emitted by ingress when endpointing occurs."""

  text: str
  """The final transcript of the utterance. Sent to the LLM as the user turn."""


@experimental(FeatureName.CASCADE_LIVE)
class UserSpeechStarted(BaseModel):
  """Emitted when the start of user speech is detected."""


IngressEvent = Union[PartialTranscript, UserTurnFinished, UserSpeechStarted]

# --------------------------------------------------------------------------
# Egress events: events emitted during speech synthesis.
# --------------------------------------------------------------------------


@experimental(FeatureName.CASCADE_LIVE)
class AudioChunk(BaseModel):
  """Synthesized audio bound for the client."""

  blob: types.Blob
  """The audio bytes and their MIME type, e.g. `audio/pcm;rate=24000`."""


@experimental(FeatureName.CASCADE_LIVE)
class AgentSpokenOutput(BaseModel):
  """Assistant text synthesized and delivered to the user, appended to history."""

  text: str
  """The text just synthesized, e.g. one sentence, with its whitespace.

  Emitted after its audio. The connection forwards it as a partial output
  transcription and concatenates these as-is into the turn's history entry,
  so keep the whitespace between sentences.
  """


EgressEvent = Union[AudioChunk, AgentSpokenOutput]
