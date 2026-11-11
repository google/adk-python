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

"""The transforms that adapt audio and text streams in a cascaded live agent.

::

    wire -> STT -> MODEL -> TTS -> wire

- **STT** (`LiveIngress`) consumes client audio blobs and emits typed text and
  turn-boundary events (`IngressEvent`).
- **TTS** (`LiveEgress`) consumes LLM text deltas and emits synthesized
  audio chunks and delivery events (`EgressEvent`).
"""

from __future__ import annotations

import abc
import asyncio
from typing import AsyncGenerator
from typing import AsyncIterator

from google.genai import types

from ..features import experimental
from ..features import FeatureName
from ._cascade_live_events import EgressEvent
from ._cascade_live_events import IngressEvent


@experimental(FeatureName.CASCADE_LIVE)
class LiveIngress(abc.ABC):
  """Transforms an incoming audio stream into text and turn-boundary events.

  Implementations consume raw audio blobs and yield `IngressEvent` instances,
  emitting `UserTurnFinished` when a complete user utterance is recognized.
  """

  @abc.abstractmethod
  def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[IngressEvent, None]:
    ...


@experimental(FeatureName.CASCADE_LIVE)
class LiveEgress(abc.ABC):
  """Transforms a text stream into synthesized audio chunks and delivery events.

  Implementations buffer and segment streamed text deltas, handle optional
  filtering, and synthesize audio chunks. Receives a ``cancel`` event to abort
  in-flight synthesis on barge-in.
  """

  @abc.abstractmethod
  def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[EgressEvent, None]:
    ...
