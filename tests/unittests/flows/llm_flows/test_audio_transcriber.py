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

"""Backward-compatibility tests for AudioTranscriber re-export."""

from __future__ import annotations

from google.adk.flows.llm_flows import audio_transcriber as legacy_module
from google.adk.flows.llm_flows.audio_transcriber import AudioTranscriber as LegacyAudioTranscriber
from google.adk.live._audio_transcriber import AudioTranscriber


def test_audio_transcriber_reexport():
  assert LegacyAudioTranscriber is AudioTranscriber
  assert getattr(legacy_module, 'AudioTranscriber') is AudioTranscriber
