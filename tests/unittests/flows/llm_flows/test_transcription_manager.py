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

"""Backward-compatibility tests for TranscriptionManager re-export."""

from __future__ import annotations

from google.adk.flows.llm_flows import transcription_manager as legacy_module
from google.adk.flows.llm_flows.transcription_manager import TranscriptionManager as LegacyTranscriptionManager
from google.adk.live._transcription_manager import TranscriptionManager


def test_transcription_manager_reexport():
  assert LegacyTranscriptionManager is TranscriptionManager
  assert getattr(legacy_module, 'TranscriptionManager') is TranscriptionManager
