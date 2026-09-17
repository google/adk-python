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

"""Unit tests for TranscriptionEntry data model."""

from google.adk.agents.transcription_entry import TranscriptionEntry as CompatTranscriptionEntry
import google.adk.live as live
from google.adk.live._transcription_entry import TranscriptionEntry
from google.genai import types
from pydantic import ValidationError
import pytest


def test_transcription_entry_backward_compat_identity():
  """Verifies that the backward compatibility export is the exact same class."""
  assert CompatTranscriptionEntry is TranscriptionEntry


def test_transcription_entry_not_in_live_facade():
  """Verifies that internal runtime model is not exported in live public facade."""
  assert "TranscriptionEntry" not in live.__all__
  assert not hasattr(live, "TranscriptionEntry")


def test_transcription_entry_with_blob():
  """Verifies creation with a Blob payload."""
  blob = types.Blob(data=b"audio_bytes", mime_type="audio/pcm")
  entry = TranscriptionEntry(role="user", data=blob)

  assert entry.role == "user"
  assert entry.data == blob


def test_transcription_entry_with_content():
  """Verifies creation with Content payload and optional role."""
  content = types.Content(
      role="model", parts=[types.Part.from_text(text="hello")]
  )
  entry = TranscriptionEntry(data=content)

  assert entry.role is None
  assert entry.data == content


def test_transcription_entry_extra_fields_forbidden():
  """Verifies that extra attributes are rejected by pydantic configuration."""
  blob = types.Blob(data=b"test", mime_type="audio/pcm")
  with pytest.raises(ValidationError):
    TranscriptionEntry(data=blob, unexpected_field="invalid")
