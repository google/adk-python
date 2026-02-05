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

"""Tests for Event model helpers."""

from google.adk.events.event import Event
from google.genai import types


def test_is_final_response_false_when_turn_incomplete():
  """Event is not final when turn_complete is explicitly False."""
  event = Event(
      author='agent',
      turn_complete=False,
      content=types.Content(role='model', parts=[types.Part(text='partial')]),
  )

  assert not event.is_final_response()


def test_is_final_response_true_when_turn_complete():
  """Event is final for plain text response when turn is complete."""
  event = Event(
      author='agent',
      turn_complete=True,
      content=types.Content(role='model', parts=[types.Part(text='done')]),
  )

  assert event.is_final_response()
