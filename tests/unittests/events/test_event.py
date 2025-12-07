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

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

def test_is_final_response_with_skip_synthesis():
  """Test that is_final_response returns True when skip_synthesis is True."""
  event = Event(
      author='agent',
      content=types.Content(role='model', parts=[types.Part(text='response')]),
      actions=EventActions(skip_synthesis=True),
  )
  assert event.is_final_response() is True

def test_is_final_response_without_skip_synthesis():
  """Test that is_final_response returns False/True correctly without skip_synthesis."""
  # Case 1: Normal text response -> True
  event = Event(
      author='agent',
      content=types.Content(role='model', parts=[types.Part(text='response')]),
  )
  assert event.is_final_response() is True

  # Case 2: Function call -> False
  event_fc = Event(
      author='agent',
      content=types.Content(
          role='model', 
          parts=[types.Part(function_call=types.FunctionCall(name='foo', args={}))]
      ),
  )
  assert event_fc.is_final_response() is False
