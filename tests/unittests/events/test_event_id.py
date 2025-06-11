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

"""Tests for Event ID generation to prevent collisions."""

from concurrent.futures import ThreadPoolExecutor
import re
import threading

from google.adk.events.event import Event
from google.genai import types
import pytest


def test_event_id_is_uuid_format():
  """Test that Event IDs follow UUID4 format."""
  event = Event(
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(text='Test message')]
      ),
  )

  # UUID4 pattern: 8-4-4-4-12 hex digits with version 4
  uuid_pattern = re.compile(
      r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
  )

  assert uuid_pattern.match(
      event.id
  ), f'Event ID {event.id} is not a valid UUID4'


def test_event_id_uniqueness_sequential():
  """Test that sequentially created Event IDs are unique."""
  event_ids = []

  for _ in range(1000):
    event = Event(
        author='test_agent',
        content=types.Content(
            role='model', parts=[types.Part(text='Test message')]
        ),
    )
    event_ids.append(event.id)

  unique_ids = set(event_ids)
  assert len(unique_ids) == len(
      event_ids
  ), f'Found {len(event_ids) - len(unique_ids)} duplicate IDs'


def test_event_id_uniqueness_concurrent():
  """Test that concurrently created Event IDs are unique."""

  def create_events(count=100):
    """Create events in a thread and return their IDs."""
    event_ids = []
    for _ in range(count):
      event = Event(
          author='test_agent',
          content=types.Content(
              role='model', parts=[types.Part(text='Test message')]
          ),
      )
      event_ids.append(event.id)
    return event_ids

  all_event_ids = []

  # Create events concurrently across multiple threads
  with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(create_events, 100) for _ in range(10)]

    for future in futures:
      batch_ids = future.result()
      all_event_ids.extend(batch_ids)

  unique_ids = set(all_event_ids)
  assert len(unique_ids) == len(
      all_event_ids
  ), f'Found {len(all_event_ids) - len(unique_ids)} duplicate IDs'


def test_event_id_streaming_scenario():
  """Test Event ID uniqueness in a streaming scenario with tool calls."""
  events = []

  # User message
  events.append(
      Event(
          author='user',
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      text='Please call a tool and inform me before running it'
                  )
              ],
          ),
      )
  )

  # Agent response before tool call
  events.append(
      Event(
          author='test_agent',
          content=types.Content(
              role='model',
              parts=[types.Part(text='I will now call the tool for you.')],
          ),
      )
  )

  # Tool call event
  events.append(
      Event(
          author='test_agent',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='test_tool', args={'param': 'value'}
                      )
                  )
              ],
          ),
      )
  )

  # Multiple intermediate streaming responses (simulating the issue scenario)
  for i in range(50):
    events.append(
        Event(
            author='test_agent',
            content=types.Content(
                role='model',
                parts=[types.Part(text=f'Intermediate response {i}')],
            ),
            partial=True,  # These are streaming partial responses
        )
    )

  # Tool response
  events.append(
      Event(
          author='test_agent',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name='test_tool', response={'result': 'success'}
                      )
                  )
              ],
          ),
      )
  )

  # Final agent response
  events.append(
      Event(
          author='test_agent',
          content=types.Content(
              role='model',
              parts=[types.Part(text='Tool execution completed successfully.')],
          ),
      )
  )

  # Check all IDs are unique
  event_ids = [event.id for event in events]
  unique_ids = set(event_ids)

  assert len(unique_ids) == len(event_ids), (
      f'Found {len(event_ids) - len(unique_ids)} duplicate IDs in streaming'
      ' scenario'
  )


def test_event_id_persistence():
  """Test that Event ID doesn't change after creation."""
  event = Event(
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(text='Test message')]
      ),
  )

  original_id = event.id

  # Accessing the ID multiple times should return the same value
  assert event.id == original_id
  assert event.id == original_id

  # Modifying the event shouldn't change the ID
  event.content.parts[0].text = 'Modified message'
  assert event.id == original_id


def test_event_id_not_empty():
  """Test that Event ID is never empty."""
  event = Event(
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(text='Test message')]
      ),
  )

  assert event.id, 'Event ID should not be empty'
  assert len(event.id) > 0, 'Event ID should have non-zero length'


def test_event_id_manual_assignment():
  """Test that manually assigned Event IDs are preserved."""
  custom_id = 'custom-event-id-123'

  event = Event(
      id=custom_id,
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(text='Test message')]
      ),
  )

  assert event.id == custom_id, 'Manually assigned Event ID should be preserved'
