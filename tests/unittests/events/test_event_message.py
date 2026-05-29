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

"""Tests for Event.message as an alias for Event.content."""

from google.adk.events.event import Event
from google.genai import types
import pytest


class TestMessageConstructor:
  """Tests for Event(message=...) constructor parameter."""

  def test_message_str_sets_content(self):
    event = Event(message='Hello!')
    assert event.content is not None
    assert event.content.parts[0].text == 'Hello!'

  def test_message_content_passes_through(self):
    content = types.Content(
        parts=[types.Part(text='from Content')], role='model'
    )
    event = Event(message=content)
    assert event.content is content

  def test_message_part_converts_to_content(self):
    part = types.Part(text='from Part')
    event = Event(message=part)
    assert event.content is not None
    assert event.content.parts[0].text == 'from Part'

  def test_message_list_of_parts(self):
    parts = [types.Part(text='part1'), types.Part(text='part2')]
    event = Event(message=parts)
    assert event.content is not None
    assert len(event.content.parts) == 2
    assert event.content.parts[0].text == 'part1'
    assert event.content.parts[1].text == 'part2'

  def test_message_and_content_raises(self):
    with pytest.raises(ValueError, match='mutually exclusive'):
      Event(
          message='hello',
          content=types.Content(parts=[types.Part(text='world')]),
      )

  def test_content_still_works(self):
    content = types.Content(
        parts=[types.Part(text='via content')], role='model'
    )
    event = Event(content=content)
    assert event.content is content
    assert event.content.parts[0].text == 'via content'

  def test_neither_message_nor_content(self):
    event = Event()
    assert event.content is None


class TestMessageProperty:
  """Tests for Event.message property getter and setter."""

  def test_message_getter_aliases_content(self):
    content = types.Content(parts=[types.Part(text='hello')], role='model')
    event = Event(content=content)
    assert event.message is event.content

  def test_message_getter_none_when_no_content(self):
    event = Event()
    assert event.message is None

  def test_message_setter_updates_content(self):
    event = Event()
    new_content = types.Content(
        parts=[types.Part(text='updated')], role='model'
    )
    event.message = new_content
    assert event.content is new_content

  def test_message_setter_accepts_str(self):
    event = Event()
    event.message = 'updated via setter'
    assert event.content is not None
    assert event.content.parts[0].text == 'updated via setter'

  def test_message_setter_none_clears_content(self):
    event = Event(message='hello')
    event.message = None
    assert event.content is None

  def test_message_from_constructor_readable_via_property(self):
    event = Event(message='Hello!')
    assert event.message is not None
    assert event.message.parts[0].text == 'Hello!'


class TestMessageSerialization:
  """Tests that serialization uses 'content', not 'message'."""

  def test_serialized_uses_content_field(self):
    event = Event(message='Hello!')
    data = event.model_dump(exclude_none=True)
    assert 'content' in data
    assert 'message' not in data

  def test_round_trip_via_content(self):
    event = Event(message='Hello!')
    data = event.model_dump()
    restored = Event.model_validate(data)
    assert restored.content is not None
    assert restored.content.parts[0].text == 'Hello!'
    assert restored.message is not None
    assert restored.message.parts[0].text == 'Hello!'


class TestMessageWithOtherKwargs:
  """Tests message combined with other convenience kwargs."""

  def test_message_with_state(self):
    event = Event(message='hello', state={'key': 'val'})
    assert event.content is not None
    assert event.content.parts[0].text == 'hello'
    assert event.actions.state_delta == {'key': 'val'}

  def test_message_with_route(self):
    event = Event(message='hello', route='next')
    assert event.content is not None
    assert event.actions.route == 'next'
