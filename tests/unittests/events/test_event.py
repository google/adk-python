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

from __future__ import annotations

"""Unit tests for the helper methods on the Event class."""

from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.genai import types
import pytest


def _text_part(text: str = 'hello') -> types.Part:
  return types.Part(text=text)


def _function_call_part(name: str = 'my_func') -> types.Part:
  return types.Part(function_call=types.FunctionCall(name=name, args={'x': 1}))


def _function_response_part(name: str = 'my_func') -> types.Part:
  return types.Part(
      function_response=types.FunctionResponse(name=name, response={'y': 2})
  )


def _code_execution_result_part(output: str = '42') -> types.Part:
  return types.Part(
      code_execution_result=types.CodeExecutionResult(
          outcome=types.Outcome.OUTCOME_OK, output=output
      )
  )


def _event(parts: list[types.Part] | None = None, **kwargs) -> Event:
  content = (
      types.Content(role='model', parts=parts) if parts is not None else None
  )
  return Event(author='agent', content=content, **kwargs)


# --- is_final_response -------------------------------------------------------


def test_is_final_response_plain_text_event_is_final():
  event = _event(parts=[_text_part()])
  assert event.is_final_response() is True


def test_is_final_response_empty_event_is_final():
  event = _event()
  assert event.is_final_response() is True


def test_is_final_response_with_function_call_is_not_final():
  event = _event(parts=[_text_part(), _function_call_part()])
  assert event.is_final_response() is False


def test_is_final_response_with_function_response_is_not_final():
  event = _event(parts=[_function_response_part()])
  assert event.is_final_response() is False


def test_is_final_response_partial_event_is_not_final():
  event = _event(parts=[_text_part()], partial=True)
  assert event.is_final_response() is False


def test_is_final_response_with_trailing_code_result_is_not_final():
  event = _event(parts=[_text_part(), _code_execution_result_part()])
  assert event.is_final_response() is False


def test_is_final_response_skip_summarization_overrides_function_response():
  event = _event(
      parts=[_function_response_part()],
      actions=EventActions(skip_summarization=True),
  )
  assert event.is_final_response() is True


def test_is_final_response_long_running_tool_ids_overrides_function_call():
  event = _event(
      parts=[_function_call_part()], long_running_tool_ids={'tool-1'}
  )
  assert event.is_final_response() is True


# --- get_function_calls ------------------------------------------------------


def test_get_function_calls_returns_calls_in_order():
  event = _event(
      parts=[
          _text_part(),
          _function_call_part('first'),
          _function_response_part(),
          _function_call_part('second'),
      ]
  )
  assert [call.name for call in event.get_function_calls()] == [
      'first',
      'second',
  ]


def test_get_function_calls_no_content_returns_empty():
  assert _event().get_function_calls() == []


def test_get_function_calls_empty_parts_returns_empty():
  assert _event(parts=[]).get_function_calls() == []


def test_get_function_calls_text_only_returns_empty():
  assert _event(parts=[_text_part()]).get_function_calls() == []


# --- get_function_responses --------------------------------------------------


def test_get_function_responses_returns_responses_in_order():
  event = _event(
      parts=[
          _function_response_part('first'),
          _text_part(),
          _function_call_part(),
          _function_response_part('second'),
      ]
  )
  assert [resp.name for resp in event.get_function_responses()] == [
      'first',
      'second',
  ]


def test_get_function_responses_no_content_returns_empty():
  assert _event().get_function_responses() == []


def test_get_function_responses_empty_parts_returns_empty():
  assert _event(parts=[]).get_function_responses() == []


# --- has_trailing_code_execution_result --------------------------------------


def test_has_trailing_code_execution_result_true_when_last():
  event = _event(parts=[_text_part(), _code_execution_result_part()])
  assert event.has_trailing_code_execution_result() is True


def test_has_trailing_code_execution_result_false_when_not_last():
  event = _event(parts=[_code_execution_result_part(), _text_part()])
  assert event.has_trailing_code_execution_result() is False


def test_has_trailing_code_execution_result_false_no_content():
  assert _event().has_trailing_code_execution_result() is False


def test_has_trailing_code_execution_result_false_empty_parts():
  assert _event(parts=[]).has_trailing_code_execution_result() is False


# --- id generation (model_post_init) -----------------------------------------


def test_event_id_auto_assigned_when_missing():
  assert _event().id != ''


def test_event_ids_are_unique():
  assert _event().id != _event().id


def test_event_id_preserved_when_provided():
  assert _event(id='fixed-id').id == 'fixed-id'


# --- state initialization ----------------------------------------------------


def test_event_constructor_with_state():
  """Tests that the event constructor handles the state argument."""
  my_event = Event(state={'key': 'value'})
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {'key': 'value'}


def test_event_constructor_without_state():
  """Tests that the event constructor works without the state argument."""
  my_event = Event()
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {}


# --- isolation scope ---------------------------------------------------------


def test_event_isolation_scope():
  """Tests Event.isolation_scope default value and serialization."""
  ev = Event()
  assert ev.isolation_scope is None

  ev2 = Event(isolation_scope='task:fc-123')
  dumped = ev2.model_dump(mode='json', by_alias=True, exclude_none=True)
  assert dumped['isolationScope'] == 'task:fc-123'


# --- serialization -----------------------------------------------------------


def test_event_serialization_always_camel_case():
  """Tests that Event serialization produces camelCase keys."""
  request_input = RequestInput(interrupt_id='fc-1', message='test')

  # Create an event with fields that would produce snake_case if not dumped by alias
  event = Event(
      invocation_id='i-1',
      node_info=NodeInfo(
          path='a/b',
          output_for=['c'],
          message_as_output=True,
      ),
      output=request_input,
  )

  dumped = event.model_dump(by_alias=True)

  def check_no_snake_case_keys(data):
    if isinstance(data, dict):
      for key, value in data.items():
        assert '_' not in key, f'Found snake_case key: {key} in {data}'
        check_no_snake_case_keys(value)
    elif isinstance(data, list):
      for item in data:
        check_no_snake_case_keys(item)

  check_no_snake_case_keys(dumped)

  # Also verify that expected keys are indeed camelCased
  assert 'invocationId' in dumped
  assert 'nodeInfo' in dumped
  assert 'outputFor' in dumped['nodeInfo']
  assert 'messageAsOutput' in dumped['nodeInfo']

  # Verify RequestInput fields are camelCased
  assert 'output' in dumped
  assert 'interruptId' in dumped['output']


# --- message alias for content -----------------------------------------------


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


class TestMessageSubclassField:
  """Tests that a subclass declaring `message` as a real field is honored.

  `_accept_convenience_kwargs` already routes construction kwargs to such a
  field; the `message` property/setter must defer to it too instead of
  aliasing `content`.
  """

  def test_subclass_field_readable_via_property(self):
    class _Sub(Event):
      message: str = ''

    event = _Sub(message='hello', author='a')
    assert event.message == 'hello'

  def test_subclass_field_serializes_and_round_trips(self):
    class _Sub(Event):
      message: str = ''

    event = _Sub(message='hello', author='a')
    data = event.model_dump()
    assert data['message'] == 'hello'
    assert _Sub.model_validate(data).message == 'hello'

  def test_subclass_field_setter_updates_field_not_content(self):
    class _Sub(Event):
      message: str = ''

    event = _Sub(message='hello', author='a')
    event.message = 'updated'
    assert event.message == 'updated'
    assert event.content is None

  def test_base_event_message_still_aliases_content(self):
    content = types.Content(parts=[types.Part(text='hi')], role='model')
    event = Event(content=content)
    assert event.message is event.content
