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
from google.adk.events.event_actions import EventActions
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
