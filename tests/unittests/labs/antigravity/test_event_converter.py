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

"""Tests for the Antigravity step-to-event converter.

Verifies that model text, function calls, and function responses map to the
expected ADK events, and that repeated steps are deduplicated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.labs.antigravity import _event_converter
from google.antigravity import types as sdk_types


def _make_ctx() -> MagicMock:
  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  return ctx


def _convert(step, *, streaming=False):
  return _event_converter.convert_step_to_events(
      step,
      ctx=_make_ctx(),
      author='agy',
      seen_tool_calls=set(),
      seen_tool_results=set(),
      streaming=streaming,
  )


def test_completed_model_text_maps_to_one_model_text_event():
  """A completed model text response becomes a single model text event."""
  step = sdk_types.Step(
      step_index=0,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      content='hello there',
      is_complete_response=True,
  )

  events = _convert(step)

  assert len(events) == 1
  assert events[0].author == 'agy'
  assert events[0].content.role == 'model'
  assert events[0].content.parts[0].text == 'hello there'


def test_partial_model_text_produces_no_event():
  """A streaming partial text step (cumulative snapshot) yields nothing."""
  step = sdk_types.Step(
      step_index=0,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      content='hello',
      content_delta='hello',
      is_complete_response=None,
  )

  assert _convert(step) == []


def test_function_call_maps_to_function_call_event():
  """A model tool-call step becomes a model function-call event."""
  step = sdk_types.Step(
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      tool_calls=[
          sdk_types.ToolCall(name='view_file', args={'path': '/x'}, id='c1')
      ],
  )

  events = _convert(step)

  assert len(events) == 1
  fc = events[0].content.parts[0].function_call
  assert events[0].author == 'agy'
  assert fc.name == 'view_file'
  assert fc.id == 'c1'
  assert fc.args == {'path': '/x'}


def test_function_response_maps_to_function_response_event():
  """A completed tool-execution step becomes a function-response event."""
  step = sdk_types.Step(
      step_index=2,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.SYSTEM,
      status=sdk_types.StepStatus.DONE,
      content='file contents',
      tool_calls=[sdk_types.ToolCall(name='view_file', args={}, id='c1')],
  )

  events = _convert(step)

  assert len(events) == 1
  fr = events[0].content.parts[0].function_response
  assert events[0].author == 'view_file'
  assert events[0].content.role == 'user'
  assert fr.name == 'view_file'
  assert fr.id == 'c1'
  assert fr.response == {'result': 'file contents'}


def test_errored_tool_step_maps_error_response():
  """A failed tool-execution step reports the error in the response payload."""
  step = sdk_types.Step(
      step_index=3,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.SYSTEM,
      status=sdk_types.StepStatus.ERROR,
      error='permission denied',
      tool_calls=[sdk_types.ToolCall(name='run_command', args={}, id='c2')],
  )

  events = _convert(step)

  assert events[0].content.parts[0].function_response.response == {
      'error': 'permission denied'
  }


def test_duplicate_tool_call_emitted_once():
  """The same tool call repeated across steps is emitted only once."""
  call = sdk_types.ToolCall(name='view_file', args={}, id='c1')
  step = sdk_types.Step(
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      tool_calls=[call],
  )
  ctx = _make_ctx()
  seen: set[str] = set()

  first = _event_converter.convert_step_to_events(
      step, ctx=ctx, author='agy', seen_tool_calls=seen, seen_tool_results=set()
  )
  second = _event_converter.convert_step_to_events(
      step, ctx=ctx, author='agy', seen_tool_calls=seen, seen_tool_results=set()
  )

  assert len(first) == 1
  assert second == []


def test_incomplete_text_step_produces_no_final_event():
  """A non-final text step yields nothing in non-streaming mode."""
  step = sdk_types.Step(
      step_index=0,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      thinking='reasoning...',
      content='',
  )

  assert _convert(step) == []


def test_streaming_emits_partial_thinking_then_text_deltas():
  """In SSE mode a step's thinking and text deltas become partial events."""
  step = sdk_types.Step(
      step_index=0,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      thinking_delta='thinking...',
      content_delta='hello',
  )

  events = _convert(step, streaming=True)

  assert len(events) == 2
  assert events[0].partial is True
  assert events[0].content.parts[0].thought is True
  assert events[0].content.parts[0].text == 'thinking...'
  assert events[1].partial is True
  assert events[1].content.parts[0].text == 'hello'


def test_non_streaming_omits_partial_deltas():
  """Without SSE mode, delta-only steps yield no events."""
  step = sdk_types.Step(
      step_index=0,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      thinking_delta='thinking...',
      content_delta='hello',
  )

  assert _convert(step, streaming=False) == []


def test_streaming_completed_step_emits_partial_then_final():
  """A completed step in SSE mode emits the partial delta then the final text."""
  step = sdk_types.Step(
      step_index=1,
      type=sdk_types.StepType.TEXT_RESPONSE,
      source=sdk_types.StepSource.MODEL,
      content_delta=' world',
      content='hello world',
      is_complete_response=True,
  )

  events = _convert(step, streaming=True)

  assert len(events) == 2
  assert events[0].partial is True
  assert events[0].content.parts[0].text == ' world'
  assert events[1].partial in (False, None)
  assert events[1].content.parts[0].text == 'hello world'
