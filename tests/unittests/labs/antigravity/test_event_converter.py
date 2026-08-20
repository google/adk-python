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

import asyncio
from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.labs.antigravity import _event_converter
from google.adk.labs.antigravity import _tool_result_capture
from google.antigravity import types as sdk_types
from google.genai import types as genai_types
import pytest


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


class _Turn:
  """Replays steps through the converter carrying one turn's state across.

  A tool call and its response arrive on different steps, so a converter
  exercised one isolated step at a time cannot show whether they pair up.
  """

  def __init__(self, *, tool_results=None):
    self.ctx = _make_ctx()
    self.seen_tool_calls: set[str] = set()
    self.seen_tool_results: set[str] = set()
    self.tool_results = tool_results

  def step(self, step, *, streaming=False):
    return _event_converter.convert_step_to_events(
        step,
        ctx=self.ctx,
        author='agy',
        seen_tool_calls=self.seen_tool_calls,
        seen_tool_results=self.seen_tool_results,
        tool_results=self.tool_results,
        streaming=streaming,
    )

  def flush(self):
    """Drains the buffer as the end of a turn does."""
    return _event_converter.drain_tool_results(
        ctx=self.ctx,
        seen_tool_calls=self.seen_tool_calls,
        seen_tool_results=self.seen_tool_results,
        tool_results=self.tool_results,
    )


def _responses(events):
  """Returns (name, id, response) for each function-response event."""
  found = []
  for event in events:
    for part in event.content.parts if event.content else []:
      if part.function_response:
        found.append((
            part.function_response.name,
            part.function_response.id,
            part.function_response.response,
        ))
  return found


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


# --- The two real tool-step shapes -------------------------------------------
#
# One tool call arrives as two updates to one step: ACTIVE when the model
# issues it, then DONE (or ERROR) when it finishes. Which update carries
# `tool_calls` is the whole difference between the two kinds of tool. Both
# shapes below are what `LocalConnectionStep.from_dict` actually produces;
# `source=MODEL` throughout because it never maps a tool step to SYSTEM.

_BUILTIN_CALL_ID = 'traj_1:2'
# What the hook is handed for that same built-in: the model's own call id, or
# a SHA-256 of the step id. Never the step id itself, hence never
# `_BUILTIN_CALL_ID` (`localharness/tool_metadata.go`, `ResolveStepCallID`).
_BUILTIN_HOOK_CALL_ID = 'toolu_01ABCDEF'
_CLIENT_CALL_ID = 'call_3'


def _builtin_tool_step(status, *, content='', error=''):
  """One update of a built-in tool's step, which always keeps its tool_calls."""
  # A built-in's call id defaults to the step id, f'{trajectory_id}:{index}'.
  return sdk_types.Step(
      id=_BUILTIN_CALL_ID,
      step_index=2,
      trajectory_id='traj_1',
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      target=sdk_types.StepTarget.ENVIRONMENT,
      status=status,
      content=content,
      error=error,
      tool_calls=[
          sdk_types.ToolCall(
              name='view_file',
              args={'file_path': '/foo'},
              id=_BUILTIN_CALL_ID,
          )
      ],
  )


def _client_tool_active_step():
  """The ACTIVE step the SDK fabricates when the harness asks us to run a tool."""
  # The only step that ever carries a client tool's call, so the only chance to
  # emit the function call.
  return sdk_types.Step(
      id=_CLIENT_CALL_ID,
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      target=sdk_types.StepTarget.ENVIRONMENT,
      status=sdk_types.StepStatus.ACTIVE,
      content='',
      tool_calls=[
          sdk_types.ToolCall(
              name='naming_reviewer',
              args={'request': 'Is "tmp2" a good variable name?'},
              id=_CLIENT_CALL_ID,
          )
      ],
  )


def _client_tool_done_step():
  """The terminal step for a client tool, with `tool_calls` blanked."""
  # `process_event` strips them so the client does not raise a second
  # ToolCallStart, leaving the Go translator's banner as the only content.
  return sdk_types.Step(
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      target=sdk_types.StepTarget.ENVIRONMENT,
      status=sdk_types.StepStatus.DONE,
      content='Calling custom tool "naming_reviewer"',
      tool_calls=[],
  )


def _tool_result(call_id, *, value=None, error=None, name='naming_reviewer'):
  """A ToolResult as the post-tool-call hook receives one."""
  return sdk_types.ToolResult(name=name, id=call_id, result=value, error=error)


class _ToolFailure(RuntimeError):
  """Stands in for the SDK's ``ToolExecutionError``.

  Structural rather than the SDK class: the capture reads a ``ToolError``
  protocol, and the open-source ``google-antigravity`` does not export that
  class yet. `test_tool_result_capture` pins the real one where it exists.
  """

  def __init__(self, message: str, tool_name: str, call_id: str | None = None):
    super().__init__(message)
    self.tool_name = tool_name
    self.call_id = call_id


def _tool_failure(
    call_id, *, message='child agent exploded', name='naming_reviewer'
):
  """The error the on-tool-error hook receives."""
  return _ToolFailure(message, name, call_id=call_id)


def test_a_built_in_tool_call_and_result_pair_up_across_its_two_steps():
  """A built-in tool is answered from the step stream alone."""
  turn = _Turn()

  active = turn.step(_builtin_tool_step(sdk_types.StepStatus.ACTIVE))
  done = turn.step(
      _builtin_tool_step(sdk_types.StepStatus.DONE, content='file contents')
  )

  assert [e.content.parts[0].function_call.id for e in active] == [
      _BUILTIN_CALL_ID
  ]
  assert _responses(done) == [
      ('view_file', _BUILTIN_CALL_ID, {'result': 'file contents'})
  ]
  assert done[0].author == 'view_file'
  assert done[0].content.role == 'user'


def test_a_built_in_tool_is_answered_with_no_buffer_at_all():
  """An agent with no sub-agents registers no hook, so tool_results is None."""
  turn = _Turn(tool_results=None)

  turn.step(_builtin_tool_step(sdk_types.StepStatus.ACTIVE))
  done = turn.step(
      _builtin_tool_step(sdk_types.StepStatus.DONE, content='file contents')
  )

  assert _responses(done) == [
      ('view_file', _BUILTIN_CALL_ID, {'result': 'file contents'})
  ]


def test_a_failed_built_in_tool_step_reports_the_error():
  """A failed tool-execution step reports the error in the response payload."""
  turn = _Turn()

  turn.step(_builtin_tool_step(sdk_types.StepStatus.ACTIVE))
  done = turn.step(
      _builtin_tool_step(sdk_types.StepStatus.ERROR, error='permission denied')
  )

  assert _responses(done) == [
      ('view_file', _BUILTIN_CALL_ID, {'error': 'permission denied'})
  ]


def test_a_client_tool_is_answered_from_the_buffered_result():
  """Neither of a client tool's steps can answer its call; the hook can."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())
  buffer.record(
      _tool_result(_CLIENT_CALL_ID, value='{"result": "Rename it to tmp2."}')
  )
  done = turn.step(_client_tool_done_step())

  assert _responses(done) == [
      ('naming_reviewer', _CLIENT_CALL_ID, {'result': 'Rename it to tmp2.'})
  ]
  assert done[0].author == 'naming_reviewer'
  assert done[0].content.role == 'user'


def test_a_client_tool_result_arriving_after_its_done_step_is_flushed():
  """Hook dispatch is backgrounded, so a result can land after its step."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())
  done = turn.step(_client_tool_done_step())
  buffer.record(
      _tool_result(_CLIENT_CALL_ID, value='{"result": "Rename it to tmp2."}')
  )
  flushed = turn.flush()

  assert not _responses(done)
  assert _responses(flushed) == [
      ('naming_reviewer', _CLIENT_CALL_ID, {'result': 'Rename it to tmp2.'})
  ]


def test_a_client_tool_result_drained_at_its_step_is_not_flushed_again():
  """A duplicate function_response is as broken as a missing one."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())
  buffer.record(_tool_result(_CLIENT_CALL_ID, value='{"result": "ok"}'))
  turn.step(_client_tool_done_step())

  assert not turn.flush()


def test_a_failed_client_tool_reports_the_error_the_error_hook_captured():
  """A client tool that raised is reported as an error, not as a null result.

  The failure arrives on `on_tool_error`, never on `post_tool_call`: the
  harness routes a failed tool to exactly one of the two.
  """
  buffer = _tool_result_capture.ToolResultBuffer()
  errors = _tool_result_capture.ToolErrorCapture(buffer)
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())
  asyncio.run(errors.run(None, _tool_failure(_CLIENT_CALL_ID)))
  done = turn.step(_client_tool_done_step())

  assert _responses(done) == [
      ('naming_reviewer', _CLIENT_CALL_ID, {'error': 'child agent exploded'})
  ]


def test_a_built_in_is_answered_from_its_step_and_its_hook_copy_is_inert():
  """The hook fires for built-ins too, under an id this side never sees.

  The SDK gives a built-in's `ToolCall.id` the step id, while the hook is
  handed the model's own call id (`ResolveStepCallID`). The copy is therefore
  neither drainable nor droppable by id here; the turn's clear collects it.
  """
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_builtin_tool_step(sdk_types.StepStatus.ACTIVE))
  buffer.record(
      _tool_result(
          _BUILTIN_HOOK_CALL_ID,
          value='{"result": "hook copy"}',
          name='view_file',
      )
  )
  builtin_done = turn.step(
      _builtin_tool_step(sdk_types.StepStatus.DONE, content='file contents')
  )
  turn.step(_client_tool_active_step())
  buffer.record(_tool_result(_CLIENT_CALL_ID, value='{"result": "ok"}'))
  client_done = turn.step(_client_tool_done_step())

  assert _responses(builtin_done) == [
      ('view_file', _BUILTIN_CALL_ID, {'result': 'file contents'})
  ]
  assert _responses(client_done) == [
      ('naming_reviewer', _CLIENT_CALL_ID, {'result': 'ok'})
  ]
  assert not turn.flush()
  # The hook copy is still held, under the id only the hook ever sees.
  assert buffer.take({_BUILTIN_HOOK_CALL_ID})


def test_a_stripped_step_with_nothing_buffered_yields_nothing():
  """The banner text is the translator's, not the tool's, so it is no answer."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())

  assert not turn.step(_client_tool_done_step())


def test_a_result_for_a_call_that_was_never_emitted_is_not_drained():
  """A response may only follow the call it answers."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)
  buffer.record(_tool_result('never_called', value='{"result": "ok"}'))

  assert not turn.step(_client_tool_done_step())
  assert not turn.flush()


@pytest.mark.parametrize(
    'value,expected',
    [
        pytest.param(
            '{"result": "the child\'s answer"}',
            {'result': "the child's answer"},
            id='wrapped_json_object_is_unwrapped',
        ),
        pytest.param(
            '{"stdout": "hi", "code": 0}',
            {'stdout': 'hi', 'code': 0},
            id='any_json_object_passes_through',
        ),
        pytest.param(
            'Calling custom tool: not json {',
            {'result': 'Calling custom tool: not json {'},
            id='malformed_json_degrades_to_the_raw_string',
        ),
        pytest.param('"bare string"', {'result': 'bare string'}, id='json_str'),
        pytest.param('[1, 2]', {'result': [1, 2]}, id='json_array_is_wrapped'),
        pytest.param({'already': 'a dict'}, {'already': 'a dict'}, id='dict'),
        pytest.param(None, {'result': 'success'}, id='none_is_not_null'),
    ],
)
def test_a_buffered_result_becomes_a_dict_payload(value, expected):
  """`FunctionResponse.response` must be a dict, whatever the tool returned."""
  buffer = _tool_result_capture.ToolResultBuffer()
  turn = _Turn(tool_results=buffer)

  turn.step(_client_tool_active_step())
  buffer.record(_tool_result(_CLIENT_CALL_ID, value=value))
  done = turn.step(_client_tool_done_step())

  assert _responses(done) == [('naming_reviewer', _CLIENT_CALL_ID, expected)]


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


def _event(author='agy', partial=False, parts=None):
  """Builds an ADK Event; `parts=None` means the event carries no content."""
  return Event(
      invocation_id='inv_1',
      author=author,
      partial=partial,
      content=(
          None
          if parts is None
          else genai_types.Content(role='model', parts=parts)
      ),
  )


_TEXT_PART = genai_types.Part.from_text(text='answer')
_THOUGHT_PART = genai_types.Part(text='thinking out loud', thought=True)
_CALL_PART = genai_types.Part(
    function_call=genai_types.FunctionCall(name='run_command', args={})
)
_RESPONSE_PART = genai_types.Part(
    function_response=genai_types.FunctionResponse(
        name='run_command', response={'result': 'ok'}
    )
)


@pytest.mark.parametrize(
    'event,expected',
    [
        pytest.param(_event(parts=[_TEXT_PART]), 'answer', id='text'),
        pytest.param(
            _event(parts=[_TEXT_PART, _TEXT_PART]),
            'answeranswer',
            id='text_parts_concatenated',
        ),
        pytest.param(
            _event(parts=[_THOUGHT_PART, _TEXT_PART]),
            'answer',
            id='thought_dropped_text_kept',
        ),
        pytest.param(
            _event(partial=True, parts=[_TEXT_PART]), None, id='partial'
        ),
        pytest.param(_event(parts=[_THOUGHT_PART]), None, id='thought_only'),
        pytest.param(_event(parts=[_CALL_PART]), None, id='function_call_only'),
        pytest.param(
            _event(author='run_command', parts=[_RESPONSE_PART]),
            None,
            id='function_response_from_tool',
        ),
        pytest.param(
            _event(author='some_other_agent', parts=[_TEXT_PART]),
            None,
            id='wrong_author',
        ),
        pytest.param(_event(parts=[]), None, id='empty_parts'),
        pytest.param(_event(parts=None), None, id='no_content'),
    ],
)
def test_final_model_text_filters(event, expected):
  """Only this agent's own, complete, user-visible text counts."""
  assert _event_converter.final_model_text(event, 'agy') == expected
