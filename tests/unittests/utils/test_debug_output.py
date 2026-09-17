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

"""Tests for the debug event printer."""

from __future__ import annotations

from google.adk.events.event import Event
from google.adk.utils._debug_output import print_event
from google.genai import types


def _event(*parts: types.Part) -> Event:
  return Event(
      author='agent', content=types.Content(role='model', parts=list(parts))
  )


def _lines(capsys) -> list[str]:
  out = capsys.readouterr().out
  return out.splitlines()


def test_print_event_without_content_prints_nothing(capsys):
  print_event(Event(author='agent'), verbose=True)
  assert _lines(capsys) == []


def test_print_event_without_parts_prints_nothing(capsys):
  event = Event(author='agent', content=types.Content(role='model', parts=[]))
  print_event(event, verbose=True)
  assert _lines(capsys) == []


def test_print_event_prints_text_with_author_prefix(capsys):
  print_event(_event(types.Part(text='hello')))
  assert _lines(capsys) == ['agent > hello']


def test_print_event_coalesces_consecutive_text_parts_into_one_line(capsys):
  # A streamed answer arrives as several text parts; repeating the author
  # prefix per part would fragment one sentence across many lines.
  print_event(
      _event(
          types.Part(text='hello '),
          types.Part(text='there '),
          types.Part(text='world'),
      )
  )
  assert _lines(capsys) == ['agent > hello there world']


def test_print_event_hides_non_text_parts_when_not_verbose(capsys):
  print_event(
      _event(
          types.Part(text='answer'),
          types.Part(
              function_call=types.FunctionCall(name='lookup', args={'a': 1})
          ),
      )
  )
  assert _lines(capsys) == ['agent > answer']


def test_print_event_verbose_flushes_pending_text_before_a_tool_call(capsys):
  # The text that preceded the call must be printed first, otherwise the
  # transcript reads out of order.
  print_event(
      _event(
          types.Part(text='let me check'),
          types.Part(
              function_call=types.FunctionCall(name='lookup', args={'a': 1})
          ),
          types.Part(text='done'),
      ),
      verbose=True,
  )
  assert _lines(capsys) == [
      'agent > let me check',
      "agent > [Calling tool: lookup({'a': 1})]",
      'agent > done',
  ]


def test_print_event_verbose_truncates_long_tool_call_args(capsys):
  print_event(
      _event(
          types.Part(
              function_call=types.FunctionCall(
                  name='lookup', args={'text': 'a' * 100}
              )
          )
      ),
      verbose=True,
  )
  # str(args) is "{'text': 'aaa...'}"; the preview keeps its first 50
  # characters - the 10-character prefix "{'text': '" plus 40 a's.
  assert _lines(capsys) == [
      "agent > [Calling tool: lookup({'text': '" + 'a' * 40 + '...)]'
  ]


def test_print_event_verbose_truncates_long_tool_response(capsys):
  print_event(
      _event(
          types.Part(
              function_response=types.FunctionResponse(
                  name='lookup', response={'r': 'b' * 200}
              )
          )
      ),
      verbose=True,
  )
  # A response preview keeps 100 characters: "{'r': '" plus 93 b's.
  assert _lines(capsys) == ["agent > [Tool result: {'r': '" + 'b' * 93 + '...]']


def test_print_event_verbose_reports_executable_code_language(capsys):
  print_event(
      _event(types.Part.from_executable_code(code='x = 1', language='PYTHON')),
      verbose=True,
  )
  # The language is an enum, and formatting a str-mixin enum renders the bare
  # value on 3.10 but ``Language.PYTHON`` on 3.11+, so only assert it is named.
  (line,) = _lines(capsys)
  assert line.startswith('agent > [Executing ')
  assert 'PYTHON' in line
  assert line.endswith(' code...]')


def test_print_event_verbose_executable_code_without_language(capsys):
  print_event(
      _event(types.Part(executable_code=types.ExecutableCode(code='x = 1'))),
      verbose=True,
  )
  # An unlabelled code block still gets a line, with a generic word for it.
  assert _lines(capsys) == ['agent > [Executing code code...]']


def test_print_event_verbose_reports_code_output(capsys):
  print_event(
      _event(
          types.Part.from_code_execution_result(
              outcome='OUTCOME_OK', output='42'
          )
      ),
      verbose=True,
  )
  assert _lines(capsys) == ['agent > [Code output: 42]']


def test_print_event_verbose_code_result_without_output(capsys):
  print_event(
      _event(
          types.Part(
              code_execution_result=types.CodeExecutionResult(
                  outcome='OUTCOME_OK'
              )
          )
      ),
      verbose=True,
  )
  assert _lines(capsys) == ['agent > [Code output: result]']


def test_print_event_verbose_reports_inline_data_mime_type(capsys):
  print_event(
      _event(
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'\x00')
          )
      ),
      verbose=True,
  )
  # The bytes are never printed, only the kind of data they are.
  assert _lines(capsys) == ['agent > [Inline data: image/png]']


def test_print_event_verbose_reports_file_uri(capsys):
  print_event(
      _event(
          types.Part(
              file_data=types.FileData(
                  file_uri='files/report', mime_type='text/plain'
              )
          )
      ),
      verbose=True,
  )
  assert _lines(capsys) == ['agent > [File: files/report]']
