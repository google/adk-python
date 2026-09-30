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

"""Tests for the model consult session-to-advisor handover."""

from typing import Sequence

from google.adk.events.event import Event
from google.adk.tools.model_consult._context import build_advisor_contents
from google.adk.tools.model_consult._context import ModelConsultContextConfig
from google.adk.tools.model_consult._context import render_transcript
from google.genai import types
from pydantic import ValidationError
import pytest

# ADK authors every event the agent produces, including tool results, with the
# agent's own name. Only the end user's turns are authored 'user'.
_AGENT = 'root_agent'


def _user_event(text: str) -> Event:
  """Builds a user turn carrying a single text part."""
  return Event(
      author='user',
      content=types.Content(role='user', parts=[types.Part(text=text)]),
  )


def _agent_event(parts: list[types.Part]) -> Event:
  """Builds an agent turn carrying the given parts."""
  return Event(author=_AGENT, content=types.Content(role='model', parts=parts))


def _tool_result_event(
    name: str, response: dict[str, object], *, call_id: str = 'fc-1'
) -> Event:
  """Builds a tool result event the way ADK's tool caller builds it.

  The author is the agent, not the user; only `content.role` is 'user'. See
  `flows/llm_flows/tools/_caller.py`, which sets `function_response.id`, builds
  the response content with `role='user'`, and authors the event with the
  agent's name.
  """
  return Event(
      author=_AGENT,
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name=name, response=response
                  )
              )
          ],
      ),
  )


def _texts(contents: Sequence[types.Content]) -> list[str]:
  """Flattens the text of every part, in order."""
  return [
      part.text or '' for content in contents for part in content.parts or []
  ]


def _chars(contents: Sequence[types.Content]) -> int:
  """Counts the characters the handover would actually send."""
  return sum(len(text) for text in _texts(contents))


def test_session_is_replayed_as_multi_turn_contents():
  """Events reach the advisor in order, with their roles preserved."""
  events = [
      _user_event('Investigate the paging alert.'),
      _agent_event([types.Part(text='Checking logs.')]),
  ]

  contents = build_advisor_contents(events)

  assert [content.role for content in contents] == ['user', 'model']
  assert _texts(contents) == ['Investigate the paging alert.', 'Checking logs.']


def test_executor_thoughts_are_withheld_by_default():
  """Thought parts do not reach the advisor unless asked for."""
  events = [
      _agent_event([
          types.Part(text='internal musing', thought=True),
          types.Part(text='visible answer'),
      ])
  ]

  contents = build_advisor_contents(events)

  assert _texts(contents) == ['visible answer']


def test_included_thoughts_are_labelled_as_thoughts():
  """Reasoning stays distinguishable from what the executor concluded."""
  events = [
      _agent_event([
          types.Part(text='internal musing', thought=True),
          types.Part(text='visible answer'),
      ])
  ]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(include_thoughts=True)
  )

  assert _texts(contents) == ['[thought] internal musing', 'visible answer']


def test_tool_calls_and_results_are_flattened_into_text():
  """Function parts become readable text the advisor can consume.

  The advisor holds none of the executor's tool declarations, so a live
  function call part would be a validation error for most providers.
  """
  events = [
      _agent_event([
          types.Part(
              function_call=types.FunctionCall(
                  id='fc-1', name='query_logs', args={'service': 'checkout'}
              )
          )
      ]),
      _tool_result_event('query_logs', {'errors': 42}),
  ]

  contents = build_advisor_contents(events)

  # The tool result is authored by the agent, so it lands in the same model
  # turn as the call that produced it.
  assert [content.role for content in contents] == ['model']
  assert _texts(contents) == [
      '[tool_call] query_logs({"service": "checkout"})',
      '[tool_result] query_logs -> {"errors": 42}',
  ]
  assert all(
      part.function_call is None and part.function_response is None
      for content in contents
      for part in content.parts or []
  )


def test_in_flight_consult_is_left_out_of_the_handover():
  """The consult that triggered the handover is not replayed back to it."""
  events = [
      _agent_event([
          types.Part(
              function_call=types.FunctionCall(
                  id='fc-current',
                  name='model_consult',
                  args={'question': 'help'},
              )
          )
      ])
  ]

  contents = build_advisor_contents(
      events, skip_function_call_ids=['fc-current']
  )

  assert not contents


def test_in_flight_consult_result_is_left_out_of_the_handover():
  """The matching tool result is skipped by the same id."""
  events = [
      _tool_result_event(
          'model_consult', {'status': 'ok'}, call_id='fc-current'
      ),
  ]

  contents = build_advisor_contents(
      events, skip_function_call_ids=['fc-current']
  )

  assert not contents


def test_consecutive_same_role_turns_are_merged():
  """Adjacent same-role turns collapse into one content.

  Advisor models reached through LiteLlm require strict role alternation.
  """
  events = [
      _agent_event([types.Part(text='one')]),
      _agent_event([types.Part(text='two')]),
  ]

  contents = build_advisor_contents(events)

  assert len(contents) == 1
  assert _texts(contents) == ['one', 'two']


def test_partial_streaming_events_are_ignored():
  """Streaming fragments are skipped so text is not duplicated."""
  streaming = _agent_event([types.Part(text='partial chunk')])
  streaming.partial = True

  contents = build_advisor_contents([streaming, _user_event('done')])

  assert _texts(contents) == ['done']


def test_max_events_keeps_only_the_most_recent_turns():
  """The event cap trims from the front, keeping the newest turns."""
  events = [_user_event(f'turn {i}') for i in range(10)]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_events=3)
  )

  assert _texts(contents) == ['turn 7', 'turn 8', 'turn 9']


def test_character_budget_drops_the_middle_and_marks_the_gap():
  """Over budget, the original task and the current state both survive."""
  events = []
  for i in range(20):
    events.append(_user_event(f'user {i} ' + 'x' * 500))
    events.append(_agent_event([types.Part(text=f'model {i} ' + 'y' * 500)]))

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_chars=4000)
  )

  texts = _texts(contents)
  assert any('omitted to fit the context budget' in text for text in texts)
  assert texts[0].startswith('user 0')
  assert texts[-1].startswith('model 19')
  assert _chars(contents) <= 4000


def test_budget_survives_one_turn_larger_than_the_whole_budget():
  """The newest turn is always kept, so it is trimmed rather than exempted."""
  events = [
      _user_event('small task'),
      _agent_event([types.Part(text='Z' * 30_000)]),
  ]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_chars=1000)
  )

  assert _chars(contents) <= 1000
  assert 'characters truncated' in _texts(contents)[-1]


@pytest.mark.parametrize('max_chars', [40, 1000])
def test_budget_holds_when_the_newest_turn_is_media(max_chars: int):
  """Media and its omission placeholder both count against the budget."""
  events = [
      _user_event('small task'),
      _agent_event([
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'x' * 40_000)
          )
      ]),
  ]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_chars=max_chars)
  )

  assert _chars(contents) <= max_chars
  assert _texts(contents)[-1].startswith('[media omitted to fit')


@pytest.mark.parametrize('max_chars', [40, 61])
def test_budget_too_small_for_the_marker_keeps_the_newest_turn(max_chars: int):
  """The newest turn outranks the omission marker, and never ships empty.

  A content with no parts is a validation error for several providers, so a
  budget that cannot carry both (including `max_chars=61`, the exact length of
  the marker itself) has to drop the marker, not the turn.
  """
  events = [
      _user_event('a' * 200),
      _agent_event([types.Part(text='b' * 200)]),
      _user_event('c' * 200),
      _agent_event([types.Part(text='d' * 400)]),
  ]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_chars=max_chars)
  )

  assert contents
  assert all(content.parts for content in contents)
  assert _chars(contents) <= max_chars
  assert 'd' in _texts(contents)[-1]


def test_rewound_invocations_are_not_handed_over():
  """The executor no longer sees a rewound turn, so neither does the advisor."""
  discarded = _user_event('wrong task')
  discarded.invocation_id = 'inv1'
  rewind = Event(author='user', invocation_id='inv2')
  rewind.actions.rewind_before_invocation_id = 'inv1'
  live = _user_event('real task')
  live.invocation_id = 'inv3'

  contents = build_advisor_contents([discarded, rewind, live])

  assert _texts(contents) == ['real task']


def test_trimming_keeps_the_roles_alternating():
  """The omission marker must not re-introduce adjacent same-role turns."""
  events = []
  for i in range(20):
    events.append(_user_event(f'user {i} ' + 'x' * 500))
    events.append(_agent_event([types.Part(text=f'model {i} ' + 'y' * 500)]))

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_chars=4000)
  )

  roles = [content.role for content in contents]
  assert all(before != after for before, after in zip(roles, roles[1:]))


def test_oversized_tool_results_are_truncated_per_part():
  """A single huge tool result cannot consume the whole handover."""
  events = [_tool_result_event('dump', {'blob': 'z' * 50_000})]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_part_chars=500)
  )

  text = _texts(contents)[0]
  assert 'characters truncated' in text
  # The 500 characters that survive, plus the prefix and the truncation note.
  assert len(text) < 600


def test_plain_text_gets_more_room_than_a_tool_result():
  """Prose receives _TEXT_CHARS_MULTIPLIER times the per-part tool cap."""
  prose = 'p' * 3000
  events = [
      _agent_event([types.Part(text=prose)]),
      _tool_result_event('dump', {'blob': 'z' * 3000}),
  ]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(max_part_chars=500)
  )

  texts = _texts(contents)
  assert texts[0] == prose
  assert 'characters truncated' in texts[1]


def test_media_reaches_the_advisor_by_default():
  """Inline media is passed through untouched."""
  media = types.Part(
      inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG fake')
  )

  contents = build_advisor_contents([_agent_event([media])])

  assert contents[0].parts[0].inline_data is not None


def test_media_is_described_in_text_for_text_only_advisors():
  """With include_media off, media becomes a placeholder instead."""
  media = types.Part(
      inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG fake')
  )

  contents = build_advisor_contents(
      [_agent_event([media])],
      config=ModelConsultContextConfig(include_media=False),
  )

  assert _texts(contents) == ['[media omitted: image/png]']


def test_code_parts_are_rendered_as_text():
  """Executed code and its output reach the advisor as readable text."""
  events = [
      _agent_event([
          types.Part(
              executable_code=types.ExecutableCode(
                  code='print(1)', language=types.Language.PYTHON
              )
          ),
          types.Part(
              code_execution_result=types.CodeExecutionResult(
                  outcome=types.Outcome.OUTCOME_OK, output='1'
              )
          ),
      ])
  ]

  contents = build_advisor_contents(events)

  assert _texts(contents) == ['[code]\nprint(1)', '[code_result] 1']


def test_whitespace_only_text_is_dropped():
  """Blank turns are not worth a slot in the handover."""
  events = [_agent_event([types.Part(text='   \n  ')])]

  contents = build_advisor_contents(events)

  assert not contents


def test_session_can_be_withheld_entirely():
  """With include_session off, the advisor sees no session content."""
  events = [_user_event('secret internal transcript')]

  contents = build_advisor_contents(
      events, config=ModelConsultContextConfig(include_session=False)
  )

  assert not contents


def test_transcript_rendering_labels_each_role():
  """Transcript mode renders contents as a labelled plain-text block."""
  events = [_user_event('question'), _agent_event([types.Part(text='answer')])]

  transcript = render_transcript(build_advisor_contents(events))

  assert transcript == 'USER: question\n\nAGENT: answer'


def test_transcript_rendering_names_media_it_cannot_write_out():
  """Media survives as a marker so the transcript is not silently lossy."""
  media = types.Part(
      inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG fake')
  )

  transcript = render_transcript(
      build_advisor_contents([_agent_event([media])])
  )

  assert transcript == 'AGENT: [media: image/png]'


def test_transcript_rendering_names_file_parts():
  """A file part carries no text and no bytes, so it is the easiest to lose."""
  file_part = types.Part(
      file_data=types.FileData(
          file_uri='gs://bucket/spec.pdf', mime_type='application/pdf'
      )
  )

  transcript = render_transcript(
      build_advisor_contents([_agent_event([file_part])])
  )

  assert transcript == 'AGENT: [file: gs://bucket/spec.pdf]'


def test_config_rejects_unknown_fields():
  """A misspelled option fails loudly instead of being silently ignored."""
  with pytest.raises(ValidationError):
    ModelConsultContextConfig(max_char=100)


@pytest.mark.parametrize('field', ['max_events', 'max_chars', 'max_part_chars'])
def test_config_rejects_degenerate_caps(field: str):
  """A cap of zero once meant 'no cap', which is the opposite of the ask."""
  with pytest.raises(ValidationError):
    ModelConsultContextConfig(**{field: 0})
