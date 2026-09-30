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

"""Tests for conformance replay comparison helpers."""

from __future__ import annotations

from google.adk.cli.conformance._replay_validators import compare_events
from google.adk.cli.conformance._replay_validators import compare_session
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
from google.genai import types


def _text_event(text: str, **overrides) -> Event:
  """Builds a minimal model event carrying a single text part."""
  kwargs = dict(
      author='agent',
      content=types.Content(role='model', parts=[types.Part(text=text)]),
  )
  kwargs.update(overrides)
  return Event(**kwargs)


def _session(**overrides) -> Session:
  kwargs = dict(id='s1', app_name='app', user_id='u1')
  kwargs.update(overrides)
  return Session(**kwargs)


def test_compare_events_equal_lists_succeed_with_no_error_message():
  result = compare_events([_text_event('hi')], [_text_event('hi')])

  assert result.success
  assert result.error_message is None

  # Zero events on both sides is a valid, matching replay.
  assert compare_events([], []).success


def test_compare_events_count_mismatch_reports_both_counts():
  actual = [_text_event('a'), _text_event('b')]
  recorded = [_text_event('a')]

  result = compare_events(actual, recorded)

  assert not result.success
  # The caller has to be able to see which side had how many events.
  assert 'Event count mismatch' in result.error_message
  assert 'Actual: \n2' in result.error_message
  assert 'Recorded: \n1' in result.error_message


def test_compare_events_ignores_per_run_identity_fields():
  """id/timestamp/invocation_id differ on every run and must not fail replay."""
  actual = _text_event(
      'same', id='id-actual', timestamp=1.0, invocation_id='inv-actual'
  )
  recorded = _text_event(
      'same', id='id-recorded', timestamp=2.0, invocation_id='inv-recorded'
  )

  assert compare_events([actual], [recorded]).success


def test_compare_events_ignores_function_call_ids_but_not_names():
  """Function call ids are regenerated per run; the call itself is not."""
  same_name_actual = Event(
      author='agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='call-actual', name='roll', args={'sides': 6}
                  )
              )
          ],
      ),
  )
  same_name_recorded = Event(
      author='agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='call-recorded', name='roll', args={'sides': 6}
                  )
              )
          ],
      ),
  )
  other_name = Event(
      author='agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='call-recorded', name='flip', args={'sides': 6}
                  )
              )
          ],
      ),
  )

  assert compare_events([same_name_actual], [same_name_recorded]).success
  assert not compare_events([same_name_actual], [other_name]).success


def test_compare_events_reports_index_of_first_differing_event():
  actual = [_text_event('a'), _text_event('b'), _text_event('c')]
  recorded = [_text_event('a'), _text_event('B'), _text_event('C')]

  result = compare_events(actual, recorded)

  assert not result.success
  # Zero-based index of the first mismatch, and it stops there.
  assert result.error_message.startswith('event 1 mismatch')
  assert 'event 2 mismatch' not in result.error_message


def test_compare_events_mismatch_message_is_a_diff_from_recorded_to_actual():
  result = compare_events([_text_event('actual-text')], [_text_event('rec')])

  assert not result.success
  # The diff runs recorded -> actual, so the recorded value is the removal
  # and the actual value is the addition. Getting this backwards would make
  # every conformance failure read inverted.
  assert '--- recorded event 0' in result.error_message
  assert '+++ actual event 0' in result.error_message
  assert '-        "text": "rec"' in result.error_message
  assert '+        "text": "actual-text"' in result.error_message


def test_compare_session_ignores_id_last_update_time_and_events():
  actual = _session(
      id='actual-id', last_update_time=1.0, events=[_text_event('x')]
  )
  recorded = _session(id='recorded-id', last_update_time=99.0, events=[])

  # Events are compared separately by compare_events, so they must not make
  # the session comparison fail here.
  assert compare_session(actual, recorded).success


def test_compare_session_detects_user_state_difference():
  actual = _session(state={'locale': 'en-US'})
  recorded = _session(state={'locale': 'fr-FR'})

  result = compare_session(actual, recorded)

  assert not result.success
  assert result.error_message.startswith('session mismatch')
  assert 'en-US' in result.error_message
  assert 'fr-FR' in result.error_message


def test_compare_session_ignores_adk_internal_state_keys():
  actual = _session(
      state={
          'locale': 'en-US',
          '_adk_recordings_config': {'mode': 'record'},
          '_adk_replay_config': {'mode': 'replay'},
      }
  )
  recorded = _session(state={'locale': 'en-US'})

  assert compare_session(actual, recorded).success


def test_compare_events_ignores_recording_config_state_delta():
  actual = _text_event(
      'x',
      actions=EventActions(
          state_delta={'_adk_replay_config': {'on': True}, 'kept': 1}
      ),
  )
  recorded = _text_event('x', actions=EventActions(state_delta={'kept': 1}))

  assert compare_events([actual], [recorded]).success

  differing = _text_event('x', actions=EventActions(state_delta={'kept': 2}))
  assert not compare_events([actual], [differing]).success
