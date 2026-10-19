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

"""Tests for the pending-commit delta a session State records."""

from __future__ import annotations

import threading
from typing import Any

from google.adk.sessions.state import State
from google.adk.sessions.state import StateSchemaError
from pydantic import BaseModel
import pytest


def _state_with(**entries: Any) -> tuple[State, dict[str, Any], dict[str, Any]]:
  """Returns a State over a session already holding entries, plus its dicts."""
  value: dict[str, Any] = dict(entries)
  delta: dict[str, Any] = {}
  return State(value=value, delta=delta), value, delta


def test_assigning_a_key_records_it():
  state, _, delta = _state_with()

  state['answer'] = 42

  assert delta == {'answer': 42}


def test_writing_into_a_nested_dict_records_the_key():
  state, _, delta = _state_with(responses={'first': 1})

  state['responses']['second'] = 2

  assert state.has_delta()
  assert delta == {'responses': {'first': 1, 'second': 2}}


def test_writing_far_down_a_nested_dict_records_the_key():
  state, _, delta = _state_with(responses={'first': {'inner': {}}})

  state['responses']['first']['inner']['leaf'] = 'v'

  assert state.has_delta()
  assert delta == {'responses': {'first': {'inner': {'leaf': 'v'}}}}


def test_deleting_from_a_nested_dict_records_the_key():
  state, _, delta = _state_with(responses={'first': 1, 'second': 2})

  del state['responses']['first']

  assert state.has_delta()
  assert delta == {'responses': {'second': 2}}


@pytest.mark.parametrize(
    'mutate, expected',
    [
        (lambda items: items.append(3), [1, 2, 3]),
        (lambda items: items.extend([3, 4]), [1, 2, 3, 4]),
        (lambda items: items.insert(0, 0), [0, 1, 2]),
        (lambda items: items.remove(1), [2]),
        (lambda items: items.__setitem__(0, 9), [9, 2]),
        (lambda items: items.reverse(), [2, 1]),
    ],
)
def test_writing_into_a_nested_list_records_the_key(mutate, expected):
  """Every in-place list mutator changes the list the key holds."""
  state, _, delta = _state_with(items=[1, 2])

  mutate(state['items'])

  assert state.has_delta()
  assert delta == {'items': expected}


def test_reading_a_container_records_nothing():
  state, _, delta = _state_with(responses={'first': 1}, items=[1])

  assert state['responses']['first'] == 1
  assert state.get('items') == [1]
  assert 'missing' not in state

  assert not state.has_delta()
  assert not delta


def test_a_read_hands_back_the_stored_container():
  """Code holding the stored container keeps writing into the session."""
  held = {'a': 1}
  state, value, delta = _state_with(cfg=held)

  assert state['cfg'] is held
  held['b'] = 2

  assert value['cfg'] is held
  assert state['cfg'] == {'a': 1, 'b': 2}
  assert state.has_delta()
  assert delta == {'cfg': {'a': 1, 'b': 2}}


def test_a_write_after_the_delta_is_cleared_is_still_recorded():
  """A caller that moves the delta into an event clears it and keeps going."""
  state, _, delta = _state_with(responses={})
  responses = state['responses']

  responses['a'] = 1
  assert state.has_delta()
  delta.clear()
  responses['b'] = 2

  assert state.has_delta()
  assert delta == {'responses': {'a': 1, 'b': 2}}


def test_a_nested_write_is_recorded_only_by_the_state_that_read_it():
  """Two contexts over one session each record only what they handed out."""
  session_state: dict[str, Any] = {'responses': {}}
  first_delta: dict[str, Any] = {}
  first = State(value=session_state, delta=first_delta)
  second_delta: dict[str, Any] = {}
  second = State(value=session_state, delta=second_delta)

  second['responses']['b'] = 2

  assert not first.has_delta()
  assert second.has_delta()
  assert second_delta == {'responses': {'b': 2}}
  assert first['responses'] is session_state['responses']


def test_a_container_that_cannot_be_copied_is_left_alone():
  lock = threading.Lock()
  state, _, delta = _state_with(handles={'lock': lock})

  state['handles']['other'] = 1

  assert not state.has_delta()
  assert not delta
  assert state['handles'] == {'lock': lock, 'other': 1}


class _ComparesByIdentity:
  """A value whose copy never equals it, as a plain object does not."""


def test_a_container_whose_copy_never_matches_is_left_alone():
  """Such a container would otherwise look changed on every read."""
  state, _, delta = _state_with(handles={'h': _ComparesByIdentity()})

  state['handles']

  assert not state.has_delta()
  assert not delta


def test_a_key_another_state_reassigned_is_not_recorded_by_the_reader():
  """The assignment wins; the reader must not put its own view in the delta."""
  session_state: dict[str, Any] = {'items': [0]}
  reader_delta: dict[str, Any] = {}
  reader = State(value=session_state, delta=reader_delta)
  writer = State(value=session_state, delta={})

  reader['items']
  writer['items'] = [1]

  assert not reader.has_delta()
  assert not reader_delta


class _UserProfileSchema(BaseModel):
  user_id: str
  profile: dict[str, Any]
  tags: list[str]


def _state_under_schema() -> tuple[State, dict[str, Any]]:
  """Returns a schema-bound State seeded with one of each field, plus its delta."""
  value: dict[str, Any] = {
      'user_id': 'u123',
      'profile': {'tier': 'free'},
      'tags': ['alpha'],
  }
  delta: dict[str, Any] = {}
  return State(value=value, delta=delta, schema=_UserProfileSchema), delta


def test_writing_into_a_nested_dict_records_the_key_under_a_schema():
  state, delta = _state_under_schema()

  state['profile']['tier'] = 'premium'

  assert state.has_delta()
  assert delta['profile'] == {'tier': 'premium'}


def test_assigning_a_key_is_still_validated_against_the_schema():
  state, _ = _state_under_schema()

  with pytest.raises(StateSchemaError):
    state['user_id'] = 12345
  with pytest.raises(StateSchemaError):
    state['unknown_field'] = 'bad'
