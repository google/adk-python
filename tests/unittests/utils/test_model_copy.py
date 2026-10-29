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

"""Tests that deep_copy_model returns what pydantic's deep copy returns."""

import copy
import enum
from typing import Any
from unittest import mock

from fastapi.openapi.models import APIKey
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
from google.adk.utils._model_copy import deep_copy_model
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

_ATOMIC_TYPES = (type(None), bool, int, float, str, bytes)


class _Color(enum.Enum):
  RED = 'red'


class _TaggedEvent(Event):
  tags: list[str] = Field(default_factory=list)


class _Leaf(BaseModel):
  values: list[int] = Field(default_factory=list)


class _Tree(BaseModel):
  model_config = ConfigDict(extra='allow')

  left: _Leaf | None = None
  right: _Leaf | None = None
  data: dict[str, Any] = Field(default_factory=dict)


class _CountingLeaf(_Leaf):

  def __deepcopy__(self, memo: dict[int, Any] | None = None) -> '_CountingLeaf':
    copied = _CountingLeaf(values=list(self.values))
    copied.values.append(-1)
    return copied


def _pydantic_deepcopy(value: Any) -> Any:
  """Copies with pydantic's own `__deepcopy__` on Event and Session."""
  with mock.patch.object(Event, '__deepcopy__', BaseModel.__deepcopy__):
    with mock.patch.object(Session, '__deepcopy__', BaseModel.__deepcopy__):
      return copy.deepcopy(value)


def _assert_same_copy(
    actual: Any,
    expected: Any,
    original: Any,
    pairs: dict[int, Any] | None = None,
    reverse: dict[int, Any] | None = None,
) -> None:
  """Asserts two copies match in values, sharing and reuse of originals."""
  if pairs is None:
    pairs, reverse = {}, {}
  assert type(actual) is type(expected)
  assert (actual is original) == (expected is original)
  if type(actual) in _ATOMIC_TYPES:
    assert actual == expected
    return
  if id(actual) in pairs or id(expected) in reverse:
    assert pairs.get(id(actual)) is expected
    assert reverse.get(id(expected)) is actual
    return
  pairs[id(actual)] = expected
  reverse[id(expected)] = actual
  if isinstance(actual, mock.NonCallableMock):
    return
  if isinstance(actual, BaseModel):
    for name in ('__dict__', '__pydantic_extra__', '__pydantic_private__'):
      _assert_same_copy(
          getattr(actual, name),
          getattr(expected, name),
          getattr(original, name),
          pairs,
          reverse,
      )
    assert actual.__pydantic_fields_set__ == expected.__pydantic_fields_set__
    assert (
        actual.__pydantic_fields_set__ is not original.__pydantic_fields_set__
    )
  elif isinstance(actual, dict):
    assert list(actual) == list(expected) == list(original)
    for key in actual:
      _assert_same_copy(
          actual[key], expected[key], original[key], pairs, reverse
      )
  elif isinstance(actual, (list, tuple)):
    assert len(actual) == len(expected) == len(original)
    for a, e, o in zip(actual, expected, original):
      _assert_same_copy(a, e, o, pairs, reverse)
  else:
    assert actual == expected


def _auth_config(**extra: Any) -> AuthConfig:
  return AuthConfig(
      auth_scheme=APIKey(**{'in': 'header', 'name': 'x-key'}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY, api_key='key'
      ),
      **extra,
  )


def _session() -> Session:
  shared = {'n': [1, 2]}
  events = [
      Event(
          author='user',
          content=types.Content(
              role='user',
              parts=[
                  types.Part(text='hi'),
                  types.Part(
                      inline_data=types.Blob(
                          mime_type='image/png', data=b'\x89PNG'
                      )
                  ),
              ],
          ),
      ),
      Event(
          author='agent',
          content=types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='call-1',
                          name='lookup',
                          args={'q': shared, 'nested': {'k': [shared]}},
                      )
                  )
              ],
          ),
          long_running_tool_ids={'call-1'},
      ),
      Event(
          author='agent',
          actions=EventActions(
              state_delta={'shared': shared, 'list': [1, shared]},
              requested_auth_configs={
                  'call-2': _auth_config(),
                  'call-3': _auth_config(note={'shared': shared}),
              },
          ),
          custom_metadata={'color': _Color.RED, 'pair': (1, shared)},
      ),
      _TaggedEvent(author='agent'),
  ]
  session = Session(
      id='s',
      app_name='app',
      user_id='u',
      state={'shared': shared, 'again': shared},
      events=events,
  )
  session._storage_update_marker = 'rev-1'
  return session


def test_pydantic_model_state_is_the_four_slots_the_copier_carries():
  """A pydantic release that adds model state fails here, not silently."""
  assert BaseModel.__slots__ == (
      '__dict__',
      '__pydantic_fields_set__',
      '__pydantic_extra__',
      '__pydantic_private__',
  )


def test_session_deepcopy_matches_pydantic():
  """Deep-copying a session returns what pydantic's own copier returns."""
  session = _session()

  _assert_same_copy(
      copy.deepcopy(session), _pydantic_deepcopy(session), session
  )


def test_event_deepcopy_and_model_copy_match_pydantic():
  """Both copy entry points on an event return what pydantic's copier does."""
  event = _session().events[2]
  expected = _pydantic_deepcopy(event)

  _assert_same_copy(copy.deepcopy(event), expected, event)
  _assert_same_copy(event.model_copy(deep=True), expected, event)


def test_session_copy_equals_original_and_shares_nothing_mutable():
  """Mutating any part of the copy leaves the original session untouched."""
  session = _session()

  copied = copy.deepcopy(session)

  assert copied == session
  copied.state['shared']['n'].append(3)
  copied.events[1].content.parts[0].function_call.args['nested']['k'].clear()
  copied.events[2].actions.requested_auth_configs['call-2'].note = 'set later'
  copied.events[1].long_running_tool_ids.add('copy only')
  copied.events[3].tags.append('copy only')
  assert session.state['shared'] == {'n': [1, 2]}
  assert session.events[1].content.parts[0].function_call.args['nested'][
      'k'
  ] == [{'n': [1, 2]}]
  assert (
      session.events[2]
      .actions.requested_auth_configs['call-2']
      .__pydantic_extra__
      == {}
  )
  assert session.events[1].long_running_tool_ids == {'call-1'}
  assert session.events[3].tags == []


def test_objects_shared_in_the_original_stay_shared_in_the_copy():
  """One value reached by two paths is still one value in the copy."""
  session = _session()

  copied = copy.deepcopy(session)

  shared = copied.state['shared']
  assert copied.state['again'] is shared
  assert copied.events[1].content.parts[0].function_call.args['q'] is shared
  assert copied.events[2].actions.state_delta['list'][1] is shared
  assert shared is not session.state['shared']


def test_mock_events_copy_as_with_pydantic():
  """A session holding mock events copies the way pydantic copies it."""
  session = Session(id='s', app_name='app', user_id='u')
  session.events.append(mock.Mock(spec=Event))
  session.events.append(mock.create_autospec(Event, instance=True))

  _assert_same_copy(
      copy.deepcopy(session), _pydantic_deepcopy(session), session
  )


def test_dict_cycle_in_state_copies_as_with_pydantic():
  """State that contains itself copies to a cycle instead of recursing."""
  cycle: dict[str, Any] = {}
  cycle['self'] = cycle
  session = Session(id='s', app_name='app', user_id='u', state={'c': cycle})

  copied = copy.deepcopy(session)

  _assert_same_copy(copied, _pydantic_deepcopy(session), session)
  assert copied.state['c']['self'] is copied.state['c']


def test_generic_models_match_pydantic_deepcopy():
  """Any model, not just Event and Session, copies as pydantic copies it."""
  leaf = _Leaf(values=[1])
  tree = _Tree(left=leaf, right=leaf, data={'leaf': leaf}, extra_field=[leaf])

  _assert_same_copy(deep_copy_model(tree), copy.deepcopy(tree), tree)


def test_nested_custom_deepcopy_is_used():
  """A nested model with its own __deepcopy__ still gets to run it."""
  tree = _Tree(left=_CountingLeaf(values=[1]))

  copied = deep_copy_model(tree)

  assert copied.left.values == [1, -1]


def test_shared_memo_keeps_sharing_across_calls():
  """Passing one memo to two calls keeps a value shared between the copies."""
  shared = _Leaf(values=[1])
  first = _Tree(left=shared)
  second = _Tree(right=shared)
  memo: dict[int, Any] = {}

  copied_first = deep_copy_model(first, memo)
  copied_second = deep_copy_model(second, memo)

  assert copied_second.right is copied_first.left
  assert any(kept is shared for kept in memo[id(memo)])
