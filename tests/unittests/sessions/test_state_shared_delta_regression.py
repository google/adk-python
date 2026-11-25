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

import datetime

from google.adk.sessions.state import State
from google.protobuf import duration_pb2
import pytest


@pytest.mark.parametrize(
    'non_json_value',
    [datetime.datetime(2026, 1, 1), duration_pb2.Duration(seconds=3)],
    ids=['datetime', 'protobuf'],
)
def test_shared_delta_withdraws_container_after_other_context_changes_it(
    non_json_value,
):
  value = {'config': {'initial': 1}}
  delta = {}
  first = State(value=value, delta=delta)
  second = State(value=value, delta=delta)

  first['config']['added'] = 2
  assert first.has_delta()
  assert delta == {'config': {'initial': 1, 'added': 2}}

  # Both contexts refer to the same event delta. The second context makes the
  # already-recorded container unequal to its JSON representation.
  second['config']['non_json'] = non_json_value

  assert not second.has_delta()
  assert 'config' not in delta


def test_explicit_assignment_in_other_context_is_not_withdrawn():
  value = {'config': {}}
  delta = {}
  first = State(value=value, delta=delta)
  second = State(value=value, delta=delta)
  first['config']['added'] = 1
  assert first.has_delta()

  second['config'] = {'explicit': 2}
  second['config']['non_json'] = datetime.datetime(2026, 1, 1)

  assert second.has_delta()
  assert 'config' in delta


def test_explicit_assignment_of_same_container_is_not_withdrawn():
  value = {'config': {}}
  delta = {}
  first = State(value=value, delta=delta)
  second = State(value=value, delta=delta)
  first['config']['added'] = 1
  assert first.has_delta()

  second['config'] = second['config']
  second['config']['non_json'] = datetime.datetime(2026, 1, 1)

  assert second.has_delta()
  assert 'config' in delta


def test_stale_reader_does_not_forget_another_contexts_recorded_value():
  value = {'config': {}}
  delta = {}
  reader = State(value=value, delta=delta)
  writer = State(value=value, delta=delta)
  invalidator = State(value=value, delta=delta)
  reader['config']
  value['config'] = {'replacement': 1}
  writer['config']['added'] = 2
  assert writer.has_delta()

  assert reader.has_delta()
  invalidator['config']['non_json'] = datetime.datetime(2026, 1, 1)

  assert not invalidator.has_delta()
  assert 'config' not in delta


def test_contexts_sharing_event_actions_withdraw_nested_delta():
  from unittest.mock import MagicMock

  from google.adk.agents.context import Context
  from google.adk.events.event_actions import EventActions

  invocation = MagicMock()
  invocation.session.state = {'config': {'initial': 1}}
  invocation._state_schema = None
  actions = EventActions()
  first = Context(invocation, event_actions=actions)
  second = Context(invocation, event_actions=actions)

  first.state['config']['added'] = 2
  assert first.state.has_delta()
  second.state['config']['non_json'] = datetime.datetime(2026, 1, 1)

  assert not second.state.has_delta()
  assert actions.state_delta == {}
