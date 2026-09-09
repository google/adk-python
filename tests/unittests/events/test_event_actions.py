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

"""Unit tests for EventActions serialization and its fallback helper."""

import datetime
import logging

from google.adk.events.event_actions import _make_json_serializable
from google.adk.events.event_actions import EventActions
from pydantic import BaseModel


class _Sample(BaseModel):
  x: int = 5
  label: str = 'hi'


class TestMakeJsonSerializable:
  """Tests for the `_make_json_serializable` fallback helper."""

  def test_plain_values_are_unchanged(self):
    value = {'a': 1, 'b': [1, 2], 'c': {'d': 'e'}, 'f': None, 'g': True}
    assert _make_json_serializable(value) == value

  def test_datetime_is_preserved_not_discarded(self):
    dt = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    assert _make_json_serializable(dt) == '2024-01-02T03:04:05Z'

  def test_pydantic_model_is_serialized_to_dict(self):
    assert _make_json_serializable(_Sample()) == {'x': 5, 'label': 'hi'}

  def test_nested_rich_types_are_serialized(self):
    dt = datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc)
    result = _make_json_serializable({'when': dt, 'model': _Sample(), 'n': [1]})
    assert result == {
        'when': '2024-05-06T00:00:00Z',
        'model': {'x': 5, 'label': 'hi'},
        'n': [1],
    }

  def test_unserializable_value_is_replaced_with_repr(self):
    result = _make_json_serializable(lambda: 1)
    assert isinstance(result, str)
    assert 'function' in result

  def test_unserializable_value_nested(self):
    result = _make_json_serializable({'cb': lambda: 1, 'ok': 2})
    assert result['ok'] == 2
    assert isinstance(result['cb'], str)


class TestStateDeltaSerialization:
  """Tests for the `state_delta` wrap serializer."""

  def test_serializable_state_delta_round_trips(self):
    actions = EventActions(state_delta={'a': 1, 'b': [1, 2], 'c': {'d': 'e'}})
    dumped = actions.model_dump(mode='json')
    assert dumped['state_delta'] == {'a': 1, 'b': [1, 2], 'c': {'d': 'e'}}

  def test_non_serializable_state_delta_does_not_raise(self):
    actions = EventActions(state_delta={'cb': lambda: 1, 'ok': 2})
    dumped = actions.model_dump(mode='json')
    assert dumped['state_delta']['ok'] == 2
    assert isinstance(dumped['state_delta']['cb'], str)

  def test_non_serializable_state_delta_logs_warning(self, caplog):
    actions = EventActions(state_delta={'cb': lambda: 1})
    with caplog.at_level(logging.WARNING):
      actions.model_dump(mode='json')
    assert any(
        'Failed to serialize `state_delta`' in record.message
        for record in caplog.records
    )

  def test_datetime_preserved_when_fallback_triggered(self):
    # A callable forces the fallback path; the datetime must still serialize
    # faithfully rather than being discarded.
    dt = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    actions = EventActions(state_delta={'when': dt, 'cb': lambda: 1})
    dumped = actions.model_dump(mode='json')
    assert dumped['state_delta']['when'] == '2024-01-02T03:04:05Z'

  def test_exclude_is_respected_for_serializable_state(self):
    actions = EventActions(
        state_delta={'_adk_replay_config': {'dir': '/x'}, 'foo': 1}
    )
    dumped = actions.model_dump(
        mode='json', exclude={'state_delta': {'_adk_replay_config': True}}
    )
    assert dumped['state_delta'] == {'foo': 1}

  def test_exclude_is_respected_in_fallback_path(self):
    # Even when sanitization is required (callable present), caller `exclude`
    # directives must still be applied to the fallback output.
    actions = EventActions(
        state_delta={
            '_adk_replay_config': {'dir': '/x'},
            'cb': lambda: 1,
            'ok': 2,
        }
    )
    dumped = actions.model_dump(
        mode='json', exclude={'state_delta': {'_adk_replay_config': True}}
    )
    assert '_adk_replay_config' not in dumped['state_delta']
    assert dumped['state_delta']['ok'] == 2
    assert isinstance(dumped['state_delta']['cb'], str)


class TestAgentStateSerialization:
  """Tests for the `agent_state` wrap serializer."""

  def test_none_agent_state_serializes_to_none(self):
    assert EventActions().model_dump(mode='json')['agent_state'] is None

  def test_serializable_agent_state_round_trips(self):
    actions = EventActions(agent_state={'a': 1, 'b': 'two'})
    assert actions.model_dump(mode='json')['agent_state'] == {
        'a': 1,
        'b': 'two',
    }

  def test_non_serializable_agent_state_does_not_raise(self):
    actions = EventActions(agent_state={'cb': lambda: 1, 'n': 3})
    dumped = actions.model_dump(mode='json')
    assert dumped['agent_state']['n'] == 3
    assert isinstance(dumped['agent_state']['cb'], str)

  def test_non_serializable_agent_state_logs_warning(self, caplog):
    actions = EventActions(agent_state={'cb': lambda: 1})
    with caplog.at_level(logging.WARNING):
      actions.model_dump(mode='json')
    assert any(
        'Failed to serialize `agent_state`' in record.message
        for record in caplog.records
    )
