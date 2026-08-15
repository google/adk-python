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

"""Tests for the event normalization used to replay recorded agent sessions."""

from __future__ import annotations

from google.adk.cli.agent_test_runner import make_sort_key
from google.adk.cli.agent_test_runner import normalize_events
from google.adk.events.event import Event
from google.genai import types


def test_normalize_events_drops_volatile_fields_and_nulls_from_json_events():
  event = {
      'id': 'e-1',
      'timestamp': 1234.5,
      'invocationId': 'i-1',
      'invocation_id': 'i-1',
      'usageMetadata': {'totalTokenCount': 7},
      'interactionId': 'server-token',
      'turnComplete': True,
      'author': 'agent',
      'output': None,
  }

  # Everything that differs between two identical runs has to go, in either
  # naming convention, and null-valued keys must not survive either.
  assert normalize_events([event], is_json=True) == [{'author': 'agent'}]


def test_normalize_events_agrees_between_event_objects_and_recorded_json():
  event = Event(
      author='agent',
      invocation_id='i-1',
      content=types.Content(role='model', parts=[types.Part(text='hello')]),
      long_running_tool_ids={'b', 'a'},
  )
  recorded = event.model_dump(mode='json', by_alias=True, exclude_none=True)

  # This equality is the whole point of the function: a live run and the
  # fixture it is compared against must normalize to the same shape.
  assert normalize_events([event], is_json=False) == normalize_events(
      [recorded], is_json=True
  )
  assert normalize_events([event], is_json=False) == [{
      'author': 'agent',
      'content': {'role': 'model', 'parts': [{'text': 'hello'}]},
      'nodeInfo': {'path': ''},
      'longRunningToolIds': ['a', 'b'],
  }]


def test_normalize_events_strips_thought_signatures_from_parts():
  event = {
      'author': 'agent',
      'content': {
          'role': 'model',
          'parts': [{'text': 'hi', 'thoughtSignature': 'opaque-blob'}],
      },
  }

  normalized = normalize_events([event], is_json=True)

  assert normalized[0]['content']['parts'] == [{'text': 'hi'}]


def test_normalize_events_drops_role_only_for_human_in_the_loop_requests():
  hitl = {
      'author': 'agent',
      'content': {
          'role': 'model',
          'parts': [{'functionCall': {'name': 'adk_request_confirmation'}}],
      },
  }
  ordinary = {
      'author': 'agent',
      'content': {
          'role': 'model',
          'parts': [{'functionCall': {'name': 'roll_dice'}}],
      },
  }

  normalized = normalize_events([hitl, ordinary], is_json=True)

  # The role of a HITL request is not stable across runs; every other event
  # keeps it.
  assert 'role' not in normalized[0]['content']
  assert normalized[1]['content']['role'] == 'model'


def test_normalize_events_sorts_long_running_tool_ids_and_drops_empty_lists():
  unordered = {'author': 'agent', 'longRunningToolIds': ['z', 'a', 'm']}
  empty = {'author': 'agent', 'longRunningToolIds': []}

  normalized = normalize_events([unordered, empty], is_json=True)

  # The ids come from a set, so only the sorted form is reproducible.
  assert normalized[0]['longRunningToolIds'] == ['a', 'm', 'z']
  assert 'longRunningToolIds' not in normalized[1]


def test_normalize_events_prunes_empty_action_groups():
  partly_empty = {
      'author': 'agent',
      'actions': {'stateDelta': {}, 'artifactDelta': {'report.md': 1}},
  }
  all_empty = {
      'author': 'agent',
      'actions': {'stateDelta': {}, 'artifactDelta': {}},
  }

  normalized = normalize_events([partly_empty, all_empty], is_json=True)

  assert normalized[0]['actions'] == {'artifactDelta': {'report.md': 1}}
  assert 'actions' not in normalized[1]


def test_normalize_events_drops_join_state_keys_from_state_delta():
  event = {
      'author': 'agent',
      'actions': {
          'stateDelta': {
              'answer': 42,
              'fanout_join_state': {'pending': 2},
          }
      },
  }

  normalized = normalize_events([event], is_json=True)

  # Join bookkeeping is an implementation detail of parallel execution.
  assert normalized[0]['actions']['stateDelta'] == {'answer': 42}


def test_make_sort_key_orders_by_author_then_node_path():
  events = [
      {'author': 'b', 'nodeInfo': {'path': 'a'}},
      {'author': 'a', 'nodeInfo': {'path': 'z'}},
      {'author': 'a', 'nodeInfo': {'path': 'a'}},
      {'author': 'a'},
  ]

  ordered = sorted(events, key=make_sort_key)

  assert [
      (event['author'], event.get('nodeInfo', {}).get('path', ''))
      for event in ordered
  ] == [('a', ''), ('a', 'a'), ('a', 'z'), ('b', 'a')]


def test_make_sort_key_ignores_dict_key_order_but_separates_content():
  same_content_a = {'author': 'a', 'first': 1, 'second': 2}
  same_content_b = {'author': 'a', 'second': 2, 'first': 1}
  other_content = {'author': 'a', 'first': 1, 'second': 3}

  # Two events that only differ in insertion order must sort as one value,
  # otherwise fixture comparison depends on dict ordering.
  assert make_sort_key(same_content_a) == make_sort_key(same_content_b)
  assert make_sort_key(same_content_a) < make_sort_key(other_content)
