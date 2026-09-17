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

import json
from types import SimpleNamespace
from unittest import mock

from google.adk.cli import agent_test_runner
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


def test_rebuild_tests_preserves_non_ascii_event_text(
    tmp_path, monkeypatch, capsys
):
  """Rebuilt test files preserve non-ASCII event text."""
  agent_dir = tmp_path / 'test_agent'
  tests_dir = agent_dir / 'tests'
  tests_dir.mkdir(parents=True)
  (agent_dir / 'agent.py').write_text('', encoding='utf-8')
  test_file = tests_dir / 'unicode.json'
  session_data = {
      'events': [{
          'author': 'user',
          'content': {
              'role': 'user',
              'parts': [{'text': '日本語の質問'}],
          },
      }]
  }
  test_file.write_text(
      json.dumps(session_data, ensure_ascii=False), encoding='utf-8'
  )

  class _Runner:

    def __init__(self):
      self.session = SimpleNamespace(user_id='test_user', id='test_session')
      self.runner = self

    async def run_async(self, **kwargs):
      del kwargs
      yield Event(
          author='test_agent',
          invocation_id='live-invocation',
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='日本語の回答')],
          ),
      )

  loader = mock.create_autospec(
      agent_test_runner.AgentLoader, instance=True, spec_set=True
  )
  loader.load_agent.return_value = object()
  loader_factory = mock.create_autospec(
      agent_test_runner.AgentLoader, spec_set=True, return_value=loader
  )
  monkeypatch.setattr(
      agent_test_runner,
      'AgentLoader',
      loader_factory,
  )
  runner_factory = mock.create_autospec(
      agent_test_runner.InMemoryRunner,
      spec_set=True,
      return_value=_Runner(),
  )
  monkeypatch.setattr(
      agent_test_runner,
      'InMemoryRunner',
      runner_factory,
  )

  agent_test_runner.rebuild_tests(str(agent_dir))

  # rebuild_tests swallows per-fixture exceptions into a printed line, so check
  # it here; otherwise any breakage surfaces as an opaque substring mismatch.
  stdout = capsys.readouterr().out
  assert 'Error rebuilding' not in stdout, stdout

  rebuilt = test_file.read_text(encoding='utf-8')
  assert '日本語の質問' in rebuilt
  assert '日本語の回答' in rebuilt
  assert '\\u' not in rebuilt
