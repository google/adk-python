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

"""Tests for seeding empty session state from agent instructions."""

from __future__ import annotations

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.utils.state import create_empty_state


def test_create_empty_state_seeds_every_instruction_placeholder():
  agent = LlmAgent(
      name='root',
      instruction='Greet {user_name} about {topic} in {user_name} style.',
  )

  assert create_empty_state(agent) == {'user_name': '', 'topic': ''}


def test_create_empty_state_walks_the_whole_sub_agent_tree():
  grandchild = LlmAgent(name='grandchild', instruction='deep {deep_key}')
  child = LlmAgent(
      name='child', instruction='mid {mid_key}', sub_agents=[grandchild]
  )
  root = LlmAgent(name='root', instruction='top {top_key}', sub_agents=[child])

  assert create_empty_state(root) == {
      'top_key': '',
      'mid_key': '',
      'deep_key': '',
  }


def test_create_empty_state_omits_keys_already_initialized():
  agent = LlmAgent(name='root', instruction='{a} {b} {c}')

  result = create_empty_state(agent, {'b': 'set', 'unrelated': 'x'})

  # Only the keys the caller has not supplied are seeded, and an initialized
  # key is not echoed back with an empty value.
  assert result == {'a': '', 'c': ''}


def test_create_empty_state_only_matches_bare_word_placeholders():
  agent = LlmAgent(
      name='root',
      instruction='{ok_key} {user.name} {with-dash} {} {a b} {{escaped}}',
  )

  # The placeholder syntax is a single \\w+ run; anything else is left alone.
  assert create_empty_state(agent) == {'ok_key': '', 'escaped': ''}


def test_create_empty_state_ignores_non_llm_agents():
  class _Plain(BaseAgent):
    pass

  root = _Plain(
      name='root',
      sub_agents=[
          _Plain(name='plain_child'),
          LlmAgent(name='llm_child', instruction='{from_llm}'),
      ],
  )

  assert create_empty_state(root) == {'from_llm': ''}


def test_create_empty_state_ignores_callable_instruction_providers():
  def _instruction(_ctx):
    return 'dynamic {never_seeded}'

  root = LlmAgent(
      name='root',
      instruction=_instruction,
      sub_agents=[LlmAgent(name='child', instruction='{static_key}')],
  )

  assert create_empty_state(root) == {'static_key': ''}


def test_create_empty_state_returns_empty_dict_when_nothing_to_seed():
  assert create_empty_state(LlmAgent(name='root', instruction='no slots')) == {}
