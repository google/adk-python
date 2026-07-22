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

"""Regression tests for _get_transfer_targets in agent_transfer module."""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.flows.llm_flows.agent_transfer import _get_transfer_targets

from ... import testing_utils


def test_loop_agent_peer_does_not_raise_attribute_error():
  """LoopAgent as a peer agent must not raise AttributeError.

  LoopAgent has no 'mode' attribute; _get_transfer_targets must not crash
  when iterating over peer agents of an LlmAgent.
  """
  mock_model = testing_utils.MockModel.create(responses=['response'])

  loop_peer = LoopAgent(name='loop_peer', sub_agents=[])
  llm_agent = Agent(name='llm_agent', model=mock_model)

  Agent(
      name='root',
      model=mock_model,
      sub_agents=[llm_agent, loop_peer],
  )

  targets = _get_transfer_targets(llm_agent)
  target_names = [t.name for t in targets]
  assert 'loop_peer' in target_names


def test_loop_agent_sub_agent_does_not_raise_attribute_error():
  """LoopAgent as a sub-agent of the current agent must not raise AttributeError."""
  mock_model = testing_utils.MockModel.create(responses=['response'])

  loop_sub = LoopAgent(name='loop_sub', sub_agents=[])
  llm_agent = Agent(
      name='llm_agent',
      model=mock_model,
      sub_agents=[loop_sub],
  )

  targets = _get_transfer_targets(llm_agent)
  target_names = [t.name for t in targets]
  assert 'loop_sub' in target_names


def test_sequential_agent_peer_does_not_raise_attribute_error():
  """SequentialAgent as a peer agent must not raise AttributeError.

  SequentialAgent has no 'mode' attribute; the hasattr guard must cover it.
  """
  mock_model = testing_utils.MockModel.create(responses=['response'])

  seq_peer = SequentialAgent(name='seq_peer', sub_agents=[])
  llm_agent = Agent(name='llm_agent', model=mock_model)

  Agent(
      name='root',
      model=mock_model,
      sub_agents=[llm_agent, seq_peer],
  )

  targets = _get_transfer_targets(llm_agent)
  target_names = [t.name for t in targets]
  assert 'seq_peer' in target_names


def test_parallel_agent_peer_does_not_raise_attribute_error():
  """ParallelAgent as a peer agent must not raise AttributeError.

  ParallelAgent has no 'mode' attribute; the hasattr guard must cover it.
  """
  mock_model = testing_utils.MockModel.create(responses=['response'])

  par_peer = ParallelAgent(name='par_peer', sub_agents=[])
  llm_agent = Agent(name='llm_agent', model=mock_model)

  Agent(
      name='root',
      model=mock_model,
      sub_agents=[llm_agent, par_peer],
  )

  targets = _get_transfer_targets(llm_agent)
  target_names = [t.name for t in targets]
  assert 'par_peer' in target_names


def test_single_turn_peer_is_excluded_from_transfer_targets():
  """LlmAgent with mode='single_turn' must be excluded from peer targets.

  Verifies the filtering logic: agents in single_turn mode should not be
  offered as transfer destinations since they are not interactive.
  """
  mock_model = testing_utils.MockModel.create(responses=['response'])

  single_turn_peer = Agent(
      name='single_turn_peer', model=mock_model, mode='single_turn'
  )
  llm_agent = Agent(name='llm_agent', model=mock_model)

  Agent(
      name='root',
      model=mock_model,
      sub_agents=[llm_agent, single_turn_peer],
  )

  targets = _get_transfer_targets(llm_agent)
  target_names = [t.name for t in targets]
  assert 'single_turn_peer' not in target_names
