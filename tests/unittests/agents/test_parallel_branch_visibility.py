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

"""Which earlier agent outputs reach an agent's LLM request (issue #3470)."""

import random

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
import pytest

from .. import testing_utils


def _agent(name: str) -> LlmAgent:
  return LlmAgent(
      name=name, model=testing_utils.MockModel.create(responses=[f'{name} out'])
  )


def _seen_outputs(agent: LlmAgent) -> set[str]:
  """Names of agents whose output is in the agent's last LLM request."""
  seen = set()
  for content in agent.model.requests[-1].contents:
    for part in content.parts or []:
      if part.text and part.text.rstrip().endswith(' out'):
        seen.add(part.text.split(' out')[0].split()[-1])
  return seen


@pytest.mark.asyncio
async def test_reducer_nested_in_parallel_sees_its_workers():
  """The architecture from issue #3470."""
  a, b, c, d, e, f = (_agent(n) for n in 'abcdef')
  r1, r2, r3 = _agent('r1'), _agent('r2'), _agent('r3')
  root = SequentialAgent(
      name='final_seq',
      sub_agents=[
          ParallelAgent(
              name='final_par',
              sub_agents=[
                  SequentialAgent(
                      name='group1',
                      sub_agents=[
                          ParallelAgent(name='abc', sub_agents=[a, b, c]),
                          r1,
                      ],
                  ),
                  SequentialAgent(
                      name='group2',
                      sub_agents=[
                          ParallelAgent(name='def', sub_agents=[d, e, f]),
                          r2,
                      ],
                  ),
              ],
          ),
          r3,
      ],
  )

  await testing_utils.InMemoryRunner(root).run_async('go')

  assert _seen_outputs(r1) == {'a', 'b', 'c'}
  assert _seen_outputs(r2) == {'d', 'e', 'f'}
  assert _seen_outputs(r3) == {'a', 'b', 'c', 'd', 'e', 'f', 'r1', 'r2'}


@pytest.mark.asyncio
async def test_sequence_of_parallel_stages_sees_earlier_stages():
  """The follow-up case from issue #3470."""
  a, b, c, d = (_agent(n) for n in 'abcd')
  root = SequentialAgent(
      name='seq',
      sub_agents=[
          ParallelAgent(name='stage1', sub_agents=[a, b]),
          ParallelAgent(name='stage2', sub_agents=[c, d]),
      ],
  )

  await testing_utils.InMemoryRunner(root).run_async('go')

  assert _seen_outputs(a) == set()
  assert _seen_outputs(c) == {'a', 'b'}
  assert _seen_outputs(d) == {'a', 'b'}


# A path is the list of (container_kind, child_index) from the root to a leaf.
_Path = tuple[tuple[type[BaseAgent], int], ...]


def _random_tree(
    rng: random.Random,
    depth: int,
    leaves: dict[str, tuple[LlmAgent, _Path]],
    path: _Path = (),
) -> BaseAgent:
  """Builds a random Sequential/Parallel tree with LlmAgent leaves."""
  if depth == 0 or (path and rng.random() < 0.25):
    leaf = _agent(f'a{len(leaves)}')
    leaves[leaf.name] = (leaf, path)
    return leaf
  kind = rng.choice([SequentialAgent, ParallelAgent])
  children = [
      _random_tree(rng, depth - 1, leaves, path + ((kind, i),))
      for i in range(rng.randint(2, 3))
  ]
  name = '_'.join([kind.__name__] + [str(i) for _, i in path])
  return kind(name=name, sub_agents=children)


def _expected_seen(viewer: _Path, others: dict[str, _Path]) -> set[str]:
  """Outputs the viewer should see, derived from the tree alone.

  Walk both paths from the root to the first container where they take
  different children. If it is a ParallelAgent the two are in sibling lanes and
  the output is hidden. If it is a SequentialAgent the output is visible
  exactly when the other agent's child came first.
  """
  seen = set()
  for name, other in others.items():
    for (kind, i), (_, j) in zip(viewer, other):
      if i != j:
        if kind is SequentialAgent and j < i:
          seen.add(name)
        break
  return seen


@pytest.mark.asyncio
@pytest.mark.parametrize('seed', range(20))
async def test_visibility_matches_tree_structure(seed):
  """Hidden across ParallelAgent lanes, visible otherwise, for any tree."""
  leaves: dict[str, tuple[LlmAgent, _Path]] = {}
  root = _random_tree(random.Random(seed), 3, leaves)

  await testing_utils.InMemoryRunner(root).run_async('go')

  paths = {name: path for name, (_, path) in leaves.items()}
  for name, (leaf, path) in leaves.items():
    assert _seen_outputs(leaf) == _expected_seen(path, paths), name
