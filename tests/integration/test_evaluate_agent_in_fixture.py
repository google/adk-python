# Copyright 2025 Google LLC
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

"""Evaluate all agents in fixture folder if evaluation test files exist."""

import os

from google.adk.evaluation.agent_evaluator import AgentEvaluator
import pytest


def agent_eval_artifacts_in_fixture():
  """Get all agents from fixture folder."""
  agent_eval_artifacts = []
  fixture_dir = os.path.join(os.path.dirname(__file__), 'fixture')
  for agent_name in os.listdir(fixture_dir):
    agent_dir = os.path.join(fixture_dir, agent_name)
    if not os.path.isdir(agent_dir):
      continue
    for filename in os.listdir(agent_dir):
      # Evaluation test files end with test.json
      if not filename.endswith('test.json'):
        continue
      agent_eval_artifacts.append((
          f'tests.integration.fixture.{agent_name}',
          f'tests/integration/fixture/{agent_name}/{filename}',
      ))

  # This method gets invoked twice, sorting helps ensure that both the
  # invocations have the same view.
  agent_eval_artifacts = sorted(
      agent_eval_artifacts, key=lambda item: f'{item[0]}|{item[1]}'
  )
  return agent_eval_artifacts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_name, evalfile',
    agent_eval_artifacts_in_fixture(),
    ids=[agent_name for agent_name, _ in agent_eval_artifacts_in_fixture()],
)
async def test_evaluate_agents_long_running_4_runs_per_eval_item(
    agent_name, evalfile
):
  """Test agents evaluation in fixture folder.

  After querying the fixture folder, we have 5 eval items. For each eval item
  we use 4 runs.

  A single eval item is a session that can have multiple queries in it.
  """
  await AgentEvaluator.evaluate(
      agent_module=agent_name,
      eval_dataset_file_path_or_dir=evalfile,
      # Using a slightly higher value helps us manage the variances that may
      # happen in each eval.
      # This, of course, comes at a cost of increased test run times.
      num_runs=4,
  )
