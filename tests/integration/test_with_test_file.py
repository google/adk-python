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

from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
import pytest


@pytest.mark.asyncio
async def test_with_single_test_file():
  """Test the agent's basic ability via session file."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir="tests/integration/fixture/home_automation_agent/simple_test.test.json",
  )


@pytest.mark.asyncio
async def test_with_custom_metric():
  """Test eval with a custom metric."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir=(
          "tests/integration/fixture/home_automation_agent/test_files/custom_metrics/simple_custom_metric.test.json"
      ),
      num_runs=1,
  )


@pytest.mark.asyncio
async def test_with_folder_of_test_files_long_running():
  """Test the agent's basic ability via a folder of session files."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir=(
          "tests/integration/fixture/home_automation_agent/test_files"
      ),
      num_runs=4,
  )


@pytest.mark.asyncio
async def test_with_single_test_file_saves_eval_set_result(
    tmp_path,
):
  """Persists eval set results under the explicitly provided app_name."""
  eval_set_results_manager = LocalEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir=(
          "tests/integration/fixture/home_automation_agent/simple_test.test.json"
      ),
      num_runs=2,
      app_name="home_automation_agent",
      eval_set_results_manager=eval_set_results_manager,
  )

  # Results are aggregated into a single eval set result file (matching
  # LocalEvalService), containing one EvalCaseResult per run.
  saved_result_files = list(
      (tmp_path / "home_automation_agent" / ".adk" / "eval_history").glob(
          "*.evalset_result.json"
      )
  )
  assert len(saved_result_files) == 1

  saved_result_ids = eval_set_results_manager.list_eval_set_results(
      "home_automation_agent"
  )
  assert len(saved_result_ids) == 1
  eval_set_result = eval_set_results_manager.get_eval_set_result(
      "home_automation_agent", saved_result_ids[0]
  )
  assert len(eval_set_result.eval_case_results) == 2
