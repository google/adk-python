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

from google.adk.evaluation.agent_evaluator import AgentEvaluator
import pytest


@pytest.mark.asyncio
async def test_eval_agent():
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.trip_planner_agent",
      eval_dataset_file_path_or_dir=(
          "tests/integration/fixture/trip_planner_agent/trip_inquiry.test.json"
      ),
      num_runs=4,
  )
