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
async def test_simple_multi_turn_conversation():
  """Test a simple multi-turn conversation."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir="tests/integration/fixture/home_automation_agent/test_files/simple_multi_turn_conversation.test.json",
      num_runs=4,
  )


@pytest.mark.asyncio
async def test_dependent_tool_calls():
  """Test subsequent tool calls that are dependent on previous tool calls."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir="tests/integration/fixture/home_automation_agent/test_files/dependent_tool_calls.test.json",
      num_runs=4,
  )


@pytest.mark.asyncio
async def test_memorizing_past_events():
  """Test memorizing past events."""
  await AgentEvaluator.evaluate(
      agent_module="tests.integration.fixture.home_automation_agent",
      eval_dataset_file_path_or_dir="tests/integration/fixture/home_automation_agent/test_files/memorizing_past_events/eval_data.test.json",
      num_runs=4,
  )
