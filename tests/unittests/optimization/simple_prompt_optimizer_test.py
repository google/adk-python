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

"""Tests for simple_prompt_optimizer."""

from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import LlmResponse
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizer
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizerConfig
from google.genai import types as genai_types
import pytest


@pytest.fixture
def mock_sampler() -> mock.MagicMock:
  sampler = mock.MagicMock(spec=Sampler)
  sampler.get_train_example_ids.return_value = ["1", "2", "3", "4", "5"]
  sampler.get_validation_example_ids.return_value = ["v1", "v2"]

  async def mock_sample_and_score(
      agent: Agent,
      example_set: str,
      batch: list[str] | None = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    # Determine the actual batch to use
    if batch is None:
      if example_set == "train":
        current_batch = sampler.get_train_example_ids()
      else:  # "validation"
        current_batch = sampler.get_validation_example_ids()
    else:
      current_batch = batch

    # Simulate better score for "improved" prompt
    if "IMPROVED" in agent.instruction:
      scores = {uid: 0.9 for uid in current_batch}
    else:
      scores = {uid: 0.5 for uid in current_batch}
    return UnstructuredSamplingResult(scores=scores)

  sampler.sample_and_score.side_effect = mock_sample_and_score
  return sampler


@pytest.fixture
def mock_llm_class() -> mock.MagicMock:
  mock_llm = mock.MagicMock()

  async def mock_generate_content_async(*args, **kwargs):
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="IMPROVED PROMPT")]
        )
    )

  mock_llm.generate_content_async.side_effect = mock_generate_content_async
  mock_class = mock.MagicMock(return_value=mock_llm)
  return mock_class


@mock.patch(
    "google.adk.optimization.simple_prompt_optimizer.LLMRegistry.resolve"
)
@pytest.mark.asyncio
async def test_simple_prompt_optimizer(
    mock_llm_resolve: mock.MagicMock,
    mock_llm_class: mock.MagicMock,
    mock_sampler: mock.MagicMock,
):
  """Test the SimplePromptOptimizer."""
  mock_llm_resolve.return_value = mock_llm_class
  config = SimplePromptOptimizerConfig(num_iterations=2, batch_size=2)
  optimizer = SimplePromptOptimizer(config)

  initial_agent = Agent(name="test_agent", instruction="Initial Prompt")
  result = await optimizer.optimize(initial_agent, mock_sampler)

  # Assertions
  assert len(result.optimized_agents) == 1
  optimized_agent = result.optimized_agents[0].optimized_agent
  assert optimized_agent.instruction == "IMPROVED PROMPT"
  assert result.optimized_agents[0].overall_score == 0.9

  # Check mock calls
  assert mock_sampler.get_train_example_ids.call_count == 1
  # 1 initial, 2 iterations, 1 final validation
  assert mock_sampler.sample_and_score.call_count == 4
  assert mock_llm_class.return_value.generate_content_async.call_count == 2
