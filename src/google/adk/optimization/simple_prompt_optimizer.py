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

"""A simple iterative prompt optimizer."""

from __future__ import annotations

import logging
import random

from google.adk.agents.llm_agent import Agent
from google.adk.evaluation._retry_options_utils import add_default_retry_options_if_not_present
from google.adk.models import google_llm
from google.adk.models import llm_request
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.optimization.agent_optimizer import AgentOptimizer
from google.adk.optimization.data_types import BaseAgentWithScores
from google.adk.optimization.data_types import OptimizerResult
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.sampler import Sampler
from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger("google_adk." + __name__)

_OPTIMIZER_PROMPT_TEMPLATE = """
You are an expert prompt engineer. Your task is to improve the system prompt for an AI agent.
The agent's current prompt achieved an average score of {current_score:.2f} on a set of evaluation tasks. A higher score is better.

Here is the current prompt:
<current_prompt>
{current_prompt_text}
</current_prompt>

Based on the current prompt, rewrite it to create a new, improved version that is likely to achieve a higher score.
The agent needs to solve customer support tasks by using tools correctly and following policies.
Focus on clarity, structure, and providing actionable guidance for the agent.

**Output only the new, full, improved agent prompt. Do not add any other text, explanations, or markdown formatting.**
"""


class SimplePromptOptimizerConfig(BaseModel):
  """Configuration for the IterativePromptOptimizer."""

  optimizer_model: str = Field(
      default="gemini-2.5-flash",
      description=(
          "The model used to analyze the eval results and optimize the agent."
      ),
  )

  model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=10240,
          )
      ),
      description="The configuration for the optimizer model.",
  )

  num_iterations: int = Field(
      default=10,
      description="The number of optimization rounds to run.",
  )
  batch_size: int = Field(
      default=5,
      description=(
          "The number of training examples to use for scoring each candidate."
      ),
  )


class SimplePromptOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, BaseAgentWithScores]
):
  """A naive optimizer that iteratively tries to improve an agent's prompt."""

  def __init__(self, config: SimplePromptOptimizerConfig):
    self._config = config
    llm_registry = LLMRegistry()
    self._llm = llm_registry.new_llm(self._config.optimizer_model)

  async def _generate_candidate_prompt(
      self, best_agent: Agent, best_score: float
  ) -> str:
    """Generates a new prompt candidate using the optimizer LLM."""
    prompt_for_optimizer = _OPTIMIZER_PROMPT_TEMPLATE.format(
        current_score=best_score,
        current_prompt_text=best_agent.instruction,
    )
    llm_request = LlmRequest(
        model=self._config.optimizer_model,
        config=self._config.model_configuration,
        contents=[
            genai_types.Content(
                parts=[genai_types.Part(text=prompt_for_optimizer)],
                role="user",
            ),
        ],
    )
    add_default_retry_options_if_not_present(llm_request)

    response_text = ""
    async for llm_response in self._llm.generate_content_async(llm_request):
      if not (llm_response.content and llm_response.content.parts):
        continue
      for part in llm_response.content.parts:
        if part.text and not part.thought:
          response_text += part.text
    return response_text

  async def _score_agent_on_batch(
      self,
      agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
      example_ids: list[str],
  ) -> float:
    """Scores the agent on a random batch of training examples."""
    eval_batch = random.sample(example_ids, self._config.batch_size)
    eval_results = await sampler.sample_and_score(
        agent, "train", eval_batch, capture_full_eval_data=False
    )
    if not eval_results.scores:
      return 0.0
    return sum(eval_results.scores.values()) / len(eval_results.scores)

  async def _run_optimization_iterations(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
      train_example_ids: list[str],
  ) -> tuple[Agent, float]:
    """Runs the optimization loop and returns the best agent and score."""
    best_agent = initial_agent
    logger.info("Evaluating initial agent to get baseline score...")
    best_score = await self._score_agent_on_batch(
        best_agent, sampler, train_example_ids
    )
    logger.info("Initial agent baseline score: %f", best_score)

    for i in range(self._config.num_iterations):
      logger.info(
          "--- Starting optimization iteration %d/%d ---",
          i + 1,
          self._config.num_iterations,
      )
      new_prompt_text = await self._generate_candidate_prompt(
          best_agent, best_score
      )
      candidate_agent = best_agent.clone(
          update={"instruction": new_prompt_text}
      )
      logger.info("Generated new candidate prompt:\n%s", new_prompt_text)
      candidate_score = await self._score_agent_on_batch(
          candidate_agent, sampler, train_example_ids
      )
      logger.info(
          "Candidate score: %f (vs. best score: %f)",
          candidate_score,
          best_score,
      )
      if candidate_score > best_score:
        logger.info("New candidate is better. Updating best agent.")
        best_agent = candidate_agent
        best_score = candidate_score
      else:
        logger.info("New candidate is not better. Discarding.")
    return best_agent, best_score

  async def _run_final_validation(
      self,
      best_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
  ) -> float:
    """Runs final validation on the best agent found."""
    logger.info(
        "Optimization loop finished. Running final validation on the best agent"
        " found."
    )
    validation_results = await sampler.sample_and_score(
        best_agent, "validation"
    )
    if not validation_results.scores:
      return 0.0
    return sum(validation_results.scores.values()) / len(
        validation_results.scores
    )

  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
  ) -> OptimizerResult[BaseAgentWithScores]:
    train_example_ids = sampler.get_train_example_ids()

    if self._config.batch_size > len(train_example_ids):
      logger.warning(
          "Batch size (%d) is larger than the number of training examples"
          " (%d). Using all training examples for each evaluation.",
          self._config.batch_size,
          len(train_example_ids),
      )
      self._config.batch_size = len(train_example_ids)

    best_agent, _ = await self._run_optimization_iterations(
        initial_agent, sampler, train_example_ids
    )

    final_score = await self._run_final_validation(best_agent, sampler)
    logger.info("Final validation score: %f", final_score)

    return OptimizerResult(
        optimized_agents=[
            BaseAgentWithScores(
                optimized_agent=best_agent, overall_score=final_score
            )
        ]
    )
