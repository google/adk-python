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

import asyncio
import logging
import random
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.evaluation._retry_options_utils import add_default_retry_options_if_not_present
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.optimization._run_context import OptimizationBudgetExceeded
from google.adk.optimization._run_context import OptimizationProviderError
from google.adk.optimization._run_context import OptimizationRunContext
from google.adk.optimization._run_context import OptimizerCapabilities
from google.adk.optimization._run_context import RunStatus
from google.adk.optimization._run_context import STAGE_CANDIDATE_GENERATION
from google.adk.optimization.agent_optimizer import AgentOptimizer
from google.adk.optimization.data_types import AgentWithScores
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
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
  """A naive optimizer that iteratively tries to improve an agent's prompt."""

  def __init__(self, config: SimplePromptOptimizerConfig):
    self._config = config
    llm_registry = LLMRegistry()
    self._llm = llm_registry.new_llm(self._config.optimizer_model)

  async def _generate_candidate_prompt(
      self,
      best_agent: Agent,
      best_score: float,
      run_context: Optional[OptimizationRunContext] = None,
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

    # One logical optimizer-owned model invocation: atomic admission checks
    # cancellation and the call budget before the call starts; the terminal
    # outcome (usage, provider error, or cancellation) is committed after.
    handle = None
    if run_context is not None:
      handle = run_context.begin_model_call(
          STAGE_CANDIDATE_GENERATION,
          requested_model=self._config.optimizer_model,
      )

    response_text = ""
    last_usage = None
    model_version = None
    try:
      async for llm_response in self._llm.generate_content_async(llm_request):
        if getattr(llm_response, "usage_metadata", None) is not None:
          last_usage = llm_response.usage_metadata
        if getattr(llm_response, "model_version", None):
          model_version = llm_response.model_version
        error_code = getattr(llm_response, "error_code", None)
        if error_code and run_context is not None and handle is not None:
          # Governed runs preserve usage-so-far and terminate on an in-band
          # provider error instead of silently succeeding. end_model_call
          # transitions the run to FAILED and raises the typed error.
          run_context.end_model_call(
              handle,
              usage_metadata=last_usage,
              returned_model_version=model_version,
              error_code=str(error_code),
              error_type=type(llm_response).__name__,
          )
        if not (llm_response.content and llm_response.content.parts):
          continue
        for part in llm_response.content.parts:
          if part.text and not part.thought:
            response_text += part.text
    except asyncio.CancelledError:
      # Native task cancellation: settle the open call WITH the usage
      # evidence accumulated so far, finalize the governed run as cancelled,
      # and re-raise promptly.
      if run_context is not None:
        if handle is not None:
          run_context.end_model_call(
              handle,
              usage_metadata=last_usage,
              returned_model_version=model_version,
              cancelled=True,
          )
        run_context.finalize_cancelled("task_cancelled")
      raise
    except OptimizationProviderError:
      raise
    except Exception as e:
      if run_context is not None and handle is not None:
        run_context.end_model_call(
            handle,
            usage_metadata=last_usage,
            error_code=str(
                getattr(e, "error_code", None) or "PROVIDER_EXCEPTION"
            ),
            error_type=type(e).__name__,
        )
      raise
    if run_context is not None and handle is not None:
      run_context.end_model_call(
          handle,
          usage_metadata=last_usage,
          returned_model_version=model_version,
      )
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
      run_context: Optional[OptimizationRunContext] = None,
  ) -> tuple[Agent, float]:
    """Runs the optimization loop and returns the best agent and score."""
    best_agent = initial_agent
    if run_context is not None:
      # Boundary: before the baseline sampler evaluation. A pre-cancelled
      # run performs zero sampler work.
      run_context.raise_if_cancelled()
    logger.info("Evaluating initial agent to get baseline score...")
    best_score = await self._score_agent_on_batch(
        best_agent, sampler, train_example_ids
    )
    logger.info("Initial agent baseline score: %f", best_score)

    for i in range(self._config.num_iterations):
      if run_context is not None:
        # Boundary: after the previous sampler operation / before this
        # iteration's model call.
        run_context.raise_if_cancelled()
      logger.info(
          "--- Starting optimization iteration %d/%d ---",
          i + 1,
          self._config.num_iterations,
      )
      try:
        new_prompt_text = await self._generate_candidate_prompt(
            best_agent, best_score, run_context
        )
      except OptimizationBudgetExceeded:
        if (
            run_context is not None
            and run_context.budgets.on_budget_exceeded == "return_partial"
        ):
          # Stop scheduling immediately after the final in-flight call
          # settled; the best-so-far agent survives as the partial result.
          logger.warning(
              "Optimization budget exhausted at iteration %d; stopping.",
              i + 1,
          )
          return best_agent, best_score
        raise
      if run_context is not None:
        # Boundary: after candidate generation / before candidate scoring.
        run_context.raise_if_cancelled()
      candidate_agent = best_agent.clone(
          update={"instruction": new_prompt_text}
      )
      logger.info("Generated new candidate prompt:\n%s", new_prompt_text)
      candidate_score = await self._score_agent_on_batch(
          candidate_agent, sampler, train_example_ids
      )
      if run_context is not None:
        # Boundary: after candidate scoring.
        run_context.raise_if_cancelled()
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

  @property
  def capabilities(self) -> OptimizerCapabilities:
    return OptimizerCapabilities(
        accepts_run_context=True,
        model_calls_observable=True,
        logical_call_limits_enforceable=True,
        reported_token_limits_enforceable=True,
        cooperative_cancellation=True,
        sampler_usage_included=False,
    )

  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
      *,
      run_context: Optional[OptimizationRunContext] = None,
  ) -> OptimizerResult[AgentWithScores]:
    if run_context is not None:
      run_context.attach(owner=self)

    try:
      if run_context is not None:
        # A pending cancellation wins over sampler discovery -- observed
        # before any sampler method is touched.
        run_context.raise_if_cancelled()
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
          initial_agent, sampler, train_example_ids, run_context
      )

      if (
          run_context is not None
          and run_context.snapshot().run_status == RunStatus.BUDGET_EXCEEDED
      ):
        # return_partial stop: no further optimizer-owned or sampler work is
        # scheduled -- final validation included, so overall_score stays
        # None. The caller-owned snapshot (run_status=budget_exceeded) is
        # the authoritative record; the public result schema is unchanged.
        return OptimizerResult(
            optimized_agents=[
                AgentWithScores(optimized_agent=best_agent, overall_score=None)
            ],
        )

      if run_context is not None:
        # Boundary: before final validation.
        run_context.raise_if_cancelled()
      final_score = await self._run_final_validation(best_agent, sampler)
      logger.info("Final validation score: %f", final_score)

      if run_context is not None:
        # Boundary: after final validation -- cancellation requested during
        # validation must not end as COMPLETED (finalize_success is also
        # defensive, but the explicit boundary keeps evidence and ordering
        # clear).
        run_context.raise_if_cancelled()
        run_context.finalize_success()
      return OptimizerResult(
          optimized_agents=[
              AgentWithScores(
                  optimized_agent=best_agent, overall_score=final_score
              )
          ]
      )
    except asyncio.CancelledError:
      if run_context is not None:
        run_context.finalize_cancelled("task_cancelled")
      raise
    except Exception as e:
      if run_context is not None:
        # Non-provider failures (sampler crashes, etc.) still finalize the
        # authoritative snapshot; typed run-context errors already did.
        run_context.finalize_failed(
            error_code=getattr(e, "error_code", None) or "OPTIMIZER_FAILURE",
            error_type=type(e).__name__,
        )
      raise
