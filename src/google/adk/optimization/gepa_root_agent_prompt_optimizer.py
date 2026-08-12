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

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field

from ..agents.llm_agent import Agent
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..models.registry import LLMRegistry
from ..utils.feature_decorator import experimental
from ._gepa_utils import generate_reflection_response
from ._gepa_utils import GEPAPrompt
from ._gepa_utils import require_static_instruction
from .agent_optimizer import AgentOptimizer
from .data_types import AgentWithScores
from .data_types import OptimizerResult
from .data_types import UnstructuredSamplingResult
from .sampler import _ExampleSet
from .sampler import Sampler

if TYPE_CHECKING:
  from gepa.core.result import GEPAResult

_logger = logging.getLogger("google_adk." + __name__)

_AGENT_PROMPT_NAME = "agent_prompt"


class GEPARootAgentPromptOptimizerConfig(BaseModel):
  """Contains configuration options required by the GEPARootAgentPromptOptimizer."""

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

  max_metric_calls: int = Field(
      default=100,
      description="The maximum number of metric calls (evaluations) to make.",
  )

  reflection_minibatch_size: int = Field(
      default=3,
      description="The number of examples to use for reflection.",
  )

  run_dir: str | None = Field(
      default=None,
      description=(
          "The directory to save the intermediate/final optimization results."
      ),
  )


class GEPARootAgentPromptOptimizerResult(OptimizerResult[AgentWithScores]):
  """The final result of the GEPARootAgentPromptOptimizer."""

  gepa_result: dict[str, Any] | None = Field(
      default=None,
      description="The raw result dictionary from the GEPA optimizer.",
  )


def _create_agent_gepa_adapter_class() -> type[Any]:
  """Creates the _AgentGEPAAdapter class dynamically to avoid top-level gepa imports."""
  from gepa.core.adapter import EvaluationBatch
  from gepa.core.adapter import GEPAAdapter

  class _AgentGEPAAdapter(  # type: ignore[misc]
      GEPAAdapter[str, dict[str, Any], dict[str, Any]]
  ):
    """A GEPA adapter for ADK agents."""

    def __init__(
        self,
        initial_agent: Agent,
        sampler: Sampler[UnstructuredSamplingResult],
        main_loop: asyncio.AbstractEventLoop,
    ):
      self._initial_agent = initial_agent
      self._sampler = sampler
      self._main_loop = main_loop

      self._train_example_ids = set(sampler.get_train_example_ids())
      self._validation_example_ids = set(sampler.get_validation_example_ids())

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
      prompt = candidate[_AGENT_PROMPT_NAME]
      _logger.info(
          "Evaluating agent on batch:\n%s\nwith prompt:\n%s", batch, prompt
      )
      # Clone the agent and update the instruction
      new_agent = self._initial_agent.clone(update={"instruction": prompt})

      if set(batch) <= self._train_example_ids:
        example_set: _ExampleSet = "train"
      elif set(batch) <= self._validation_example_ids:
        example_set = "validation"
      else:
        raise ValueError(f"Invalid batch composition: {batch}")

      # Run the evaluation in the main loop
      future = asyncio.run_coroutine_threadsafe(
          self._sampler.sample_and_score(
              new_agent,
              example_set=example_set,
              batch=batch,
              capture_full_eval_data=capture_traces,
          ),
          self._main_loop,
      )
      result: UnstructuredSamplingResult = future.result()

      scores = []
      outputs = []
      trajectories = []

      for example_id in batch:
        score = result.scores[example_id]
        scores.append(score)

        eval_data = result.data.get(example_id, {}) if result.data else {}
        outputs.append(eval_data)
        trajectories.append(eval_data)

      return EvaluationBatch(
          outputs=outputs, scores=scores, trajectories=trajectories
      )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
      trajectories = eval_batch.trajectories
      if trajectories is None:
        raise ValueError(
            "GEPA cannot build a reflective dataset without captured"
            " trajectories."
        )
      dataset: list[dict[str, Any]] = []
      trace_instances: list[tuple[float, dict[str, Any]]] = list(
          zip(
              eval_batch.scores,
              trajectories,
              strict=True,
          )
      )
      for trace_instance in trace_instances:
        score, eval_data = trace_instance

        dataset.append({
            _AGENT_PROMPT_NAME: candidate[_AGENT_PROMPT_NAME],
            "score": score,
            "eval_data": eval_data,
        })

      # same data for all components (should be only one)
      result = {comp: dataset for comp in components_to_update}

      return result

  return _AgentGEPAAdapter


@experimental
class GEPARootAgentPromptOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
  """An optimizer that improves the root agent prompt using the GEPA framework."""

  def __init__(
      self,
      config: GEPARootAgentPromptOptimizerConfig,
  ):
    self._config = config
    llm_registry = LLMRegistry()
    self._llm_class = llm_registry.resolve(self._config.optimizer_model)

  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
  ) -> GEPARootAgentPromptOptimizerResult:
    """Runs the GEPARootAgentPromptOptimizer.

    Args:
      initial_agent: The initial agent whose prompt is to be optimized. Only the
        root agent prompt will be optimized.
      sampler: The interface used to get training and validation example UIDs,
        request agent evaluations, and get useful data for optimizing the agent.

    Returns:
      The final result of the optimization process, containing the optimized
      agent instance, its scores on the validation examples, and other metrics.
    """
    if initial_agent.sub_agents:
      _logger.warning(
          "The GEPARootAgentPromptOptimizer will not optimize prompts for"
          " sub-agents."
      )

    _logger.info("Setting up the GEPA optimizer...")

    try:
      import gepa  # lazy import as gepa is not in core ADK package

      adapter_class = _create_agent_gepa_adapter_class()
    except ImportError as e:
      raise ImportError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

    loop = asyncio.get_running_loop()

    adapter = adapter_class(
        initial_agent=initial_agent,
        sampler=sampler,
        main_loop=loop,
    )

    llm = self._llm_class(model=self._config.optimizer_model)

    def reflection_lm(prompt: GEPAPrompt) -> str:
      future = asyncio.run_coroutine_threadsafe(
          generate_reflection_response(
              llm=llm,
              model=self._config.optimizer_model,
              config=self._config.model_configuration,
              prompt=prompt,
          ),
          loop,
      )
      return future.result()

    train_ids = sampler.get_train_example_ids()
    val_ids = sampler.get_validation_example_ids()

    if set(train_ids).intersection(val_ids):
      _logger.warning(
          "The training and validation example UIDs overlap. This WILL cause"
          " aliasing issues unless each common UID refers to the same example"
          " in both sets."
      )

    initial_instruction = require_static_instruction(initial_agent)

    def run_gepa() -> GEPAResult[dict[str, Any], int]:
      return gepa.optimize(
          seed_candidate={_AGENT_PROMPT_NAME: initial_instruction},
          trainset=train_ids,
          valset=val_ids,
          adapter=adapter,
          max_metric_calls=self._config.max_metric_calls,
          reflection_lm=reflection_lm,
          reflection_minibatch_size=self._config.reflection_minibatch_size,
          run_dir=self._config.run_dir,
      )

    _logger.info("Running the GEPA optimizer...")

    ctx = contextvars.copy_context()
    gepa_results = await loop.run_in_executor(None, lambda: ctx.run(run_gepa))

    _logger.info("GEPA optimization finished. Preparing final results...")

    optimized_prompts = [
        candidate[_AGENT_PROMPT_NAME] for candidate in gepa_results.candidates
    ]
    scores = gepa_results.val_aggregate_scores

    optimized_agents = [
        AgentWithScores(
            optimized_agent=initial_agent.clone(
                update={"instruction": optimized_prompt},
            ),
            overall_score=score,
        )
        for optimized_prompt, score in zip(
            optimized_prompts, scores, strict=True
        )
    ]

    return GEPARootAgentPromptOptimizerResult(
        optimized_agents=optimized_agents,
        gepa_result=gepa_results.to_dict(),
    )
