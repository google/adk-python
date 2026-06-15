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
from typing import Callable

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field

from ..agents.llm_agent import Agent
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..tools.skill_toolset import SkillToolset
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .agent_optimizer import AgentOptimizer
from .data_types import AgentWithScores
from .data_types import OptimizerResult
from .data_types import UnstructuredSamplingResult
from .sampler import Sampler

logger = logging.getLogger("google_adk." + __name__)

_AGENT_PROMPT_KEY = "agent_prompt"
_SKILL_KEY_PREFIX = "skill_instructions:"
_SKILL_KEY_TEMPLATE = _SKILL_KEY_PREFIX + "{skill_name}"

_AGENT_PROMPT_UPDATOR_INST_TEMPLATE = """\
I provided an AI agent with the following core instructions:
```
<curr_param>
```

I then evaluated the agent.
The following are examples of different task inputs provided to the agent along with the agent's response and some external feedback for each input:
```
<side_info>
```

Your task is to write a new version of the agent core instructions.
During evaluation, the agent may have loaded skills containing additional instructions.
Do NOT include or attempt to fix instructions loaded through skills (instructions for deciding which skills to load are acceptable in the core instructions).
Focus only on the agent's general behavior, reasoning processes, and tool/skill selection.

Read the evaluation data carefully to identify the format of the user input, agent response, and feedback.
Identify any factual information about the task which belongs in the core instructions.
If such information is omitted or incorrect, update the core instructions accordingly.
Unless there are clear contradictions, avoid removing existing information from the core instructions as it may be relevant to other tasks.

Provide the new instructions within ``` blocks."""

_SKILL_INST_UPDATOR_INST_TEMPLATE = """\
I provided an AI agent with access to a skill named `{skill_name}` which provides the following skill instructions:
```
<curr_param>
```

I then evaluated the agent.
The following are examples of different task inputs provided to the agent along with the agent's response and some external feedback for each input:
```
<side_info>
```

Your task is to write a new version of the skill instructions.
Do NOT include or attempt to fix the agent's core instructions.
If NONE of the evaluation tasks exercised this skill, do not update the skill instructions.
If at least some of the evaluation tasks exercised this skill, then update the skill instructions based on the evaluation data for those tasks.
During evaluation, the agent may have loaded other skills besides this one.
Do NOT include or attempt to fix instructions related to other skills.

Read the evaluation data carefully to identify the format of the user input, agent response, and feedback.
Identify any factual information about the task which belongs in the skill instructions.
If such information is omitted or incorrect, update the skill instructions accordingly.
Unless there are clear contradictions, avoid removing existing information from the skill instructions as it may be relevant to other tasks.
Also note that the eval data may contain multiple copies and different versions of the skill instructions; disregard them and focus on updating the skill instructions provided at the start.

Provide the new instructions within ``` blocks."""


class GEPARootAgentOptimizerConfig(BaseModel):
  """Contains configuration options required by the GEPARootAgentOptimizer."""

  optimizer_model: str = Field(
      default="gemini-3.5-flash",
      description=(
          "The model used to analyze the eval results and optimize the agent."
      ),
  )

  model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_level=genai_types.ThinkingLevel.HIGH,
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
          " Providing this enables resuming the optimization process from a"
          " checkpoint if it is interrupted. Otherwise, the process will start"
          " from scratch."
      ),
  )


class GEPARootAgentOptimizerResult(OptimizerResult[AgentWithScores]):
  """The final result of the GEPARootAgentOptimizer."""

  gepa_result: dict[str, Any] | None = Field(
      default=None,
      description="The raw result dictionary from the GEPA optimizer.",
  )


def _update_skill_toolset(
    toolset: SkillToolset, candidate: dict[str, str]
) -> SkillToolset:
  """Clones the SkillToolset with skills updated from the candidate."""
  new_skills = []
  for skill in toolset.skills:
    skill_key = _SKILL_KEY_TEMPLATE.format(skill_name=skill.name)
    if skill_key in candidate:
      new_skill = skill.model_copy(
          update={"instructions": candidate[skill_key]}
      )
      new_skills.append(new_skill)
    else:
      new_skills.append(skill)
  return toolset.clone_with_updated_skills(new_skills)


def _create_agent_from_candidate(
    initial_agent: Agent, candidate: dict[str, str]
) -> Agent:
  """Reconstructs the agent using the provided candidate."""
  prompt = candidate.get(_AGENT_PROMPT_KEY, initial_agent.instruction)
  new_agent = initial_agent.clone(update={"instruction": prompt})

  new_tools = []
  for tool in initial_agent.tools:
    if isinstance(tool, SkillToolset):
      new_tools.append(_update_skill_toolset(tool, candidate))
    else:
      new_tools.append(tool)

  new_agent.tools = new_tools
  return new_agent


def _create_agent_gepa_adapter_class():
  """Creates the _AgentGEPAAdapter class dynamically to avoid top-level gepa imports."""
  from gepa.core.adapter import EvaluationBatch
  from gepa.core.adapter import GEPAAdapter
  from gepa.strategies.instruction_proposal import InstructionProposalSignature

  class _AgentGEPAAdapter(GEPAAdapter[str, dict[str, Any], dict[str, Any]]):
    """A GEPA adapter for ADK agents."""

    def __init__(
        self,
        initial_agent: Agent,
        sampler: Sampler[UnstructuredSamplingResult],
        main_loop: asyncio.AbstractEventLoop,
        reflection_lm: Callable[[str], str],
    ):
      self._initial_agent = initial_agent
      self._sampler = sampler
      self._main_loop = main_loop
      self._reflection_lm = reflection_lm

      self._train_example_ids = set(sampler.get_train_example_ids())
      self._validation_example_ids = set(sampler.get_validation_example_ids())

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
      logger.info("Evaluating agent on batch:\n%r", batch)
      new_agent = _create_agent_from_candidate(self._initial_agent, candidate)

      if set(batch) <= self._train_example_ids:
        example_set = "train"
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
      """Selects the relevant parts of the eval data for reflection."""
      trace_instances: list[tuple[float, dict[str, Any]]] = list(
          zip(
              eval_batch.scores,
              eval_batch.trajectories,
              strict=True,
          )
      )

      result = {comp: [] for comp in components_to_update}

      for score, eval_data in trace_instances:
        entry = {"score": score, "eval_data": eval_data}

        eval_data_str = str(eval_data)  # to check for skill name presence

        # filter examples relevant to each skill
        for component in components_to_update:
          if component.startswith(_SKILL_KEY_PREFIX):
            skill_name = component.removeprefix(_SKILL_KEY_PREFIX)
            if skill_name in eval_data_str:
              result[component].append(entry)
          else:  # agent core instructions - all examples are relevant
            result[component].append(entry)

      return result

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
      new_texts = {}
      for component in components_to_update:
        if component == _AGENT_PROMPT_KEY:
          prompt_template = _AGENT_PROMPT_UPDATOR_INST_TEMPLATE
        elif component.startswith(_SKILL_KEY_PREFIX):
          skill_name = component.removeprefix(_SKILL_KEY_PREFIX)
          prompt_template = _SKILL_INST_UPDATOR_INST_TEMPLATE.format(
              skill_name=skill_name
          )
        else:
          raise ValueError(f"Unknown component type for update: {component}")

        input_dict = {
            "current_instruction_doc": candidate[component],
            "dataset_with_feedback": reflective_dataset[component],
            "prompt_template": prompt_template,
        }
        prompt = InstructionProposalSignature.prompt_renderer(input_dict)
        lm_out = self._reflection_lm(prompt)
        output_dict = InstructionProposalSignature.output_extractor(lm_out)
        new_texts[component] = output_dict["new_instruction"]

      return new_texts

  return _AgentGEPAAdapter


@experimental
class GEPARootAgentOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
  """An optimizer that improves the root agent using the GEPA framework."""

  def __init__(
      self,
      config: GEPARootAgentOptimizerConfig,
  ):
    self._config = config
    llm_registry = LLMRegistry()
    self._llm_class = llm_registry.resolve(self._config.optimizer_model)

  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
  ) -> GEPARootAgentOptimizerResult:
    """Runs the GEPARootAgentOptimizer.

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
      logger.warning(
          "The GEPARootAgentOptimizer will not optimize prompts for sub-agents."
      )

    logger.info("Setting up the GEPA optimizer...")

    try:
      import gepa  # lazy import as gepa is not in core ADK package

      _AgentGEPAAdapter = _create_agent_gepa_adapter_class()
    except ImportError as e:
      raise ImportError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

    loop = asyncio.get_running_loop()

    llm = self._llm_class(model=self._config.optimizer_model)

    def reflection_lm(prompt: str) -> str:
      llm_request = LlmRequest(
          model=self._config.optimizer_model,
          config=self._config.model_configuration,
          contents=[
              genai_types.Content(
                  parts=[genai_types.Part(text=prompt)],
                  role="user",
              )
          ],
      )

      async def _generate() -> str:
        async with Aclosing(llm.generate_content_async(llm_request)) as agen:
          # only one yield expected so no need to loop
          llm_response: LlmResponse = await agen.__anext__()
          generated_content = llm_response.content
          if not generated_content or not generated_content.parts:
            return ""
          return "".join(
              part.text
              for part in generated_content.parts
              if part.text and not part.thought
          )

      future = asyncio.run_coroutine_threadsafe(_generate(), loop)
      return future.result()

    adapter = _AgentGEPAAdapter(
        initial_agent=initial_agent,
        sampler=sampler,
        main_loop=loop,
        reflection_lm=reflection_lm,
    )

    train_ids = sampler.get_train_example_ids()
    val_ids = sampler.get_validation_example_ids()

    if set(train_ids).intersection(val_ids):
      logger.warning(
          "The training and validation example UIDs overlap. This WILL cause"
          " aliasing issues unless each common UID refers to the same example"
          " in both sets."
      )

    def run_gepa():
      seed_candidate = {}
      for tool in initial_agent.tools:
        if isinstance(tool, SkillToolset):
          for skill in tool.skills:
            seed_candidate[
                _SKILL_KEY_TEMPLATE.format(skill_name=skill.name)
            ] = skill.instructions
      # added last so skills will be optimized first when components are
      # selected by for loops (due to dict ordering)
      seed_candidate[_AGENT_PROMPT_KEY] = initial_agent.instruction

      return gepa.optimize(
          seed_candidate=seed_candidate,
          trainset=train_ids,
          valset=val_ids,
          adapter=adapter,
          max_metric_calls=self._config.max_metric_calls,
          reflection_lm=reflection_lm,
          reflection_minibatch_size=self._config.reflection_minibatch_size,
          run_dir=self._config.run_dir,
      )

    logger.info("Running the GEPA optimizer...")

    ctx = contextvars.copy_context()
    gepa_results = await loop.run_in_executor(None, lambda: ctx.run(run_gepa))

    logger.info("GEPA optimization finished. Preparing final results...")

    scores = gepa_results.val_aggregate_scores

    optimized_agents = [
        AgentWithScores(
            optimized_agent=_create_agent_from_candidate(
                initial_agent, candidate
            ),
            overall_score=score,
        )
        for candidate, score in zip(gepa_results.candidates, scores)
    ]

    return GEPARootAgentOptimizerResult(
        optimized_agents=optimized_agents,
        gepa_result=gepa_results.to_dict(),
    )
