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

import logging
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ..agents.llm_agent import Agent
from ..evaluation.base_eval_service import EvaluateConfig
from ..evaluation.base_eval_service import EvaluateRequest
from ..evaluation.base_eval_service import InferenceConfig
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.base_eval_service import InferenceResult
from ..evaluation.eval_case import get_all_tool_calls_with_responses
from ..evaluation.eval_case import IntermediateData
from ..evaluation.eval_case import Invocation
from ..evaluation.eval_case import InvocationEvents
from ..evaluation.eval_config import EvalConfig
from ..evaluation.eval_config import get_eval_metrics_from_config
from ..evaluation.eval_metrics import EvalStatus
from ..evaluation.eval_result import EvalCaseResult
from ..evaluation.eval_sets_manager import EvalSetsManager
from ..evaluation.local_eval_service import LocalEvalService
from ..evaluation.simulation.user_simulator_provider import UserSimulatorProvider
from ..utils.context_utils import Aclosing
from .data_types import UnstructuredSamplingResult
from .sampler import Sampler

logger = logging.getLogger("google_adk." + __name__)


def _log_eval_summary(eval_results: list[EvalCaseResult]):
  """Logs a summary of eval results."""
  num_pass, num_fail, num_other = 0, 0, 0
  for eval_result in eval_results:
    eval_result: EvalCaseResult
    if eval_result.final_eval_status == EvalStatus.PASSED:
      num_pass += 1
    elif eval_result.final_eval_status == EvalStatus.FAILED:
      num_fail += 1
    else:
      num_other += 1
  log_str = f"Evaluation summary: {num_pass} PASSED, {num_fail} FAILED"
  if num_other:
    log_str += f", {num_other} OTHER"
  logger.info(log_str)


def extract_tool_call_data(
    intermediate_data: IntermediateData | InvocationEvents,
) -> list[dict[str, Any]]:
  """Extracts tool calls and their responses from intermediate data."""
  call_response_pairs = get_all_tool_calls_with_responses(intermediate_data)
  result = []
  for tool_call, tool_response in call_response_pairs:
    result.append({
        "name": tool_call.name,
        "args": tool_call.args,
        "response": tool_response.response if tool_response else None,
    })
  return result


def extract_single_invocation_info(
    invocation: Invocation,
) -> dict[str, Any]:
  """Extracts useful information from a single invocation."""
  user_prompt = ""
  for part in invocation.user_content.parts:
    if part.text and not part.thought:
      user_prompt += part.text
  agent_response = ""
  if invocation.final_response:
    for part in invocation.final_response.parts:
      if part.text and not part.thought:
        agent_response += part.text
  result = {"user_prompt": user_prompt, "agent_response": agent_response}
  if invocation.intermediate_data:
    tool_call_data = extract_tool_call_data(invocation.intermediate_data)
    result["tool_calls"] = tool_call_data
  return result


class LocalEvalSamplerConfig(BaseModel):
  """Contains configuration options required by the LocalEvalServiceInterface."""

  eval_config: EvalConfig = Field(
      required=True,
      description="The configuration for the evaluation.",
  )

  app_name: str = Field(
      required=True,
      description="The app name to use for evaluation.",
  )

  train_eval_set: str = Field(
      required=True,
      description="The name of the eval set to use for optimization.",
  )

  train_eval_case_ids: Optional[list[str]] = Field(
      default=None,
      description=(
          "The ids of the eval cases to use for optimization. If not provided,"
          " all eval cases in the train_eval_set will be used."
      ),
  )

  validation_eval_set: Optional[str] = Field(
      default=None,
      description=(
          "The name of the eval set to use for validating the optimized agent."
          " If not provided, the train_eval_set will also be used for"
          " validation."
      ),
  )

  validation_eval_case_ids: Optional[list[str]] = Field(
      default=None,
      description=(
          "The ids of the eval cases to use for validating the optimized agent."
          " If not provided, all eval cases in the validation_eval_set will be"
          " used. If validation_eval_set is also not provided, all train eval"
          " cases will be used."
      ),
  )


class LocalEvalSampler(Sampler[UnstructuredSamplingResult]):
  """Evaluates candidate agents with the ADK's LocalEvalService."""

  def __init__(
      self,
      config: LocalEvalSamplerConfig,
      eval_sets_manager: EvalSetsManager,
  ):
    self._config = config
    self._eval_sets_manager = eval_sets_manager

    self._train_eval_set = self._config.train_eval_set
    self._train_eval_case_ids = (
        self._config.train_eval_case_ids
        or self._get_eval_case_ids(self._train_eval_set)
    )

    self._validation_eval_set = (
        self._config.validation_eval_set or self._train_eval_set
    )
    if self._config.validation_eval_case_ids:
      self._validation_eval_case_ids = self._config.validation_eval_case_ids
    elif self._config.validation_eval_set:
      self._validation_eval_case_ids = self._get_eval_case_ids(
          self._validation_eval_set
      )
    else:
      self._validation_eval_case_ids = self._train_eval_case_ids

  def _get_selected_example_set_id(
      self, example_set: Literal[Sampler.TRAIN_SET, Sampler.VALIDATION_SET]
  ) -> str:
    """Returns the ID of the selected example set."""
    return {
        Sampler.TRAIN_SET: self._train_eval_set,
        Sampler.VALIDATION_SET: self._validation_eval_set,
    }[example_set]

  def _get_all_example_ids(
      self, example_set: Literal[Sampler.TRAIN_SET, Sampler.VALIDATION_SET]
  ) -> list[str]:
    """Returns the IDs of all examples in the selected example set."""
    return {
        Sampler.TRAIN_SET: self._train_eval_case_ids,
        Sampler.VALIDATION_SET: self._validation_eval_case_ids,
    }[example_set]

  def _get_eval_case_ids(self, eval_set_id: str) -> list[str]:
    """Returns the ids of eval cases in the given eval set."""
    eval_set = self._eval_sets_manager.get_eval_set(
        app_name=self._config.app_name,
        eval_set_id=eval_set_id,
    )
    if eval_set:
      return [eval_case.eval_id for eval_case in eval_set.eval_cases]
    else:
      raise ValueError(
          f"Eval set `{eval_set_id}` does not exist for app"
          f" `{self._config.app_name}`."
      )

  async def _evaluate_agent(
      self,
      agent: Agent,
      eval_set_id: str,
      eval_case_ids: list[str],
  ) -> list[EvalCaseResult]:
    """Evaluates the agent on the requested eval cases and returns the results.

    Args:
      agent: The agent to evaluate.
      eval_set_id: The id of the eval set to use for evaluation.
      eval_case_ids: The ids of the eval cases to use for evaluation.

    Returns:
      A list of EvalCaseResult, one per eval case.
    """
    # create the inference request
    inference_request = InferenceRequest(
        app_name=self._config.app_name,
        eval_set_id=eval_set_id,
        eval_case_ids=eval_case_ids,
        inference_config=InferenceConfig(),
    )

    # create the LocalEvalService
    user_simulator_provider = UserSimulatorProvider(
        self._config.eval_config.user_simulator_config
    )
    eval_service = LocalEvalService(
        root_agent=agent,
        eval_sets_manager=self._eval_sets_manager,
        user_simulator_provider=user_simulator_provider,
    )

    # inference/sampling
    async with Aclosing(
        eval_service.perform_inference(inference_request=inference_request)
    ) as agen:
      inference_results: list[InferenceResult] = [
          inference_result async for inference_result in agen
      ]

    # evaluation
    eval_metrics = get_eval_metrics_from_config(self._config.eval_config)
    evaluate_request = EvaluateRequest(
        inference_results=inference_results,
        evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
    )
    async with Aclosing(
        eval_service.evaluate(evaluate_request=evaluate_request)
    ) as agen:
      eval_results: list[EvalCaseResult] = [
          eval_result async for eval_result in agen
      ]

    return eval_results

  def _extract_eval_data(
      self,
      eval_set_id: str,
      eval_results: list[EvalCaseResult],
  ) -> dict[str, dict[str, Any]]:
    """Extracts evaluation data from the eval results."""
    eval_data = {}
    for eval_result in eval_results:
      eval_result_dict = {}
      eval_case = self._eval_sets_manager.get_eval_case(
          app_name=self._config.app_name,
          eval_set_id=eval_set_id,
          eval_case_id=eval_result.eval_id,
      )
      if eval_case and eval_case.conversation_scenario:
        eval_result_dict["conversation_scenario"] = (
            eval_case.conversation_scenario
        )

      per_invocation_results = []
      for (
          per_invocation_result
      ) in eval_result.eval_metric_result_per_invocation:
        eval_metric_results = []
        for eval_metric_result in per_invocation_result.eval_metric_results:
          eval_metric_results.append({
              "metric_name": eval_metric_result.metric_name,
              "score": round(eval_metric_result.score, 2),  # accurate enough
              "eval_status": eval_metric_result.eval_status.name,
          })
        per_invocation_result_dict = {
            "actual_invocation": extract_single_invocation_info(
                per_invocation_result.actual_invocation
            ),
            "eval_metric_results": eval_metric_results,
        }
        if per_invocation_result.expected_invocation:
          per_invocation_result_dict["expected_invocation"] = (
              extract_single_invocation_info(
                  per_invocation_result.expected_invocation
              )
          )
        per_invocation_results.append(per_invocation_result_dict)
      eval_result_dict["invocations"] = per_invocation_results
      eval_data[eval_result.eval_id] = eval_result_dict

    return eval_data

  def get_train_example_ids(self) -> list[str]:
    """Returns the UIDs of examples to use for training the agent."""
    return self._train_eval_case_ids

  def get_validation_example_ids(self) -> list[str]:
    """Returns the UIDs of examples to use for validating the optimized agent."""
    return self._validation_eval_case_ids

  async def sample_and_score(
      self,
      candidate: Agent,
      example_set: Literal[
          Sampler.TRAIN_SET, Sampler.VALIDATION_SET
      ] = Sampler.VALIDATION_SET,
      batch: Optional[list[str]] = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    """Evaluates the candidate agent on the batch of examples using the ADK LocalEvalService.

    Args:
      candidate: The candidate agent to be evaluated.
      example_set: The set of examples to evaluate the candidate agent on.
        Possible values are "train" and "validation".
      batch: UIDs of examples to evaluate the candidate agent on. If not
        provided, all examples from the chosen set will be used.
      capture_full_eval_data: If false, it is enough to only calculate the
        scores for each example. If true, this method should also capture all
        other data required for optimizing the agent (e.g., outputs,
        trajectories, and tool calls).

    Returns:
      The evaluation results, containing the scores for each example and (if
      requested) other data required for optimization.
    """
    eval_set_id = self._get_selected_example_set_id(example_set)
    if batch is None:
      batch = self._get_all_example_ids(example_set)

    eval_results = await self._evaluate_agent(candidate, eval_set_id, batch)
    _log_eval_summary(eval_results)

    scores = {
        eval_result.eval_id: (
            1.0 if eval_result.final_eval_status == EvalStatus.PASSED else 0.0
        )
        for eval_result in eval_results
    }

    eval_data = (
        self._extract_eval_data(eval_set_id, eval_results)
        if capture_full_eval_data
        else None
    )

    return UnstructuredSamplingResult(scores=scores, data=eval_data)
