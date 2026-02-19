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

import re
from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...models.registry import LLMRegistry
from ...utils.context_utils import Aclosing
from ...utils.feature_decorator import experimental
from .._retry_options_utils import add_default_retry_options_if_not_present
from ..eval_case import ConversationScenario
from ..eval_case import Invocation
from ..eval_metrics import BaseCriterion
from ..eval_metrics import EvalMetric
from ..eval_metrics import EvalStatus
from ..eval_metrics import LlmBackedUserSimulatorCriterion
from ..evaluator import EvaluationResult
from ..evaluator import Evaluator
from ..evaluator import PerInvocationResult
from ..llm_as_judge import AutoRaterScore
from ..llm_as_judge_utils import get_eval_status
from ..llm_as_judge_utils import get_text_from_content
from ..llm_as_judge_utils import Label
from .per_turn_user_simulator_quality_prompts import get_per_turn_user_simulator_quality_prompt


def _parse_llm_response(response: str) -> Label:
  """Parses the LLM response and extracts the final label.

  Args:
    response: LLM response.

  Returns:
    The extracted label, either VALID, INVALID, or NOT_FOUND.
  """
  # Regex matching the label field in the response.
  is_valid_match = re.search(
      r'"is_valid":\s*\[*[\n\s]*"*([^"\]]*)"*[\n\s]*\]*\s*[,\n\}]',
      response,
  )

  # If there was no match for "is_valid", return NOT_FOUND
  if is_valid_match is None:
    return Label.NOT_FOUND

  # Remove any trailing whitespace, commas, or end-brackets from the label.
  label = is_valid_match.group(1).strip("}").replace(",", "").strip().lower()
  if label in [
      Label.INVALID.value,
      Label.ALMOST.value,
      Label.FALSE.value,
      *Label.PARTIALLY_VALID.value,
  ]:
    return Label.INVALID
  elif label in [Label.VALID.value, Label.TRUE.value]:
    return Label.VALID
  else:
    return Label.NOT_FOUND


def _format_conversation_history(invocations: list[Invocation]) -> str:
  conversation_history = []
  for invocation in invocations:
    if invocation.user_content is not None and invocation.user_content.parts:
      conversation_history.append(
          f"user: {get_text_from_content(invocation.user_content)}"
      )

    final_response = invocation.final_response
    if final_response is not None:
      conversation_history.append(
          f"{final_response.role}: {get_text_from_content(final_response)}"
      )
  return "\n\n".join(conversation_history)


def _get_stop_signal_invocation(stop_signal: str) -> Invocation:
  return Invocation(
      invocation_id="stop_signal_proxy_invocation",
      user_content=genai_types.Content(
          parts=[genai_types.Part(text=stop_signal)]
      ),
  )


@experimental
class PerTurnUserSimulatorQualityV1(Evaluator):
  """Per turn user simulator evaluator.

  This evaluator verifies that the conversation from a user simulator sticks
  to the given conversation scenario:
  - In the first turn, it verifies that the user simulator output the
    specified starting prompt.
  - For all the other turns, it verifies that the user simulator stuck to the
    conversation plan.
  - It also verifies that the user simulator finished the conversation
    appropriately.
  This evaluator uses an LLM to verify all turns except the first one. It
  aggregates repeated invocation samples by taking majority vote. The overall
  score is the fraction of turns of the conversation before the verifier
  detects an issue with the user simulator.
  """

  criterion_type: ClassVar[type[LlmBackedUserSimulatorCriterion]] = (
      LlmBackedUserSimulatorCriterion
  )

  def __init__(
      self,
      eval_metric: EvalMetric,
  ):
    self._eval_metric = eval_metric
    self._criterion = self._deserialize_criterion(eval_metric)

    self._llm_options = self._criterion.judge_model_options
    self._stop_signal = self._criterion.stop_signal
    self._llm = self._setup_llm()

  def _deserialize_criterion(self, eval_metric: EvalMetric) -> BaseCriterion:
    expected_criterion_type_error = ValueError(
        f"`{eval_metric.metric_name}` metric expects a criterion of type"
        f" `{self.criterion_type}`."
    )
    try:
      if self._eval_metric.criterion is None:
        raise expected_criterion_type_error

      return self.criterion_type.model_validate(
          self._eval_metric.criterion.model_dump()
      )
    except ValidationError as e:
      raise expected_criterion_type_error from e

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    del expected_invocations  # not used by this metric.
    if conversation_scenario is None:
      raise ValueError("conversation_scenario is needed by this metric.")

    # Evaluate the first invocation contains the given starting prompt.
    results = [
        self._evaluate_first_turn(actual_invocations[0], conversation_scenario)
    ]

    # Evaluate the rest of the invocations.
    for i, invocation in enumerate(actual_invocations):
      # skip the first invocation.
      if i == 0:
        continue

      result = await self._evaluate_intermediate_turn(
          invocation_at_step=invocation,
          invocation_history=actual_invocations[:i],
          conversation_scenario=conversation_scenario,
      )
      results.append(result)

    if not results:
      return EvaluationResult()

    # Evaluate whether the conversation ended correctly.
    stop_signal_evaluation = await self._evaluate_stop_signal_turn(
        invocation_history=actual_invocations,
        conversation_scenario=conversation_scenario,
    )

    # If the conversation did not end correctly, indicate so by marking the
    # last user turn as failed.
    if stop_signal_evaluation.eval_status == EvalStatus.FAILED:
      results[-1] = stop_signal_evaluation

    return self._aggregate_conversation_results(results)

  def _setup_llm(self) -> BaseLlm:
    model_id = self._llm_options.judge_model
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(model_id)
    return llm_class(model=model_id)

  def _format_llm_prompt(
      self,
      invocation: Invocation,
      conversation_scenario: ConversationScenario,
      previous_invocations: Optional[list[Invocation]],
  ) -> str:
    if previous_invocations is None:
      raise ValueError(
          "Previous invocations should have a set value when "
          "formatting the LLM prompt. "
          f"Encountered: {previous_invocations}"
      )

    if conversation_scenario is None:
      raise ValueError(
          "Conversation scenario should have a set value when "
          "formatting the LLM prompt. "
          f"Encountered: {conversation_scenario}"
      )

    return get_per_turn_user_simulator_quality_prompt(
        conversation_plan=conversation_scenario.conversation_plan,
        conversation_history=_format_conversation_history(previous_invocations),
        generated_user_response=get_text_from_content(invocation.user_content),
        stop_signal=self._stop_signal,
        user_persona=conversation_scenario.user_persona,
    )

  def _convert_llm_response_to_score(
      self, auto_rater_response: LlmResponse
  ) -> AutoRaterScore:
    response_text = get_text_from_content(auto_rater_response.content)
    if response_text is None or not response_text:
      return AutoRaterScore()
    label = _parse_llm_response(response_text)

    if label == Label.VALID:
      return AutoRaterScore(score=1.0)
    elif label == Label.INVALID:
      return AutoRaterScore(score=0.0)
    else:
      return AutoRaterScore()

  def _aggregate_samples(
      self,
      per_invocation_samples: list[PerInvocationResult],
  ) -> PerInvocationResult:
    """Aggregates samples by taking majority vote."""
    if not per_invocation_samples:
      raise ValueError("No samples to aggregate into a result.")

    positive_results = [s for s in per_invocation_samples if s.score == 1.0]
    negative_results = [s for s in per_invocation_samples if s.score == 0.0]

    if not positive_results and not negative_results:
      return per_invocation_samples[0]
    elif len(positive_results) > len(negative_results):
      return positive_results[0]
    else:  # len(negative_results) >= len(positive_results)
      return negative_results[0]

  def _aggregate_conversation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Computes the fraction of results that resulted in a pass status."""
    num_valid = 0
    num_evaluated = 0
    for result in per_invocation_results:
      if result.eval_status == EvalStatus.PASSED:
        num_valid += result.score

      num_evaluated += 1

    # If no invocation was evaluated, we mark the score as None.
    if num_evaluated == 0:
      return EvaluationResult(
          per_invocation_results=per_invocation_results,
      )

    overall_score = num_valid / num_evaluated
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=get_eval_status(
            overall_score, self._criterion.threshold
        ),
        per_invocation_results=per_invocation_results,
    )

  def _evaluate_first_turn(
      self,
      first_invocation: Invocation,
      conversation_scenario: ConversationScenario,
  ) -> PerInvocationResult:
    if first_invocation.user_content is None:
      return PerInvocationResult(
          actual_invocation=first_invocation,
          eval_status=EvalStatus.NOT_EVALUATED,
      )

    score = int(
        get_text_from_content(first_invocation.user_content).strip()
        == conversation_scenario.starting_prompt.strip()
    )
    return PerInvocationResult(
        actual_invocation=first_invocation,
        score=score,
        eval_status=get_eval_status(score, self._eval_metric.threshold),
    )

  async def _evaluate_intermediate_turn(
      self,
      invocation_at_step: Invocation,
      invocation_history: list[Invocation],
      conversation_scenario: Optional[ConversationScenario],
  ) -> PerInvocationResult:

    auto_rater_prompt = self._format_llm_prompt(
        invocation=invocation_at_step,
        conversation_scenario=conversation_scenario,
        previous_invocations=invocation_history,
    )

    llm_request = LlmRequest(
        model=self._llm_options.judge_model,
        contents=[
            genai_types.Content(
                parts=[genai_types.Part(text=auto_rater_prompt)],
                role="user",
            )
        ],
        config=self._llm_options.judge_model_config,
    )
    add_default_retry_options_if_not_present(llm_request)
    num_samples = self._llm_options.num_samples
    samples = []
    for _ in range(num_samples):
      llm_score = await self._sample_llm(llm_request)
      samples.append(
          PerInvocationResult(
              eval_status=get_eval_status(
                  llm_score.score, self._eval_metric.threshold
              ),
              score=llm_score.score,
              actual_invocation=invocation_at_step,
          )
      )
    if not samples:
      return PerInvocationResult(
          eval_status=EvalStatus.NOT_EVALUATED,
          actual_invocation=invocation_at_step,
      )

    return self._aggregate_samples(samples)

  async def _evaluate_stop_signal_turn(
      self,
      invocation_history: list[Invocation],
      conversation_scenario: ConversationScenario,
  ) -> PerInvocationResult:
    return await self._evaluate_intermediate_turn(
        invocation_at_step=_get_stop_signal_invocation(self._stop_signal),
        invocation_history=invocation_history,
        conversation_scenario=conversation_scenario,
    )

  async def _sample_llm(self, llm_request: LlmRequest) -> AutoRaterScore:
    async with Aclosing(self._llm.generate_content_async(llm_request)) as agen:
      async for llm_response in agen:
        # Non-streaming call, so there is only one response content.
        return self._convert_llm_response_to_score(llm_response)
