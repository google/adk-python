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

from abc import abstractmethod
import asyncio
from collections import defaultdict
from collections.abc import Sequence
import logging
from typing import Generic
from typing import Optional
from typing import TypeVar

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from ._retry_options_utils import add_default_retry_options_if_not_present
from .common import EvalBaseModel
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import _get_metric_threshold
from .eval_metrics import EvalMetric
from .eval_metrics import EvalStatus
from .eval_metrics import LlmAsAJudgeCriterion
from .eval_metrics import RubricsBasedCriterion
from .eval_metrics import RubricScore
from .evaluator import _validate_invocation_lengths
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult
from .llm_as_judge_utils import get_eval_status

logger = logging.getLogger("google_adk." + __name__)


class AutoRaterScore(EvalBaseModel):
  score: Optional[float] = None
  rubric_scores: Optional[list[RubricScore]] = None


# RubricsBasedCriterion is a sibling of LlmAsAJudgeCriterion, not a subclass,
# so the two are spelled as a value restriction rather than as a union bound;
# both declare judge_model_options, which is all this class reads.
_CriterionT = TypeVar(
    "_CriterionT", LlmAsAJudgeCriterion, RubricsBasedCriterion
)


@experimental
class LlmAsJudge(Evaluator, Generic[_CriterionT]):
  """Evaluator based on a LLM.

  It is meant to be extended by specific auto-raters for different evaluation
  tasks:
    - Provide the prompt template, and implement format_auto_rater_prompt to
      format the auto-rater prompt for a given invocation.
    - Implement convert_auto_rater_response_to_score to parse the auto-rater
      response and return the corresponding score.
    - Implement aggregate_invocation_results to aggregate the per-invocation
      results to get the overall score.
    - (Optional) Override aggregate_per_invocation_result_samples to aggregate
      multiple auto-rater samples of the same invocation.
  """

  def __init__(
      self,
      eval_metric: EvalMetric,
      criterion_type: type[_CriterionT],
      expected_invocations_required: bool = False,
  ):
    self._eval_metric = eval_metric
    self._expected_invocations_required = expected_invocations_required

    expected_criterion_type_error = ValueError(
        f"`{eval_metric.metric_name}` metric expects a criterion of type"
        f" `{criterion_type}`."
    )

    try:
      if self._eval_metric.criterion is None:
        raise expected_criterion_type_error

      self._criterion: _CriterionT = criterion_type.model_validate(
          self._eval_metric.criterion.model_dump()
      )
    except ValidationError as e:
      raise expected_criterion_type_error from e

    self._judge_model_options = self._criterion.judge_model_options
    self._threshold = _get_metric_threshold(eval_metric)
    self._judge_model = self._setup_auto_rater()

  @abstractmethod
  def format_auto_rater_prompt(
      self, actual: Invocation, expected: Optional[Invocation]
  ) -> str:
    """Formats the auto-rater prompt to evaluate the given invocation."""

  @abstractmethod
  def convert_auto_rater_response_to_score(
      self, auto_rater_response: LlmResponse
  ) -> AutoRaterScore:
    """Parses auto_rater_response and returns the corresponding score, or None if the score cannot be determined."""

  @abstractmethod
  def aggregate_per_invocation_samples(
      self,
      per_invocation_samples: list[PerInvocationResult],
  ) -> PerInvocationResult:
    """Aggregates repeated per-invocation samples to get the final result for the invocation."""

  @abstractmethod
  def aggregate_invocation_results(
      self,
      per_invocation_results: list[PerInvocationResult],
  ) -> EvaluationResult:
    """Aggregates the per invocation results to get the overall score."""

  async def _evaluate_single_sample(
      self,
      llm_request: LlmRequest,
      actual: Invocation,
      expected: Optional[Invocation],
  ) -> PerInvocationResult:
    """Evaluates a single sample for an invocation.

    Args:
      llm_request: The LLM request to execute.
      actual: The actual invocation to evaluate.
      expected: The expected invocation (optional).

    Returns:
      A PerInvocationResult containing the evaluation score and status.
    """
    async with Aclosing(
        self._judge_model.generate_content_async(llm_request)
    ) as agen:
      async for llm_response in agen:
        # Non-streaming call, so there is only one response content.
        auto_rater_score = self.convert_auto_rater_response_to_score(
            llm_response
        )
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=auto_rater_score.score,
            eval_status=get_eval_status(
                auto_rater_score.score, self._threshold
            ),
            rubric_scores=auto_rater_score.rubric_scores,
        )
    # If we reach here, the LLM didn't return any response
    raise RuntimeError(
        "LLM evaluation failed: no response received from judge model"
    )

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if self._expected_invocations_required and expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")
    _validate_invocation_lengths(actual_invocations, expected_invocations)
    del conversation_scenario  # not supported for per-invocation evaluation.

    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide a list of None.
    # Sequence rather than list: it is covariant, so the supplied
    # list[Invocation] is accepted without copying it.
    resolved_expected: Sequence[Optional[Invocation]] = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    # Build all LLM evaluation tasks for parallel execution
    tasks = []
    invocation_indices = []  # Track which invocation each task belongs to
    parallelism_limit = self._judge_model_options.parallelism_limit
    semaphore = asyncio.Semaphore(parallelism_limit)

    async def _evaluate_single_sample_with_sem(
        llm_request: LlmRequest,
        actual: Invocation,
        expected: Optional[Invocation],
    ) -> PerInvocationResult:
      async with semaphore:
        return await self._evaluate_single_sample(llm_request, actual, expected)

    for invocation_idx, (actual, expected) in enumerate(
        zip(actual_invocations, resolved_expected, strict=True)
    ):
      auto_rater_prompt = self.format_auto_rater_prompt(actual, expected)
      llm_request = LlmRequest(
          model=self._judge_model_options.judge_model,
          contents=[
              genai_types.Content(
                  parts=[genai_types.Part(text=auto_rater_prompt)],
                  role="user",
              )
          ],
          config=self._judge_model_options.judge_model_config
          or genai_types.GenerateContentConfig(),
      )
      add_default_retry_options_if_not_present(llm_request)
      num_samples = self._judge_model_options.num_samples

      # Create tasks for all samples of this invocation
      for _ in range(num_samples):
        tasks.append(
            _evaluate_single_sample_with_sem(llm_request, actual, expected)
        )
        invocation_indices.append(invocation_idx)

    # Execute all tasks in parallel
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Group results by invocation
    results_by_invocation = defaultdict(list)
    for invocation_idx, result in zip(
        invocation_indices, all_results, strict=True
    ):
      if isinstance(result, Exception):
        logger.warning(
            "Evaluation sample failed for invocation %d: %s",
            invocation_idx,
            result,
            exc_info=result,
        )
      results_by_invocation[invocation_idx].append(result)

    # Aggregate samples for each invocation
    per_invocation_results = []
    for invocation_idx in sorted(results_by_invocation.keys()):
      invocation_result_samples = results_by_invocation[invocation_idx]
      actual = actual_invocations[invocation_idx]
      expected = resolved_expected[invocation_idx]
      if any(isinstance(r, Exception) for r in invocation_result_samples):
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=None,
                eval_status=EvalStatus.NOT_EVALUATED,
            )
        )
      elif invocation_result_samples:
        per_invocation_results.append(
            self.aggregate_per_invocation_samples(invocation_result_samples)
        )

    if per_invocation_results:
      return self.aggregate_invocation_results(per_invocation_results)
    return EvaluationResult()

  def _setup_auto_rater(self) -> BaseLlm:
    model_id = self._judge_model_options.judge_model
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(model_id)
    return llm_class(model=model_id)
