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

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from google.genai import types as genai_types
from typing_extensions import override

from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..utils.context_utils import Aclosing
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult
from .llm_as_judge_utils import get_eval_status


class LlmAsJudge(Evaluator):
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
  ):
    self._eval_metric = eval_metric
    if not eval_metric.judge_model_options:
      raise ValueError("Judge model options is required for LlmAsJudge.")
    self._judge_model_options = eval_metric.judge_model_options
    if self._judge_model_options.judge_model_config is None:
      self._judge_model_options.judge_model_config = (
          genai_types.GenerateContentConfig()
      )
    self._judge_model = self._setup_auto_rater()

  @abstractmethod
  def format_auto_rater_prompt(
      self, actual: Invocation, expected: Invocation
  ) -> str:
    """Formats the auto-rater prompt to evaluate the given invocation."""

  @abstractmethod
  def convert_auto_rater_response_to_score(
      self, auto_rater_response: LlmResponse
  ) -> Optional[float]:
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

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      auto_rater_prompt = self.format_auto_rater_prompt(actual, expected)
      llm_request = LlmRequest(
          model=self._judge_model_options.judge_model,
          contents=[
              genai_types.Content(
                  parts=[genai_types.Part(text=auto_rater_prompt)],
                  role="user",
              )
          ],
          config=self._judge_model_options.judge_model_config,
      )
      num_samples = self._judge_model_options.num_samples
      invocation_result_samples = []
      for _ in range(num_samples):
        async with Aclosing(
            self._judge_model.generate_content_async(llm_request)
        ) as agen:
          async for llm_response in agen:
            # Non-streaming call, so there is only one response content.
            score = self.convert_auto_rater_response_to_score(llm_response)
            invocation_result_samples.append(
                PerInvocationResult(
                    actual_invocation=actual,
                    expected_invocation=expected,
                    score=score,
                    eval_status=get_eval_status(
                        score, self._eval_metric.threshold
                    ),
                )
            )
      if not invocation_result_samples:
        continue
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
