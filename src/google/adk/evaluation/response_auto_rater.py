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

from typing import Optional

from google.genai import types as genai_types
from typing_extensions import override

from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult


def get_text_from_content(content: Optional[genai_types.Content]) -> str:
  if content and content.parts:
    return "\n".join([p.text for p in content.parts if p.text])


def get_eval_status(score: Optional[float], threshold: float) -> EvalStatus:
  if score is None:
    return EvalStatus.NOT_EVALUATED
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


class ResponseAutoRater(Evaluator):
  """Response evaluator based on an auto-rater (LLM).

  It is meant to be extended by specific auto-raters for different evaluation
  tasks:
    - Provide the prompt template, and implement format_auto_rater_prompt to
      format the auto-rater prompt for a given invocation.
    - Implement convert_auto_rater_response_to_score to parse the auto-rater
      response and return the corresponding score.
    - Implement aggregate_invocation_results to aggregate the per-invocation
      results to get the overall score.
  """

  def __init__(
      self,
      eval_metric: EvalMetric,
      auto_rater_prompt_template: str,
      auto_rater_num_samples: int = 5,
  ):
    self._eval_metric = eval_metric
    self._auto_rater_prompt_template = auto_rater_prompt_template
    if not eval_metric.judge_model_config:
      raise ValueError(
          "Judge model config is required for AutoRater-based evaluator."
      )
    self._generation_config = eval_metric.judge_model_config
    self._model_name = eval_metric.judge_model
    self._judge_model = self._setup_auto_rater()
    self._auto_rater_num_samples = auto_rater_num_samples

  def format_auto_rater_prompt(
      self, actual: Invocation, expected: Invocation
  ) -> str:
    """Formats the auto-rater prompt to evaluate the given invocation."""
    raise NotImplementedError()

  def convert_auto_rater_response_to_score(
      self, auto_rater_response: LlmResponse
  ) -> Optional[float]:
    """Parses auto_rater_response and returns the corresponding score, or None if the score cannot be determined."""
    raise NotImplementedError()

  def aggregate_invocation_results(
      self,
      per_invocation_results: list[PerInvocationResult],
  ) -> EvaluationResult:
    """Aggregates the per invocation results to get the overall score."""
    raise NotImplementedError()

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
          model=self._model_name,
          contents=[
              genai_types.Content(
                  parts=[genai_types.Part(text=auto_rater_prompt)],
                  role="user",
              )
          ],
          config=self._generation_config,
      )
      for _ in range(self._auto_rater_num_samples):
        async for llm_response in self._judge_model.generate_content_async(
            llm_request
        ):
          # Non-streaming call, so there is only one response content.
          score = self.convert_auto_rater_response_to_score(llm_response)
          per_invocation_results.append(
              PerInvocationResult(
                  actual_invocation=actual,
                  expected_invocation=expected,
                  score=score,
                  eval_status=get_eval_status(
                      score, self._eval_metric.threshold
                  ),
              )
          )

    if per_invocation_results:
      return self.aggregate_invocation_results(per_invocation_results)
    return EvaluationResult()

  def _setup_auto_rater(self) -> BaseLlm:
    if not self._model_name:
      raise ValueError("Model name is required for AutoRater-based evaluator.")
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(self._model_name)
    return llm_class(model=self._model_name)
