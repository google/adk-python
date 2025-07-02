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

import os
from typing import Optional

from google.genai import types as genai_types
import pandas as pd
from typing_extensions import override
from vertexai import Client as VertexAiClient
from vertexai import types as vertexai_types

from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult
from .final_response_match_v1 import RougeEvaluator


class ResponseEvaluator(Evaluator):
  """Runs response evaluation for agents."""

  def __init__(
      self,
      threshold: Optional[float] = None,
      metric_name: Optional[str] = None,
      eval_metric: Optional[EvalMetric] = None,
  ):
    if (threshold is not None and eval_metric) or (
        metric_name is not None and eval_metric
    ):
      raise ValueError(
          "Either eval_metric should be specified or both threshold and"
          " metric_name should be specified."
      )

    if eval_metric:
      threshold = eval_metric.threshold
      metric_name = eval_metric.metric_name

    if "response_evaluation_score" == metric_name:
      self._metric_name = vertexai_types.PrebuiltMetric.COHERENCE
    elif "response_match_score" == metric_name:
      self._metric_name = "response_match_score"
    else:
      raise ValueError(f"`{metric_name}` is not supported.")

    self._threshold = threshold

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    # If the metric is response_match_score, just use the RougeEvaluator.
    if self._metric_name == "response_match_score":
      rouge_evaluator = RougeEvaluator(
          EvalMetric(metric_name=self._metric_name, threshold=self._threshold)
      )
      return rouge_evaluator.evaluate_invocations(
          actual_invocations, expected_invocations
      )

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      prompt = self._get_text(expected.user_content)
      reference = self._get_text(expected.final_response)
      response = self._get_text(actual.final_response)

      eval_case = {
          "prompt": prompt,
          "reference": reference,
          "response": response,
      }

      eval_case_result = ResponseEvaluator._perform_eval(
          pd.DataFrame([eval_case]), [self._metric_name]
      )
      score = self._get_score(eval_case_result)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
          )
      )

      if score:
        total_score += score
        num_invocations += 1

    if per_invocation_results:
      overall_score = (
          total_score / num_invocations if num_invocations > 0 else None
      )
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status(overall_score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()

  def _get_text(self, content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  def _get_score(self, eval_result) -> Optional[float]:
    if eval_result and eval_result.summary_metrics:
      return eval_result.summary_metrics[0].mean_score

    return None

  def _get_eval_status(self, score: Optional[float]):
    if score:
      return (
          EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
      )

    return EvalStatus.NOT_EVALUATED

  @staticmethod
  def _perform_eval(dataset, metrics):
    """This method hides away the call to external service.

    Primarily helps with unit testing.
    """
    project_id = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    location = os.environ.get("GOOGLE_CLOUD_REGION")
    client = VertexAiClient(project=project_id, location=location)

    return client.evals.evaluate(
        dataset=vertexai_types.EvaluationDataset(eval_dataset_df=dataset),
        metrics=metrics,
    )
