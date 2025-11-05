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

import math
import os
from typing import Optional

from google.genai import types as genai_types
import pandas as pd
from typing_extensions import override

from ..dependencies.vertexai import vertexai
from .eval_case import Invocation
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

vertexai_types = vertexai.types
VertexAiClient = vertexai.Client

_ERROR_MESSAGE_SUFFIX = """
You should specify both project id and location. This metric uses Vertex Gen AI
Eval SDK, and it requires google cloud credentials.

If using an .env file add the values there, or explicitly set in the code using
the template below:

os.environ['GOOGLE_CLOUD_LOCATION'] = <LOCATION>
os.environ['GOOGLE_CLOUD_PROJECT'] = <PROJECT ID>
"""


class _VertexAiEvalFacade(Evaluator):
  """Simple facade for Vertex Gen AI Eval SDK.

  Vertex Gen AI Eval SDK exposes quite a few metrics that are valuable for
  agentic evals. This class helps us to access those metrics.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.
  """

  def __init__(
      self,
      threshold: float,
      metric_name: vertexai_types.PrebuiltMetric,
      expected_invocations_required=False,
  ):
    self._threshold = threshold
    self._metric_name = metric_name
    self._expected_invocations_required = expected_invocations_required

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ) -> EvaluationResult:
    if self._expected_invocations_required and expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")

    # If expected_invocation are not required by the metric and if they are not
    # supplied, we provide an a list of None.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      prompt = self._get_text(actual.user_content)
      reference = self._get_text(expected.final_response) if expected else None
      response = self._get_text(actual.final_response)
      eval_case = {
          "prompt": prompt,
          "reference": reference,
          "response": response,
      }

      eval_case_result = _VertexAiEvalFacade._perform_eval(
          dataset=pd.DataFrame([eval_case]), metrics=[self._metric_name]
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
    if (
        eval_result
        and eval_result.summary_metrics
        and isinstance(eval_result.summary_metrics[0].mean_score, float)
        and not math.isnan(eval_result.summary_metrics[0].mean_score)
    ):
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
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", None)

    if not project_id:
      raise ValueError("Missing project id." + _ERROR_MESSAGE_SUFFIX)
    if not location:
      raise ValueError("Missing location." + _ERROR_MESSAGE_SUFFIX)

    client = VertexAiClient(project=project_id, location=location)

    return client.evals.evaluate(
        dataset=vertexai_types.EvaluationDataset(eval_dataset_df=dataset),
        metrics=metrics,
    )
