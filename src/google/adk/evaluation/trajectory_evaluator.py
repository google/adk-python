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

from .eval_case import get_all_tool_calls
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult


class TrajectoryEvaluator(Evaluator):
  """Evaluates tool use trajectories for accuracy."""

  def __init__(
      self,
      threshold: Optional[float] = None,
      eval_metric: Optional[EvalMetric] = None,
  ):
    if threshold is not None and eval_metric:
      raise ValueError(
          "Either eval_metric should be specified or threshold should be"
          " specified."
      )

    if eval_metric:
      threshold = eval_metric.threshold

    self._threshold = threshold

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
        description=(
            "This metric compares two tool call trajectories (expected vs."
            " actual) for the same user interaction. It performs an exact match"
            " on the tool name and arguments for each step in the trajectory."
            " A score of 1.0 indicates a perfect match, while 0.0 indicates a"
            " mismatch. Higher values are better."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ) -> EvaluationResult:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations."""
    if expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")

    total_tool_use_accuracy = 0.0
    num_invocations = 0
    per_invocation_results = []

    for actual, expected in zip(actual_invocations, expected_invocations):
      actual_tool_uses = get_all_tool_calls(actual.intermediate_data)
      expected_tool_uses = get_all_tool_calls(expected.intermediate_data)

      tool_use_accuracy = (
          1.0
          if self._are_tool_calls_equal(actual_tool_uses, expected_tool_uses)
          else 0.0
      )
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=tool_use_accuracy,
              eval_status=self._get_eval_status(tool_use_accuracy),
          )
      )
      total_tool_use_accuracy += tool_use_accuracy
      num_invocations += 1

    if per_invocation_results:
      overall_score = total_tool_use_accuracy / num_invocations
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status(overall_score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()

  def _are_tool_calls_equal(
      self,
      actual_tool_calls: list[genai_types.FunctionCall],
      expected_tool_calls: list[genai_types.FunctionCall],
  ) -> bool:
    if len(actual_tool_calls) != len(expected_tool_calls):
      return False

    for actual, expected in zip(actual_tool_calls, expected_tool_calls):
      if actual.name != expected.name or actual.args != expected.args:
        return False

    return True

  def _get_eval_status(self, score: float):
    return EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
