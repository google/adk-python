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

import logging
from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from .eval_case import get_all_tool_calls
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .eval_metrics import ToolTrajectoryCriterion
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)


class TrajectoryEvaluator(Evaluator):
  """Evaluates tool use trajectories for accuracy.

  This evaluator compares the sequence of tools called by the agent against a
  list of expected calls and computes an average score based on one of the match
  types: `EXACT`, `IN_ORDER`, or `ANY_ORDER`.

  For each invocation being evaluated, this evaluator compares the list of
  tool calls produced by the agent with the list of expected tool calls using
  one of three match types. If the tool calls match based on the selected match
  type, a score of 1.0 is awarded for that invocation, otherwise the score is
  0.0. The final value is the average of these scores across all
  invocations in the eval case.

  The comparison can be done using one of following match types:
    - `EXACT`: Requires a perfect match between the actual and expected tool
      calls, with no extra or missing tool calls.
    - `IN_ORDER`: Requires all tool calls from the expected list to be present
      in the actual list, in the same order, but allows for other tool calls
      to appear in between.
    - `ANY_ORDER`: Requires all tool calls from the expected list to be
      present in the actual list, in any order, and allows for other tool
      calls to appear in between.
  """

  criterion_type: ClassVar[type[ToolTrajectoryCriterion]] = (
      ToolTrajectoryCriterion
  )

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

    if eval_metric and eval_metric.criterion:
      try:
        criterion = TrajectoryEvaluator.criterion_type.model_validate(
            eval_metric.criterion.model_dump()
        )
        self._threshold = criterion.threshold
        self._match_type = criterion.match_type
      except ValidationError as e:
        expected_criterion_type_error = ValueError(
            f"`{eval_metric.metric_name}` metric expects a criterion of type"
            f" `{TrajectoryEvaluator.criterion_type}`."
        )
        raise expected_criterion_type_error from e
    elif eval_metric:
      self._threshold = eval_metric.threshold
      self._match_type = ToolTrajectoryCriterion.MatchType.EXACT
    else:
      self._threshold = threshold
      self._match_type = ToolTrajectoryCriterion.MatchType.EXACT

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
      tool_use_accuracy = self._calculate_tool_use_accuracy(actual, expected)
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

  def _calculate_tool_use_accuracy(
      self,
      actual_invocation: Invocation,
      expected_invocation: Invocation,
  ) -> float:
    """Calculates tool use accuracy for a single invocation."""
    actual_tool_uses = get_all_tool_calls(actual_invocation.intermediate_data)
    expected_tool_uses = get_all_tool_calls(
        expected_invocation.intermediate_data
    )

    tool_use_match_status = False
    if self._match_type == ToolTrajectoryCriterion.MatchType.EXACT:
      tool_use_match_status = self._are_tool_calls_exact_match(
          actual_tool_uses, expected_tool_uses
      )
    elif self._match_type == ToolTrajectoryCriterion.MatchType.IN_ORDER:
      tool_use_match_status = self._are_tool_calls_in_order_match(
          actual_tool_uses, expected_tool_uses
      )
    elif self._match_type == ToolTrajectoryCriterion.MatchType.ANY_ORDER:
      tool_use_match_status = self._are_tool_calls_any_order_match(
          actual_tool_uses, expected_tool_uses
      )
    else:
      raise ValueError(f"Unsupported match type {self._match_type}")

    return 1.0 if tool_use_match_status else 0.0

  def _are_tool_calls_in_order_match(
      self,
      actual_tool_calls: list[genai_types.FunctionCall],
      expected_tool_calls: list[genai_types.FunctionCall],
  ) -> bool:
    """Checks if expected tool calls appear in actual tool calls in order.

    This method implements IN_ORDER match type. It allows for additional
    tool calls in actual_tool_calls, as long as all expected tool calls are
    present in the same order.

    Args:
      actual_tool_calls: A list of tool calls that actually happened.
      expected_tool_calls: A list of tool calls that were expected to happen.

    Returns:
      True if actual tool calls match expected tool calls in order,
      False otherwise.
    """
    if not expected_tool_calls:
      return True
    if not actual_tool_calls and expected_tool_calls:
      return False

    expected_it = iter(expected_tool_calls)
    try:
      current_expected = next(expected_it)
      for actual in actual_tool_calls:
        if (
            actual.name == current_expected.name
            and actual.args == current_expected.args
        ):
          current_expected = next(expected_it)
    except StopIteration:
      return True

    return False

  def _are_tool_calls_any_order_match(
      self,
      actual_tool_calls: list[genai_types.FunctionCall],
      expected_tool_calls: list[genai_types.FunctionCall],
  ) -> bool:
    """Checks if expected tool calls appear in actual tool calls in any order.

    This method implements ANY_ORDER match type. It allows for additional
    tool calls in actual_tool_calls, as long as all expected tool calls are
    present.

    Args:
      actual_tool_calls: A list of tool calls that actually happened.
      expected_tool_calls: A list of tool calls that were expected to happen.

    Returns:
      True if actual tool calls contain all expected tool calls,
      False otherwise.
    """
    if not expected_tool_calls:
      return True
    if not actual_tool_calls and expected_tool_calls:
      return False

    actual_tool_calls_copy = list(actual_tool_calls)
    for expected in expected_tool_calls:
      found = False
      for i, actual in enumerate(actual_tool_calls_copy):
        if actual.name == expected.name and actual.args == expected.args:
          actual_tool_calls_copy.pop(i)
          found = True
          break
      if not found:
        return False
    return True

  def _are_tool_calls_exact_match(
      self,
      actual_tool_calls: list[genai_types.FunctionCall],
      expected_tool_calls: list[genai_types.FunctionCall],
  ) -> bool:
    """Checks if actual tool calls exactly match expected tool calls.

    This method implements EXACT match type. It requires that
    actual_tool_calls and expected_tool_calls have the same tool calls in
    the same order, with no extra or missing tool calls.

    Args:
      actual_tool_calls: A list of tool calls that actually happened.
      expected_tool_calls: A list of tool calls that were expected to happen.

    Returns:
      True if actual tool calls exactly match expected tool calls,
      False otherwise.
    """
    if len(actual_tool_calls) != len(expected_tool_calls):
      return False

    for actual, expected in zip(actual_tool_calls, expected_tool_calls):
      if actual.name != expected.name or actual.args != expected.args:
        return False

    return True

  def _get_eval_status(self, score: float):
    return EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
