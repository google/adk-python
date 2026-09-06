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

from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from typing_extensions import override

from .eval_case import ConversationScenario
from .eval_case import get_all_tool_calls
from .eval_case import Invocation
from .eval_metrics import _get_metric_threshold
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .evaluator import _validate_invocation_lengths
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult


class _ToolParameterMatchEvaluator(Evaluator):
  """Scores expected arguments for tool calls aligned by name in order."""

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

  def __init__(self, *, eval_metric: EvalMetric):
    self._threshold = _get_metric_threshold(eval_metric)

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if expected_invocations is None:
      return EvaluationResult()
    _validate_invocation_lengths(actual_invocations, expected_invocations)
    del conversation_scenario

    per_invocation_results = []
    evaluated_scores = []
    for actual, expected in zip(
        actual_invocations, expected_invocations, strict=True
    ):
      expected_calls = get_all_tool_calls(expected.intermediate_data)
      if not expected_calls:
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
            )
        )
        continue

      score = self._score_invocation(actual, expected_calls)
      evaluated_scores.append(score)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
          )
      )

    if not evaluated_scores:
      return EvaluationResult(
          per_invocation_results=per_invocation_results,
      )

    overall_score = sum(evaluated_scores) / len(evaluated_scores)
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=self._get_eval_status(overall_score),
        per_invocation_results=per_invocation_results,
    )

  def _score_invocation(
      self,
      actual_invocation: Invocation,
      expected_calls: list[genai_types.FunctionCall],
  ) -> float:
    actual_calls = get_all_tool_calls(actual_invocation.intermediate_data)
    actual_index = 0
    call_scores = []
    for expected_call in expected_calls:
      actual_call = None
      for index in range(actual_index, len(actual_calls)):
        if actual_calls[index].name == expected_call.name:
          actual_call = actual_calls[index]
          actual_index = index + 1
          break
      call_scores.append(
          self._score_call(actual_call, expected_call)
          if actual_call is not None
          else 0.0
      )
    return sum(call_scores) / len(call_scores)

  @staticmethod
  def _score_call(
      actual_call: genai_types.FunctionCall,
      expected_call: genai_types.FunctionCall,
  ) -> float:
    expected_args = expected_call.args or {}
    if not expected_args:
      return 1.0
    actual_args = actual_call.args or {}
    matched_args = sum(
        name in actual_args and actual_args[name] == value
        for name, value in expected_args.items()
    )
    return matched_args / len(expected_args)

  def _get_eval_status(self, score: float) -> EvalStatus:
    return EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED
