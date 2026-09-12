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

"""Tests for the tool parameter match evaluator."""

from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
from google.genai import types as genai_types

_USER_CONTENT = genai_types.Content(parts=[genai_types.Part(text="test")])


def _invocation(*calls: genai_types.FunctionCall) -> Invocation:
  return Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=list(calls)),
  )


def _evaluate(
    actual: list[Invocation], expected: list[Invocation] | None
) -> EvaluationResult:
  evaluator = MetricEvaluatorRegistry().get_evaluator(
      EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_PARAMETER_MATCH.value,
          threshold=0.5,
      )
  )
  return evaluator.evaluate_invocations(actual, expected)


def test_scores_expected_arguments_independently():
  """A partially correct tool call receives partial credit."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome", "rooms": 2})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome", "rooms": 1})
  )

  result = _evaluate([actual], [expected])

  assert result.overall_score == 0.5
  assert result.overall_eval_status == EvalStatus.PASSED


def test_call_without_expected_arguments_receives_full_credit():
  """A matched call with no expected arguments scores one."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  expected = _invocation(genai_types.FunctionCall(name="search", args={}))

  result = _evaluate([actual], [expected])

  assert result.overall_score == 1.0


def test_aligns_repeated_calls_by_name_in_order():
  """Extra calls do not displace later expected calls with matching names."""
  actual = _invocation(
      genai_types.FunctionCall(name="log", args={}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
  )

  result = _evaluate([actual], [expected])

  assert result.overall_score == 1.0


def test_unmatched_expected_call_receives_no_credit():
  """An expected tool call absent from the actual trajectory scores zero."""
  actual = _invocation()
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )

  result = _evaluate([actual], [expected])

  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED


def test_unmatched_call_does_not_consume_later_match():
  """An absent expected call does not hide a later matching actual call."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="log", args={}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = _evaluate([actual], [expected])

  assert result.overall_score == 0.5


def test_empty_expected_trajectory_is_not_evaluated():
  """An invocation without expected tool calls is not evaluated."""
  invocation = _invocation()

  result = _evaluate([invocation], [invocation])

  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
  )


def test_missing_expected_invocations_is_not_evaluated():
  """Missing reference invocations produce a not-evaluated result."""
  result = _evaluate([_invocation()], None)

  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
