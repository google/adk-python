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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.genai import types as genai_types
import pytest


def _create_test_rouge_evaluator(threshold: float) -> RougeEvaluator:
  return RougeEvaluator(
      EvalMetric(metric_name="response_match_score", threshold=threshold)
  )


def _create_test_invocations(
    candidate: str, reference: str
) -> tuple[Invocation, Invocation]:
  """Returns tuple of (actual_invocation, expected_invocation)."""
  return Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=candidate)]
      ),
  ), Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=reference)]
      ),
  )


def test_calculate_rouge_1_scores_empty_candidate_and_reference():
  candidate = ""
  reference = ""
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores_empty_candidate():
  candidate = ""
  reference = "This is a test reference."
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores_empty_reference():
  candidate = "This is a test candidate response."
  reference = ""
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores():
  candidate = "This is a test candidate response."
  reference = "This is a test reference."
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == pytest.approx(2 / 3)
  assert rouge_1_score.recall == pytest.approx(4 / 5)
  assert rouge_1_score.fmeasure == pytest.approx(8 / 11)


@pytest.mark.parametrize(
    "candidates, references, expected_score, expected_status",
    [
        (
            ["The quick brown fox jumps.", "hello world"],
            ["The quick brown fox jumps over the lazy dog.", "hello"],
            0.69048,  # (5/7 + 2/3) / 2
            EvalStatus.FAILED,
        ),
        (
            ["This is a test.", "Another test case."],
            ["This is a test.", "This is a different test."],
            0.625,  # (1 + 1/4) / 2
            EvalStatus.FAILED,
        ),
        (
            ["No matching words here.", "Second candidate."],
            ["Completely different text.", "Another reference."],
            0.0,  # (0 + 1/2) / 2
            EvalStatus.FAILED,
        ),
        (
            ["Same words", "Same words"],
            ["Same words", "Same words"],
            1.0,
            EvalStatus.PASSED,
        ),
    ],
)
def test_rouge_evaluator_multiple_invocations(
    candidates: list[str],
    references: list[str],
    expected_score: float,
    expected_status: EvalStatus,
):
  rouge_evaluator = _create_test_rouge_evaluator(threshold=0.8)
  actual_invocations = []
  expected_invocations = []
  for candidate, reference in zip(candidates, references):
    actual_invocation, expected_invocation = _create_test_invocations(
        candidate, reference
    )
    actual_invocations.append(actual_invocation)
    expected_invocations.append(expected_invocation)

  evaluation_result = rouge_evaluator.evaluate_invocations(
      actual_invocations, expected_invocations
  )
  assert evaluation_result.overall_score == pytest.approx(
      expected_score, rel=1e-3
  )
  assert evaluation_result.overall_eval_status == expected_status


def test_get_metric_info():
  """Test get_metric_info function for response match metric."""
  metric_info = RougeEvaluator.get_metric_info()
  assert metric_info.metric_name == PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
  assert metric_info.metric_value_info.interval.min_value == 0.0
  assert metric_info.metric_value_info.interval.max_value == 1.0
