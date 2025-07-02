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

"""Tests for the Response Evaluator."""
import random
from unittest.mock import patch

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.response_evaluator import ResponseEvaluator
from google.genai import types as genai_types
import pytest
from vertexai import types as vertexai_types


@patch(
    "google.adk.evaluation.response_evaluator.ResponseEvaluator._perform_eval"
)
class TestResponseEvaluator:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_evaluate_invocations_rouge_metric(self, mock_perform_eval):
    """Test evaluate_invocations function for Rouge metric."""
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = ResponseEvaluator(
        threshold=0.8, metric_name="response_match_score"
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == pytest.approx(8 / 11)
    # ROUGE-1 F1 is approx. 0.73 < 0.8 threshold, so eval status is FAILED.
    assert evaluation_result.overall_eval_status == EvalStatus.FAILED
    mock_perform_eval.assert_not_called()  # Ensure _perform_eval was not called

  def test_evaluate_invocations_coherence_metric_passed(
      self, mock_perform_eval
  ):
    """Test evaluate_invocations function for Coherence metric."""
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = ResponseEvaluator(
        threshold=0.8, metric_name="response_evaluation_score"
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.9)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == 0.9
    assert evaluation_result.overall_eval_status == EvalStatus.PASSED
    mock_perform_eval.assert_called_once()

  def test_evaluate_invocations_coherence_metric_failed(
      self, mock_perform_eval
  ):
    """Test evaluate_invocations function for Coherence metric."""
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = ResponseEvaluator(
        threshold=0.8, metric_name="response_evaluation_score"
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.7)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == 0.7
    assert evaluation_result.overall_eval_status == EvalStatus.FAILED
    mock_perform_eval.assert_called_once()

  def test_evaluate_invocations_coherence_metric_no_score(
      self, mock_perform_eval
  ):
    """Test evaluate_invocations function for Coherence metric."""
    actual_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[
                    genai_types.Part(text="This is a test candidate response.")
                ]
            ),
        )
    ]
    expected_invocations = [
        Invocation(
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="This is a test query.")]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="This is a test reference.")]
            ),
        )
    ]
    evaluator = ResponseEvaluator(
        threshold=0.8, metric_name="response_evaluation_score"
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score is None
    assert evaluation_result.overall_eval_status == EvalStatus.NOT_EVALUATED
    mock_perform_eval.assert_called_once()

  def test_evaluate_invocations_coherence_metric_multiple_invocations(
      self, mock_perform_eval
  ):
    """Test evaluate_invocations function for Coherence metric with multiple invocations."""
    num_invocations = 6
    actual_invocations = []
    expected_invocations = []
    mock_eval_results = []
    random.seed(61553)
    scores = [random.random() for _ in range(num_invocations)]

    for i in range(num_invocations):
      actual_invocations.append(
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text=f"Query {i+1}")]
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text=f"Response {i+1}")]
              ),
          )
      )
      expected_invocations.append(
          Invocation(
              user_content=genai_types.Content(
                  parts=[genai_types.Part(text=f"Query {i+1}")]
              ),
              final_response=genai_types.Content(
                  parts=[genai_types.Part(text=f"Reference {i+1}")]
              ),
          )
      )
      mock_eval_results.append(
          vertexai_types.EvaluationResult(
              summary_metrics=[
                  vertexai_types.AggregatedMetricResult(mean_score=scores[i])
              ],
              eval_case_results=[],
          )
      )

    evaluator = ResponseEvaluator(
        threshold=0.8, metric_name="response_evaluation_score"
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.side_effect = mock_eval_results

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score == pytest.approx(
        sum(scores) / num_invocations
    )
    assert evaluation_result.overall_eval_status == EvalStatus.FAILED
    assert mock_perform_eval.call_count == num_invocations
