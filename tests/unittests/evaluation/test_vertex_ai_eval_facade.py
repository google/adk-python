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
import math
import random
from unittest.mock import patch

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.vertex_ai_eval_facade import _VertexAiEvalFacade
from google.genai import types as genai_types
import pytest
from vertexai import types as vertexai_types


@patch(
    "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
)
class TestVertexAiEvalFacade:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_evaluate_invocations_metric_passed(self, mock_perform_eval):
    """Test evaluate_invocations function for a metric."""
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
    evaluator = _VertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
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
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  def test_evaluate_invocations_metric_failed(self, mock_perform_eval):
    """Test evaluate_invocations function for a metric."""
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
    evaluator = _VertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
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
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  @pytest.mark.parametrize(
      "summary_metric_with_no_score",
      [
          ([]),
          ([vertexai_types.AggregatedMetricResult(mean_score=float("nan"))]),
          ([vertexai_types.AggregatedMetricResult(mean_score=None)]),
          ([vertexai_types.AggregatedMetricResult(mean_score=math.nan)]),
      ],
  )
  def test_evaluate_invocations_metric_no_score(
      self, mock_perform_eval, summary_metric_with_no_score
  ):
    """Test evaluate_invocations function for a metric."""
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
    evaluator = _VertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=summary_metric_with_no_score,
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations, expected_invocations
    )

    assert evaluation_result.overall_score is None
    assert evaluation_result.overall_eval_status == EvalStatus.NOT_EVALUATED
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  def test_evaluate_invocations_metric_multiple_invocations(
      self, mock_perform_eval
  ):
    """Test evaluate_invocations function for a metric with multiple invocations."""
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

    evaluator = _VertexAiEvalFacade(
        threshold=0.8, metric_name=vertexai_types.PrebuiltMetric.COHERENCE
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
