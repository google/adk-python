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
from unittest.mock import patch

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.response_evaluator import ResponseEvaluator
from google.genai import types as genai_types
import pytest
from vertexai import types as vertexai_types


@patch(
    "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
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
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.PrebuiltMetric.COHERENCE.name
    ]

  def test_get_metric_info_response_evaluation_score(self, mock_perform_eval):
    """Test get_metric_info function for response evaluation metric."""
    metric_info = ResponseEvaluator.get_metric_info(
        PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
    )
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
    )
    assert metric_info.metric_value_info.interval.min_value == 1.0
    assert metric_info.metric_value_info.interval.max_value == 5.0

  def test_get_metric_info_response_match_score(self, mock_perform_eval):
    """Test get_metric_info function for response match metric."""
    metric_info = ResponseEvaluator.get_metric_info(
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
    )
    assert metric_info.metric_name == PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_get_metric_info_invalid(self, mock_perform_eval):
    """Test get_metric_info function for invalid metric."""
    with pytest.raises(ValueError):
      ResponseEvaluator.get_metric_info("invalid_metric")
