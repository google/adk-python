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

"""Tests for the Response Evaluator."""
import math
import os
import random

from google.adk.dependencies.vertexai import vertexai
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.vertex_ai_eval_facade import _VertexAiEvalFacade
from google.genai import types as genai_types
import pandas as pd
import pytest

vertexai_types = vertexai.types


class TestVertexAiEvalFacade:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_evaluate_invocations_metric_passed(self, mocker):
    """Test evaluate_invocations function for a metric."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
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

  def test_evaluate_invocations_metric_failed(self, mocker):
    """Test evaluate_invocations function for a metric."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
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
      self, mocker, summary_metric_with_no_score
  ):
    """Test evaluate_invocations function for a metric."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
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

  def test_evaluate_invocations_metric_multiple_invocations(self, mocker):
    """Test evaluate_invocations function for a metric with multiple invocations."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
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

  def test_perform_eval_with_api_key(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_API_KEY": "test_api_key"}, clear=True
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client_instance = mock_client_cls.return_value
    dummy_dataset = pd.DataFrame(
        [{"prompt": "p", "reference": "r", "response": "r"}]
    )
    dummy_metrics = [vertexai_types.PrebuiltMetric.COHERENCE]

    _VertexAiEvalFacade._perform_eval(dummy_dataset, dummy_metrics)

    mock_client_cls.assert_called_once_with(api_key="test_api_key")
    mock_client_instance.evals.evaluate.assert_called_once()

  def test_perform_eval_with_project_and_location(self, mocker):
    mocker.patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test_project",
            "GOOGLE_CLOUD_LOCATION": "test_location",
        },
        clear=True,
    )
    mock_client_cls = mocker.patch(
        "google.adk.dependencies.vertexai.vertexai.Client"
    )
    mock_client_instance = mock_client_cls.return_value
    dummy_dataset = pd.DataFrame(
        [{"prompt": "p", "reference": "r", "response": "r"}]
    )
    dummy_metrics = [vertexai_types.PrebuiltMetric.COHERENCE]

    _VertexAiEvalFacade._perform_eval(dummy_dataset, dummy_metrics)

    mock_client_cls.assert_called_once_with(
        project="test_project", location="test_location"
    )
    mock_client_instance.evals.evaluate.assert_called_once()

  def test_perform_eval_with_project_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_PROJECT": "test_project"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    dummy_dataset = pd.DataFrame(
        [{"prompt": "p", "reference": "r", "response": "r"}]
    )
    dummy_metrics = [vertexai_types.PrebuiltMetric.COHERENCE]

    with pytest.raises(ValueError, match="Missing location."):
      _VertexAiEvalFacade._perform_eval(dummy_dataset, dummy_metrics)

  def test_perform_eval_with_location_only_raises_error(self, mocker):
    mocker.patch.dict(
        os.environ, {"GOOGLE_CLOUD_LOCATION": "test_location"}, clear=True
    )
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    dummy_dataset = pd.DataFrame(
        [{"prompt": "p", "reference": "r", "response": "r"}]
    )
    dummy_metrics = [vertexai_types.PrebuiltMetric.COHERENCE]

    with pytest.raises(ValueError, match="Missing project id."):
      _VertexAiEvalFacade._perform_eval(dummy_dataset, dummy_metrics)

  def test_perform_eval_with_no_env_vars_raises_error(self, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("google.adk.dependencies.vertexai.vertexai.Client")
    dummy_dataset = pd.DataFrame(
        [{"prompt": "p", "reference": "r", "response": "r"}]
    )
    dummy_metrics = [vertexai_types.PrebuiltMetric.COHERENCE]

    with pytest.raises(
        ValueError,
        match=(
            "Either API Key or Google cloud Project id and location should be"
            " specified."
        ),
    ):
      _VertexAiEvalFacade._perform_eval(dummy_dataset, dummy_metrics)
