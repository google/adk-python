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
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.evaluation.response_evaluator import ResponseEvaluator
import pandas as pd
import pytest
from vertexai.preview.evaluation import MetricPromptTemplateExamples

# Mock object for the result normally returned by _perform_eval
MOCK_EVAL_RESULT = MagicMock()
MOCK_EVAL_RESULT.summary_metrics = {"mock_metric": 0.75, "another_mock": 3.5}
# Add a metrics_table for testing _print_results interaction
MOCK_EVAL_RESULT.metrics_table = pd.DataFrame({
    "prompt": ["mock_query1"],
    "response": ["mock_resp1"],
    "mock_metric": [0.75],
})

SAMPLE_TURN_1_ALL_KEYS = {
    "query": "query1",
    "response": "response1",
    "actual_tool_use": [{"tool_name": "tool_a", "tool_input": {}}],
    "expected_tool_use": [{"tool_name": "tool_a", "tool_input": {}}],
    "reference": "reference1",
}
SAMPLE_TURN_2_MISSING_REF = {
    "query": "query2",
    "response": "response2",
    "actual_tool_use": [],
    "expected_tool_use": [],
    # "reference": "reference2" # Missing
}
SAMPLE_TURN_3_MISSING_EXP_TOOLS = {
    "query": "query3",
    "response": "response3",
    "actual_tool_use": [{"tool_name": "tool_b", "tool_input": {}}],
    # "expected_tool_use": [], # Missing
    "reference": "reference3",
}
SAMPLE_TURN_4_MINIMAL = {
    "query": "query4",
    "response": "response4",
    # Minimal keys, others missing
}


@patch(
    "google.adk.evaluation.response_evaluator.ResponseEvaluator._perform_eval"
)
class TestResponseEvaluator:
  """A class to help organize "patch" that are applicabple to all tests."""

  def test_evaluate_none_dataset_raises_value_error(self, mock_perform_eval):
    """Test evaluate function raises ValueError for an empty list."""
    with pytest.raises(ValueError, match="The evaluation dataset is empty."):
      ResponseEvaluator.evaluate(None, ["response_evaluation_score"])
    mock_perform_eval.assert_not_called()  # Ensure _perform_eval was not called

  def test_evaluate_empty_dataset_raises_value_error(self, mock_perform_eval):
    """Test evaluate function raises ValueError for an empty list."""
    with pytest.raises(ValueError, match="The evaluation dataset is empty."):
      ResponseEvaluator.evaluate([], ["response_evaluation_score"])
    mock_perform_eval.assert_not_called()  # Ensure _perform_eval was not called

  def test_evaluate_determines_metrics_correctly_for_perform_eval(
      self, mock_perform_eval
  ):
    """Test that the correct metrics list is passed to _perform_eval based on criteria/keys."""
    mock_perform_eval.return_value = MOCK_EVAL_RESULT

    # Test case 1: Only Coherence
    raw_data_1 = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria_1 = ["response_evaluation_score"]
    ResponseEvaluator.evaluate(raw_data_1, criteria_1)
    _, kwargs = mock_perform_eval.call_args
    assert kwargs["metrics"] == [
        MetricPromptTemplateExamples.Pointwise.COHERENCE
    ]
    mock_perform_eval.reset_mock()  # Reset mock for next call

    # Test case 2: Only Rouge
    raw_data_2 = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria_2 = ["response_match_score"]
    ResponseEvaluator.evaluate(raw_data_2, criteria_2)
    _, kwargs = mock_perform_eval.call_args
    assert kwargs["metrics"] == ["rouge_1"]
    mock_perform_eval.reset_mock()

    # Test case 3: No metrics if keys missing in first turn
    raw_data_3 = [[SAMPLE_TURN_4_MINIMAL, SAMPLE_TURN_1_ALL_KEYS]]
    criteria_3 = ["response_evaluation_score", "response_match_score"]
    ResponseEvaluator.evaluate(raw_data_3, criteria_3)
    _, kwargs = mock_perform_eval.call_args
    assert kwargs["metrics"] == []
    mock_perform_eval.reset_mock()

    # Test case 4: No metrics if criteria empty
    raw_data_4 = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria_4 = []
    ResponseEvaluator.evaluate(raw_data_4, criteria_4)
    _, kwargs = mock_perform_eval.call_args
    assert kwargs["metrics"] == []
    mock_perform_eval.reset_mock()

  def test_evaluate_calls_perform_eval_correctly_all_metrics(
      self, mock_perform_eval
  ):
    """Test evaluate function calls _perform_eval with expected args when all criteria/keys are present."""
    # Arrange
    mock_perform_eval.return_value = (
        MOCK_EVAL_RESULT  # Configure the mock return value
    )

    raw_data = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria = ["response_evaluation_score", "response_match_score"]

    # Act
    summary = ResponseEvaluator.evaluate(raw_data, criteria)

    # Assert
    # 1. Check metrics determined by _get_metrics (passed to _perform_eval)
    expected_metrics_list = [
        MetricPromptTemplateExamples.Pointwise.COHERENCE,
        "rouge_1",
    ]
    # 2. Check DataFrame prepared (passed to _perform_eval)
    expected_df_data = [{
        "prompt": "query1",
        "response": "response1",
        "actual_tool_use": [{"tool_name": "tool_a", "tool_input": {}}],
        "reference_trajectory": [{"tool_name": "tool_a", "tool_input": {}}],
        "reference": "reference1",
    }]
    expected_df = pd.DataFrame(expected_df_data)

    # Assert _perform_eval was called once
    mock_perform_eval.assert_called_once()
    # Get the arguments passed to the mocked _perform_eval
    _, kwargs = mock_perform_eval.call_args
    # Check the 'dataset' keyword argument
    pd.testing.assert_frame_equal(kwargs["dataset"], expected_df)
    # Check the 'metrics' keyword argument
    assert kwargs["metrics"] == expected_metrics_list

    # 3. Check the correct summary metrics are returned
    # (from mock_perform_eval's return value)
    assert summary == MOCK_EVAL_RESULT.summary_metrics

  def test_evaluate_prepares_dataframe_correctly_for_perform_eval(
      self, mock_perform_eval
  ):
    """Test that the DataFrame is correctly flattened and renamed before passing to _perform_eval."""
    mock_perform_eval.return_value = MOCK_EVAL_RESULT

    raw_data = [
        [SAMPLE_TURN_1_ALL_KEYS],  # Conversation 1
        [
            SAMPLE_TURN_2_MISSING_REF,
            SAMPLE_TURN_3_MISSING_EXP_TOOLS,
        ],  # Conversation 2
    ]
    criteria = [
        "response_match_score"
    ]  # Doesn't affect the DataFrame structure

    ResponseEvaluator.evaluate(raw_data, criteria)

    # Expected flattened and renamed data
    expected_df_data = [
        # Turn 1 (from SAMPLE_TURN_1_ALL_KEYS)
        {
            "prompt": "query1",
            "response": "response1",
            "actual_tool_use": [{"tool_name": "tool_a", "tool_input": {}}],
            "reference_trajectory": [{"tool_name": "tool_a", "tool_input": {}}],
            "reference": "reference1",
        },
        # Turn 2 (from SAMPLE_TURN_2_MISSING_REF)
        {
            "prompt": "query2",
            "response": "response2",
            "actual_tool_use": [],
            "reference_trajectory": [],
            # "reference": None # Missing key results in NaN in DataFrame
            # usually
        },
        # Turn 3 (from SAMPLE_TURN_3_MISSING_EXP_TOOLS)
        {
            "prompt": "query3",
            "response": "response3",
            "actual_tool_use": [{"tool_name": "tool_b", "tool_input": {}}],
            # "reference_trajectory": None, # Missing key results in NaN
            "reference": "reference3",
        },
    ]
    # Need to be careful with missing keys -> NaN when creating DataFrame
    # Pandas handles this automatically when creating from list of dicts
    expected_df = pd.DataFrame(expected_df_data)

    mock_perform_eval.assert_called_once()
    _, kwargs = mock_perform_eval.call_args
    # Compare the DataFrame passed to the mock
    pd.testing.assert_frame_equal(kwargs["dataset"], expected_df)

  @patch(
      "google.adk.evaluation.response_evaluator.ResponseEvaluator._print_results"
  )  # Mock the private print method
  def test_evaluate_print_detailed_results(
      self, mock_print_results, mock_perform_eval
  ):
    """Test _print_results function is called when print_detailed_results=True."""
    mock_perform_eval.return_value = (
        MOCK_EVAL_RESULT  # Ensure _perform_eval returns our mock result
    )

    raw_data = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria = ["response_match_score"]

    ResponseEvaluator.evaluate(raw_data, criteria, print_detailed_results=True)

    # Assert _perform_eval was called
    mock_perform_eval.assert_called_once()
    # Assert _print_results was called once with the result object
    # from _perform_eval
    mock_print_results.assert_called_once_with(MOCK_EVAL_RESULT)

  @patch(
      "google.adk.evaluation.response_evaluator.ResponseEvaluator._print_results"
  )
  def test_evaluate_no_print_detailed_results(
      self, mock_print_results, mock_perform_eval
  ):
    """Test _print_results function is NOT called when print_detailed_results=False (default)."""
    mock_perform_eval.return_value = MOCK_EVAL_RESULT

    raw_data = [[SAMPLE_TURN_1_ALL_KEYS]]
    criteria = ["response_match_score"]

    ResponseEvaluator.evaluate(raw_data, criteria, print_detailed_results=False)

    # Assert _perform_eval was called
    mock_perform_eval.assert_called_once()
    # Assert _print_results was NOT called
    mock_print_results.assert_not_called()
