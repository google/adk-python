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

"""Testings for the Trajectory Evaluator."""

import math

from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
import pytest

# Define reusable tool call structures
TOOL_ROLL_DICE_16 = {"tool_name": "roll_die", "tool_input": {"sides": 16}}
TOOL_ROLL_DICE_6 = {"tool_name": "roll_die", "tool_input": {"sides": 6}}
TOOL_GET_WEATHER = {
    "tool_name": "get_weather",
    "tool_input": {"location": "Paris"},
}
TOOL_GET_WEATHER_SF = {
    "tool_name": "get_weather",
    "tool_input": {"location": "SF"},
}

# Sample data for turns
TURN_MATCH = {
    "query": "Q1",
    "response": "R1",
    "actual_tool_use": [TOOL_ROLL_DICE_16],
    "expected_tool_use": [TOOL_ROLL_DICE_16],
}
TURN_MISMATCH_INPUT = {
    "query": "Q2",
    "response": "R2",
    "actual_tool_use": [TOOL_ROLL_DICE_6],
    "expected_tool_use": [TOOL_ROLL_DICE_16],
}
TURN_MISMATCH_NAME = {
    "query": "Q3",
    "response": "R3",
    "actual_tool_use": [TOOL_GET_WEATHER],
    "expected_tool_use": [TOOL_ROLL_DICE_16],
}
TURN_MATCH_MULTIPLE = {
    "query": "Q4",
    "response": "R4",
    "actual_tool_use": [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6],
    "expected_tool_use": [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6],
}
TURN_MISMATCH_ORDER = {
    "query": "Q5",
    "response": "R5",
    "actual_tool_use": [TOOL_ROLL_DICE_6, TOOL_GET_WEATHER],
    "expected_tool_use": [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6],
}
TURN_MISMATCH_LENGTH_ACTUAL_LONGER = {
    "query": "Q6",
    "response": "R6",
    "actual_tool_use": [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6],
    "expected_tool_use": [TOOL_GET_WEATHER],
}
TURN_MISMATCH_LENGTH_EXPECTED_LONGER = {
    "query": "Q7",
    "response": "R7",
    "actual_tool_use": [TOOL_GET_WEATHER],
    "expected_tool_use": [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6],
}
TURN_MATCH_WITH_MOCK_OUTPUT = {
    "query": "Q8",
    "response": "R8",
    "actual_tool_use": [TOOL_GET_WEATHER_SF],
    "expected_tool_use": [
        {**TOOL_GET_WEATHER_SF, "mock_tool_output": "Sunny"}
    ],  # Add mock output to expected
}
TURN_MATCH_EMPTY_TOOLS = {
    "query": "Q9",
    "response": "R9",
    "actual_tool_use": [],
    "expected_tool_use": [],
}
TURN_MISMATCH_EMPTY_VS_NONEMPTY = {
    "query": "Q10",
    "response": "R10",
    "actual_tool_use": [],
    "expected_tool_use": [TOOL_GET_WEATHER],
}


def test_evaluate_none_dataset_raises_value_error():
  """Tests evaluate function raises ValueError for an empty list."""
  with pytest.raises(ValueError, match="The evaluation dataset is empty."):
    TrajectoryEvaluator.evaluate(None)


def test_evaluate_empty_dataset_raises_value_error():
  """Tests evaluate function raises ValueError for an empty list."""
  with pytest.raises(ValueError, match="The evaluation dataset is empty."):
    TrajectoryEvaluator.evaluate([])


def test_evaluate_single_turn_match():
  """Tests evaluate function with one conversation, one turn, perfect match."""
  eval_dataset = [[TURN_MATCH]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 1.0


def test_evaluate_single_turn_mismatch():
  """Tests evaluate function with one conversation, one turn, mismatch."""
  eval_dataset = [[TURN_MISMATCH_INPUT]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 0.0


def test_evaluate_multiple_turns_all_match():
  """Tests evaluate function with one conversation, multiple turns, all match."""
  eval_dataset = [[TURN_MATCH, TURN_MATCH_MULTIPLE, TURN_MATCH_EMPTY_TOOLS]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 1.0


def test_evaluate_multiple_turns_mixed():
  """Tests evaluate function with one conversation, mixed match/mismatch turns."""
  eval_dataset = [
      [TURN_MATCH, TURN_MISMATCH_NAME, TURN_MATCH_MULTIPLE, TURN_MISMATCH_ORDER]
  ]
  # Expected: (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 0.5


def test_evaluate_multiple_conversations_mixed():
  """Tests evaluate function with multiple conversations, mixed turns."""
  eval_dataset = [
      [TURN_MATCH, TURN_MISMATCH_INPUT],  # Conv 1: 1.0, 0.0 -> Avg 0.5
      [TURN_MATCH_MULTIPLE],  # Conv 2: 1.0 -> Avg 1.0
      [
          TURN_MISMATCH_ORDER,
          TURN_MISMATCH_LENGTH_ACTUAL_LONGER,
          TURN_MATCH,
      ],  # Conv 3: 0.0, 0.0, 1.0 -> Avg 1/3
  ]
  # Expected: (1.0 + 0.0 + 1.0 + 0.0 + 0.0 + 1.0) / 6 = 3.0 / 6 = 0.5
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 0.5


def test_evaluate_ignores_mock_tool_output_in_expected():
  """Tests evaluate function correctly compares even if expected has mock_tool_output."""
  eval_dataset = [[TURN_MATCH_WITH_MOCK_OUTPUT]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 1.0


def test_evaluate_match_empty_tool_lists():
  """Tests evaluate function correctly matches empty tool lists."""
  eval_dataset = [[TURN_MATCH_EMPTY_TOOLS]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 1.0


def test_evaluate_mismatch_empty_vs_nonempty():
  """Tests evaluate function correctly mismatches empty vs non-empty tool lists."""
  eval_dataset = [[TURN_MISMATCH_EMPTY_VS_NONEMPTY]]
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 0.0
  eval_dataset_rev = [[{
      **TURN_MISMATCH_EMPTY_VS_NONEMPTY,  # Swap actual/expected
      "actual_tool_use": [TOOL_GET_WEATHER],
      "expected_tool_use": [],
  }]]
  assert TrajectoryEvaluator.evaluate(eval_dataset_rev) == 0.0


def test_evaluate_dataset_with_empty_conversation():
  """Tests evaluate function handles dataset containing an empty conversation list."""
  eval_dataset = [[TURN_MATCH], []]  # One valid conversation, one empty
  # Should only evaluate the first conversation -> 1.0 / 1 turn = 1.0
  assert TrajectoryEvaluator.evaluate(eval_dataset) == 1.0


def test_evaluate_dataset_only_empty_conversation():
  """Tests evaluate function handles dataset with only an empty conversation."""
  eval_dataset = [[]]
  # No rows evaluated, mean of empty series is NaN
  # Depending on desired behavior, this could be 0.0 or NaN. The code returns
  # NaN.
  assert math.isnan(TrajectoryEvaluator.evaluate(eval_dataset))


def test_evaluate_print_detailed_results(capsys):
  """Tests evaluate function runs with print_detailed_results=True and prints something."""
  eval_dataset = [[TURN_MATCH, TURN_MISMATCH_INPUT]]
  TrajectoryEvaluator.evaluate(eval_dataset, print_detailed_results=True)
  captured = capsys.readouterr()
  assert "query" in captured.out  # Check if the results table header is printed
  assert "R1" in captured.out  # Check if some data is printed
  assert "Failures:" in captured.out  # Check if failures header is printed
  assert "Q2" in captured.out  # Check if the failing query is printed


def test_evaluate_no_failures_print(capsys):
  """Tests evaluate function does not print Failures section when all turns match."""
  eval_dataset = [[TURN_MATCH]]
  TrajectoryEvaluator.evaluate(eval_dataset, print_detailed_results=True)
  captured = capsys.readouterr()
  assert "query" in captured.out  # Results table should still print
  assert "Failures:" not in captured.out  # Failures section should NOT print


def test_are_tools_equal_identical():
  """Tests are_tools_equal function with identical lists."""
  list_a = [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6]
  list_b = [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6]
  assert TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_empty():
  """Tests are_tools_equal function with empty lists."""
  assert TrajectoryEvaluator.are_tools_equal([], [])


def test_are_tools_equal_different_order():
  """Tests are_tools_equal function with same tools, different order."""
  list_a = [TOOL_ROLL_DICE_6, TOOL_GET_WEATHER]
  list_b = [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6]
  assert not TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_different_length():
  """Tests are_tools_equal function with lists of different lengths."""
  list_a = [TOOL_GET_WEATHER, TOOL_ROLL_DICE_6]
  list_b = [TOOL_GET_WEATHER]
  assert not TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_different_input_values():
  """Tests are_tools_equal function with different input values."""
  list_a = [TOOL_ROLL_DICE_16]
  list_b = [TOOL_ROLL_DICE_6]
  assert not TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_different_tool_names():
  """Tests are_tools_equal function with different tool names."""
  list_a = [TOOL_ROLL_DICE_16]
  list_b = [TOOL_GET_WEATHER]
  assert not TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_ignores_extra_keys():
  """Tests are_tools_equal function ignores keys other than tool_name/tool_input."""
  list_a = [{
      "tool_name": "get_weather",
      "tool_input": {"location": "Paris"},
      "extra_key": "abc",
  }]
  list_b = [{
      "tool_name": "get_weather",
      "tool_input": {"location": "Paris"},
      "other_key": 123,
  }]
  assert TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_are_tools_equal_one_empty_one_not():
  """Tests are_tools_equal function with one empty list and one non-empty list."""
  list_a = []
  list_b = [TOOL_GET_WEATHER]
  assert not TrajectoryEvaluator.are_tools_equal(list_a, list_b)


def test_get_metric_info():
  """Test get_metric_info function for tool trajectory avg metric."""
  metric_info = TrajectoryEvaluator.get_metric_info()
  assert (
      metric_info.metric_name == PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
  )
  assert metric_info.metric_value_info.interval.min_value == 0.0
  assert metric_info.metric_value_info.interval.max_value == 1.0
