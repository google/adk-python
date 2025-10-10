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


from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types as genai_types
import pytest

_USER_CONTENT = genai_types.Content(
    parts=[genai_types.Part(text="User input here.")]
)


def test_get_metric_info():
  """Test get_metric_info function for tool trajectory avg metric."""
  metric_info = TrajectoryEvaluator.get_metric_info()
  assert (
      metric_info.metric_name == PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
  )
  assert metric_info.metric_value_info.interval.min_value == 0.0
  assert metric_info.metric_value_info.interval.max_value == 1.0


@pytest.fixture
def evaluator() -> TrajectoryEvaluator:
  """Returns a TrajectoryEvaluator."""
  return TrajectoryEvaluator(threshold=0.5)


def test_evaluate_invocations_equal_tool_calls(evaluator: TrajectoryEvaluator):
  """Tests evaluate_invocations with equal tool calls."""
  tool_call = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
  intermediate_data = IntermediateData(tool_uses=[tool_call])
  invocation = Invocation(
      user_content=_USER_CONTENT, intermediate_data=intermediate_data
  )
  result = evaluator.evaluate_invocations([invocation], [invocation])
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 1
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_different_tool_call_names(
    evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with different tool call names."""
  tool_call1 = genai_types.FunctionCall(
      name="test_func1", args={"arg1": "val1"}
  )
  tool_call2 = genai_types.FunctionCall(
      name="test_func2", args={"arg1": "val1"}
  )
  invocation1 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  invocation2 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call2]),
  )
  result = evaluator.evaluate_invocations([invocation1], [invocation2])
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_tool_call_args(
    evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with different tool call args."""
  tool_call1 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
  tool_call2 = genai_types.FunctionCall(name="test_func", args={"arg1": "val2"})
  invocation1 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  invocation2 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call2]),
  )
  result = evaluator.evaluate_invocations([invocation1], [invocation2])
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_different_number_of_tool_calls(
    evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with different number of tool calls."""
  tool_call1 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
  tool_call2 = genai_types.FunctionCall(name="test_func", args={"arg1": "val1"})
  invocation1 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  invocation2 = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1, tool_call2]),
  )
  result = evaluator.evaluate_invocations([invocation1], [invocation2])
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_no_tool_calls(evaluator: TrajectoryEvaluator):
  """Tests evaluate_invocations with no tool calls."""
  invocation = Invocation(
      user_content=_USER_CONTENT, intermediate_data=IntermediateData()
  )
  result = evaluator.evaluate_invocations([invocation], [invocation])
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_multiple_invocations(
    evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with multiple invocations."""
  tool_call1 = genai_types.FunctionCall(
      name="test_func1", args={"arg1": "val1"}
  )
  tool_call2 = genai_types.FunctionCall(
      name="test_func2", args={"arg1": "val1"}
  )
  inv1_actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  inv1_expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  inv2_actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1]),
  )
  inv2_expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call2]),
  )
  result = evaluator.evaluate_invocations(
      [inv1_actual, inv2_actual], [inv1_expected, inv2_expected]
  )
  assert result.overall_score == 0.5
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 2
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[1].score == 0.0
  assert result.per_invocation_results[1].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_no_invocations(evaluator: TrajectoryEvaluator):
  """Tests evaluate_invocations with no invocations."""
  result = evaluator.evaluate_invocations([], [])
  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert not result.per_invocation_results
