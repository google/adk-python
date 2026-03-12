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

"""Testings for the Trajectory Evaluator."""

from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types as genai_types
from pydantic import ValidationError
import pytest

_USER_CONTENT = genai_types.Content(
    parts=[genai_types.Part(text="User input here.")]
)


def test_tool_trajectory_criterion_accepts_string_match_type():
  criterion = ToolTrajectoryCriterion(threshold=0.5, match_type="in_order")
  assert criterion.match_type == ToolTrajectoryCriterion.MatchType.IN_ORDER


@pytest.mark.parametrize(
    ("match_type", "expected"),
    [
        ("exact", ToolTrajectoryCriterion.MatchType.EXACT),
        ("EXACT", ToolTrajectoryCriterion.MatchType.EXACT),
        (" exact ", ToolTrajectoryCriterion.MatchType.EXACT),
        ("in order", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("IN ORDER", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("In OrDeR", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("in-order", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("IN-ORDER", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("in_order", ToolTrajectoryCriterion.MatchType.IN_ORDER),
        ("any order", ToolTrajectoryCriterion.MatchType.ANY_ORDER),
        ("ANY ORDER", ToolTrajectoryCriterion.MatchType.ANY_ORDER),
        ("any-order", ToolTrajectoryCriterion.MatchType.ANY_ORDER),
        ("ANY-ORDER", ToolTrajectoryCriterion.MatchType.ANY_ORDER),
        ("any_order", ToolTrajectoryCriterion.MatchType.ANY_ORDER),
    ],
)
def test_tool_trajectory_criterion_normalizes_string_match_type(
    match_type: str, expected: ToolTrajectoryCriterion.MatchType
):
  criterion = ToolTrajectoryCriterion(threshold=0.5, match_type=match_type)
  assert criterion.match_type == expected


def test_tool_trajectory_criterion_rejects_unknown_string_match_type():
  with pytest.raises(ValidationError):
    ToolTrajectoryCriterion(threshold=0.5, match_type="random string")


def test_trajectory_evaluator_accepts_string_match_type_from_eval_metric_dict():
  eval_metric = EvalMetric(
      threshold=0.5,
      metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
      criterion={
          "threshold": 0.5,
          "match_type": "ANY_ORDER",
      },
  )
  evaluator = TrajectoryEvaluator(eval_metric=eval_metric)

  tool_call1 = genai_types.FunctionCall(name="test_func1", args={})
  tool_call2 = genai_types.FunctionCall(name="test_func2", args={})

  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call1, tool_call2]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[tool_call2, tool_call1]),
  )

  result = evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 1.0


@pytest.fixture
def evaluator() -> TrajectoryEvaluator:
  """Returns a TrajectoryEvaluator."""
  return TrajectoryEvaluator(
      eval_metric=EvalMetric(
          threshold=0.5,
          metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
          criterion=ToolTrajectoryCriterion(
              threshold=0.5,
              match_type=ToolTrajectoryCriterion.MatchType.EXACT,
          ),
      )
  )


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


@pytest.fixture
def in_order_evaluator() -> TrajectoryEvaluator:
  """Returns a TrajectoryEvaluator for IN_ORDER match."""
  return TrajectoryEvaluator(
      eval_metric=EvalMetric(
          threshold=0.5,
          metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
          criterion=ToolTrajectoryCriterion(
              threshold=0.5,
              match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER,
          ),
      )
  )


def test_evaluate_invocations_in_order_match_with_extra_tool_calls(
    in_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with IN_ORDER match type and extra tool calls."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t1_1 = genai_types.FunctionCall(name="t1_1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t2_1 = genai_types.FunctionCall(name="t2_1", args={})
  t3 = genai_types.FunctionCall(name="t3", args={})
  t3_1 = genai_types.FunctionCall(name="t3_1", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[t1, t1_1, t2, t2_1, t3, t3_1]
      ),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t3]),
  )
  result = in_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_in_order_match_fails_with_missing_tool_call(
    in_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with IN_ORDER match type and missing tool call."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t1_1 = genai_types.FunctionCall(name="t1_1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t2_1 = genai_types.FunctionCall(name="t2_1", args={})
  t3_1 = genai_types.FunctionCall(name="t3_1", args={})
  t4 = genai_types.FunctionCall(name="t4", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t1_1, t2, t2_1, t3_1]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t4]),
  )
  result = in_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_in_order_match_fails_with_wrong_order(
    in_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with IN_ORDER match type and wrong order."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t3 = genai_types.FunctionCall(name="t3", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t3, t2]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t3]),
  )
  result = in_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


@pytest.fixture
def any_order_evaluator() -> TrajectoryEvaluator:
  """Returns a TrajectoryEvaluator for ANY_ORDER match."""
  return TrajectoryEvaluator(
      eval_metric=EvalMetric(
          threshold=0.5,
          metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
          criterion=ToolTrajectoryCriterion(
              threshold=0.5,
              match_type=ToolTrajectoryCriterion.MatchType.ANY_ORDER,
          ),
      )
  )


def test_evaluate_invocations_any_order_match_with_extra_tool_calls_different_order(
    any_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with ANY_ORDER match type and extra tool calls."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t1_1 = genai_types.FunctionCall(name="t1_1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t2_1 = genai_types.FunctionCall(name="t2_1", args={})
  t3 = genai_types.FunctionCall(name="t3", args={})
  t3_1 = genai_types.FunctionCall(name="t3_1", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[t2, t2_1, t1, t1_1, t3, t3_1]
      ),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t3]),
  )
  result = any_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_any_order_match_fails_with_missing_tool_call(
    any_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with ANY_ORDER match type and missing tool call."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t1_1 = genai_types.FunctionCall(name="t1_1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t2_1 = genai_types.FunctionCall(name="t2_1", args={})
  t3_1 = genai_types.FunctionCall(name="t3_1", args={})
  t4 = genai_types.FunctionCall(name="t4", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t1_1, t2, t2_1, t3_1]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t4]),
  )
  result = any_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_any_order_match_with_duplicates(
    any_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with ANY_ORDER match type with duplicates."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t3 = genai_types.FunctionCall(name="t3", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t3, t1]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t1]),
  )
  result = any_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.PASSED


def test_evaluate_invocations_any_order_match_fails_with_duplicates_missing(
    any_order_evaluator: TrajectoryEvaluator,
):
  """Tests evaluate_invocations with ANY_ORDER match type with missing duplicates."""
  t1 = genai_types.FunctionCall(name="t1", args={})
  t2 = genai_types.FunctionCall(name="t2", args={})
  t3 = genai_types.FunctionCall(name="t3", args={})
  actual_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t3]),
  )
  expected_invocation = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[t1, t2, t1]),
  )
  result = any_order_evaluator.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED
  assert result.per_invocation_results[0].score == 0.0
  assert result.per_invocation_results[0].eval_status == EvalStatus.FAILED


def test_evaluate_invocations_no_invocations(evaluator: TrajectoryEvaluator):
  """Tests evaluate_invocations with no invocations."""
  result = evaluator.evaluate_invocations([], [])
  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert not result.per_invocation_results


# --- ignore_args tests ---


def _make_ignore_args_evaluator(
    match_type: ToolTrajectoryCriterion.MatchType,
) -> TrajectoryEvaluator:
  return TrajectoryEvaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
          criterion=ToolTrajectoryCriterion(
              threshold=0.5,
              match_type=match_type,
              ignore_args=True,
          ),
      )
  )


def test_exact_ignore_args_passes_with_different_args():
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.EXACT)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={"a": 1}),
              genai_types.FunctionCall(name="t2", args={"b": 2}),
          ]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={"x": 99}),
              genai_types.FunctionCall(name="t2", args={"y": 100}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 1.0


def test_exact_ignore_args_fails_with_different_names():
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.EXACT)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t2", args={})]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_in_order_ignore_args_passes_with_different_args():
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.IN_ORDER)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={"a": 1}),
              genai_types.FunctionCall(name="extra", args={}),
              genai_types.FunctionCall(name="t2", args={"b": 2}),
          ]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={"x": 99}),
              genai_types.FunctionCall(name="t2", args={"y": 100}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 1.0


def test_any_order_ignore_args_passes_with_different_args():
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.ANY_ORDER)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t2", args={"b": 2}),
              genai_types.FunctionCall(name="t1", args={"a": 1}),
          ]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={"x": 99}),
              genai_types.FunctionCall(name="t2", args={"y": 100}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 1.0


def test_any_order_ignore_args_fails_with_missing_tool():
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.ANY_ORDER)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={}),
              genai_types.FunctionCall(name="t2", args={}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_ignore_args_from_dict_config():
  """Tests that ignore_args works when passed as a dict criterion."""
  eval_metric = EvalMetric(
      metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
      criterion={
          "threshold": 0.5,
          "match_type": "ANY_ORDER",
          "ignore_args": True,
      },
  )
  ev = TrajectoryEvaluator(eval_metric=eval_metric)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"a": 1})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"z": 999})]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 1.0


def test_exact_ignore_args_fails_with_different_count():
  """EXACT + ignore_args still fails when tool call counts differ."""
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.EXACT)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={}),
              genai_types.FunctionCall(name="t2", args={}),
          ]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={})]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_in_order_ignore_args_fails_with_wrong_order():
  """IN_ORDER + ignore_args still fails when order is wrong."""
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.IN_ORDER)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t2", args={}),
              genai_types.FunctionCall(name="t1", args={}),
          ]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={}),
              genai_types.FunctionCall(name="t2", args={}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_in_order_ignore_args_fails_with_missing_tool():
  """IN_ORDER + ignore_args still fails when expected tool is missing."""
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.IN_ORDER)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[
              genai_types.FunctionCall(name="t1", args={}),
              genai_types.FunctionCall(name="t2", args={}),
          ]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_ignore_args_false_still_checks_args():
  """Confirm ignore_args=False (default) still enforces arg matching."""
  ev = TrajectoryEvaluator(
      eval_metric=EvalMetric(
          metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
          criterion=ToolTrajectoryCriterion(
              threshold=0.5,
              match_type=ToolTrajectoryCriterion.MatchType.EXACT,
              ignore_args=False,
          ),
      )
  )
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"a": 1})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"a": 2})]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 0.0


def test_ignore_args_empty_tool_lists():
  """ignore_args with empty tool lists on both sides should pass."""
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.EXACT)
  inv = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=[]),
  )
  result = ev.evaluate_invocations([inv], [inv])
  assert result.overall_score == 1.0


def test_ignore_args_multiple_invocations_mixed():
  """ignore_args with multiple invocations: one matches, one doesn't."""
  ev = _make_ignore_args_evaluator(ToolTrajectoryCriterion.MatchType.EXACT)
  inv1_actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"a": 1})]
      ),
  )
  inv1_expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"z": 99})]
      ),
  )
  inv2_actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={})]
      ),
  )
  inv2_expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t2", args={})]
      ),
  )
  result = ev.evaluate_invocations(
      [inv1_actual, inv2_actual], [inv1_expected, inv2_expected]
  )
  assert result.overall_score == 0.5
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[1].score == 0.0


def test_ignore_args_with_camel_case_dict_config():
  """Tests ignore_args works via camelCase key (ignoreArgs) in dict."""
  eval_metric = EvalMetric(
      metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
      criterion={
          "threshold": 0.5,
          "matchType": "EXACT",
          "ignoreArgs": True,
      },
  )
  ev = TrajectoryEvaluator(eval_metric=eval_metric)
  actual = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"a": 1})]
      ),
  )
  expected = Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(
          tool_uses=[genai_types.FunctionCall(name="t1", args={"z": 999})]
      ),
  )
  result = ev.evaluate_invocations([actual], [expected])
  assert result.overall_score == 1.0
