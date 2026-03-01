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

import sys
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.evaluation.agent_evaluator import _EvalMetricResultWithInvocation
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalStatus
from google.genai import types as genai_types


def _make_actual_invocation(
    query: str = "user query", response: str = "agent response"
) -> Invocation:
  return Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text=query)], role="user"
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=response)], role="model"
      ),
  )


def _make_eval_metric_result(
    score: float = 0.9, status: EvalStatus = EvalStatus.PASSED
) -> EvalMetricResult:
  return EvalMetricResult(
      metric_name="test_metric",
      threshold=0.8,
      score=score,
      eval_status=status,
  )


def _call_print_details(
    items: list[_EvalMetricResultWithInvocation],
) -> MagicMock:
  """Calls _print_details with mocked pandas/tabulate, returns the mock DataFrame class."""
  mock_pandas = MagicMock()
  mock_tabulate_module = MagicMock()
  mock_tabulate_module.tabulate = MagicMock(return_value="table")

  with patch.dict(
      sys.modules,
      {"pandas": mock_pandas, "tabulate": mock_tabulate_module},
  ):
    AgentEvaluator._print_details(
        eval_metric_result_with_invocations=items,
        overall_eval_status=EvalStatus.PASSED,
        overall_score=0.9,
        metric_name="test_metric",
        threshold=0.8,
    )

  return mock_pandas.pandas.DataFrame


class TestPrintDetailsWithNoExpectedInvocation:
  """Tests for _print_details when expected_invocation is None."""

  def test_does_not_raise(self):
    items = [
        _EvalMetricResultWithInvocation(
            actual_invocation=_make_actual_invocation(),
            expected_invocation=None,
            eval_metric_result=_make_eval_metric_result(),
        )
    ]
    _call_print_details(items)  # should not raise

  def test_multiple_invocations_all_without_expected(self):
    items = [
        _EvalMetricResultWithInvocation(
            actual_invocation=_make_actual_invocation(response=f"response {i}"),
            expected_invocation=None,
            eval_metric_result=_make_eval_metric_result(),
        )
        for i in range(3)
    ]
    mock_df_cls = _call_print_details(items)
    data = mock_df_cls.call_args[0][0]
    assert len(data) == 3
    for row in data:
      assert row["prompt"] == ""
      assert row["expected_response"] == ""
      assert row["expected_tool_calls"] == ""
