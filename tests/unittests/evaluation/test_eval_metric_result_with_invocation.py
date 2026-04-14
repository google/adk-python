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

"""Regression tests for _EvalMetricResultWithInvocation None-handling.

Covers the bug described in https://github.com/google/adk-python/issues/5214
where passing expected_invocation=None (the normal path for
conversation_scenario eval cases) caused a pydantic ValidationError.
"""

from __future__ import annotations

from unittest.mock import patch

from google.adk.evaluation.agent_evaluator import _EvalMetricResultWithInvocation
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.genai import types as genai_types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invocation(**overrides) -> Invocation:
  """Return a minimal Invocation instance."""
  defaults = {
      "user_content": genai_types.Content(
          role="user", parts=[genai_types.Part(text="hello")]
      ),
  }
  defaults.update(overrides)
  return Invocation(**defaults)


def _make_eval_metric_result(
    metric_name: str = "test_metric",
    score: float = 1.0,
    status: EvalStatus = EvalStatus.PASSED,
) -> EvalMetricResult:
  return EvalMetricResult(
      metric_name=metric_name,
      score=score,
      eval_status=status,
  )


# ---------------------------------------------------------------------------
# Tests: _EvalMetricResultWithInvocation accepts None
# ---------------------------------------------------------------------------


class TestEvalMetricResultWithInvocationNone:
  """Regression: expected_invocation=None must be accepted (issue #5214)."""

  def test_construction_with_none_expected_invocation(self):
    """_EvalMetricResultWithInvocation should accept None for expected_invocation."""
    result = _EvalMetricResultWithInvocation(
        actual_invocation=_make_invocation(),
        expected_invocation=None,
        eval_metric_result=_make_eval_metric_result(),
    )
    assert result.expected_invocation is None

  def test_construction_with_omitted_expected_invocation(self):
    """expected_invocation should default to None when omitted."""
    result = _EvalMetricResultWithInvocation(
        actual_invocation=_make_invocation(),
        eval_metric_result=_make_eval_metric_result(),
    )
    assert result.expected_invocation is None

  def test_construction_with_real_expected_invocation(self):
    """Normal case: providing a real Invocation should still work."""
    inv = _make_invocation()
    result = _EvalMetricResultWithInvocation(
        actual_invocation=_make_invocation(),
        expected_invocation=inv,
        eval_metric_result=_make_eval_metric_result(),
    )
    assert result.expected_invocation is inv


# ---------------------------------------------------------------------------
# Tests: _get_eval_metric_results_with_invocation passes None through
# ---------------------------------------------------------------------------


class TestGetEvalMetricResultsWithNone:
  """_get_eval_metric_results_with_invocation must propagate None."""

  def test_none_expected_invocation_propagated(self):
    actual = _make_invocation()
    metric_result = _make_eval_metric_result(metric_name="m1")

    eval_case_result = EvalCaseResult(
        eval_set_id="test_set",
        eval_id="scenario_1",
        final_eval_status=EvalStatus.PASSED,
        overall_eval_metric_results=[metric_result],
        eval_metric_result_per_invocation=[
            EvalMetricResultPerInvocation(
                actual_invocation=actual,
                expected_invocation=None,
                eval_metric_results=[metric_result],
            )
        ],
        session_id="sess-1",
    )

    grouped = AgentEvaluator._get_eval_metric_results_with_invocation(
        [eval_case_result]
    )

    assert "m1" in grouped
    assert len(grouped["m1"]) == 1
    assert grouped["m1"][0].expected_invocation is None
    assert grouped["m1"][0].actual_invocation is actual


# ---------------------------------------------------------------------------
# Tests: _print_details does not crash when expected_invocation is None
# ---------------------------------------------------------------------------


class TestPrintDetailsNoneExpected:
  """_print_details must handle None expected_invocation gracefully."""

  def test_print_details_with_none_expected(self):
    actual = _make_invocation()
    metric_result = _make_eval_metric_result(score=0.9)

    items = [
        _EvalMetricResultWithInvocation(
            actual_invocation=actual,
            expected_invocation=None,
            eval_metric_result=metric_result,
        )
    ]

    # _print_details prints to stdout via tabulate/pandas — we just
    # verify it doesn't raise.
    with patch("builtins.print"):
      AgentEvaluator._print_details(
          eval_metric_result_with_invocations=items,
          overall_eval_status=EvalStatus.PASSED,
          overall_score=0.9,
          metric_name="test_metric",
          threshold=0.5,
      )
