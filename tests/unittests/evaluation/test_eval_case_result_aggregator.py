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

from google.adk.evaluation._eval_case_result_aggregator import aggregate_eval_case_results
from google.adk.evaluation.base_eval_service import AggregationStrategy
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.genai import types as genai_types
import pytest

_METRIC = "response_match_score"
_THRESHOLD = 0.7


def _invocation(score: float) -> EvalMetricResultPerInvocation:
  metric_result = EvalMetricResult(
      metric_name=_METRIC,
      threshold=_THRESHOLD,
      score=score,
      eval_status=EvalStatus.PASSED
      if score >= _THRESHOLD
      else EvalStatus.FAILED,
  )
  return EvalMetricResultPerInvocation(
      actual_invocation=Invocation(
          user_content=genai_types.Content(parts=[genai_types.Part(text="hi")])
      ),
      eval_metric_results=[metric_result],
  )


def _run(
    *,
    eval_set_id: str,
    eval_id: str,
    invocation_scores: list[float],
    session_id: str,
    final_eval_status: EvalStatus = EvalStatus.PASSED,
) -> EvalCaseResult:
  per_invocation = [_invocation(score) for score in invocation_scores]
  # `overall_eval_metric_results` is deliberately left empty: the aggregator
  # must derive the aggregate from per-invocation results, not from a run's
  # already-overall score.
  return EvalCaseResult(
      eval_set_id=eval_set_id,
      eval_id=eval_id,
      final_eval_status=final_eval_status,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=per_invocation,
      session_id=session_id,
  )


def test_mean_is_taken_over_invocations_not_over_runs():
  # Run A: 1 invocation @ 0.0. Run B: 3 invocations @ 1.0.
  # Mean over invocations: (0 + 1 + 1 + 1) / 4 = 0.75 -> PASSED (>= 0.7).
  # Mean of per-run means would be (0.0 + 1.0) / 2 = 0.5 -> FAILED.
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[0.0],
          session_id="s1",
      ),
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0, 1.0, 1.0],
          session_id="s2",
      ),
  ]

  aggregated = aggregate_eval_case_results(
      results,
      aggregation_strategy=AggregationStrategy.MEAN_OVER_INVOCATIONS,
  )

  assert len(aggregated) == 1
  metric_result = aggregated[0].overall_eval_metric_results[0]
  assert metric_result.score == pytest.approx(0.75)
  assert metric_result.eval_status == EvalStatus.PASSED
  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_groups_by_eval_set_id_and_eval_id_and_sorts():
  results = [
      _run(
          eval_set_id="set2",
          eval_id="caseB",
          invocation_scores=[1.0],
          session_id="s1",
      ),
      _run(
          eval_set_id="set1",
          eval_id="caseA",
          invocation_scores=[1.0],
          session_id="s2",
      ),
      _run(
          eval_set_id="set1",
          eval_id="caseA",
          invocation_scores=[1.0],
          session_id="s3",
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  assert [(r.eval_set_id, r.eval_id) for r in aggregated] == [
      ("set1", "caseA"),
      ("set2", "caseB"),
  ]


def test_retains_all_runs_invocations():
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0, 1.0],
          session_id="s1",
      ),
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0],
          session_id="s2",
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  assert len(aggregated[0].eval_metric_result_per_invocation) == 3


def test_hard_failed_run_forces_failure():
  # One run passes; another crashed (FAILED, no metric results). The crashed
  # run contributes no scores, so honor its failure explicitly.
  passing = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[1.0],
      session_id="s1",
  )
  crashed = EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id="s2",
  )

  aggregated = aggregate_eval_case_results([passing, crashed])

  assert len(aggregated) == 1
  assert aggregated[0].final_eval_status == EvalStatus.FAILED


def test_below_threshold_fails():
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[0.0, 0.5],
          session_id="s1",
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  assert aggregated[0].overall_eval_metric_results[0].score == pytest.approx(
      0.25
  )
  assert aggregated[0].final_eval_status == EvalStatus.FAILED


def test_unsupported_strategy_raises():
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0],
          session_id="s1",
      ),
  ]

  class _Other:
    pass

  with pytest.raises(ValueError):
    aggregate_eval_case_results(
        results, aggregation_strategy=_Other()  # type: ignore[arg-type]
    )
