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

"""Tests for aggregating the per-run results of an eval case run many times."""

from __future__ import annotations

from collections.abc import Sequence
import statistics
from typing import Optional

from google.adk.agents.common_configs import CodeConfig
from google.adk.evaluation._eval_case_result_multi_run_aggregator import aggregate_eval_case_results
from google.adk.evaluation._eval_case_result_multi_run_aggregator import count_runs_failed_without_metric_results
from google.adk.evaluation._eval_case_result_multi_run_aggregator import get_run_status
from google.adk.evaluation._eval_case_result_multi_run_aggregator import has_pass_count_rules
from google.adk.evaluation._eval_case_result_multi_run_aggregator import load_aggregation_methods
from google.adk.evaluation._eval_case_result_multi_run_aggregator import MethodResult
from google.adk.evaluation._eval_case_result_multi_run_aggregator import MultiRunCaseResult
from google.adk.evaluation._multi_run_aggregation_methods import get_built_in_aggregation_methods
from google.adk.evaluation._multi_run_aggregation_methods import RunResult
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_config import CustomCrossRunMethodConfig
from google.adk.evaluation.eval_config import MultiRunConfig
from google.adk.evaluation.eval_config import MultiRunCriterion
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultDetails
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_rubrics import RubricScore
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
    repetition_index: Optional[int] = None,
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
      repetition_index=repetition_index,
  )


def _crashed_run(*, session_id: str) -> EvalCaseResult:
  """Returns a run that failed before it produced any metric results."""
  return EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id=session_id,
  )


def _with_metric(
    run: EvalCaseResult, metric_result: EvalMetricResult
) -> EvalCaseResult:
  """Returns `run` with `metric_result` added to each of its invocations."""
  for invocation in run.eval_metric_result_per_invocation:
    invocation.eval_metric_results.append(metric_result.model_copy())
  return run


def _judged_run(
    repetition_index: int, **scores: Optional[float]
) -> EvalCaseResult:
  """Returns a run of case1 with one invocation, scored as given per metric.

  Each metric passes at `_THRESHOLD`, or was not evaluated if its score is
  None. Its overall result matches its invocation's, and the run's status
  follows from the metrics as in `LocalEvalService`.
  """
  metric_results = []
  for metric_name, score in scores.items():
    if score is None:
      eval_status = EvalStatus.NOT_EVALUATED
    elif score >= _THRESHOLD:
      eval_status = EvalStatus.PASSED
    else:
      eval_status = EvalStatus.FAILED
    metric_results.append(
        EvalMetricResult(
            metric_name=metric_name,
            threshold=_THRESHOLD,
            score=score,
            eval_status=eval_status,
        )
    )
  statuses = [metric_result.eval_status for metric_result in metric_results]
  if EvalStatus.FAILED in statuses:
    final_eval_status = EvalStatus.FAILED
  elif EvalStatus.PASSED in statuses:
    final_eval_status = EvalStatus.PASSED
  else:
    final_eval_status = EvalStatus.NOT_EVALUATED
  return EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=final_eval_status,
      overall_eval_metric_results=metric_results,
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=Invocation(
                  user_content=genai_types.Content(
                      parts=[genai_types.Part(text="hi")]
                  )
              ),
              eval_metric_results=[m.model_copy() for m in metric_results],
          )
      ],
      session_id=f"s{repetition_index}",
      repetition_index=repetition_index,
  )


def _run_scored(
    repetition_index: int,
    *,
    score: float,
    eval_status: EvalStatus,
    metric_name: str = _METRIC,
    threshold: float = _THRESHOLD,
    per_invocation: bool = True,
) -> EvalCaseResult:
  """Returns a run of case1 in which one metric got `score` and `eval_status`.

  The status is given rather than derived from the score, so the metric's
  evaluator can judge in either direction. The metric is scored for the whole
  run, and for the run's one invocation too unless `per_invocation` is False.
  """
  metric_result = EvalMetricResult(
      metric_name=metric_name,
      threshold=threshold,
      score=score,
      eval_status=eval_status,
  )
  return EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=eval_status,
      overall_eval_metric_results=[metric_result],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=Invocation(
                  user_content=genai_types.Content(
                      parts=[genai_types.Part(text="hi")]
                  )
              ),
              eval_metric_results=[metric_result.model_copy()],
          )
      ]
      if per_invocation
      else [],
      session_id=f"s{repetition_index}",
      repetition_index=repetition_index,
  )


def test_mean_is_taken_over_invocations_not_over_runs():
  """Scores are pooled over every invocation, not averaged per run first."""
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

  aggregated = aggregate_eval_case_results(results)

  assert len(aggregated) == 1
  metric_result = aggregated[0].overall_eval_metric_results[0]
  assert metric_result.score == pytest.approx(0.75)
  assert metric_result.eval_status == EvalStatus.PASSED
  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_groups_by_eval_set_id_and_eval_id_and_sorts():
  """One result comes back per eval case, sorted by eval set and case id."""
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


def test_aggregate_keeps_every_run():
  """The aggregate keeps the result of every run, for detailed inspection."""
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0, 1.0],
          session_id="s1",
          repetition_index=0,
      ),
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0],
          session_id="s2",
          repetition_index=1,
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  assert aggregated[0].runs == results


def test_crashed_run_fails_the_case():
  """A run that crashed before scoring fails the case, though it adds no score."""
  passing = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[1.0],
      session_id="s1",
  )

  aggregated = aggregate_eval_case_results(
      [passing, _crashed_run(session_id="s2")]
  )

  assert len(aggregated) == 1
  assert aggregated[0].final_eval_status == EvalStatus.FAILED


def test_counts_only_runs_that_failed_without_metric_results():
  """A crashed run is counted; a run that was scored and failed is not.

  That holds whether the failed run was scored per invocation or only as a
  whole.
  """
  passing = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[1.0],
      session_id="s1",
  )
  failed_with_scores = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[0.0],
      session_id="s2",
      final_eval_status=EvalStatus.FAILED,
  )
  failed_with_overall_score_only = EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[
          EvalMetricResult(
              metric_name=_METRIC,
              threshold=_THRESHOLD,
              score=0.0,
              eval_status=EvalStatus.FAILED,
          )
      ],
      eval_metric_result_per_invocation=[],
      session_id="s4",
  )

  count = count_runs_failed_without_metric_results([
      passing,
      failed_with_scores,
      _crashed_run(session_id="s3"),
      failed_with_overall_score_only,
  ])

  assert count == 1


def test_get_run_status_leaves_a_crashed_run_undecided():
  """A run that crashed has no verdict, so it neither passed nor failed."""
  assert (
      get_run_status(_crashed_run(session_id="s1")) == EvalStatus.NOT_EVALUATED
  )


def test_cross_run_criterion_leaves_crashed_runs_out():
  """A crashed run counts neither for nor against a cross-run criterion."""
  runs = [
      *(_judged_run(index, response_match_score=1.0) for index in range(3)),
      _crashed_run(session_id="s3"),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.75)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].cross_run_results == [
      MethodResult(
          name="pass_rate",
          value=1.0,
          threshold=0.75,
          status=EvalStatus.PASSED,
          note="3 of 4 runs passed, 1 crashed",
      )
  ]


@pytest.mark.parametrize(
    ("max_crashed_runs", "crashed", "expected"),
    [
        (0, 1, EvalStatus.FAILED),
        (1, 1, EvalStatus.PASSED),
        (1, 2, EvalStatus.FAILED),
    ],
    ids=["no_crash_allowed", "crash_allowed", "more_crashes_than_allowed"],
)
def test_case_fails_when_more_runs_crashed_than_allowed(
    max_crashed_runs, crashed, expected
):
  """Crashed runs fail a case only when there are more than it allows.

  Setup: three runs pass, on their metric and as a whole, and `crashed` more
    runs crash. A cross-run criterion asks for 75% of the runs to pass.
  Assert: the metric and the criterion pass either way, so the verdict follows
    `max_crashed_runs` alone.
  """
  runs = [
      *(_judged_run(index, response_match_score=1.0) for index in range(3)),
      *(_crashed_run(session_id=f"crash{index}") for index in range(crashed)),
  ]
  multi_run_config = MultiRunConfig(
      max_crashed_runs=max_crashed_runs,
      cross_run_criteria=[
          MultiRunCriterion(method="pass_rate", threshold=0.75)
      ],
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].final_eval_status == expected


@pytest.mark.parametrize(
    "max_crashed_runs", [2, 3], ids=["as_many_as_runs", "more_than_runs"]
)
def test_case_whose_runs_all_crashed_is_not_evaluated(max_crashed_runs):
  """A case needs a run that did not crash to pass, however many may crash."""
  runs = [_crashed_run(session_id=f"s{index}") for index in range(2)]
  multi_run_config = MultiRunConfig(
      max_crashed_runs=max_crashed_runs,
      cross_run_criteria=[
          MultiRunCriterion(method="pass_rate", threshold=0.75)
      ],
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].final_eval_status == EvalStatus.NOT_EVALUATED


def test_cross_run_criterion_note_counts_crashed_runs_without_a_value():
  """With every run crashed, the criterion has no value and says why."""
  runs = [_crashed_run(session_id=f"s{index}") for index in range(2)]
  multi_run_config = MultiRunConfig(
      max_crashed_runs=2,
      cross_run_criteria=[
          MultiRunCriterion(method="pass_rate", threshold=0.75)
      ],
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].cross_run_results[0].note == (
      "0 of 2 runs passed or failed, 2 crashed"
  )


@pytest.mark.parametrize(
    ("max_crashed_runs", "expected"),
    [(0, EvalStatus.FAILED), (1, EvalStatus.PASSED)],
    ids=["no_crash_allowed", "crash_allowed"],
)
def test_metric_pass_count_rule_leaves_a_crashed_run_out(
    max_crashed_runs, expected
):
  """A crashed run does not lower a metric's pass rate; the tolerance decides.

  Setup: one run passes and one crashes, and the metric is judged by its pass
    rate, with a threshold of 0.9.
  Assert: the pass rate is 1.0 either way, and `max_crashed_runs` alone
    decides whether the crash fails the case.
  """
  runs = [
      _judged_run(0, response_match_score=1.0),
      _crashed_run(session_id="s1"),
  ]
  multi_run_config = MultiRunConfig(
      max_crashed_runs=max_crashed_runs,
      cross_run_scoring_by_metric={
          "response_match_score": MultiRunCriterion(
              method="pass_rate", threshold=0.9
          )
      },
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert (
      aggregated[0].metric_methods["response_match_score"].value,
      aggregated[0].final_eval_status,
  ) == (1.0, expected)


def test_metric_pass_count_note_counts_a_crashed_run():
  """The note shows the crashed run the pass rate leaves out, as the header does."""
  runs = [
      _judged_run(0, response_match_score=1.0),
      _crashed_run(session_id="s1"),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring_by_metric={
          "response_match_score": MultiRunCriterion(
              method="pass_rate", threshold=0.9
          )
      },
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="pass_rate",
      value=1.0,
      threshold=0.9,
      status=EvalStatus.PASSED,
      note="1 of 2 runs passed, 1 crashed",
  )


def test_aggregated_case_carries_how_many_crashed_runs_it_allows():
  """The verdict keeps the crash tolerance it was judged by, for the output."""
  runs = [_judged_run(index, response_match_score=1.0) for index in range(2)]

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=MultiRunConfig(max_crashed_runs=1)
  )

  assert aggregated[0].max_crashed_runs == 1


def test_below_threshold_fails():
  """A pooled mean below the threshold fails the case."""
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
          invocation_scores=[0.5],
          session_id="s2",
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  assert aggregated[0].overall_eval_metric_results[0].score == pytest.approx(
      0.25
  )
  assert aggregated[0].final_eval_status == EvalStatus.FAILED


def test_single_run_keeps_its_own_verdict():
  """A case run once keeps the evaluator's own overall score and status."""
  # The evaluator's overall score can differ from a plain mean over
  # invocations; a single run must not be re-scored.
  overall = EvalMetricResult(
      metric_name=_METRIC,
      threshold=_THRESHOLD,
      score=0.9,
      eval_status=EvalStatus.PASSED,
  )
  run = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[0.0, 0.5],
      session_id="s1",
      repetition_index=0,
  )
  run.overall_eval_metric_results = [overall]

  aggregated = aggregate_eval_case_results([run])

  assert aggregated == [
      MultiRunCaseResult(
          eval_set_id="set1",
          eval_id="case1",
          final_eval_status=EvalStatus.PASSED,
          overall_eval_metric_results=[overall],
          runs=[run],
      )
  ]


def test_informational_metric_is_reported_but_does_not_fail():
  """An informational metric is averaged across runs but never gates."""
  results = [
      _with_metric(
          _run(
              eval_set_id="set1",
              eval_id="case1",
              invocation_scores=[1.0],
              session_id=f"s{index}",
          ),
          EvalMetricResult(
              metric_name="llm_call_count",
              score=value,
              eval_status=EvalStatus.INFORMATIONAL,
          ),
      )
      for index, value in enumerate([2.0, 4.0])
  ]

  aggregated = aggregate_eval_case_results(results)

  by_name = {
      m.metric_name: m for m in aggregated[0].overall_eval_metric_results
  }
  assert by_name["llm_call_count"].score == pytest.approx(3.0)
  assert by_name["llm_call_count"].eval_status == EvalStatus.INFORMATIONAL
  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_metric_no_run_could_score_does_not_decide_the_case():
  """A metric that no run could score neither passes nor fails the case.

  It is reported as NOT_EVALUATED, the way a single run of `LocalEvalService`
  reports it; `AgentEvaluator` would fail the case instead.
  """
  results = [
      _with_metric(
          _run(
              eval_set_id="set1",
              eval_id="case1",
              invocation_scores=[1.0],
              session_id=f"s{index}",
              repetition_index=index,
          ),
          EvalMetricResult(
              metric_name="final_response_match_v2",
              threshold=0.5,
              eval_status=EvalStatus.NOT_EVALUATED,
          ),
      )
      for index in range(2)
  ]

  aggregated = aggregate_eval_case_results(results)

  by_name = {
      m.metric_name: m for m in aggregated[0].overall_eval_metric_results
  }
  assert by_name["final_response_match_v2"].score is None
  assert (
      by_name["final_response_match_v2"].eval_status == EvalStatus.NOT_EVALUATED
  )
  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_metric_with_nothing_to_pass_is_informational():
  """A metric with no threshold or criterion is averaged but never gates.

  Setup: the first run could not score the metric, so its result there is
    NOT_EVALUATED rather than INFORMATIONAL; the second run scored it.
  Assert: the pooled metric is INFORMATIONAL with the second run's score, and
    the case passes on its other metric.
  """
  results = [
      _with_metric(
          _run(
              eval_set_id="set1",
              eval_id="case1",
              invocation_scores=[1.0],
              session_id="s0",
              repetition_index=0,
          ),
          EvalMetricResult(
              metric_name="llm_call_count",
              eval_status=EvalStatus.NOT_EVALUATED,
          ),
      ),
      _with_metric(
          _run(
              eval_set_id="set1",
              eval_id="case1",
              invocation_scores=[1.0],
              session_id="s1",
              repetition_index=1,
          ),
          EvalMetricResult(
              metric_name="llm_call_count",
              score=4.0,
              eval_status=EvalStatus.INFORMATIONAL,
          ),
      ),
  ]

  aggregated = aggregate_eval_case_results(results)

  by_name = {
      m.metric_name: m for m in aggregated[0].overall_eval_metric_results
  }
  assert by_name["llm_call_count"].score == pytest.approx(4.0)
  assert by_name["llm_call_count"].eval_status == EvalStatus.INFORMATIONAL
  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_runs_are_put_back_in_run_order():
  """The aggregate lists the runs in run order, not arrival order.

  Setup: two runs of one case arrive with run 1 first. Their session ids sort
    the opposite way from their run indices.
  Assert: run 0 comes first.
  """
  run_0 = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[0.0],
      session_id="s9",
      repetition_index=0,
  )
  run_1 = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[1.0],
      session_id="s1",
      repetition_index=1,
  )

  aggregated = aggregate_eval_case_results([run_1, run_0])[0]

  assert aggregated.runs == [run_0, run_1]


def test_runs_without_run_index_are_ordered_by_session_id():
  """Runs saved without a run index are listed by session id in any order."""

  def _results(order: list[str]) -> list[EvalCaseResult]:
    return [
        _run(
            eval_set_id="set1",
            eval_id="case1",
            invocation_scores=[1.0],
            session_id=session_id,
        )
        for session_id in order
    ]

  forward = aggregate_eval_case_results(_results(["s1", "s2"]))[0]
  backward = aggregate_eval_case_results(_results(["s2", "s1"]))[0]

  assert [run.session_id for run in forward.runs] == ["s1", "s2"]
  assert [run.session_id for run in backward.runs] == ["s1", "s2"]


def test_pooled_metric_does_not_carry_one_invocations_details():
  """The pooled score drops details, such as rubric scores, of one invocation."""
  results = [
      _run(
          eval_set_id="set1",
          eval_id="case1",
          invocation_scores=[1.0],
          session_id=f"s{index}",
          repetition_index=index,
      )
      for index in range(2)
  ]
  first_metric = results[0].eval_metric_result_per_invocation[0]
  first_metric.eval_metric_results[0].details = EvalMetricResultDetails(
      rubric_scores=[RubricScore(rubric_id="r1", score=1.0)]
  )

  aggregated = aggregate_eval_case_results(results)

  pooled = aggregated[0].overall_eval_metric_results[0]
  assert pooled.details == EvalMetricResultDetails()
  # The run's own invocation keeps its details.
  assert first_metric.eval_metric_results[0].details.rubric_scores


@pytest.mark.parametrize(
    ("final_eval_status", "metric_statuses", "expected"),
    [
        (
            EvalStatus.FAILED,
            [EvalStatus.FAILED, EvalStatus.NOT_EVALUATED],
            EvalStatus.FAILED,
        ),
        (
            EvalStatus.PASSED,
            [EvalStatus.PASSED, EvalStatus.NOT_EVALUATED],
            EvalStatus.NOT_EVALUATED,
        ),
        (
            EvalStatus.PASSED,
            [EvalStatus.PASSED, EvalStatus.INFORMATIONAL],
            EvalStatus.PASSED,
        ),
        (
            EvalStatus.NOT_EVALUATED,
            [EvalStatus.INFORMATIONAL],
            EvalStatus.NOT_EVALUATED,
        ),
    ],
    ids=[
        "failure_comes_first",
        "unevaluated_metric_leaves_it_undecided",
        "passed",
        "no_metric_gated_it",
    ],
)
def test_get_run_status_judges_a_run_as_a_whole(
    final_eval_status, metric_statuses, expected
):
  """A run fails on any failure, else is undecided on any unevaluated metric."""
  run = EvalCaseResult(
      eval_set_id="set1",
      eval_id="case1",
      final_eval_status=final_eval_status,
      overall_eval_metric_results=[
          EvalMetricResult(metric_name=f"metric_{index}", eval_status=status)
          for index, status in enumerate(metric_statuses)
      ],
      eval_metric_result_per_invocation=[],
      session_id="s1",
  )

  assert get_run_status(run) == expected


@pytest.mark.parametrize(
    ("multi_run_config", "expected"),
    [
        (None, False),
        (MultiRunConfig(), False),
        (
            MultiRunConfig(
                cross_run_scoring_by_metric={_METRIC: "mean_of_runs"}
            ),
            False,
        ),
        (
            MultiRunConfig(
                cross_run_scoring=MultiRunCriterion(
                    method="pass_rate", threshold=0.5
                )
            ),
            True,
        ),
        (
            MultiRunConfig(
                cross_run_scoring_by_metric={
                    _METRIC: MultiRunCriterion(
                        method="pass_rate", threshold=0.5
                    )
                }
            ),
            True,
        ),
        (
            MultiRunConfig(
                cross_run_criteria=[
                    MultiRunCriterion(method="pass_rate", threshold=0.5)
                ]
            ),
            True,
        ),
    ],
    ids=[
        "no_config",
        "defaults",
        "score_rule",
        "pass_count_default",
        "pass_count_metric_rule",
        "cross_run_criterion",
    ],
)
def test_has_pass_count_rules_finds_rules_given_as_objects(
    multi_run_config, expected
):
  """Only a rule with its own threshold counts passing runs."""
  assert has_pass_count_rules(multi_run_config) is expected


def test_metric_rule_replaces_the_default_for_that_metric_only():
  """A metric named in cross_run_scoring_by_metric uses its own method."""
  runs = [
      _judged_run(0, response_match_score=1.0, tool_trajectory_avg_score=1.0),
      _judged_run(1, response_match_score=0.5, tool_trajectory_avg_score=1.0),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring_by_metric={"response_match_score": "min_over_runs"}
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods == {
      "response_match_score": MethodResult(
          name="min_over_runs",
          value=0.5,
          threshold=_THRESHOLD,
          status=EvalStatus.FAILED,
      ),
      "tool_trajectory_avg_score": MethodResult(
          name="mean_over_invocations",
          value=1.0,
          threshold=_THRESHOLD,
          status=EvalStatus.PASSED,
      ),
  }


def test_pass_count_rule_judges_a_metric_by_its_passing_runs():
  """A pass-count rule compares the share of passing runs with its threshold.

  Setup: the metric passes one of two runs; its mean, 0.75, would pass.
  Assert: the pass rate, 0.5, fails the rule's threshold of 0.9.
  """
  runs = [
      _judged_run(0, response_match_score=1.0),
      _judged_run(1, response_match_score=0.5),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring_by_metric={
          "response_match_score": MultiRunCriterion(
              method="pass_rate", threshold=0.9
          )
      }
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="pass_rate",
      value=0.5,
      threshold=0.9,
      status=EvalStatus.FAILED,
      note="1 of 2 runs passed",
  )


def test_metric_result_carries_the_pass_count_rule_value_and_threshold():
  """The metric's overall result reports the pass rate against the rule."""
  runs = [
      _judged_run(0, response_match_score=1.0),
      _judged_run(1, response_match_score=0.5),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring=MultiRunCriterion(method="pass_rate", threshold=0.9)
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  metric_result = aggregated[0].overall_eval_metric_results[0]
  assert (
      metric_result.score,
      metric_result.threshold,
      metric_result.eval_status,
  ) == (0.5, 0.9, EvalStatus.FAILED)


def test_pass_count_rule_without_decided_runs_is_not_evaluated():
  """With no run that passed or failed, the rule has no value, and says why."""
  runs = [
      _judged_run(index, response_match_score=1.0, final_response_match_v2=None)
      for index in range(2)
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring_by_metric={
          "final_response_match_v2": MultiRunCriterion(
              method="pass_rate", threshold=0.5
          )
      }
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods[
      "final_response_match_v2"
  ] == MethodResult(
      name="pass_rate",
      value=None,
      threshold=0.5,
      status=EvalStatus.NOT_EVALUATED,
      note="0 of 2 runs passed or failed",
  )


def test_score_rule_without_scores_says_why():
  """A metric that no run could score is not evaluated, with the reason."""
  runs = [
      _judged_run(index, response_match_score=1.0, final_response_match_v2=None)
      for index in range(2)
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert aggregated[0].metric_methods[
      "final_response_match_v2"
  ] == MethodResult(
      name="mean_over_invocations",
      value=None,
      threshold=_THRESHOLD,
      status=EvalStatus.NOT_EVALUATED,
      note="no run produced a score",
  )


def test_informational_metric_stays_informational_under_a_pass_count_rule():
  """A metric that never passes or fails reports its mean whatever the rule."""
  runs = [
      _with_metric(
          _run(
              eval_set_id="set1",
              eval_id="case1",
              invocation_scores=[1.0],
              session_id=f"s{index}",
              repetition_index=index,
          ),
          EvalMetricResult(
              metric_name="llm_call_count",
              score=value,
              eval_status=EvalStatus.INFORMATIONAL,
          ),
      )
      for index, value in enumerate([2.0, 4.0])
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring=MultiRunCriterion(method="pass_rate", threshold=0.5)
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["llm_call_count"] == MethodResult(
      name="mean_over_invocations",
      value=3.0,
      threshold=None,
      status=EvalStatus.INFORMATIONAL,
  )


def test_metric_with_only_overall_results_is_aggregated():
  """A metric scored only for the whole run is combined like any other."""
  runs = [
      EvalCaseResult(
          eval_set_id="set1",
          eval_id="case1",
          final_eval_status=EvalStatus.PASSED,
          overall_eval_metric_results=[
              EvalMetricResult(
                  metric_name="multi_turn_task_success_v1",
                  threshold=_THRESHOLD,
                  score=score,
                  eval_status=EvalStatus.PASSED,
              )
          ],
          eval_metric_result_per_invocation=[],
          session_id=f"s{index}",
          repetition_index=index,
      )
      for index, score in enumerate([1.0, 0.75])
  ]

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=MultiRunConfig(cross_run_scoring="mean_of_runs")
  )

  assert aggregated[0].metric_methods == {
      "multi_turn_task_success_v1": MethodResult(
          name="mean_of_runs",
          value=0.875,
          threshold=_THRESHOLD,
          status=EvalStatus.PASSED,
      )
  }


def test_default_rule_averages_a_metric_scored_only_per_run():
  """A metric with no per-invocation score is judged on its run scores.

  Setup: two runs failed `multi_turn_task_success_v1`, which is scored only
    for the whole conversation, with 0.25 and 0.5 at a threshold of 0.7.
  Assert: the default rule averages the run scores to 0.375, so the metric,
    and with it the case, fail.
  """
  runs = [
      _run_scored(
          index,
          score=score,
          eval_status=EvalStatus.FAILED,
          metric_name="multi_turn_task_success_v1",
          per_invocation=False,
      )
      for index, score in enumerate([0.25, 0.5])
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert (
      aggregated[0].metric_methods["multi_turn_task_success_v1"],
      aggregated[0].final_eval_status,
  ) == (
      MethodResult(
          name="mean_over_invocations",
          value=0.375,
          threshold=_THRESHOLD,
          status=EvalStatus.FAILED,
      ),
      EvalStatus.FAILED,
  )


def test_metric_where_lower_is_better_passes_at_or_below_its_threshold():
  """A metric whose evaluator passes low scores is judged that way over runs.

  Setup: `response_length` passes at or below 20, and each of three runs
    passed it with a score of 5.0.
  Assert: the mean, 5.0, passes, the output says why, and the case passes.
  """
  runs = [
      _run_scored(
          index,
          score=5.0,
          eval_status=EvalStatus.PASSED,
          metric_name="response_length",
          threshold=20.0,
      )
      for index in range(3)
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert (
      aggregated[0].metric_methods["response_length"],
      aggregated[0].final_eval_status,
  ) == (
      MethodResult(
          name="mean_over_invocations",
          value=5.0,
          threshold=20.0,
          status=EvalStatus.PASSED,
          note="lower is better",
      ),
      EvalStatus.PASSED,
  )


def test_metric_where_higher_is_better_passes_at_or_above_its_threshold():
  """A metric whose evaluator passes high scores is judged as before.

  Setup: one run passed at 1.0 and one failed at 0.5, at a threshold of 0.7.
  Assert: the mean, 0.75, passes, with no note.
  """
  runs = [
      _run_scored(0, score=1.0, eval_status=EvalStatus.PASSED),
      _run_scored(1, score=0.5, eval_status=EvalStatus.FAILED),
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert aggregated[0].metric_methods[_METRIC] == MethodResult(
      name="mean_over_invocations",
      value=0.75,
      threshold=_THRESHOLD,
      status=EvalStatus.PASSED,
  )


def test_metric_with_mixed_signals_is_judged_higher_is_better():
  """When the runs disagree on the direction, higher is better.

  Setup: one run passed below the 0.7 threshold, at 0.5, and one passed above
    it, at 1.0, so they show opposite directions.
  Assert: the mean, 0.75, is judged at or above the threshold, so it passes.
  """
  runs = [
      _run_scored(0, score=0.5, eval_status=EvalStatus.PASSED),
      _run_scored(1, score=1.0, eval_status=EvalStatus.PASSED),
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert aggregated[0].metric_methods[_METRIC] == MethodResult(
      name="mean_over_invocations",
      value=0.75,
      threshold=_THRESHOLD,
      status=EvalStatus.PASSED,
  )


def test_score_at_the_threshold_does_not_show_a_direction():
  """A score equal to the threshold is left out when inferring the direction.

  Setup: `response_length` passes below 20 only. One run passed at 5.0, and
    one failed at exactly 20.0, which taken alone would read as higher is
    better.
  Assert: lower is better, so the mean, 12.5, passes.
  """
  runs = [
      _run_scored(
          0,
          score=5.0,
          eval_status=EvalStatus.PASSED,
          metric_name="response_length",
          threshold=20.0,
      ),
      _run_scored(
          1,
          score=20.0,
          eval_status=EvalStatus.FAILED,
          metric_name="response_length",
          threshold=20.0,
      ),
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert aggregated[0].metric_methods["response_length"] == MethodResult(
      name="mean_over_invocations",
      value=12.5,
      threshold=20.0,
      status=EvalStatus.PASSED,
      note="lower is better",
  )


def test_direction_of_a_metric_scored_only_per_run_is_inferred():
  """With no per-invocation result, the runs' overall results show the way."""
  runs = [
      _run_scored(
          index,
          score=5.0,
          eval_status=EvalStatus.PASSED,
          metric_name="response_length",
          threshold=20.0,
          per_invocation=False,
      )
      for index in range(2)
  ]

  aggregated = aggregate_eval_case_results(runs)

  assert aggregated[0].metric_methods["response_length"] == MethodResult(
      name="mean_over_invocations",
      value=5.0,
      threshold=20.0,
      status=EvalStatus.PASSED,
      note="lower is better",
  )


def test_pass_count_rule_on_a_lower_is_better_metric_counts_passing_runs():
  """A rule that counts passing runs is met by enough of them, in any metric.

  Setup: `response_length` passes at or below 20, and two of three runs
    passed it. Its rule asks for 60% of the runs to pass.
  Assert: the share of passing runs, 2/3, passes at or above 0.6.
  """
  runs = [
      _run_scored(
          index,
          score=score,
          eval_status=eval_status,
          metric_name="response_length",
          threshold=20.0,
      )
      for index, (score, eval_status) in enumerate([
          (5.0, EvalStatus.PASSED),
          (5.0, EvalStatus.PASSED),
          (30.0, EvalStatus.FAILED),
      ])
  ]
  multi_run_config = MultiRunConfig(
      cross_run_scoring_by_metric={
          "response_length": MultiRunCriterion(
              method="pass_rate", threshold=0.6
          )
      }
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_length"] == MethodResult(
      name="pass_rate",
      value=2 / 3,
      threshold=0.6,
      status=EvalStatus.PASSED,
      note="2 of 3 runs passed",
  )


@pytest.mark.parametrize(
    ("second_run", "expected_value", "expected_status"),
    [
        ((30.0, EvalStatus.FAILED), 30.0, EvalStatus.FAILED),
        ((10.0, EvalStatus.PASSED), 10.0, EvalStatus.PASSED),
    ],
    ids=["worst_run_over_the_limit", "every_run_within_the_limit"],
)
def test_min_over_runs_takes_the_highest_score_where_lower_is_better(
    second_run, expected_value, expected_status
):
  """`min_over_runs` judges the worst run, which is the highest scoring one.

  Setup: `response_length` passes at or below 20. One run scored 5.0, and the
    other `second_run`'s score.
  Assert: the value is the higher score, so the metric passes only if both
    runs are within the limit.
  """
  second_score, second_status = second_run
  runs = [
      _run_scored(
          0,
          score=5.0,
          eval_status=EvalStatus.PASSED,
          metric_name="response_length",
          threshold=20.0,
      ),
      _run_scored(
          1,
          score=second_score,
          eval_status=second_status,
          metric_name="response_length",
          threshold=20.0,
      ),
  ]
  multi_run_config = MultiRunConfig(cross_run_scoring="min_over_runs")

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_length"] == MethodResult(
      name="min_over_runs",
      value=expected_value,
      threshold=20.0,
      status=expected_status,
      note="lower is better",
  )


def test_cross_run_criterion_passes_when_enough_runs_pass_as_a_whole():
  """Every run passed, so the share of passing runs meets the criterion."""
  runs = [_judged_run(index, response_match_score=1.0) for index in range(2)]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].cross_run_results == [
      MethodResult(
          name="pass_rate",
          value=1.0,
          threshold=0.8,
          status=EvalStatus.PASSED,
          note="2 of 2 runs passed",
      )
  ]


def test_failing_cross_run_criterion_fails_a_case_whose_metrics_pass():
  """The case must pass its cross-run criteria as well as its metrics.

  Setup: the metric's mean over two runs, 0.75, passes, but only one of the
  two runs passed, short of the criterion's 0.8.
  Assert: the metric passes and the criterion fails, so the case fails.
  """
  runs = [
      _judged_run(0, response_match_score=1.0),
      _judged_run(1, response_match_score=0.5),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert (
      aggregated[0].metric_methods["response_match_score"].status,
      aggregated[0].cross_run_results[0].status,
      aggregated[0].final_eval_status,
  ) == (EvalStatus.PASSED, EvalStatus.FAILED, EvalStatus.FAILED)


def test_passing_cross_run_criterion_does_not_rescue_a_failing_metric():
  """A cross-run criterion adds a condition; a failing metric still fails."""
  runs = [
      _judged_run(0, response_match_score=0.6),
      _judged_run(1, response_match_score=0.75),
  ]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.5)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert (
      aggregated[0].cross_run_results[0].status,
      aggregated[0].final_eval_status,
  ) == (EvalStatus.PASSED, EvalStatus.FAILED)


def test_cross_run_criterion_without_decided_runs_is_not_evaluated():
  """With no run decided as a whole, the criterion has no value, and says why.

  A run with a metric that was not evaluated is undecided, though its other
  metric passed.
  """
  runs = [
      _judged_run(index, response_match_score=1.0, final_response_match_v2=None)
      for index in range(2)
  ]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].cross_run_results == [
      MethodResult(
          name="pass_rate",
          value=None,
          threshold=0.8,
          status=EvalStatus.NOT_EVALUATED,
          note="0 of 2 runs passed or failed",
      )
  ]


def test_unevaluated_cross_run_criterion_leaves_the_verdict_to_the_metrics():
  """A criterion that could not be evaluated is skipped, like such a metric."""
  runs = [
      _judged_run(index, response_match_score=1.0, final_response_match_v2=None)
      for index in range(2)
  ]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].final_eval_status == EvalStatus.PASSED


def test_single_run_is_judged_by_cross_run_criteria():
  """A case run once is still held to the cross-run criteria."""
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      [_judged_run(0, response_match_score=1.0)],
      multi_run_config=multi_run_config,
  )

  assert aggregated[0].cross_run_results == [
      MethodResult(
          name="pass_rate",
          value=1.0,
          threshold=0.8,
          status=EvalStatus.PASSED,
          note="1 of 1 runs passed",
      )
  ]


def test_single_run_metric_is_judged_by_its_pass_count_rule():
  """A case run once is judged by a metric's pass-count rule."""
  multi_run_config = MultiRunConfig(
      cross_run_scoring=MultiRunCriterion(method="pass_rate", threshold=0.5)
  )

  aggregated = aggregate_eval_case_results(
      [_judged_run(0, response_match_score=0.9)],
      multi_run_config=multi_run_config,
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="pass_rate",
      value=1.0,
      threshold=0.5,
      status=EvalStatus.PASSED,
      note="1 of 1 runs passed",
  )


def test_single_run_keeps_its_own_score_under_a_score_rule():
  """With pass-count rules configured, a score rule still does not re-score.

  Setup: the run's own overall score, 0.9, differs from the mean of its
    invocation scores, 0.25, and a cross-run criterion is configured.
  Assert: the metric keeps the run's own overall result.
  """
  overall = EvalMetricResult(
      metric_name=_METRIC,
      threshold=_THRESHOLD,
      score=0.9,
      eval_status=EvalStatus.PASSED,
  )
  run = _run(
      eval_set_id="set1",
      eval_id="case1",
      invocation_scores=[0.0, 0.5],
      session_id="s1",
      repetition_index=0,
  )
  run.overall_eval_metric_results = [overall]
  multi_run_config = MultiRunConfig(
      cross_run_criteria=[MultiRunCriterion(method="pass_rate", threshold=0.8)]
  )

  aggregated = aggregate_eval_case_results(
      [run], multi_run_config=multi_run_config
  )

  assert aggregated[0].overall_eval_metric_results == [overall]
  assert aggregated[0].metric_methods[_METRIC] == MethodResult(
      name="mean_over_invocations",
      value=0.9,
      threshold=_THRESHOLD,
      status=EvalStatus.PASSED,
  )


# A custom method is imported by its fully qualified name, so the functions
# the tests below declare as custom methods must be attributes of this module.


def _median_of_run_scores(runs: Sequence[RunResult]) -> float | None:
  scores = [run.score for run in runs if run.score is not None]
  return statistics.median(scores) if scores else None


def _at_least_two_runs_passed(runs: Sequence[RunResult]) -> float | None:
  passed = sum(1 for run in runs if run.status == EvalStatus.PASSED)
  return 1.0 if passed >= 2 else 0.0


def _no_value(runs: Sequence[RunResult]) -> float | None:
  del runs
  return None


def _custom_method(
    function, *, uses_scores: bool
) -> CustomCrossRunMethodConfig:
  """Returns the config of a custom method whose function is in this module."""
  return CustomCrossRunMethodConfig(
      code_config=CodeConfig(name=f"{__name__}.{function.__name__}"),
      uses_scores=uses_scores,
  )


def test_load_aggregation_methods_adds_custom_methods_to_the_built_in_ones():
  """Every built-in method can be named, and each custom one by its key."""
  multi_run_config = MultiRunConfig(
      custom_methods={
          "median_of_runs": _custom_method(
              _median_of_run_scores, uses_scores=True
          )
      }
  )

  methods = load_aggregation_methods(multi_run_config)

  assert set(methods) == {*get_built_in_aggregation_methods(), "median_of_runs"}


def test_custom_score_method_is_judged_against_the_metric_threshold():
  """A custom method that combines scores is held to the metric's threshold.

  Setup: the run scores 0.0, 0.8 and 0.9 have a mean of 0.57, short of the
    0.7 threshold, and a median of 0.8.
  Assert: the median passes, under the name the config gives the method.
  """
  runs = [
      _judged_run(0, response_match_score=0.0),
      _judged_run(1, response_match_score=0.8),
      _judged_run(2, response_match_score=0.9),
  ]
  multi_run_config = MultiRunConfig(
      custom_methods={
          "median_of_runs": _custom_method(
              _median_of_run_scores, uses_scores=True
          )
      },
      cross_run_scoring_by_metric={"response_match_score": "median_of_runs"},
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="median_of_runs",
      value=0.8,
      threshold=_THRESHOLD,
      status=EvalStatus.PASSED,
  )


def test_custom_pass_count_method_judges_a_metric_by_its_own_threshold():
  """A custom method that counts passing runs is held to the rule's threshold."""
  runs = [
      _judged_run(0, response_match_score=1.0),
      _judged_run(1, response_match_score=0.5),
      _judged_run(2, response_match_score=1.0),
  ]
  multi_run_config = MultiRunConfig(
      custom_methods={
          "passed_at_least_twice": _custom_method(
              _at_least_two_runs_passed, uses_scores=False
          )
      },
      cross_run_scoring_by_metric={
          "response_match_score": MultiRunCriterion(
              method="passed_at_least_twice", threshold=1.0
          )
      },
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="passed_at_least_twice",
      value=1.0,
      threshold=1.0,
      status=EvalStatus.PASSED,
      note="2 of 3 runs passed",
  )


def test_custom_cross_run_criterion_fails_a_case_whose_metrics_pass():
  """A custom method that counts passing runs can judge the whole case.

  Setup: the metric's mean over three runs, 0.73, passes the 0.7 threshold,
    but only one of the runs passed as a whole.
  Assert: the criterion, under the name the config gives the method, fails,
    and so does the case.
  """
  runs = [
      _judged_run(0, response_match_score=1.0),
      _judged_run(1, response_match_score=0.6),
      _judged_run(2, response_match_score=0.6),
  ]
  multi_run_config = MultiRunConfig(
      custom_methods={
          "passed_at_least_twice": _custom_method(
              _at_least_two_runs_passed, uses_scores=False
          )
      },
      cross_run_criteria=[
          MultiRunCriterion(method="passed_at_least_twice", threshold=1.0)
      ],
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert (aggregated[0].cross_run_results, aggregated[0].final_eval_status) == (
      [
          MethodResult(
              name="passed_at_least_twice",
              value=0.0,
              threshold=1.0,
              status=EvalStatus.FAILED,
              note="1 of 3 runs passed",
          )
      ],
      EvalStatus.FAILED,
  )


def test_custom_method_without_a_value_is_not_evaluated():
  """A custom method that returns None leaves its rule NOT_EVALUATED.

  The runs have scores, so the note puts the missing value down to the method
  rather than to the runs.
  """
  runs = [_judged_run(index, response_match_score=1.0) for index in range(2)]
  multi_run_config = MultiRunConfig(
      custom_methods={"no_value": _custom_method(_no_value, uses_scores=True)},
      cross_run_scoring_by_metric={"response_match_score": "no_value"},
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert aggregated[0].metric_methods["response_match_score"] == MethodResult(
      name="no_value",
      value=None,
      threshold=_THRESHOLD,
      status=EvalStatus.NOT_EVALUATED,
      note="the method returned None",
  )


def _share_of_runs_not_failed(runs: Sequence[RunResult]) -> float | None:
  if not runs:
    return None
  return sum(1 for run in runs if run.status != EvalStatus.FAILED) / len(runs)


def test_custom_cross_run_criterion_cannot_pass_a_case_whose_runs_all_crashed():
  """A case with no run that did not crash cannot pass, whatever its rules say.

  Setup: both runs crashed, and up to two may. The custom criterion counts
    every run that did not fail, so it gives 1.0 although no run was decided.
  Assert: the criterion passes, but the case is NOT_EVALUATED.
  """
  runs = [_crashed_run(session_id=f"s{index}") for index in range(2)]
  multi_run_config = MultiRunConfig(
      max_crashed_runs=2,
      custom_methods={
          "not_failed": _custom_method(
              _share_of_runs_not_failed, uses_scores=False
          )
      },
      cross_run_criteria=[
          MultiRunCriterion(method="not_failed", threshold=0.5)
      ],
  )

  aggregated = aggregate_eval_case_results(
      runs, multi_run_config=multi_run_config
  )

  assert (
      aggregated[0].cross_run_results[0].status,
      aggregated[0].final_eval_status,
  ) == (EvalStatus.PASSED, EvalStatus.NOT_EVALUATED)
