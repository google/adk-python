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

"""Aggregation of per-run EvalCaseResults into a single result per eval case."""

from __future__ import annotations

import statistics

from .base_eval_service import AggregationStrategy
from .eval_metrics import _get_metric_threshold
from .eval_metrics import EvalMetricResult
from .eval_result import EvalCaseResult
from .evaluator import EvalStatus


def _generate_final_eval_status(
    overall_eval_metric_results: list[EvalMetricResult],
) -> EvalStatus:
  """Returns the final eval status for a case from its overall metric results."""
  final_eval_status = EvalStatus.NOT_EVALUATED
  for overall_eval_metric_result in overall_eval_metric_results:
    overall_eval_status = overall_eval_metric_result.eval_status
    if overall_eval_status == EvalStatus.PASSED:
      final_eval_status = EvalStatus.PASSED
    elif overall_eval_status in (
        EvalStatus.NOT_EVALUATED,
        EvalStatus.INFORMATIONAL,
    ):
      # Informational metrics (e.g. the efficiency metrics) report a value but
      # never pass or fail, so they do not affect the case's status.
      continue
    elif overall_eval_status == EvalStatus.FAILED:
      return EvalStatus.FAILED
    else:
      raise ValueError(f"Unknown eval status: {overall_eval_status}.")
  return final_eval_status


def _is_informational(metric_result: EvalMetricResult) -> bool:
  """Returns True for a metric that reports a value but never passes or fails.

  Detected by its INFORMATIONAL status, falling back to the absence of any
  threshold or criterion. Mirrors `AgentEvaluator`.
  """
  return metric_result.eval_status == EvalStatus.INFORMATIONAL or (
      metric_result.criterion is None and metric_result.threshold is None
  )


def _has_hard_failed_run(per_case_results: list[EvalCaseResult]) -> bool:
  """Returns True if any run failed without producing metric results.

  A run whose inferencing raised is recorded as FAILED with no per-invocation
  metric results. Such a run contributes no scores to the mean, so it must be
  honored explicitly or a crashed run would be silently dropped from the
  aggregate verdict. Mirrors
  `AgentEvaluator._get_failures_from_final_eval_status`.
  """
  for result in per_case_results:
    if result.final_eval_status != EvalStatus.FAILED:
      continue
    if not any(
        invocation.eval_metric_results
        for invocation in result.eval_metric_result_per_invocation
    ):
      return True
  return False


def _mean_over_invocations(
    per_case_results: list[EvalCaseResult],
) -> list[EvalMetricResult]:
  """Returns overall metric results pooled over every invocation of every run.

  Every per-invocation metric result across all runs is pooled by metric name
  and its scores are averaged into a single overall score for that metric. This
  matches `AgentEvaluator`, whereas averaging each run's already-overall score
  (mean-of-means) would weight runs equally regardless of invocation count.
  """
  # Pool per-invocation metric results across all runs, keyed by metric name,
  # preserving first-seen order (which is metric evaluation order).
  results_by_metric: dict[str, list[EvalMetricResult]] = {}
  for result in per_case_results:
    for invocation in result.eval_metric_result_per_invocation:
      for metric_result in invocation.eval_metric_results:
        results_by_metric.setdefault(metric_result.metric_name, []).append(
            metric_result
        )

  overall_eval_metric_results: list[EvalMetricResult] = []
  for metric_results in results_by_metric.values():
    aggregate_metric_result = metric_results[0].model_copy(deep=True)
    scores = [m.score for m in metric_results if m.score is not None]
    if _is_informational(aggregate_metric_result):
      aggregate_metric_result.score = (
          statistics.mean(scores) if scores else None
      )
      aggregate_metric_result.eval_status = EvalStatus.INFORMATIONAL
    elif scores:
      aggregate_metric_result.score = statistics.mean(scores)
      aggregate_metric_result.eval_status = (
          EvalStatus.PASSED
          if aggregate_metric_result.score
          >= _get_metric_threshold(aggregate_metric_result)
          else EvalStatus.FAILED
      )
    else:
      aggregate_metric_result.score = None
      aggregate_metric_result.eval_status = EvalStatus.NOT_EVALUATED
    overall_eval_metric_results.append(aggregate_metric_result)

  return overall_eval_metric_results


def aggregate_eval_case_results(
    eval_case_results: list[EvalCaseResult],
    aggregation_strategy: AggregationStrategy = (
        AggregationStrategy.MEAN_OVER_INVOCATIONS
    ),
) -> list[EvalCaseResult]:
  """Aggregates per-run EvalCaseResults into one result per eval case.

  Results are grouped by (eval_set_id, eval_id); each group holds the runs for a
  single eval case. The returned list has one EvalCaseResult per case, sorted by
  (eval_set_id, eval_id). A case run once is returned unchanged, so a plain
  single-run eval keeps the evaluator's own overall scores and status.

  Args:
    eval_case_results: The per-run results to aggregate.
    aggregation_strategy: How to combine per-run results. Only
      `MEAN_OVER_INVOCATIONS` is currently supported.
  """
  if aggregation_strategy != AggregationStrategy.MEAN_OVER_INVOCATIONS:
    raise ValueError(
        f"Unsupported aggregation strategy: {aggregation_strategy}."
    )

  results_by_case: dict[tuple[str, str], list[EvalCaseResult]] = {}
  for result in eval_case_results:
    key = (result.eval_set_id, result.eval_id)
    results_by_case.setdefault(key, []).append(result)

  aggregate_results: list[EvalCaseResult] = []
  for per_case_results in results_by_case.values():
    if len(per_case_results) == 1:
      aggregate_results.append(per_case_results[0])
      continue

    # Runs arrive in completion order, so sort them to make the representative
    # run (and the order of the retained invocations) deterministic. The
    # representative is not necessarily the first run that was started.
    per_case_results = sorted(
        per_case_results, key=lambda result: result.session_id or ""
    )
    overall_eval_metric_results = _mean_over_invocations(per_case_results)
    final_eval_status = _generate_final_eval_status(overall_eval_metric_results)

    # A run that crashed before producing any metric results contributes no
    # scores to the mean above, so honor its failure explicitly.
    if final_eval_status != EvalStatus.FAILED and _has_hard_failed_run(
        per_case_results
    ):
      final_eval_status = EvalStatus.FAILED

    representative_result = per_case_results[0]
    aggregate_result = representative_result.model_copy(deep=True)
    aggregate_result.overall_eval_metric_results = overall_eval_metric_results
    aggregate_result.final_eval_status = final_eval_status
    # Retain every run's invocations so detailed inspection still has them.
    aggregate_result.eval_metric_result_per_invocation = [
        invocation
        for result in per_case_results
        for invocation in result.eval_metric_result_per_invocation
    ]
    aggregate_results.append(aggregate_result)

  return sorted(aggregate_results, key=lambda x: (x.eval_set_id, x.eval_id))
