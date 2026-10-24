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

"""Aggregation of per-run EvalCaseResults into a single result per eval case.

When an eval case is run more than once, the eval service returns (and saves)
one EvalCaseResult per run. This module folds those runs into one verdict per
eval case for reporting; the per-run results remain the record.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses

from ._multi_run_aggregation_methods import AggregationMethod
from ._multi_run_aggregation_methods import DEFAULT_AGGREGATION_METHOD
from ._multi_run_aggregation_methods import get_built_in_aggregation_methods
from ._multi_run_aggregation_methods import load_custom_aggregation_method
from ._multi_run_aggregation_methods import min_over_runs
from ._multi_run_aggregation_methods import RunResult
from .eval_config import MultiRunConfig
from .eval_config import MultiRunCriterion
from .eval_metrics import _get_metric_threshold
from .eval_metrics import EvalMetricResult
from .eval_metrics import EvalMetricResultDetails
from .eval_result import EvalCaseResult
from .evaluator import EvalStatus


@dataclasses.dataclass(frozen=True)
class MethodResult:
  """How one rule judged the runs of an eval case.

  The rule's target is one metric, or the whole eval case.
  """

  name: str
  """The name of the aggregation method, e.g. "mean_over_invocations"."""

  value: float | None
  """The method's value over the runs, or None if the runs gave it none."""

  threshold: float | None
  """The value needed to pass. None for a metric that never passes or fails."""

  status: EvalStatus
  """PASSED or FAILED by comparing the value with the threshold.

  The value passes at or above the threshold, or at or below it for a metric
  whose evaluator treats lower scores as better. NOT_EVALUATED when there is
  no value, and INFORMATIONAL for a metric that never passes or fails.
  """

  note: str | None = None
  """What the value rests on, such as how many runs passed, or why it is None.

  For a metric judged lower-is-better, with nothing else to say, it says so.
  """


@dataclasses.dataclass(frozen=True)
class MultiRunCaseResult:
  """The verdict for one eval case, judged on all of its runs."""

  eval_set_id: str
  """The eval set id."""

  eval_id: str
  """The eval case id."""

  final_eval_status: EvalStatus
  """The eval case's status across all of its runs."""

  overall_eval_metric_results: list[EvalMetricResult]
  """One result per metric, judged on all of the runs.

  Each result's score is the value of the metric's aggregation method. For a
  case run once, these are that run's own overall results, except for a metric
  judged by counting passing runs.
  """

  runs: list[EvalCaseResult]
  """The result of every run of the eval case, in run order."""

  metric_methods: dict[str, MethodResult] = dataclasses.field(
      default_factory=dict
  )
  """How each metric was judged on the runs, keyed by metric name.

  Empty for a case run once when no rule counts passing runs, since that run's
  own results are kept as they are.
  """

  cross_run_results: list[MethodResult] = dataclasses.field(
      default_factory=list
  )
  """How each of the configured cross-run criteria judged the runs, in order."""

  max_crashed_runs: int = 0
  """How many of the runs may crash before the case fails.

  The config's `max_crashed_runs` for a case judged on its runs. 0 for a case
  run once and kept as it is, which a crash fails through the run's own status.
  """


def _is_crashed(eval_case_result: EvalCaseResult) -> bool:
  """Returns True for a run that failed without producing any metric results.

  That is how a run whose inference raised is recorded: FAILED, with no metric
  result overall or for any invocation.
  """
  return (
      eval_case_result.final_eval_status == EvalStatus.FAILED
      and not eval_case_result.overall_eval_metric_results
      and not any(
          invocation.eval_metric_results
          for invocation in eval_case_result.eval_metric_result_per_invocation
      )
  )


def get_run_status(eval_case_result: EvalCaseResult) -> EvalStatus:
  """Returns whether one run of an eval case passed, failed or was undecided.

  A run that crashed, failing without producing any metric results, has no
  verdict, so it is undecided and left out of every count of passing runs.
  Otherwise a failure takes precedence: if any metric failed, the run failed.
  Then a metric that was not evaluated makes the run undecided, even when the
  other metrics passed, because the missing verdict could have been a fail.
  This is stricter than `final_eval_status`, which reports such a run as
  passed. A run in which no metric passed or failed, for example one scored
  only by informational metrics, is undecided as well.

  Args:
    eval_case_result: The result of one run of one eval case.

  Returns:
    `EvalStatus.PASSED`, `EvalStatus.FAILED` or `EvalStatus.NOT_EVALUATED`.
  """
  if _is_crashed(eval_case_result):
    return EvalStatus.NOT_EVALUATED
  if eval_case_result.final_eval_status == EvalStatus.FAILED:
    return EvalStatus.FAILED
  if any(
      metric_result.eval_status == EvalStatus.NOT_EVALUATED
      for metric_result in eval_case_result.overall_eval_metric_results
  ):
    return EvalStatus.NOT_EVALUATED
  if eval_case_result.final_eval_status == EvalStatus.PASSED:
    return EvalStatus.PASSED
  # No metric gated the case, for example only informational metrics ran.
  return EvalStatus.NOT_EVALUATED


def has_pass_count_rules(multi_run_config: MultiRunConfig | None) -> bool:
  """Returns True if the config judges anything by counting passing runs.

  Such a rule is given as an object: as a metric's rule, or in
  `cross_run_criteria`. It judges a case run once as well, so that run is
  judged by the config rather than reported as it is.

  Args:
    multi_run_config: The multi-run config, or None if there is none.
  """
  if multi_run_config is None:
    return False
  metric_rules = [
      multi_run_config.cross_run_scoring,
      *(multi_run_config.cross_run_scoring_by_metric or {}).values(),
  ]
  return bool(multi_run_config.cross_run_criteria) or any(
      isinstance(rule, MultiRunCriterion) for rule in metric_rules
  )


def load_aggregation_methods(
    multi_run_config: MultiRunConfig | None,
) -> dict[str, AggregationMethod]:
  """Returns every aggregation method the config's rules can name.

  These are the built-in methods and the config's custom methods, whose
  functions are imported here. Calling this before any eval case runs makes a
  custom method that cannot be imported fail at once, rather than after every
  run has been scored.

  Args:
    multi_run_config: The multi-run config, or None if there is none.

  Returns:
    The methods, keyed by the names the rules use.

  Raises:
    ImportError: If a custom method's function cannot be imported.
    TypeError: If a custom method's name refers to an object that is not
      callable.
  """
  methods = get_built_in_aggregation_methods()
  if multi_run_config is not None and multi_run_config.custom_methods:
    for name, config in multi_run_config.custom_methods.items():
      methods[name] = load_custom_aggregation_method(name, config)
  return methods


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


def count_runs_failed_without_metric_results(
    per_case_results: Sequence[EvalCaseResult],
) -> int:
  """Returns how many of the runs crashed, failing without metric results.

  A run whose inferencing raised is recorded as FAILED with no metric results,
  overall or per invocation. Such a run has no verdict, so it is left out of
  every count of passing runs, and the case fails if more of its runs crashed
  than `MultiRunConfig.max_crashed_runs` allows. `AgentEvaluator`, which does
  not read that config, fails a case on any such run.
  """
  return sum(1 for result in per_case_results if _is_crashed(result))


def _get_overall_result(
    run: EvalCaseResult, metric_name: str
) -> EvalMetricResult | None:
  """Returns the run's overall result for the metric, or None if it has none."""
  return next(
      (
          metric_result
          for metric_result in run.overall_eval_metric_results
          if metric_result.metric_name == metric_name
      ),
      None,
  )


def _get_metric_run_result(
    run: EvalCaseResult, metric_name: str
) -> RunResult | None:
  """Returns one metric's result in one run, or None if the run lacks it.

  A run that crashed before it was scored has no result for the metric, and
  is left out rather than counted as a run the metric did not score.
  """
  invocation_scores = [
      metric_result.score
      for invocation in run.eval_metric_result_per_invocation
      for metric_result in invocation.eval_metric_results
      if metric_result.metric_name == metric_name
  ]
  overall = _get_overall_result(run, metric_name)
  if overall is None and not invocation_scores:
    return None
  return RunResult(
      status=overall.eval_status if overall else EvalStatus.NOT_EVALUATED,
      score=overall.score if overall else None,
      invocation_scores=invocation_scores,
  )


def _get_first_metric_results(
    runs: Sequence[EvalCaseResult],
) -> dict[str, EvalMetricResult]:
  """Returns the first result of each metric in the runs, keyed by its name.

  Both the overall and the per-invocation results are searched, so a metric
  with only one kind is still found. Metrics are in first-seen order, which is
  the order they were evaluated in. Each result supplies its metric's name,
  threshold and criterion.
  """
  first_results: dict[str, EvalMetricResult] = {}
  for run in runs:
    for metric_result in run.overall_eval_metric_results:
      first_results.setdefault(metric_result.metric_name, metric_result)
    for invocation in run.eval_metric_result_per_invocation:
      for metric_result in invocation.eval_metric_results:
        first_results.setdefault(metric_result.metric_name, metric_result)
  return first_results


def _compare(
    value: float | None, threshold: float, *, higher_is_better: bool = True
) -> EvalStatus:
  """Returns whether `value` meets `threshold`; NOT_EVALUATED without a value.

  A value meets the threshold when it is at least the threshold, or, if lower
  is better, when it is at most the threshold.
  """
  if value is None:
    return EvalStatus.NOT_EVALUATED
  meets_threshold = (
      value >= threshold if higher_is_better else value <= threshold
  )
  return EvalStatus.PASSED if meets_threshold else EvalStatus.FAILED


def _higher_is_better(
    runs: Sequence[EvalCaseResult], metric_name: str, threshold: float
) -> bool:
  """Returns whether the metric's evaluator treats higher scores as better.

  Each result of the metric that has a score other than the threshold, and
  that passed or failed, shows the direction its evaluator judged in: higher
  is better if a score above the threshold passed or one below it failed, and
  lower is better otherwise. The per-invocation results of every run are read,
  or, if none of them shows a direction, the runs' overall results, so that a
  metric scored only for the whole run is covered too. Lower is better only
  if every result read shows it. This mirrors `AgentEvaluator`.

  Args:
    runs: Every run of the eval case.
    metric_name: The metric.
    threshold: The metric's threshold in `criteria`.
  """

  def directions(results: Iterable[EvalMetricResult]) -> set[bool]:
    """Returns the directions the results show; True means higher is better."""
    return {
        (result.score > threshold) == (result.eval_status == EvalStatus.PASSED)
        for result in results
        if result.score is not None
        and result.score != threshold
        and result.eval_status in (EvalStatus.PASSED, EvalStatus.FAILED)
    }

  shown = directions(
      metric_result
      for run in runs
      for invocation in run.eval_metric_result_per_invocation
      for metric_result in invocation.eval_metric_results
      if metric_result.metric_name == metric_name
  ) or directions(
      overall
      for run in runs
      if (overall := _get_overall_result(run, metric_name)) is not None
  )
  return shown != {False}


def _compute(
    method: AggregationMethod,
    run_results: Sequence[RunResult],
    *,
    higher_is_better: bool,
) -> float | None:
  """Returns a method's value over one metric's runs.

  `min_over_runs` stands for the worst run's score, so for a metric where
  lower is better it takes the highest run score rather than the lowest. Every
  other method, custom ones included, gets the scores as they are.

  Args:
    method: The metric's aggregation method.
    run_results: The metric's result in each run that has one.
    higher_is_better: Whether higher scores are better for the metric.
  """
  if method.compute is min_over_runs and not higher_is_better:
    scores = [run.score for run in run_results if run.score is not None]
    return max(scores) if scores else None
  return method.compute(run_results)


def describe_run_statuses(
    statuses: Sequence[EvalStatus], *, crashed: int = 0
) -> str:
  """Returns how many of the runs passed, and how many were not evaluated.

  For example "2 of 3 runs passed", "1 of 3 runs passed, 1 not evaluated"
  when a run neither passed nor failed, or "3 of 4 runs passed, 1 crashed".

  Args:
    statuses: The status of each run, of one metric or of the whole eval case.
    crashed: How many of the runs crashed. Their statuses are NOT_EVALUATED,
      but they are counted as crashed rather than as not evaluated.
  """
  passed = statuses.count(EvalStatus.PASSED)
  decided = passed + statuses.count(EvalStatus.FAILED)
  not_evaluated = len(statuses) - decided - crashed
  description = f"{passed} of {len(statuses)} runs passed"
  if not_evaluated:
    description += f", {not_evaluated} not evaluated"
  if crashed:
    description += f", {crashed} crashed"
  return description


def _pass_count_note(
    statuses: Sequence[EvalStatus], value: float | None, *, crashed: int = 0
) -> str:
  """Returns how many runs passed, behind the value of a pass-count method.

  Without a value, it says instead how many runs passed or failed, since the
  method had too few of them to give one. `crashed` is as in
  `describe_run_statuses`.
  """
  if value is None:
    decided = statuses.count(EvalStatus.PASSED) + statuses.count(
        EvalStatus.FAILED
    )
    note = f"{decided} of {len(statuses)} runs passed or failed"
    return note + (f", {crashed} crashed" if crashed else "")
  return describe_run_statuses(statuses, crashed=crashed)


def _judge_metric(
    *,
    first_result: EvalMetricResult,
    runs: Sequence[EvalCaseResult],
    rule: str | MultiRunCriterion,
    methods: Mapping[str, AggregationMethod],
) -> tuple[EvalMetricResult, MethodResult]:
  """Returns a metric's overall result and how its rule judged the runs.

  Args:
    first_result: The metric's first result in the runs, which supplies its
      name, threshold and criterion.
    runs: Every run of the eval case, in run order.
    rule: The name of a method that combines scores, judged against the
      metric's threshold in the direction the metric's evaluator judged its
      scores, or a method that counts passing runs, with a threshold of its
      own.
    methods: Every method a rule can name, keyed by name.

  Returns:
    The metric's overall result, whose score is the method's value, and the
    method's result.
  """
  method: AggregationMethod
  threshold: float | None
  informational = _is_informational(first_result)
  if informational:
    # Such a metric has no threshold, and no passing runs to count, so a rule
    # that counts them reports the default method's value instead.
    method = (
        methods[rule] if isinstance(rule, str) else DEFAULT_AGGREGATION_METHOD
    )
    threshold = None
  elif isinstance(rule, str):
    method = methods[rule]
    threshold = _get_metric_threshold(first_result)
  elif isinstance(rule, MultiRunCriterion):
    method = methods[rule.method]
    threshold = rule.threshold
  else:
    raise TypeError(f"Unsupported aggregation rule: {rule!r}.")

  metric_name = first_result.metric_name
  # One entry per run, None for a run with no result for the metric, such as
  # one that crashed.
  results_by_run = [_get_metric_run_result(run, metric_name) for run in runs]
  run_results = [result for result in results_by_run if result is not None]
  # A case run once keeps the evaluator's own result for a metric unless a
  # rule that counts passing runs judges it: combining the scores of a single
  # run would only re-score it.
  own_result = (
      _get_overall_result(runs[0], metric_name)
      if len(runs) == 1 and (informational or isinstance(rule, str))
      else None
  )
  higher_is_better = True
  if own_result is not None:
    value = own_result.score
    status = own_result.eval_status
  else:
    if threshold is not None and isinstance(rule, str):
      # A rule that combines scores judges them the way the metric's
      # evaluator did, so a metric where lower is better keeps that. More
      # passing runs are always better, so a rule that counts them does not.
      higher_is_better = _higher_is_better(runs, metric_name, threshold)
    value = _compute(method, run_results, higher_is_better=higher_is_better)
    status = (
        EvalStatus.INFORMATIONAL
        if threshold is None
        else _compare(value, threshold, higher_is_better=higher_is_better)
    )

  note: str | None = None
  if threshold is not None and not method.uses_scores:
    # The note counts every run, so a run left out of the value, such as one
    # that crashed, still shows, as it does in the case's header.
    note = _pass_count_note(
        [
            result.status if result is not None else EvalStatus.NOT_EVALUATED
            for result in results_by_run
        ],
        value,
        crashed=count_runs_failed_without_metric_results(runs),
    )
  elif threshold is not None and value is None:
    # A built-in method, like a run's own result, has no value only when no
    # run has a score to give it; a custom one can have reasons of its own.
    note = (
        "the method returned None"
        if method.custom and own_result is None
        else "no run produced a score"
    )
  elif not higher_is_better:
    note = "lower is better"
  method_result = MethodResult(
      name=method.name,
      value=value,
      threshold=threshold,
      status=status,
      note=note,
  )
  if own_result is not None:
    return own_result, method_result

  metric_result = first_result.model_copy(deep=True)
  # The copy came from one run, or one invocation, so its details (rubric
  # scores, token breakdown) describe that alone, not the combined value.
  metric_result.details = EvalMetricResultDetails()
  metric_result.score = value
  metric_result.eval_status = status
  if isinstance(rule, MultiRunCriterion) and threshold is not None:
    # The value counts passing runs, so it is judged against the rule's own
    # threshold rather than the metric's.
    metric_result.threshold = threshold
  return metric_result, method_result


def _judge_cross_run_criteria(
    runs: Sequence[EvalCaseResult],
    cross_run_criteria: Sequence[MultiRunCriterion],
    methods: Mapping[str, AggregationMethod],
) -> list[MethodResult]:
  """Returns how each cross-run criterion judged the runs, each as a whole.

  Each run's status is `get_run_status`, so a run that crashed is undecided; a
  whole run has no score. `methods` holds every method a criterion can name,
  keyed by name.
  """
  statuses = [get_run_status(run) for run in runs]
  crashed = count_runs_failed_without_metric_results(runs)
  run_results = [RunResult(status=status) for status in statuses]
  method_results = []
  for criterion in cross_run_criteria:
    method = methods[criterion.method]
    value = method.compute(run_results)
    method_results.append(
        MethodResult(
            name=method.name,
            value=value,
            threshold=criterion.threshold,
            status=_compare(value, criterion.threshold),
            note=_pass_count_note(statuses, value, crashed=crashed),
        )
    )
  return method_results


def _aggregate_runs(
    *,
    eval_set_id: str,
    eval_id: str,
    runs: list[EvalCaseResult],
    multi_run_config: MultiRunConfig,
    methods: Mapping[str, AggregationMethod],
) -> MultiRunCaseResult:
  """Returns the verdict for one eval case, judged on its runs.

  Args:
    eval_set_id: The eval set id.
    eval_id: The eval case id.
    runs: Every run of the eval case, in run order.
    multi_run_config: How the runs are combined.
    methods: Every method the config's rules can name, keyed by name.
  """
  metric_rules = multi_run_config.cross_run_scoring_by_metric or {}
  overall_eval_metric_results: list[EvalMetricResult] = []
  metric_methods: dict[str, MethodResult] = {}
  for metric_name, first_result in _get_first_metric_results(runs).items():
    metric_result, method_result = _judge_metric(
        first_result=first_result,
        runs=runs,
        rule=metric_rules.get(metric_name, multi_run_config.cross_run_scoring),
        methods=methods,
    )
    overall_eval_metric_results.append(metric_result)
    metric_methods[metric_name] = method_result
  cross_run_results = _judge_cross_run_criteria(
      runs, multi_run_config.cross_run_criteria or [], methods
  )

  # The case must pass its metrics and every cross-run criterion. As with a
  # metric, a cross-run criterion that could not be evaluated is skipped.
  statuses = [
      _generate_final_eval_status(overall_eval_metric_results),
      *(cross_run_result.status for cross_run_result in cross_run_results),
  ]
  if EvalStatus.FAILED in statuses:
    final_eval_status = EvalStatus.FAILED
  elif EvalStatus.PASSED in statuses:
    final_eval_status = EvalStatus.PASSED
  else:
    final_eval_status = EvalStatus.NOT_EVALUATED

  # A crashed run has no verdict, so the metrics and cross-run criteria above
  # left it out. How many crashed runs the case tolerates is decided here.
  crashed_runs = count_runs_failed_without_metric_results(runs)
  if crashed_runs > multi_run_config.max_crashed_runs:
    final_eval_status = EvalStatus.FAILED
  elif crashed_runs == len(runs) and final_eval_status == EvalStatus.PASSED:
    # A case needs a run that did not crash to pass. With no such run, the
    # built-in methods give no value, but a custom cross-run criterion still
    # can.
    final_eval_status = EvalStatus.NOT_EVALUATED

  return MultiRunCaseResult(
      eval_set_id=eval_set_id,
      eval_id=eval_id,
      final_eval_status=final_eval_status,
      overall_eval_metric_results=overall_eval_metric_results,
      runs=runs,
      metric_methods=metric_methods,
      cross_run_results=cross_run_results,
      max_crashed_runs=multi_run_config.max_crashed_runs,
  )


def _run_order_key(result: EvalCaseResult) -> tuple[bool, int, str]:
  """Returns a sort key that orders the runs of one eval case by run index.

  Runs without a `repetition_index` (e.g. from results saved before the field
  existed) sort last, by session id, so the order is still deterministic.
  """
  return (
      result.repetition_index is None,
      result.repetition_index or 0,
      result.session_id or "",
  )


def aggregate_eval_case_results(
    eval_case_results: list[EvalCaseResult],
    *,
    multi_run_config: MultiRunConfig | None = None,
) -> list[MultiRunCaseResult]:
  """Aggregates per-run EvalCaseResults into one result per eval case.

  Results are grouped by (eval_set_id, eval_id); each group holds the runs for a
  single eval case. Each metric's results in every run are combined into one
  value by the metric's rule in `multi_run_config`. By default that is the
  mean over every invocation of every run, as in `AgentEvaluator`, and the
  metric passes when the mean meets its threshold, from below if the metric's
  evaluator judged lower scores better; a rule that counts the runs in which
  the metric passed has a threshold of its own. The metric statuses
  then combine into the case's status as they do for a single run of
  `LocalEvalService`: a metric that was never evaluated is skipped, where
  `AgentEvaluator` would fail it. Each cross-run criterion counts the runs that
  passed as a whole (see `get_run_status`), and the case fails if any of them
  fails. A run that crashed, failing without producing any metric results, is
  left out of every count; the case fails if more of its runs crashed than
  `max_crashed_runs` allows, and needs a run that did not crash to pass.

  A case run once keeps the evaluator's own overall results and status, so a
  plain single-run eval is not re-scored, and a crash fails it whatever
  `max_crashed_runs` is. Once a rule counts passing runs, such a case is
  judged like any other, `max_crashed_runs` included, but the metrics no such
  rule judges keep the run's own results.

  A rule can name a custom method declared in `multi_run_config`, which is
  imported first and then used as a built-in method of its kind would be.

  Args:
    eval_case_results: The per-run results to aggregate. Not modified.
    multi_run_config: How the runs of each eval case are combined. None means
      the defaults of `MultiRunConfig`.

  Returns:
    One result per eval case, sorted by (eval_set_id, eval_id).

  Raises:
    ImportError: If a custom method's function cannot be imported.
    TypeError: If a custom method's name refers to an object that is not
      callable, or its function returns something other than a number or
      None. An exception the function raises propagates as it is.
  """
  config = multi_run_config or MultiRunConfig()
  methods = load_aggregation_methods(config)
  judge_single_runs = has_pass_count_rules(config)
  results_by_case: dict[tuple[str, str], list[EvalCaseResult]] = {}
  for result in eval_case_results:
    key = (result.eval_set_id, result.eval_id)
    results_by_case.setdefault(key, []).append(result)

  aggregate_results: list[MultiRunCaseResult] = []
  for (eval_set_id, eval_id), per_case_results in results_by_case.items():
    if len(per_case_results) == 1 and not judge_single_runs:
      run = per_case_results[0]
      aggregate_results.append(
          MultiRunCaseResult(
              eval_set_id=eval_set_id,
              eval_id=eval_id,
              final_eval_status=run.final_eval_status,
              overall_eval_metric_results=run.overall_eval_metric_results,
              runs=[run],
          )
      )
      continue

    # Runs arrive in completion order, so put them back in run order. This
    # makes the order of `runs` deterministic.
    aggregate_results.append(
        _aggregate_runs(
            eval_set_id=eval_set_id,
            eval_id=eval_id,
            runs=sorted(per_case_results, key=_run_order_key),
            multi_run_config=config,
            methods=methods,
        )
    )

  return sorted(aggregate_results, key=lambda x: (x.eval_set_id, x.eval_id))
