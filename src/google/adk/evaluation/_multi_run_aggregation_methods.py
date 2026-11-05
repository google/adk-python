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

"""Aggregation methods: ways to combine an eval case's runs into one number.

When an eval case is run more than once, a target (one metric, or the whole
eval case) has a result in every run. An aggregation method turns those
results into a single number. It only computes: comparing the number with a
threshold, and deciding the case's verdict, is the aggregator's job.

A method either combines scores, so it applies only to a metric, or counts the
runs that passed, so it applies to a metric or to the whole eval case. Besides
the built-in methods, an eval config can declare custom ones, each a Python
function that is imported by its fully qualified name.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import importlib
import numbers
import statistics
from typing import TYPE_CHECKING

from .evaluator import EvalStatus

if TYPE_CHECKING:
  from .eval_config import CustomCrossRunMethodConfig


@dataclasses.dataclass(frozen=True)
class RunResult:
  """One run's result for one target: a metric, or the whole eval case.

  A cross-run method, built-in or custom, receives the results of an eval
  case's runs, in run order, and combines them into one number. A run that
  crashed before it was scored has no verdict: for a metric it has no result
  and is left out, and for the whole eval case its status is NOT_EVALUATED.
  """

  status: EvalStatus
  """The target's eval status in this run."""

  score: float | None = None
  """The run's overall score for a metric. None for the whole eval case."""

  invocation_scores: Sequence[float | None] = ()
  """The metric's score for each invocation of the run, None where unscored.

  Empty for the whole eval case.
  """


AggregationMethodFn = Callable[[Sequence[RunResult]], float | None]
"""Combines a target's runs into one number, or None if the runs allow none."""


@dataclasses.dataclass(frozen=True)
class AggregationMethod:
  """A named way to combine a target's runs into one number."""

  name: str
  """The name shown in output, e.g. "mean_over_invocations"."""

  compute: AggregationMethodFn
  """Returns the number, or None when the runs can't support a value."""

  uses_scores: bool
  """True if it reads scores, so it applies to a metric but not a whole case."""

  custom: bool = False
  """True for a method declared in the eval config rather than built in."""


def mean_over_invocations(runs: Sequence[RunResult]) -> float | None:
  """Returns the mean score over every invocation of every run.

  Every invocation counts once, so a run with more invocations weighs more
  than a run with fewer. A run with no per-invocation score, such as one
  scored only for the whole conversation, counts once, with its overall score.

  Args:
    runs: One metric's result in each run of an eval case.

  Returns:
    The mean of the invocation scores that are not None, together with the
    overall scores of the runs that have no such invocation score, or None if
    there are none.
  """
  scores: list[float] = []
  for run in runs:
    invocation_scores = [
        score for score in run.invocation_scores if score is not None
    ]
    if invocation_scores:
      scores.extend(invocation_scores)
    elif run.score is not None:
      scores.append(run.score)
  return statistics.mean(scores) if scores else None


def mean_of_runs(runs: Sequence[RunResult]) -> float | None:
  """Returns the mean of the runs' overall scores.

  Every run counts once, however many invocations it has.

  Args:
    runs: One metric's result in each run of an eval case.

  Returns:
    The mean of the run scores that are not None, or None if there are none.
  """
  scores = [run.score for run in runs if run.score is not None]
  return statistics.mean(scores) if scores else None


def min_over_runs(runs: Sequence[RunResult]) -> float | None:
  """Returns the lowest of the runs' overall scores.

  The `min_over_runs` method stands for the worst run's score: the lowest, or
  the highest for a metric where lower is better, which the aggregator takes
  instead, since only it knows the metric's direction. Compared with the
  metric's threshold, it passes only if every run that has a score met the
  threshold.

  Args:
    runs: One metric's result in each run of an eval case.

  Returns:
    The lowest run score that is not None, or None if there is none.
  """
  scores = [run.score for run in runs if run.score is not None]
  return min(scores) if scores else None


def pass_rate(runs: Sequence[RunResult]) -> float | None:
  """Returns the fraction of the decided runs that passed.

  A run is decided when its status is PASSED or FAILED. Any other status, such
  as NOT_EVALUATED, carries no pass or fail signal, so that run is left out.

  Args:
    runs: One target's result in each run of an eval case: a metric's, or the
      whole eval case's.

  Returns:
    The number of passed runs divided by the number of decided runs, or None
    if no run was decided.
  """
  decided = [
      run.status
      for run in runs
      if run.status in (EvalStatus.PASSED, EvalStatus.FAILED)
  ]
  if not decided:
    return None
  return decided.count(EvalStatus.PASSED) / len(decided)


DEFAULT_AGGREGATION_METHOD = AggregationMethod(
    name="mean_over_invocations",
    compute=mean_over_invocations,
    uses_scores=True,
)
"""The method each metric uses unless configured otherwise."""

_BUILT_IN_AGGREGATION_METHODS: dict[str, AggregationMethod] = {
    method.name: method
    for method in (
        DEFAULT_AGGREGATION_METHOD,
        AggregationMethod(
            name="mean_of_runs", compute=mean_of_runs, uses_scores=True
        ),
        AggregationMethod(
            name="min_over_runs", compute=min_over_runs, uses_scores=True
        ),
        AggregationMethod(
            name="pass_rate", compute=pass_rate, uses_scores=False
        ),
    )
}


def get_built_in_aggregation_methods() -> dict[str, AggregationMethod]:
  """Returns a new dict of the built-in aggregation methods, keyed by name."""
  return dict(_BUILT_IN_AGGREGATION_METHODS)


def load_custom_aggregation_method(
    name: str, config: CustomCrossRunMethodConfig
) -> AggregationMethod:
  """Imports a custom method declared in an eval config.

  Args:
    name: The method's key in `custom_methods`: the name the rules refer to it
      by, and the name shown in output.
    config: The method's config, which names its function.

  Returns:
    The method. Its `compute` calls the function and returns the function's
    value as a float, or None. It raises TypeError if the function returns
    anything else, a bool included; an exception the function raises
    propagates.

  Raises:
    ImportError: If the function's module, or the function in it, cannot be
      found.
    TypeError: If the name refers to an object that is not callable.
  """
  function_path = config.code_config.name
  try:
    module_name, function_name = function_path.rsplit(".", 1)
    function: object = getattr(
        importlib.import_module(module_name), function_name
    )
  except (ImportError, AttributeError, ValueError) as e:
    # A caller may show only the message, not its cause, so the message
    # carries the cause's text.
    raise ImportError(
        f"Could not import custom method {name!r} from {function_path!r}: {e}"
    ) from e
  if not callable(function):
    raise TypeError(
        f"Custom method {name!r} refers to {function_path!r},"
        " which is not callable."
    )
  return AggregationMethod(
      name=name,
      compute=_with_result_check(name, function),
      uses_scores=config.uses_scores,
      custom=True,
  )


def _with_result_check(
    name: str, function: Callable[[Sequence[RunResult]], object]
) -> AggregationMethodFn:
  """Returns `function`, wrapped to check that it returns a number or None.

  Args:
    name: The custom method's name, for the error message.
    function: The custom method's function.
  """

  def compute(runs: Sequence[RunResult]) -> float | None:
    value = function(runs)
    if value is None:
      return None
    # A bool is an int, but True is a verdict rather than a value to compare
    # with a threshold.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
      raise TypeError(
          f"Custom method {name!r} returned {value!r} of type"
          f" {type(value).__name__}; it must return a number or None."
      )
    return float(value)

  return compute
