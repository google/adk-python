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

"""Tests for the aggregation methods that combine an eval case's runs."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import re
import statistics

from google.adk.agents.common_configs import CodeConfig
from google.adk.evaluation._multi_run_aggregation_methods import DEFAULT_AGGREGATION_METHOD
from google.adk.evaluation._multi_run_aggregation_methods import get_built_in_aggregation_methods
from google.adk.evaluation._multi_run_aggregation_methods import load_custom_aggregation_method
from google.adk.evaluation._multi_run_aggregation_methods import mean_of_runs
from google.adk.evaluation._multi_run_aggregation_methods import mean_over_invocations
from google.adk.evaluation._multi_run_aggregation_methods import min_over_runs
from google.adk.evaluation._multi_run_aggregation_methods import pass_at_k
from google.adk.evaluation._multi_run_aggregation_methods import pass_hat_k
from google.adk.evaluation._multi_run_aggregation_methods import pass_rate
from google.adk.evaluation._multi_run_aggregation_methods import RunResult
from google.adk.evaluation.eval_config import CustomCrossRunMethodConfig
from google.adk.evaluation.evaluator import EvalStatus
import pytest


def _run(*invocation_scores: float | None) -> RunResult:
  """Returns one run's result for a metric with the given invocation scores."""
  return RunResult(
      status=EvalStatus.PASSED, invocation_scores=list(invocation_scores)
  )


def _scored_run(score: float | None) -> RunResult:
  """Returns one run's result for a metric with the given overall score."""
  return RunResult(status=EvalStatus.PASSED, score=score)


def _judged_run(status: EvalStatus) -> RunResult:
  """Returns one run's result for a target that was judged `status`."""
  return RunResult(status=status)


def test_mean_over_invocations_weighs_every_invocation_equally():
  """Every invocation counts once, so longer runs weigh more in the mean."""
  runs = [_run(0.0), _run(1.0, 1.0, 1.0)]

  score = mean_over_invocations(runs)

  assert score == pytest.approx(0.75)


def test_mean_over_invocations_skips_unscored_invocations():
  """An invocation the metric did not score does not affect the mean."""
  assert mean_over_invocations([_run(0.5, None), _run(1.0)]) == pytest.approx(
      0.75
  )


def test_mean_over_invocations_counts_a_run_without_turn_scores_once():
  """A run with no per-invocation score adds its overall score, once.

  Setup: one run has turn scores 0.0 and 1.0, and an overall score of 0.9;
    the other has no turn score and an overall score of 0.25.
  Assert: the mean is over 0.0, 1.0 and 0.25. The first run's overall score
    is not used, since it has turn scores.
  """
  runs = [
      RunResult(
          status=EvalStatus.PASSED, score=0.9, invocation_scores=[0.0, 1.0]
      ),
      RunResult(status=EvalStatus.FAILED, score=0.25),
  ]

  assert mean_over_invocations(runs) == pytest.approx(1.25 / 3)


@pytest.mark.parametrize(
    "runs",
    [[], [_run()], [_run(None, None)]],
    ids=["no_runs", "no_invocations", "no_scored_invocations"],
)
def test_mean_over_invocations_is_none_without_scores(runs):
  """With no scored invocation there is no mean to compare with a threshold."""
  assert mean_over_invocations(runs) is None


def test_mean_of_runs_weighs_every_run_equally():
  """Every run counts once, however many invocations it has."""
  runs = [
      RunResult(status=EvalStatus.FAILED, score=0.0, invocation_scores=[0.0]),
      RunResult(
          status=EvalStatus.PASSED,
          score=1.0,
          invocation_scores=[1.0, 1.0, 1.0],
      ),
  ]

  score = mean_of_runs(runs)

  assert score == pytest.approx(0.5)


def test_mean_of_runs_skips_runs_without_a_score():
  """A run the metric did not score does not affect the mean."""
  runs = [_scored_run(0.5), _scored_run(None), _scored_run(1.0)]

  assert mean_of_runs(runs) == pytest.approx(0.75)


def test_min_over_runs_is_the_lowest_run_score():
  """The worst run decides the value."""
  runs = [_scored_run(0.9), _scored_run(0.2), _scored_run(0.7)]

  assert min_over_runs(runs) == pytest.approx(0.2)


def test_min_over_runs_skips_runs_without_a_score():
  """A run the metric did not score is not taken as the lowest."""
  runs = [_scored_run(None), _scored_run(0.6)]

  assert min_over_runs(runs) == pytest.approx(0.6)


@pytest.mark.parametrize("method", [mean_of_runs, min_over_runs])
@pytest.mark.parametrize(
    "runs",
    [[], [_scored_run(None), _scored_run(None)]],
    ids=["no_runs", "no_scored_runs"],
)
def test_run_score_methods_are_none_without_run_scores(method, runs):
  """With no run score there is no value to compare with a threshold."""
  assert method(runs) is None


def test_pass_rate_is_the_fraction_of_runs_that_passed():
  """Two passes out of three runs give 2/3."""
  runs = [
      _judged_run(EvalStatus.PASSED),
      _judged_run(EvalStatus.FAILED),
      _judged_run(EvalStatus.PASSED),
  ]

  assert pass_rate(runs) == pytest.approx(2 / 3)


def test_pass_rate_leaves_out_runs_that_were_not_evaluated():
  """A run that neither passed nor failed counts in neither total."""
  runs = [
      _judged_run(EvalStatus.PASSED),
      _judged_run(EvalStatus.NOT_EVALUATED),
      _judged_run(EvalStatus.FAILED),
  ]

  assert pass_rate(runs) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "runs",
    [
        [],
        [
            _judged_run(EvalStatus.NOT_EVALUATED),
            _judged_run(EvalStatus.NOT_EVALUATED),
        ],
    ],
    ids=["no_runs", "no_decided_runs"],
)
def test_pass_rate_is_none_without_a_decided_run(runs):
  """With no run that passed or failed there is no rate to compare."""
  assert pass_rate(runs) is None


def test_default_aggregation_method_is_a_built_in_method():
  """The default method is the built-in method of the same name."""
  assert (
      get_built_in_aggregation_methods()["mean_over_invocations"]
      is DEFAULT_AGGREGATION_METHOD
  )


@pytest.mark.parametrize("num_decided", range(1, 7))
def test_pass_k_methods_match_enumeration_of_every_k_subset(num_decided):
  """pass@k and pass^k equal the share of k-run subsets with any or all passes."""
  for num_passed in range(num_decided + 1):
    outcomes = [True] * num_passed + [False] * (num_decided - num_passed)
    runs = [
        _judged_run(EvalStatus.PASSED if passed else EvalStatus.FAILED)
        for passed in outcomes
    ]
    for k in range(1, num_decided + 1):
      subsets = list(itertools.combinations(outcomes, k))

      assert pass_at_k(runs, k) == pytest.approx(
          sum(any(subset) for subset in subsets) / len(subsets)
      )
      assert pass_hat_k(runs, k) == pytest.approx(
          sum(all(subset) for subset in subsets) / len(subsets)
      )


def test_pass_k_methods_at_k_one_equal_the_pass_rate():
  """At k = 1, pass@k and pass^k are both the pass rate."""
  runs = [_judged_run(EvalStatus.PASSED)] * 3 + [_judged_run(EvalStatus.FAILED)]

  assert pass_at_k(runs, 1) == pytest.approx(pass_rate(runs))
  assert pass_hat_k(runs, 1) == pytest.approx(pass_rate(runs))


def test_pass_k_methods_leave_out_runs_that_were_not_evaluated():
  """An undecided run is left out of n, as in pass_rate."""
  runs = [
      _judged_run(EvalStatus.PASSED),
      _judged_run(EvalStatus.NOT_EVALUATED),
      _judged_run(EvalStatus.PASSED),
      _judged_run(EvalStatus.FAILED),
  ]

  # Three decided runs, two of which passed.
  assert pass_at_k(runs, 2) == pytest.approx(1.0)
  assert pass_hat_k(runs, 2) == pytest.approx(1 / 3)


@pytest.mark.parametrize("method", [pass_at_k, pass_hat_k])
def test_pass_k_methods_are_none_with_fewer_than_k_decided_runs(method):
  """Fewer than k decided runs cannot support an estimate of k runs."""
  runs = [_judged_run(EvalStatus.PASSED), _judged_run(EvalStatus.NOT_EVALUATED)]

  assert method(runs, 2) is None


@pytest.mark.parametrize(
    ("name", "display_name", "compute_with_k"),
    [("pass_at_k", "pass@3", pass_at_k), ("pass_hat_k", "pass^3", pass_hat_k)],
)
def test_pass_k_method_with_k_is_named_and_computed_for_k(
    name, display_name, compute_with_k
):
  """A pass@k method needs k, and binding it names the method for k."""
  method = get_built_in_aggregation_methods()[name]
  runs = [_judged_run(EvalStatus.PASSED)] * 3 + [_judged_run(EvalStatus.FAILED)]

  bound = method.with_k(3)

  assert method.uses_scores is False
  with pytest.raises(TypeError, match="needs k"):
    method.compute(runs)
  assert bound.name == display_name
  assert bound.compute(runs) == pytest.approx(compute_with_k(runs, 3))


def test_with_k_on_a_method_without_k_raises_type_error():
  """Only a method that looks at k runs at a time takes k."""
  with pytest.raises(TypeError, match="does not take k"):
    get_built_in_aggregation_methods()["pass_rate"].with_k(2)


@pytest.mark.parametrize("k", [0, -1])
@pytest.mark.parametrize("method", [pass_at_k, pass_hat_k])
def test_pass_k_methods_reject_k_below_one(method, k):
  """k = 0 would give a misleading value and k < 0 would divide by zero."""
  with pytest.raises(ValueError, match=f"k must be at least 1, got {k}."):
    method([], k)


@pytest.mark.parametrize("k", [0, -1])
def test_with_k_rejects_k_below_one(k):
  """Binding a k below 1 fails at once, not when the method is computed."""
  with pytest.raises(ValueError, match=f"k must be at least 1, got {k}."):
    get_built_in_aggregation_methods()["pass_hat_k"].with_k(k)


@pytest.mark.parametrize(
    ("name", "compute", "uses_scores"),
    [
        ("mean_of_runs", mean_of_runs, True),
        ("min_over_runs", min_over_runs, True),
        ("pass_rate", pass_rate, False),
    ],
)
def test_built_in_aggregation_method_is_keyed_by_its_name(
    name, compute, uses_scores
):
  """Each built-in method is listed under its name, with the kind it is."""
  method = get_built_in_aggregation_methods()[name]

  assert (method.name, method.compute, method.uses_scores) == (
      name,
      compute,
      uses_scores,
  )


# A custom method is imported by its fully qualified name, so the functions
# the tests below load must be attributes of this module.


def _median_of_run_scores(runs: Sequence[RunResult]) -> float | None:
  scores = [run.score for run in runs if run.score is not None]
  return statistics.median(scores) if scores else None


def _returns_three(runs: Sequence[RunResult]) -> int:
  del runs
  return 3


def _returns_true(runs: Sequence[RunResult]) -> bool:
  del runs
  return True


def _returns_text(runs: Sequence[RunResult]) -> str:
  del runs
  return "0.5"


def _raises(runs: Sequence[RunResult]) -> float | None:
  del runs
  raise RuntimeError("custom method failed")


_NOT_A_FUNCTION = 0.5


def _custom_method_config(
    function_name: str, *, uses_scores: bool = True
) -> CustomCrossRunMethodConfig:
  """Returns the config of a custom method defined in this module."""
  return CustomCrossRunMethodConfig(
      code_config=CodeConfig(name=f"{__name__}.{function_name}"),
      uses_scores=uses_scores,
  )


@pytest.mark.parametrize("uses_scores", [True, False])
def test_custom_method_is_loaded_under_its_config_name(uses_scores):
  """A loaded method has the config's name and kind, and runs the function."""
  config = _custom_method_config(
      _median_of_run_scores.__name__, uses_scores=uses_scores
  )
  runs = [_scored_run(0.2), _scored_run(0.9), _scored_run(0.4)]

  method = load_custom_aggregation_method("median_of_runs", config)

  assert (method.name, method.uses_scores, method.compute(runs)) == (
      "median_of_runs",
      uses_scores,
      0.4,
  )


@pytest.mark.parametrize(
    ("function_path", "cause"),
    [
        ("no_such_module_for_adk_tests.median", ModuleNotFoundError),
        (f"{__name__}._no_such_function", AttributeError),
    ],
    ids=["unknown_module", "unknown_function"],
)
def test_custom_method_that_cannot_be_imported_raises_import_error(
    function_path, cause
):
  """The error names the method and the path, and is chained from its cause."""
  config = CustomCrossRunMethodConfig(
      code_config=CodeConfig(name=function_path), uses_scores=True
  )

  with pytest.raises(
      ImportError,
      match=re.escape(
          f"Could not import custom method 'median' from {function_path!r}"
      ),
  ) as exc_info:
    load_custom_aggregation_method("median", config)

  assert isinstance(exc_info.value.__cause__, cause)


def test_custom_method_that_is_not_callable_raises_type_error():
  """A name that refers to a value rather than a function is rejected."""
  config = _custom_method_config("_NOT_A_FUNCTION")

  with pytest.raises(
      TypeError,
      match=re.escape(
          f"'median' refers to '{__name__}._NOT_A_FUNCTION', which is not"
          " callable"
      ),
  ):
    load_custom_aggregation_method("median", config)


@pytest.mark.parametrize(
    "function_name",
    [_returns_true.__name__, _returns_text.__name__],
    ids=["bool", "str"],
)
def test_custom_method_returning_a_non_number_raises_type_error(
    function_name,
):
  """Only a number or None can be compared with a threshold."""
  method = load_custom_aggregation_method(
      "passed", _custom_method_config(function_name)
  )

  with pytest.raises(TypeError, match="Custom method 'passed' returned"):
    method.compute([])


def test_custom_method_int_result_is_returned_as_a_float():
  """An int is a number like any other, and comes back as a float."""
  method = load_custom_aggregation_method(
      "three", _custom_method_config(_returns_three.__name__)
  )

  value = method.compute([])

  assert (value, type(value)) == (3.0, float)


def test_custom_method_error_propagates():
  """An exception the custom function raises reaches the caller as it is."""
  method = load_custom_aggregation_method(
      "broken", _custom_method_config(_raises.__name__)
  )

  with pytest.raises(RuntimeError, match="custom method failed"):
    method.compute([])


def test_run_result_is_exported_from_the_evaluation_package():
  """Custom methods receive RunResult, so it is importable from the package."""
  from google.adk.evaluation import RunResult as PublicRunResult

  assert PublicRunResult is RunResult
