# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..sessions.session import Session
from .eval_metrics import EvalMetric
from .eval_metrics import EvalMetricResult
from .eval_metrics import EvalMetricResultPerInvocation
from .evaluator import EvalStatus


class EvalCaseResult(BaseModel):
  """Case level evaluation results."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  eval_set_file: Optional[str] = Field(
      deprecated=True,
      default=None,
      description="This field is deprecated, use eval_set_id instead.",
  )
  eval_set_id: str = ""
  """The eval set id."""

  eval_id: str = ""
  """The eval case id."""

  final_eval_status: EvalStatus
  """Final eval status for this eval case."""

  eval_metric_results: Optional[list[tuple[EvalMetric, EvalMetricResult]]] = (
      Field(
          deprecated=True,
          default=None,
          description=(
              "This field is deprecated, use overall_eval_metric_results"
              " instead."
          ),
      )
  )

  overall_eval_metric_results: list[EvalMetricResult]
  """Overall result for each metric for the entire eval case."""

  eval_metric_result_per_invocation: list[EvalMetricResultPerInvocation]
  """Result for each metric on a per invocation basis."""

  session_id: str
  """Session id of the session generated as result of inferencing/scraping stage of the eval."""

  session_details: Optional[Session] = None
  """Session generated as result of inferencing/scraping stage of the eval."""

  user_id: Optional[str] = None
  """User id used during inferencing/scraping stage of the eval."""

  repetition_index: Optional[int] = None
  """0-based index of the run this result is for, among the runs of the same
  eval case (see `InferenceConfig.num_runs`). None if it was not recorded.
  """


class MethodResult(BaseModel):
  """How one rule judged the runs of an eval case.

  The rule's target is one metric, or the whole eval case.
  """

  model_config = ConfigDict(
      frozen=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  name: str
  """The name of the aggregation method, e.g. "mean_over_invocations"."""

  value: Optional[float]
  """The method's value over the runs, or None if the runs gave it none."""

  threshold: Optional[float]
  """The value needed to pass. None for a metric that never passes or fails."""

  status: EvalStatus
  """PASSED or FAILED by comparing the value with the threshold.

  The value passes at or above the threshold, or at or below it for a metric
  whose evaluator treats lower scores as better. NOT_EVALUATED when there is
  no value, and INFORMATIONAL for a metric that never passes or fails, or for
  a cross-run criterion without a threshold, which only reports its value.
  """

  note: Optional[str] = None
  """What the value rests on, such as how many runs passed, or why it is None.

  For a metric judged lower-is-better, with nothing else to say, it says so.
  """


class MultiRunCaseSummary(BaseModel):
  """Persisted cross-run summary for one eval case across all of its runs."""

  model_config = ConfigDict(
      frozen=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  eval_set_id: str = ""
  """The eval set id."""

  eval_id: str = ""
  """The eval case id."""

  final_eval_status: EvalStatus
  """The eval case's final status across all of its runs."""

  num_runs: int
  """Total number of runs recorded for this eval case."""

  passed_runs: int
  """Number of runs that passed as a whole (`get_run_status == PASSED`)."""

  failed_runs: int
  """Number of runs that failed as a whole (`get_run_status == FAILED`)."""

  crashed_runs: int = 0
  """Number of runs that failed without producing any metric results."""

  not_evaluated_runs: int = 0
  """Number of non-crashed runs whose whole-run status was NOT_EVALUATED."""

  pass_rate: Optional[float] = None
  """Fraction of decided (passed or failed) runs that passed, or None if no run
  was decided.
  """

  overall_eval_metric_results: list[EvalMetricResult] = Field(
      default_factory=list
  )
  """One result per metric, aggregated across all runs."""

  metric_methods: dict[str, MethodResult] = Field(default_factory=dict)
  """How each metric was judged on the runs, keyed by metric name."""

  cross_run_results: list[MethodResult] = Field(default_factory=list)
  """How each configured cross-run criterion judged the runs, in order."""

  max_crashed_runs: int = 0
  """How many runs may crash before the case fails."""


class EvalSetResult(BaseModel):
  """Eval set level evaluation results."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  eval_set_result_id: str
  eval_set_result_name: Optional[str] = None
  eval_set_id: str
  eval_case_results: list[EvalCaseResult] = Field(default_factory=list)
  multi_run_case_summaries: Optional[list[MultiRunCaseSummary]] = None
  creation_timestamp: float = 0.0
