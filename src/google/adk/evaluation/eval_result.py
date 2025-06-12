# Copyright 2025 Google LLC
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
  creation_timestamp: float = 0.0
