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

from enum import Enum
from typing import Optional
from typing import Union

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from typing_extensions import TypeAlias

from .eval_case import Invocation
from .evaluator import EvalStatus


class MetricNames(Enum):
  TOOL_TRAJECTORY_AVG_SCORE = "tool_trajectory_avg_score"

  # This is an alias of TOOL_TRAJECTORY_AVG_SCORE.
  TOOL_TRAJECTORY_MATCH_V1 = "tool_trajectory_match_v1"

  RESPONSE_EVALUATION_SCORE = "response_evaluation_score"

  RESPONSE_MATCH_SCORE = "response_match_score"

  # This is an alias of RESPONSE_MATCH_SCORE.
  FINAL_RESPONSE_MATCH_V1 = "final_response_match_v1"


MetricName: TypeAlias = Union[str, MetricNames]


class EvalMetric(BaseModel):
  """A metric used to evaluate a particular aspect of an eval case."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  metric_name: str
  """The name of the metric."""

  threshold: float
  """A threshold value. Each metric decides how to interpret this threshold."""


class EvalMetricResult(EvalMetric):
  """The actual computed score/value of a particular EvalMetric."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  score: Optional[float] = None
  eval_status: EvalStatus


class EvalMetricResultPerInvocation(BaseModel):
  """Eval metric results per invocation."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  actual_invocation: Invocation
  """The actual invocation, usually obtained by inferencing the agent."""

  expected_invocation: Invocation
  """The expected invocation, usually the reference or golden invocation."""

  eval_metric_results: list[EvalMetricResult] = []
  """Eval resutls for each applicable metric."""
