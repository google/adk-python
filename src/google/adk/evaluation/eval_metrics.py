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

from google.genai import types as genai_types
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import TypeAlias

from .eval_case import Invocation
from .evaluator import EvalStatus


class PrebuiltMetrics(Enum):
  TOOL_TRAJECTORY_AVG_SCORE = "tool_trajectory_avg_score"

  RESPONSE_EVALUATION_SCORE = "response_evaluation_score"

  RESPONSE_MATCH_SCORE = "response_match_score"

  SAFETY_V1 = "safety_v1"

  FINAL_RESPONSE_MATCH_V2 = "final_response_match_v2"


MetricName: TypeAlias = Union[str, PrebuiltMetrics]


class JudgeModelOptions(BaseModel):
  """Options for an eval metric's judge model."""

  judge_model: str = Field(
      default="gemini-2.5-flash",
      description=(
          "The judge model to use for evaluation. It can be a model name."
      ),
  )

  judge_model_config: Optional[genai_types.GenerateContentConfig] = Field(
      default=None,
      description="The configuration for the judge model.",
  )

  num_samples: Optional[int] = Field(
      default=None,
      description=(
          "The number of times to sample the model for each invocation"
          " evaluation."
      ),
  )


class EvalMetric(BaseModel):
  """A metric used to evaluate a particular aspect of an eval case."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  metric_name: str = Field(
      description="The name of the metric.",
  )

  threshold: float = Field(
      description=(
          "A threshold value. Each metric decides how to interpret this"
          " threshold."
      ),
  )

  judge_model_options: Optional[JudgeModelOptions] = Field(
      default=None,
      description="Options for the judge model.",
  )


class EvalMetricResult(EvalMetric):
  """The actual computed score/value of a particular EvalMetric."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  score: Optional[float] = Field(
      default=None,
      description=(
          "Score obtained after evaluating the metric. Optional, as evaluation"
          " might not have happened."
      ),
  )
  eval_status: EvalStatus = Field(description="The status of this evaluation.")


class EvalMetricResultPerInvocation(BaseModel):
  """Eval metric results per invocation."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  actual_invocation: Invocation = Field(
      description=(
          "The actual invocation, usually obtained by inferencing the agent."
      )
  )

  expected_invocation: Invocation = Field(
      description=(
          "The expected invocation, usually the reference or golden invocation."
      )
  )

  eval_metric_results: list[EvalMetricResult] = Field(
      default=[],
      description="Eval resutls for each applicable metric.",
  )


class Interval(BaseModel):
  """Represents a range of numeric values, e.g. [0 ,1] or (2,3) or [-1, 6)."""

  min_value: float = Field(description="The smaller end of the interval.")

  open_at_min: bool = Field(
      default=False,
      description=(
          "The interval is Open on the min end. The default value is False,"
          " which means that we assume that the interval is Closed."
      ),
  )

  max_value: float = Field(description="The larger end of the interval.")

  open_at_max: bool = Field(
      default=False,
      description=(
          "The interval is Open on the max end. The default value is False,"
          " which means that we assume that the interval is Closed."
      ),
  )


class MetricValueInfo(BaseModel):
  """Information about the type of metric value."""

  interval: Optional[Interval] = Field(
      default=None,
      description="The values represented by the metric are of type interval.",
  )


class MetricInfo(BaseModel):
  """Information about the metric that are used for Evals."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  metric_name: str = Field(description="The name of the metric.")

  description: str = Field(
      default=None, description="A 2 to 3 line description of the metric."
  )

  metric_value_info: MetricValueInfo = Field(
      description="Information on the nature of values supported by the metric."
  )
