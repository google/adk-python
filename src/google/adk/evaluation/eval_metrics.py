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

from .common import EvalBaseModel
from .eval_case import Invocation
from .eval_rubrics import Rubric
from .eval_rubrics import RubricScore


class EvalStatus(Enum):
  PASSED = 1
  FAILED = 2
  NOT_EVALUATED = 3


class PrebuiltMetrics(Enum):
  TOOL_TRAJECTORY_AVG_SCORE = "tool_trajectory_avg_score"

  RESPONSE_EVALUATION_SCORE = "response_evaluation_score"

  RESPONSE_MATCH_SCORE = "response_match_score"

  SAFETY_V1 = "safety_v1"

  FINAL_RESPONSE_MATCH_V2 = "final_response_match_v2"

  RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1 = (
      "rubric_based_final_response_quality_v1"
  )

  HALLUCINATIONS_V1 = "hallucinations_v1"

  RUBRIC_BASED_TOOL_USE_QUALITY_V1 = "rubric_based_tool_use_quality_v1"


MetricName: TypeAlias = Union[str, PrebuiltMetrics]
Threshold: TypeAlias = float


class JudgeModelOptions(EvalBaseModel):
  """Options for an eval metric's judge model."""

  judge_model: str = Field(
      default="gemini-2.5-flash",
      description=(
          "The judge model to use for evaluation. It can be a model name."
      ),
  )

  judge_model_config: Optional[genai_types.GenerateContentConfig] = Field(
      default=genai_types.GenerateContentConfig,
      description="The configuration for the judge model.",
  )

  num_samples: int = Field(
      default=5,
      description=(
          "The number of times to sample the model for each invocation"
          " evaluation. Given that models tend to have certain degree of"
          " unreliability to them, we repeatedly sample them with the same"
          " data. These repeated invocation are them aggregated using some"
          " strategy. From experimentation, we have found 5 to be a good"
          " default."
      ),
  )


class BaseCriterion(BaseModel):
  """Base creterion to use for an Eval Metric."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      extra="allow",
  )

  threshold: Threshold = Field(
      description="The threshold to be used by the metric.",
  )


class LlmAsAJudgeCriterion(BaseCriterion):
  """Criterion when using LLM-As-A-Judge metric."""

  judge_model_options: JudgeModelOptions = Field(
      default_factory=JudgeModelOptions,
      description="Options for the judge model.",
  )


class RubricsBasedCriterion(BaseCriterion):
  """Criterion when using a rubric based metric."""

  judge_model_options: JudgeModelOptions = Field(
      default_factory=JudgeModelOptions,
      description="Options for the judge model.",
  )

  rubrics: list[Rubric] = Field(
      default_factory=list,
      description=(
          "Rubrics to be used by Metric. Not all metrics rely on rubrics, but"
          " metrics like `rubric_based_final_response_quality_v1` do. Metrics"
          " that don't use Rubrics, will just ignore this field, if specified."
          " Metrics that do use rubrics will raise an execption, if they are"
          " not specified."
      ),
  )


class HallucinationsCriterion(BaseCriterion):
  """Criterion to use when evaluating agents response for hallucinations."""

  judge_model_options: JudgeModelOptions = Field(
      default_factory=JudgeModelOptions,
      description="Options for the judge model.",
  )

  evaluate_intermediate_nl_responses: bool = Field(
      default=False,
      description=(
          "Whether any intermediate NL responses should be evaluated"
          " for hallucinations or not. By default, the metric only evaluates"
          " final response from the Agent for hallucinations."
      ),
  )


class EvalMetric(EvalBaseModel):
  """A metric used to evaluate a particular aspect of an eval case."""

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
      deprecated=True,
      default=None,
      description=(
          "[DEPRECATED] This field is deprecated in favor of `criterion`."
          " Depending on the metric you may want to one of the sub-classes of"
          " BaseCriterion."
      ),
  )

  criterion: Optional[BaseCriterion] = Field(
      default=None, description="""Evaluation criterion used by the metric."""
  )


class EvalMetricResultDetails(EvalBaseModel):
  rubric_scores: Optional[list[RubricScore]] = Field(
      default=None,
      description=(
          "The scores obtained after applying the rubrics to the Agent's"
          " response."
      ),
  )


class EvalMetricResult(EvalMetric):
  """The actual computed score/value of a particular EvalMetric."""

  score: Optional[float] = Field(
      default=None,
      description=(
          "Score obtained after evaluating the metric. Optional, as evaluation"
          " might not have happened."
      ),
  )

  eval_status: EvalStatus = Field(description="The status of this evaluation.")

  details: EvalMetricResultDetails = Field(
      default_factory=EvalMetricResultDetails, description=""""""
  )


class EvalMetricResultPerInvocation(EvalBaseModel):
  """Eval metric results per invocation."""

  actual_invocation: Invocation = Field(
      description=(
          "The actual invocation, usually obtained by inferencing the agent."
      )
  )

  expected_invocation: Optional[Invocation] = Field(
      default=None,
      description=(
          "The expected invocation, usually the reference or golden invocation."
      ),
  )

  eval_metric_results: list[EvalMetricResult] = Field(
      default=[],
      description="Eval results for each applicable metric.",
  )


class Interval(EvalBaseModel):
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


class MetricValueInfo(EvalBaseModel):
  """Information about the type of metric value."""

  interval: Optional[Interval] = Field(
      default=None,
      description="The values represented by the metric are of type interval.",
  )


class MetricInfo(EvalBaseModel):
  """Information about the metric that are used for Evals."""

  metric_name: str = Field(description="The name of the metric.")

  description: str = Field(
      default=None, description="A 2 to 3 line description of the metric."
  )

  metric_value_info: MetricValueInfo = Field(
      description="Information on the nature of values supported by the metric."
  )
