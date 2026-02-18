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

from __future__ import annotations

from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricInfoProvider
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics


class TrajectoryEvaluatorMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for TrajectoryEvaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
        description=(
            "This metric compares two tool call trajectories (expected vs."
            " actual) for the same user interaction. It performs an exact match"
            " on the tool name and arguments for each step in the trajectory."
            " A score of 1.0 indicates a perfect match, while 0.0 indicates a"
            " mismatch. Higher values are better."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class ResponseEvaluatorMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for ResponseEvaluator."""

  def __init__(self, metric_name: str):
    self._metric_name = metric_name

  def get_metric_info(self) -> MetricInfo:
    """Returns MetricInfo for the given metric name."""
    if PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value == self._metric_name:
      return MetricInfo(
          metric_name=PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value,
          description=(
              "This metric evaluates how coherent agent's response was. Value"
              " range of this metric is [1,5], with values closer to 5 more"
              " desirable."
          ),
          metric_value_info=MetricValueInfo(
              interval=Interval(min_value=1.0, max_value=5.0)
          ),
      )
    elif PrebuiltMetrics.RESPONSE_MATCH_SCORE.value == self._metric_name:
      return MetricInfo(
          metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
          description=(
              "This metric evaluates if the agent's final response matches a"
              " golden/expected final response using Rouge_1 metric. Value"
              " range for this metric is [0,1], with values closer to 1 more"
              " desirable."
          ),
          metric_value_info=MetricValueInfo(
              interval=Interval(min_value=0.0, max_value=1.0)
          ),
      )
    else:
      raise ValueError(f"`{self._metric_name}` is not supported.")


class SafetyEvaluatorV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for SafetyEvaluatorV1."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.SAFETY_V1.value,
        description=(
            "This metric evaluates the safety (harmlessness) of an Agent's"
            " Response. Value range of the metric is [0, 1], with values closer"
            " to 1 to be more desirable (safe)."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class FinalResponseMatchV2EvaluatorMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for FinalResponseMatchV2Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
        description=(
            "This metric evaluates if the agent's final response matches a"
            " golden/expected final response using LLM as a judge. Value range"
            " for this metric is [0,1], with values closer to 1 more desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider(
    MetricInfoProvider
):
  """Metric info provider for RubricBasedFinalResponseQualityV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
        description=(
            "This metric assess if the agent's final response against a set of"
            " rubrics using LLM as a judge. Value range for this metric is"
            " [0,1], with values closer to 1 more desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class HallucinationsV1EvaluatorMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for HallucinationsV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.HALLUCINATIONS_V1.value,
        description=(
            "This metric assesses whether a model response contains any false,"
            " contradictory, or unsupported claims using a LLM as judge. Value"
            " range for this metric is [0,1], with values closer to 1 more"
            " desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class RubricBasedToolUseV1EvaluatorMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for RubricBasedToolUseV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
        description=(
            "This metric assess if the agent's usage of tools against a set of"
            " rubrics using LLM as a judge. Value range for this metric is"
            " [0,1], with values closer to 1 more desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class PerTurnUserSimulatorQualityV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for PerTurnUserSimulatorQualityV1."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.PER_TURN_USER_SIMULATOR_QUALITY_V1,
        description=(
            "This metric evaluates if the user messages generated by a "
            "user simulator follow the given conversation scenario. It "
            "validates each message separately. The resulting metric "
            "computes the percentage of user messages that we mark as "
            "valid. The value range for this metric is [0,1], with values "
            "closer to 1 more desirable. "
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )
