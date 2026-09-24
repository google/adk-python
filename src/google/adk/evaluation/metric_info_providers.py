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


class MultiTurnTaskSuccessV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for MultiTurnTaskSuccessV1."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.MULTI_TURN_TASK_SUCCESS_V1.value,
        description=(
            "Evaluates if the agent was able to achieve the goal or goals of"
            " the conversation."
            " Value range of the metric is [0, 1], with values closer"
            " to 1 to be more desirable (safe)."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class MultiTurnTrajectoryQualityV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for MultiTurnTrajectoryQualityV1."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
        description=(
            "Evaluates the overall trajectory of the conversation. Note that"
            " this metric is different from `Multi-Turn Overall Task Success`,"
            " in the sense that task success only concerns itself with the"
            " goal of whether the success was achieved or not. How that was"
            " achieved is not its concern. This metric on the other hand does"
            " care about the path that agent took to achieve the goal. This is"
            " a reference free metric."
            " Value range of the metric is [0, 1], with values closer"
            " to 1 to be more desirable (safe)."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class MultiTurnToolUseQualityV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for MultiTurnToolUseQualityV1."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.MULTI_TURN_TOOL_USE_QUALITY_V1.value,
        description=(
            "Evaluates the function calls made during a multi-turn"
            " conversation. This is a reference free metric."
            " Value range of the metric is [0, 1], with values closer"
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


class RubricBasedMultiTurnTrajectoryMetricInfoProvider(MetricInfoProvider):
  """Metric info provider for RubricBasedMultiTurnTrajectory."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1,
        description=(
            "This metric evaluates the agent's multi-turn trajectory against"
            " a set of user-provided rubrics using an LLM as a judge. Value"
            " range for this metric is [0,1], with values closer to 1 more"
            " desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )


class ToolCallCountV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for _ToolCallCountV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value,
        description=(
            "This metric counts the number of tool (function) calls the agent"
            " made per invocation, averaged across the eval case. It is an"
            " informational efficiency metric: it reports the value for"
            " tracking and does not pass or fail the eval case."
        ),
        metric_value_info=MetricValueInfo(),
        # Informational: reports a value, never gates, so no threshold
        # is required and no value interval bounds it.
        requires_threshold=False,
    )


class InferenceCallCountV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for _InferenceCallCountV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.INFERENCE_CALL_COUNT_V1.value,
        description=(
            "This metric counts the number of inference (model) calls the agent"
            " made per invocation, averaged across the eval case. It is a proxy"
            " for the number of attempts or reasoning steps taken, and read"
            " alongside token usage it separates the two ways a turn gets"
            " expensive: more calls, or a larger context per call. It is an"
            " informational efficiency metric: it reports the value for"
            " tracking and does not pass or fail the eval case."
        ),
        metric_value_info=MetricValueInfo(),
        # Informational: reports a value, never gates, so no threshold
        # is required and no value interval bounds it.
        requires_threshold=False,
    )


class InvocationDurationV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for _InvocationDurationV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value,
        description=(
            "This metric reports the wall-clock seconds an invocation took,"
            " averaged across the eval case. The duration is measured while the"
            " agent runs; an invocation not produced by this eval run reports"
            " no value. Wall-clock time is noisier than the token and call"
            " counts, since it moves with model-server load and network, so"
            " read it as an indication rather than a regression signal. It is"
            " an informational efficiency metric: it reports the value for"
            " tracking and does not pass or fail the eval case."
        ),
        metric_value_info=MetricValueInfo(),
        # Informational: reports a value, never gates, so no threshold
        # is required and no value interval bounds it.
        requires_threshold=False,
    )


class TokenUsageV1MetricInfoProvider(MetricInfoProvider):
  """Metric info provider for _TokenUsageV1Evaluator."""

  def get_metric_info(self) -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
        description=(
            "This metric sums the tokens consumed by the model across all model"
            " calls in an invocation, averaged across the eval case. The score"
            " is the total; every token type is reported alongside it as a"
            " nested breakdown -- total, then input (prompt, of which cached,"
            " plus tool use) and output (candidates plus reasoning) -- using"
            " the same definitions as ADK's telemetry token metrics. It is an"
            " informational efficiency metric: it reports the value for"
            " tracking and does not pass or fail the eval case."
        ),
        metric_value_info=MetricValueInfo(),
        # Informational: reports a value, never gates, so no threshold
        # is required and no value interval bounds it.
        requires_threshold=False,
    )
