# Copyright 2025 Google LLC
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

from typing import Optional

from typing_extensions import override

from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .final_response_match_v1 import RougeEvaluator
from .vertex_ai_eval_facade import _VertexAiEvalFacade


class ResponseEvaluator(Evaluator):
  """Evaluates Agent's responses.

  This class supports two metrics:
  1) response_evaluation_score
  This metric evaluates how coherent agent's response was.

  Value range of this metric is [1,5], with values closer to 5 more desirable.

  2) response_match_score:
  This metric evaluates if agent's final response matches a golden/expected
  final response using Rouge_1 metric.

  Value range for this metric is [0,1], with values closer to 1 more desirable.
  """

  def __init__(
      self,
      threshold: Optional[float] = None,
      metric_name: Optional[str] = None,
      eval_metric: Optional[EvalMetric] = None,
  ):
    if (threshold is not None and eval_metric) or (
        metric_name is not None and eval_metric
    ):
      raise ValueError(
          "Either eval_metric should be specified or both threshold and"
          " metric_name should be specified."
      )

    if eval_metric:
      threshold = eval_metric.threshold
      metric_name = eval_metric.metric_name

    if PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value == metric_name:
      from ..dependencies.vertexai import vertexai

      self._metric_name = vertexai.types.PrebuiltMetric.COHERENCE
    elif PrebuiltMetrics.RESPONSE_MATCH_SCORE.value == metric_name:
      self._metric_name = metric_name
    else:
      raise ValueError(f"`{metric_name}` is not supported.")

    self._threshold = threshold

  @staticmethod
  def get_metric_info(metric_name: str) -> MetricInfo:
    """Returns MetricInfo for the given metric name."""
    if PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value == metric_name:
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
    elif PrebuiltMetrics.RESPONSE_MATCH_SCORE.value == metric_name:
      return RougeEvaluator.get_metric_info()
    else:
      raise ValueError(f"`{metric_name}` is not supported.")

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ) -> EvaluationResult:
    # If the metric is response_match_score, just use the RougeEvaluator.
    if self._metric_name == PrebuiltMetrics.RESPONSE_MATCH_SCORE.value:
      rouge_evaluator = RougeEvaluator(
          EvalMetric(metric_name=self._metric_name, threshold=self._threshold)
      )
      return rouge_evaluator.evaluate_invocations(
          actual_invocations, expected_invocations
      )

    return _VertexAiEvalFacade(
        threshold=self._threshold,
        metric_name=self._metric_name,
        expected_invocations_required=True,
    ).evaluate_invocations(actual_invocations, expected_invocations)
