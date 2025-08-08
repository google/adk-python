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

from typing_extensions import override
from vertexai import types as vertexai_types

from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .vertex_ai_eval_facade import _VertexAiEvalFacade


class SafetyEvaluatorV1(Evaluator):
  """Evaluates safety (harmlessness) of an Agent's Response.

  The class delegates the responsibility to Vertex Gen AI Eval SDK. The V1
  suffix in the class name is added to convey that there could be other versions
  of the safety metric as well, and those metrics could use a different strategy
  to evaluate safety.

  Using this class requires a GCP project. Please set GOOGLE_CLOUD_PROJECT and
  GOOGLE_CLOUD_LOCATION in your .env file.

  Value range of the metric is [0, 1], with values closer to 1 to be more
  desirable (safe).
  """

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @staticmethod
  def get_metric_info() -> MetricInfo:
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

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    return _VertexAiEvalFacade(
        threshold=self._eval_metric.threshold,
        metric_name=vertexai_types.PrebuiltMetric.SAFETY,
    ).evaluate_invocations(actual_invocations, expected_invocations)
