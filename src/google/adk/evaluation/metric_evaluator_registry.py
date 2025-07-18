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

import logging

from ..errors.not_found_error import NotFoundError
from .eval_metrics import EvalMetric
from .eval_metrics import MetricName
from .eval_metrics import PrebuiltMetrics
from .evaluator import Evaluator
from .final_response_match_v2 import FinalResponseMatchV2Evaluator
from .response_evaluator import ResponseEvaluator
from .safety_evaluator import SafetyEvaluatorV1
from .trajectory_evaluator import TrajectoryEvaluator

logger = logging.getLogger("google_adk." + __name__)


class MetricEvaluatorRegistry:
  """A registry for metric Evaluators."""

  _registry: dict[str, type[Evaluator]] = {}

  def get_evaluator(self, eval_metric: EvalMetric) -> Evaluator:
    """Returns an Evaluator for the given metric.

    A new instance of the Evaluator is returned.

    Args:
      eval_metric: The metric for which we need the Evaluator.

    Raises:
      NotFoundError: If there is no evaluator for the metric.
    """
    if eval_metric.metric_name not in self._registry:
      raise NotFoundError(f"{eval_metric.metric_name} not found in registry.")

    return self._registry[eval_metric.metric_name](eval_metric=eval_metric)

  def register_evaluator(
      self, metric_name: MetricName, evaluator: type[Evaluator]
  ):
    """Registers an evaluator given the metric name.

    If a mapping already exist, then it is updated.
    """
    if metric_name in self._registry:
      logger.info(
          "Updating Evaluator class for %s from %s to %s",
          metric_name,
          self._registry[metric_name],
          evaluator,
      )

    self._registry[str(metric_name)] = evaluator


def _get_default_metric_evaluator_registry() -> MetricEvaluatorRegistry:
  """Returns an instance of MetricEvaluatorRegistry with standard metrics already registered in it."""
  metric_evaluator_registry = MetricEvaluatorRegistry()

  metric_evaluator_registry.register_evaluator(
      metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
      evaluator=TrajectoryEvaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_name=PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value,
      evaluator=ResponseEvaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
      evaluator=ResponseEvaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_name=PrebuiltMetrics.SAFETY_V1.value,
      evaluator=SafetyEvaluatorV1,
  )
  metric_evaluator_registry.register_evaluator(
      metric_name=PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
      evaluator=FinalResponseMatchV2Evaluator,
  )

  return metric_evaluator_registry


DEFAULT_METRIC_EVALUATOR_REGISTRY = _get_default_metric_evaluator_registry()
