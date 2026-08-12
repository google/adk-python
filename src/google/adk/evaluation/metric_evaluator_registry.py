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

import logging
from typing import cast
from typing import Optional
from typing import Protocol

from ..errors.not_found_error import NotFoundError
from ..utils.feature_decorator import experimental
from .custom_metric_evaluator import _CustomMetricEvaluator
from .eval_config import EvalConfig
from .eval_metrics import EvalMetric
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import Evaluator
from .final_response_match_v2 import FinalResponseMatchV2Evaluator
from .hallucinations_v1 import HallucinationsV1Evaluator
from .metric_info_providers import FinalResponseMatchV2EvaluatorMetricInfoProvider
from .metric_info_providers import HallucinationsV1EvaluatorMetricInfoProvider
from .metric_info_providers import MultiTurnTaskSuccessV1MetricInfoProvider
from .metric_info_providers import MultiTurnToolUseQualityV1MetricInfoProvider
from .metric_info_providers import MultiTurnTrajectoryQualityV1MetricInfoProvider
from .metric_info_providers import PerTurnUserSimulatorQualityV1MetricInfoProvider
from .metric_info_providers import ResponseEvaluatorMetricInfoProvider
from .metric_info_providers import RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider
from .metric_info_providers import RubricBasedMultiTurnTrajectoryMetricInfoProvider
from .metric_info_providers import RubricBasedToolUseV1EvaluatorMetricInfoProvider
from .metric_info_providers import SafetyEvaluatorV1MetricInfoProvider
from .metric_info_providers import TrajectoryEvaluatorMetricInfoProvider
from .multi_turn_task_success_evaluator import MultiTurnTaskSuccessV1Evaluator
from .multi_turn_tool_use_quality_evaluator import MultiTurnToolUseQualityV1Evaluator
from .multi_turn_trajectory_quality_evaluator import MultiTurnTrajectoryQualityV1Evaluator
from .response_evaluator import ResponseEvaluator
from .rubric_based_final_response_quality_v1 import RubricBasedFinalResponseQualityV1Evaluator
from .rubric_based_multi_turn_trajectory_evaluator import RubricBasedMultiTurnTrajectoryEvaluator
from .rubric_based_tool_use_quality_v1 import RubricBasedToolUseV1Evaluator
from .safety_evaluator import SafetyEvaluatorV1
from .simulation.per_turn_user_simulator_quality_v1 import PerTurnUserSimulatorQualityV1
from .trajectory_evaluator import TrajectoryEvaluator

logger = logging.getLogger("google_adk." + __name__)


class _EvalMetricEvaluatorFactory(Protocol):

  def __call__(self, *, eval_metric: EvalMetric) -> Evaluator:
    """Creates an evaluator for one metric configuration."""


@experimental
class MetricEvaluatorRegistry:
  """A registry for metric Evaluators."""

  def __init__(self) -> None:
    # Each registry instance owns its mappings, so a custom metric registered
    # for one app is not resolvable from another app's registry. The standard
    # metrics are seeded into every instance, as they are the same everywhere.
    self._registry: dict[str, tuple[type[Evaluator], MetricInfo]] = {}
    # Module path of the custom function backing a metric, keyed by metric
    # name. Only ever written from an eval config.
    self._custom_function_paths: dict[str, str] = {}
    _register_standard_metrics(self)

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

    evaluator_type, _ = self._registry[eval_metric.metric_name]
    if issubclass(evaluator_type, _CustomMetricEvaluator):
      custom_function_path = self._custom_function_path(eval_metric)
      if custom_function_path is None:
        raise NotFoundError(
            f"No custom function registered for {eval_metric.metric_name}."
        )
      return evaluator_type(
          eval_metric=eval_metric,
          custom_function_path=custom_function_path,
      )
    evaluator_factory = cast(_EvalMetricEvaluatorFactory, evaluator_type)
    return evaluator_factory(eval_metric=eval_metric)

  def _custom_function_path(self, eval_metric: EvalMetric) -> Optional[str]:
    """Returns the module path to import for a custom metric, if known.

    Both sources are eval config entries: one recorded when the metric was
    registered from a config, the other carried on a metric built from a
    config. The `custom_function_path` field on the incoming metric is not
    consulted, as it can be set by whoever built the request.

    Args:
      eval_metric: The metric whose custom function is being resolved.
    """
    if path := self._custom_function_paths.get(eval_metric.metric_name):
      return path
    return eval_metric._config_custom_function_path  # pylint: disable=protected-access

  def register_evaluator(
      self,
      metric_info: MetricInfo,
      evaluator: type[Evaluator],
  ) -> None:
    """Registers an evaluator given the metric info.

    If a mapping already exist, then it is updated.
    """
    self._register(metric_info, evaluator, custom_function_path=None)

  def _register(
      self,
      metric_info: MetricInfo,
      evaluator: type[Evaluator],
      custom_function_path: Optional[str],
  ) -> None:
    """Registers an evaluator, along with the function path it may need.

    A path already recorded for the metric is kept when this registration does
    not carry one, so re-registering an evaluator does not drop it.

    Args:
      metric_info: Info for the metric the evaluator is registered against.
      evaluator: The evaluator class to register.
      custom_function_path: Module path of the function backing a custom
        metric, taken from an eval config, or None.
    """
    metric_name = metric_info.metric_name
    if metric_name in self._registry:
      logger.info(
          "Updating Evaluator class for %s from %s to %s",
          metric_name,
          self._registry[metric_name],
          evaluator,
      )

    self._registry[str(metric_name)] = (evaluator, metric_info)
    if custom_function_path is not None:
      self._custom_function_paths[str(metric_name)] = custom_function_path

  def get_registered_metrics(
      self,
  ) -> list[MetricInfo]:
    """Returns a list of MetricInfo about the metrics registered so far."""
    return [
        evaluator_and_metric_info[1].model_copy(deep=True)
        for _, evaluator_and_metric_info in self._registry.items()
    ]


def _register_standard_metrics(
    metric_evaluator_registry: MetricEvaluatorRegistry,
) -> None:
  """Registers the metrics that ship with ADK into the given registry."""
  metric_evaluator_registry.register_evaluator(
      metric_info=TrajectoryEvaluatorMetricInfoProvider().get_metric_info(),
      evaluator=TrajectoryEvaluator,
  )

  metric_evaluator_registry.register_evaluator(
      metric_info=ResponseEvaluatorMetricInfoProvider(
          PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
      ).get_metric_info(),
      evaluator=ResponseEvaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=ResponseEvaluatorMetricInfoProvider(
          PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
      ).get_metric_info(),
      evaluator=ResponseEvaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=SafetyEvaluatorV1MetricInfoProvider().get_metric_info(),
      evaluator=SafetyEvaluatorV1,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=MultiTurnTaskSuccessV1MetricInfoProvider().get_metric_info(),
      evaluator=MultiTurnTaskSuccessV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=MultiTurnTrajectoryQualityV1MetricInfoProvider().get_metric_info(),
      evaluator=MultiTurnTrajectoryQualityV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=MultiTurnToolUseQualityV1MetricInfoProvider().get_metric_info(),
      evaluator=MultiTurnToolUseQualityV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=FinalResponseMatchV2EvaluatorMetricInfoProvider().get_metric_info(),
      evaluator=FinalResponseMatchV2Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider().get_metric_info(),
      evaluator=RubricBasedFinalResponseQualityV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=HallucinationsV1EvaluatorMetricInfoProvider().get_metric_info(),
      evaluator=HallucinationsV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=RubricBasedToolUseV1EvaluatorMetricInfoProvider().get_metric_info(),
      evaluator=RubricBasedToolUseV1Evaluator,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=PerTurnUserSimulatorQualityV1MetricInfoProvider().get_metric_info(),
      evaluator=PerTurnUserSimulatorQualityV1,
  )
  metric_evaluator_registry.register_evaluator(
      metric_info=RubricBasedMultiTurnTrajectoryMetricInfoProvider().get_metric_info(),
      evaluator=RubricBasedMultiTurnTrajectoryEvaluator,
  )


def _get_default_metric_evaluator_registry() -> MetricEvaluatorRegistry:
  """Returns an instance of MetricEvaluatorRegistry with standard metrics already registered in it."""
  return MetricEvaluatorRegistry()


DEFAULT_METRIC_EVALUATOR_REGISTRY = _get_default_metric_evaluator_registry()


def _get_default_metric_info(
    metric_name: str, description: str = ""
) -> MetricInfo:
  """Returns a default MetricInfo for a metric."""
  return MetricInfo(
      metric_name=metric_name,
      description=description,
      metric_value_info=MetricValueInfo(
          interval=Interval(min_value=0.0, max_value=1.0)
      ),
  )


def register_custom_metrics_from_config(
    eval_config: EvalConfig,
    metric_evaluator_registry: Optional[MetricEvaluatorRegistry] = None,
) -> MetricEvaluatorRegistry:
  """Registers custom metrics declared in the given eval config.

  Args:
    eval_config: The eval config whose custom_metrics entries should be
      registered. Entries without a metric_info get a default one with a
      [0.0, 1.0] value interval.
    metric_evaluator_registry: The registry to register the metrics in.
      Defaults to DEFAULT_METRIC_EVALUATOR_REGISTRY.

  Returns:
    The registry the metrics were registered in.
  """
  if metric_evaluator_registry is None:
    metric_evaluator_registry = DEFAULT_METRIC_EVALUATOR_REGISTRY
  if not eval_config.custom_metrics:
    return metric_evaluator_registry

  for metric_name, config in eval_config.custom_metrics.items():
    if config.metric_info:
      metric_info = config.metric_info.model_copy()
      metric_info.metric_name = metric_name
    else:
      metric_info = _get_default_metric_info(
          metric_name=metric_name, description=config.description
      )
    metric_evaluator_registry._register(  # pylint: disable=protected-access
        metric_info, _CustomMetricEvaluator, config.code_config.name
    )
  return metric_evaluator_registry
