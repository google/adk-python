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

import math

from google.adk.agents.common_configs import CodeConfig
from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.eval_config import CustomMetricConfig
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from google.adk.evaluation.metric_evaluator_registry import FinalResponseMatchV2EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import HallucinationsV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
from google.adk.evaluation.metric_evaluator_registry import PerTurnUserSimulatorQualityV1MetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import register_custom_metrics_from_config
from google.adk.evaluation.metric_evaluator_registry import ResponseEvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import RubricBasedMultiTurnTrajectoryMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import RubricBasedToolUseV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import SafetyEvaluatorV1MetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import TrajectoryEvaluator
from google.adk.evaluation.metric_evaluator_registry import TrajectoryEvaluatorMetricInfoProvider
from pydantic import ValidationError
import pytest

_DUMMY_METRIC_NAME = "dummy_metric_name"
_DUMMY_METRIC_INFO = MetricInfo(
    metric_name=_DUMMY_METRIC_NAME,
    description="Dummy metric description",
    metric_value_info=MetricValueInfo(
        interval=Interval(min_value=0.0, max_value=1.0)
    ),
)
_ANOTHER_DUMMY_METRIC_INFO = MetricInfo(
    metric_name=_DUMMY_METRIC_NAME,
    description="Another dummy metric description",
    metric_value_info=MetricValueInfo(
        interval=Interval(min_value=0.0, max_value=1.0)
    ),
)


class DummyEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  def evaluate_invocations(self, actual_invocations, expected_invocations):
    return "dummy_result"


class AnotherDummyEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  def evaluate_invocations(self, actual_invocations, expected_invocations):
    return "another_dummy_result"


class TestMetricEvaluatorRegistry:
  """Test cases for MetricEvaluatorRegistry."""

  @pytest.fixture
  def registry(self):
    return MetricEvaluatorRegistry()

  def test_register_evaluator(self, registry):
    registry.register_evaluator(
        _DUMMY_METRIC_INFO,
        DummyEvaluator,
    )
    assert _DUMMY_METRIC_NAME in registry._registry
    assert registry._registry[_DUMMY_METRIC_NAME] == (
        DummyEvaluator,
        _DUMMY_METRIC_INFO,
    )

  def test_register_evaluator_updates_existing(self, registry):
    registry.register_evaluator(
        _DUMMY_METRIC_INFO,
        DummyEvaluator,
    )

    assert registry._registry[_DUMMY_METRIC_NAME] == (
        DummyEvaluator,
        _DUMMY_METRIC_INFO,
    )

    registry.register_evaluator(
        _ANOTHER_DUMMY_METRIC_INFO, AnotherDummyEvaluator
    )
    assert registry._registry[_DUMMY_METRIC_NAME] == (
        AnotherDummyEvaluator,
        _ANOTHER_DUMMY_METRIC_INFO,
    )

  def test_a_new_registry_has_the_standard_metrics(self):
    registry = MetricEvaluatorRegistry()

    registered = {
        metric_info.metric_name
        for metric_info in registry.get_registered_metrics()
    }
    assert {
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
        PrebuiltMetrics.SAFETY_V1.value,
        PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
        PrebuiltMetrics.HALLUCINATIONS_V1.value,
    } <= registered
    assert isinstance(
        registry.get_evaluator(
            EvalMetric(
                metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
                threshold=0.5,
            )
        ),
        TrajectoryEvaluator,
    )

  def test_registrations_are_not_shared_across_instances(self, registry):
    registry.register_evaluator(
        _DUMMY_METRIC_INFO,
        DummyEvaluator,
    )

    other_registry = MetricEvaluatorRegistry()

    assert _DUMMY_METRIC_NAME not in other_registry._registry
    with pytest.raises(NotFoundError):
      other_registry.get_evaluator(
          EvalMetric(metric_name=_DUMMY_METRIC_NAME, threshold=0.5)
      )

  def test_get_evaluator(self, registry):
    registry.register_evaluator(
        _DUMMY_METRIC_INFO,
        DummyEvaluator,
    )
    eval_metric = EvalMetric(metric_name=_DUMMY_METRIC_NAME, threshold=0.5)
    evaluator = registry.get_evaluator(eval_metric)
    assert isinstance(evaluator, DummyEvaluator)

  def test_get_evaluator_not_found(self, registry):
    eval_metric = EvalMetric(metric_name="non_existent_metric", threshold=0.5)
    with pytest.raises(NotFoundError):
      registry.get_evaluator(eval_metric)


class TestRegisterCustomMetricsFromConfig:
  """Test cases for register_custom_metrics_from_config."""

  _CUSTOM_METRIC_NAME = "custom_metric_for_registry_test"

  @pytest.fixture
  def registry(self):
    return MetricEvaluatorRegistry()

  def _registered_metric_info(self, registry, metric_name):
    return next(
        metric_info
        for metric_info in registry.get_registered_metrics()
        if metric_info.metric_name == metric_name
    )

  def test_registers_custom_metric_with_provided_metric_info(self, registry):
    metric_info = MetricInfo(
        metric_name="name_to_be_overridden",
        description="Custom metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=5.0)
        ),
    )
    eval_config = EvalConfig(
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt"),
                metric_info=metric_info,
            )
        }
    )

    result = register_custom_metrics_from_config(eval_config, registry)

    assert result is registry
    registered_info = self._registered_metric_info(
        registry, self._CUSTOM_METRIC_NAME
    )
    assert registered_info.metric_value_info.interval.max_value == 5.0
    assert all(
        metric_info.metric_name != "name_to_be_overridden"
        for metric_info in registry.get_registered_metrics()
    )
    evaluator = registry.get_evaluator(
        EvalMetric(
            metric_name=self._CUSTOM_METRIC_NAME,
            threshold=0.5,
            custom_function_path="math.sqrt",
        )
    )
    assert isinstance(evaluator, _CustomMetricEvaluator)

  def test_registers_custom_metric_with_default_metric_info(self, registry):
    eval_config = EvalConfig(
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt"),
                description="A custom metric",
            )
        }
    )

    register_custom_metrics_from_config(eval_config, registry)

    registered_info = self._registered_metric_info(
        registry, self._CUSTOM_METRIC_NAME
    )
    assert registered_info.description == "A custom metric"
    assert registered_info.metric_value_info.interval.min_value == 0.0
    assert registered_info.metric_value_info.interval.max_value == 1.0

  def test_ignores_custom_function_path_on_the_eval_metric(self, registry):
    eval_config = EvalConfig(
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt"),
            )
        }
    )
    register_custom_metrics_from_config(eval_config, registry)

    evaluator = registry.get_evaluator(
        EvalMetric(
            metric_name=self._CUSTOM_METRIC_NAME,
            threshold=0.5,
            custom_function_path="math.floor",
        )
    )

    assert evaluator._metric_function is math.sqrt

  def test_no_custom_metrics_is_a_no_op(self, registry):
    registered_before = registry.get_registered_metrics()

    result = register_custom_metrics_from_config(EvalConfig(), registry)

    assert result is registry
    assert registry.get_registered_metrics() == registered_before

  def test_defaults_to_the_default_registry(self):
    eval_config = EvalConfig(
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt"),
            )
        }
    )

    try:
      result = register_custom_metrics_from_config(eval_config)

      assert result is DEFAULT_METRIC_EVALUATOR_REGISTRY
      registered_info = self._registered_metric_info(
          DEFAULT_METRIC_EVALUATOR_REGISTRY, self._CUSTOM_METRIC_NAME
      )
      assert registered_info.metric_name == self._CUSTOM_METRIC_NAME
    finally:
      DEFAULT_METRIC_EVALUATOR_REGISTRY._registry.pop(
          self._CUSTOM_METRIC_NAME, None
      )
      DEFAULT_METRIC_EVALUATOR_REGISTRY._custom_function_paths.pop(
          self._CUSTOM_METRIC_NAME, None
      )


def _custom_metric_info(metric_name: str) -> MetricInfo:
  return MetricInfo(
      metric_name=metric_name,
      description="Custom metric registered by hand.",
      metric_value_info=MetricValueInfo(
          interval=Interval(min_value=0.0, max_value=1.0)
      ),
  )


class TestCustomFunctionPathResolution:
  """How the module path imported for a custom metric is chosen.

  Agents in the repo register `_CustomMetricEvaluator` by hand against a
  registry and then run with an eval config that names the function. The
  function has to come from that config, and never from the metric handed to
  `get_evaluator`, which on a served eval is built from the request.
  """

  _CUSTOM_METRIC_NAME = "custom_metric_for_resolution_test"

  @pytest.fixture
  def registry(self):
    return MetricEvaluatorRegistry()

  def test_hand_registered_evaluator_uses_the_float_criterion_config(
      self, registry
  ):
    registry.register_evaluator(
        _custom_metric_info(self._CUSTOM_METRIC_NAME), _CustomMetricEvaluator
    )
    eval_config = EvalConfig(
        criteria={self._CUSTOM_METRIC_NAME: 0.8},
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt")
            )
        },
    )

    [eval_metric] = get_eval_metrics_from_config(eval_config)
    evaluator = registry.get_evaluator(eval_metric)

    assert evaluator._metric_function is math.sqrt

  def test_hand_registered_evaluator_uses_the_criterion_object_config(
      self, registry
  ):
    registry.register_evaluator(
        _custom_metric_info(self._CUSTOM_METRIC_NAME), _CustomMetricEvaluator
    )
    eval_config = EvalConfig(
        criteria={self._CUSTOM_METRIC_NAME: BaseCriterion(threshold=1.0)},
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt")
            )
        },
    )

    [eval_metric] = get_eval_metrics_from_config(eval_config)
    evaluator = registry.get_evaluator(eval_metric)

    assert evaluator._metric_function is math.sqrt

  def test_each_metric_resolves_its_own_config_function(self, registry):
    other_metric_name = "another_custom_metric_for_resolution_test"
    for metric_name in (self._CUSTOM_METRIC_NAME, other_metric_name):
      registry.register_evaluator(
          _custom_metric_info(metric_name), _CustomMetricEvaluator
      )
    eval_config = EvalConfig(
        criteria={self._CUSTOM_METRIC_NAME: 1.0, other_metric_name: 1.0},
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt")
            ),
            other_metric_name: CustomMetricConfig(
                code_config=CodeConfig(name="math.floor")
            ),
        },
    )

    evaluators = {
        eval_metric.metric_name: registry.get_evaluator(eval_metric)
        for eval_metric in get_eval_metrics_from_config(eval_config)
    }

    assert evaluators[self._CUSTOM_METRIC_NAME]._metric_function is math.sqrt
    assert evaluators[other_metric_name]._metric_function is math.floor

  def test_hand_registration_on_the_default_registry_resolves(self):
    eval_config = EvalConfig(
        criteria={self._CUSTOM_METRIC_NAME: 1.0},
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt")
            )
        },
    )

    try:
      DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
          _custom_metric_info(self._CUSTOM_METRIC_NAME), _CustomMetricEvaluator
      )

      [eval_metric] = get_eval_metrics_from_config(eval_config)
      evaluator = DEFAULT_METRIC_EVALUATOR_REGISTRY.get_evaluator(eval_metric)

      assert evaluator._metric_function is math.sqrt
    finally:
      DEFAULT_METRIC_EVALUATOR_REGISTRY._registry.pop(
          self._CUSTOM_METRIC_NAME, None
      )

  def test_a_metric_without_a_config_is_rejected(self, registry):
    registry.register_evaluator(
        _custom_metric_info(self._CUSTOM_METRIC_NAME), _CustomMetricEvaluator
    )

    with pytest.raises(NotFoundError):
      registry.get_evaluator(
          EvalMetric(
              metric_name=self._CUSTOM_METRIC_NAME,
              threshold=0.5,
              custom_function_path="math.floor",
          )
      )

  def test_a_config_path_does_not_carry_to_another_configs_metric(
      self, registry
  ):
    registry.register_evaluator(
        _custom_metric_info(self._CUSTOM_METRIC_NAME), _CustomMetricEvaluator
    )
    trusted_config = EvalConfig(
        criteria={self._CUSTOM_METRIC_NAME: 1.0},
        custom_metrics={
            self._CUSTOM_METRIC_NAME: CustomMetricConfig(
                code_config=CodeConfig(name="math.sqrt")
            )
        },
    )
    [trusted_metric] = get_eval_metrics_from_config(trusted_config)
    assert registry.get_evaluator(trusted_metric)._metric_function is math.sqrt

    with pytest.raises(NotFoundError):
      registry.get_evaluator(
          EvalMetric(metric_name=self._CUSTOM_METRIC_NAME, threshold=0.5)
      )

  def test_the_config_path_cannot_be_set_through_the_metric(self):
    with pytest.raises(ValidationError):
      EvalMetric.model_validate({
          "metric_name": self._CUSTOM_METRIC_NAME,
          "threshold": 0.5,
          "_config_custom_function_path": "math.floor",
      })


class TestMetricInfoProviders:
  """Test cases for MetricInfoProviders."""

  def test_trajectory_evaluator_metric_info_provider(self):
    metric_info = TrajectoryEvaluatorMetricInfoProvider().get_metric_info()
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_response_evaluator_metric_info_provider_eval_score(self):
    metric_info = ResponseEvaluatorMetricInfoProvider(
        PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
    ).get_metric_info()
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
    )
    assert metric_info.metric_value_info.interval.min_value == 1.0
    assert metric_info.metric_value_info.interval.max_value == 5.0

  def test_response_evaluator_metric_info_provider_match_score(self):
    metric_info = ResponseEvaluatorMetricInfoProvider(
        PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
    ).get_metric_info()
    assert metric_info.metric_name == PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_safety_evaluator_v1_metric_info_provider(self):
    metric_info = SafetyEvaluatorV1MetricInfoProvider().get_metric_info()
    assert metric_info.metric_name == PrebuiltMetrics.SAFETY_V1.value
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_final_response_match_v2_evaluator_metric_info_provider(self):
    metric_info = (
        FinalResponseMatchV2EvaluatorMetricInfoProvider().get_metric_info()
    )
    assert (
        metric_info.metric_name == PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_rubric_based_final_response_quality_v1_evaluator_metric_info_provider(
      self,
  ):
    metric_info = (
        RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider().get_metric_info()
    )
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_hallucinations_v1_evaluator_metric_info_provider(self):
    metric_info = (
        HallucinationsV1EvaluatorMetricInfoProvider().get_metric_info()
    )
    assert metric_info.metric_name == PrebuiltMetrics.HALLUCINATIONS_V1.value
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_rubric_based_tool_use_v1_evaluator_metric_info_provider(self):
    metric_info = (
        RubricBasedToolUseV1EvaluatorMetricInfoProvider().get_metric_info()
    )
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_per_turn_user_simulator_quality_v1_metric_info_provider(self):
    metric_info = (
        PerTurnUserSimulatorQualityV1MetricInfoProvider().get_metric_info()
    )
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.PER_TURN_USER_SIMULATOR_QUALITY_V1.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0

  def test_rubric_based_multi_turn_trajectory_metric_info_provider(self):
    metric_info = (
        RubricBasedMultiTurnTrajectoryMetricInfoProvider().get_metric_info()
    )
    assert (
        metric_info.metric_name
        == PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1.value
    )
    assert metric_info.metric_value_info.interval.min_value == 0.0
    assert metric_info.metric_value_info.interval.max_value == 1.0
