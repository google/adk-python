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

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.metric_evaluator_registry import FinalResponseMatchV2EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import HallucinationsV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
from google.adk.evaluation.metric_evaluator_registry import PerTurnUserSimulatorQualityV1MetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import ResponseEvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import RubricBasedToolUseV1EvaluatorMetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import SafetyEvaluatorV1MetricInfoProvider
from google.adk.evaluation.metric_evaluator_registry import TrajectoryEvaluatorMetricInfoProvider
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
