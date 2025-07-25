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

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
import pytest

_DUMMY_METRIC_NAME = "dummy_metric_name"


class TestMetricEvaluatorRegistry:
  """Test cases for MetricEvaluatorRegistry."""

  @pytest.fixture
  def registry(self):
    return MetricEvaluatorRegistry()

  class DummyEvaluator(Evaluator):

    def __init__(self, eval_metric: EvalMetric):
      self._eval_metric = eval_metric

    def evaluate_invocations(self, actual_invocations, expected_invocations):
      return "dummy_result"

    @staticmethod
    def get_metric_info() -> MetricInfo:
      return MetricInfo(
          metric_name=_DUMMY_METRIC_NAME,
          description="Dummy metric description",
          metric_value_info=MetricValueInfo(
              interval=Interval(min_value=0.0, max_value=1.0)
          ),
      )

  class AnotherDummyEvaluator(Evaluator):

    def __init__(self, eval_metric: EvalMetric):
      self._eval_metric = eval_metric

    def evaluate_invocations(self, actual_invocations, expected_invocations):
      return "another_dummy_result"

    @staticmethod
    def get_metric_info() -> MetricInfo:
      return MetricInfo(
          metric_name=_DUMMY_METRIC_NAME,
          description="Another dummy metric description",
          metric_value_info=MetricValueInfo(
              interval=Interval(min_value=0.0, max_value=1.0)
          ),
      )

  def test_register_evaluator(self, registry):
    metric_info = TestMetricEvaluatorRegistry.DummyEvaluator.get_metric_info()
    registry.register_evaluator(
        metric_info,
        TestMetricEvaluatorRegistry.DummyEvaluator,
    )
    assert _DUMMY_METRIC_NAME in registry._registry
    assert registry._registry[_DUMMY_METRIC_NAME] == (
        TestMetricEvaluatorRegistry.DummyEvaluator,
        metric_info,
    )

  def test_register_evaluator_updates_existing(self, registry):
    metric_info = TestMetricEvaluatorRegistry.DummyEvaluator.get_metric_info()
    registry.register_evaluator(
        metric_info,
        TestMetricEvaluatorRegistry.DummyEvaluator,
    )

    assert registry._registry[_DUMMY_METRIC_NAME] == (
        TestMetricEvaluatorRegistry.DummyEvaluator,
        metric_info,
    )

    metric_info = (
        TestMetricEvaluatorRegistry.AnotherDummyEvaluator.get_metric_info()
    )
    registry.register_evaluator(
        metric_info, TestMetricEvaluatorRegistry.AnotherDummyEvaluator
    )
    assert registry._registry[_DUMMY_METRIC_NAME] == (
        TestMetricEvaluatorRegistry.AnotherDummyEvaluator,
        metric_info,
    )

  def test_get_evaluator(self, registry):
    metric_info = TestMetricEvaluatorRegistry.DummyEvaluator.get_metric_info()
    registry.register_evaluator(
        metric_info,
        TestMetricEvaluatorRegistry.DummyEvaluator,
    )
    eval_metric = EvalMetric(metric_name=_DUMMY_METRIC_NAME, threshold=0.5)
    evaluator = registry.get_evaluator(eval_metric)
    assert isinstance(evaluator, TestMetricEvaluatorRegistry.DummyEvaluator)

  def test_get_evaluator_not_found(self, registry):
    eval_metric = EvalMetric(metric_name="non_existent_metric", threshold=0.5)
    with pytest.raises(NotFoundError):
      registry.get_evaluator(eval_metric)
