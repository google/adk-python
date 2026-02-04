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

from unittest import mock

from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.custom_metric_evaluator import _get_metric_function
from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvaluationResult
import pytest


def my_sync_metric_function(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None,
) -> EvaluationResult:
  """Sync metric function for testing."""
  return EvaluationResult(overall_score=1.0)


async def my_async_metric_function(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None,
) -> EvaluationResult:
  """Async metric function for testing."""
  return EvaluationResult(overall_score=0.5)


@mock.patch("importlib.import_module")
def test_get_metric_function_success(mock_import_module):
  """Tests that _get_metric_function successfully returns a function."""
  mock_module = mock.MagicMock()
  mock_module.my_sync_metric_function = my_sync_metric_function
  mock_import_module.return_value = mock_module
  func = _get_metric_function(
      "test_custom_metric_evaluator.my_sync_metric_function"
  )
  assert func == my_sync_metric_function


@mock.patch("importlib.import_module", side_effect=ImportError)
def test_get_metric_function_module_not_found(mock_import_module):
  """Tests that _get_metric_function raises ImportError for non-existent module."""
  with pytest.raises(ImportError):
    _get_metric_function("non_existent_module.my_sync_metric_function")


@mock.patch("importlib.import_module")
def test_get_metric_function_function_not_found(mock_import_module):
  """Tests that _get_metric_function raises ImportError for non-existent function."""
  mock_import_module.return_value = object()
  with pytest.raises(ImportError):
    _get_metric_function(
        "google.adk.tests.unittests.evaluation.test_custom_metric_evaluator.non_existent_function"
    )


def test_get_metric_function_malformed_path():
  """Tests that _get_metric_function raises ImportError for malformed path."""
  with pytest.raises(ImportError):
    _get_metric_function("malformed_path")


@mock.patch(
    "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
    return_value=my_sync_metric_function,
)
@pytest.mark.asyncio
async def test_custom_metric_evaluator_sync_function(mock_get_metric_function):
  """Tests that _CustomMetricEvaluator works with a sync metric function."""
  eval_metric = EvalMetric(metric_name="sync_metric")
  evaluator = _CustomMetricEvaluator(
      eval_metric=eval_metric,
      custom_function_path="google.adk.tests.unittests.evaluation.test_custom_metric_evaluator.my_sync_metric_function",
  )
  result = await evaluator.evaluate_invocations([], None)
  assert result.overall_score == 1.0


@mock.patch(
    "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
    return_value=my_async_metric_function,
)
@pytest.mark.asyncio
async def test_custom_metric_evaluator_async_function(mock_get_metric_function):
  """Tests that _CustomMetricEvaluator works with an async metric function."""
  eval_metric = EvalMetric(metric_name="async_metric")
  evaluator = _CustomMetricEvaluator(
      eval_metric=eval_metric,
      custom_function_path="google.adk.tests.unittests.evaluation.test_custom_metric_evaluator.my_async_metric_function",
  )
  result = await evaluator.evaluate_invocations([], None)
  assert result.overall_score == 0.5
