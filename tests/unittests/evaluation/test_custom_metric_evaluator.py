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

import asyncio
from collections.abc import Awaitable
import contextlib
import threading
from unittest import mock

from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.custom_metric_evaluator import _get_metric_function
from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.utils._client_labels_utils import client_label_context
from google.adk.utils._client_labels_utils import EVAL_CLIENT_LABEL
from google.adk.utils._client_labels_utils import get_client_labels
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


async def test_custom_metric_evaluator_sync_function_runs_off_event_loop():
  """Blocking sync custom metrics run in worker threads, off the event loop.

  Setup: A synchronous custom metric callable that synchronizes across a 2-party
    threading.Barrier and waits for an asyncio heartbeat task to release it.
  Act: Run two evaluate_invocations calls concurrently alongside a heartbeat.
  Assert: Both calls finish, which needs them to run at the same time while
    the heartbeat keeps running on the event loop.
  """
  both_workers_entered = threading.Event()
  release_workers = threading.Event()
  barrier = threading.Barrier(2, action=both_workers_entered.set, timeout=2.0)

  def _blocking_sync_metric(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> EvaluationResult:
    del (
        eval_metric,
        actual_invocations,
        expected_invocations,
        conversation_scenario,
    )
    barrier.wait()
    if not release_workers.wait(timeout=2.0):
      raise TimeoutError(
          "Event loop heartbeat did not run while workers blocked."
      )
    return EvaluationResult(overall_score=1.0)

  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_blocking_sync_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=EvalMetric(metric_name="blocking_sync_metric"),
        custom_function_path="test_module.blocking_sync_metric",
    )

  async def _heartbeat() -> None:
    while not both_workers_entered.is_set():
      await asyncio.sleep(0.001)
    release_workers.set()

  heartbeat_task = asyncio.create_task(_heartbeat())
  try:
    results = await asyncio.gather(
        evaluator.evaluate_invocations([], None),
        evaluator.evaluate_invocations([], None),
    )
  finally:
    both_workers_entered.set()
    release_workers.set()
    heartbeat_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await heartbeat_task

  assert [result.overall_score for result in results] == [1.0, 1.0]


async def test_custom_metric_evaluator_sync_function_sees_eval_client_label():
  """A sync metric on a worker thread sees the caller's eval client label."""
  labels_seen: list[bool] = []

  def _label_recording_metric(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> EvaluationResult:
    del (
        eval_metric,
        actual_invocations,
        expected_invocations,
        conversation_scenario,
    )
    labels_seen.append(EVAL_CLIENT_LABEL in get_client_labels())
    return EvaluationResult(overall_score=1.0)

  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_label_recording_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=EvalMetric(metric_name="label_metric"),
        custom_function_path="test_module.label_recording_metric",
    )

  with client_label_context(EVAL_CLIENT_LABEL):
    await evaluator.evaluate_invocations([], None)

  assert labels_seen == [True]


async def test_custom_metric_evaluator_clears_threshold_only_on_copy():
  """The metric function gets a threshold-free copy of the eval metric."""
  received_thresholds: list[float | None] = []

  def _threshold_recording_metric(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> EvaluationResult:
    del actual_invocations, expected_invocations, conversation_scenario
    received_thresholds.append(eval_metric.threshold)
    return EvaluationResult(overall_score=1.0)

  eval_metric = EvalMetric(metric_name="threshold_metric", threshold=0.8)
  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_threshold_recording_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=eval_metric,
        custom_function_path="test_module.threshold_recording_metric",
    )

  await evaluator.evaluate_invocations([], None)

  assert received_thresholds == [None]
  assert eval_metric.threshold == 0.8


async def test_custom_metric_evaluator_sync_function_returning_awaitable():
  """A sync metric that returns an awaitable still resolves to its result."""

  async def _inner() -> EvaluationResult:
    await asyncio.sleep(0)
    return EvaluationResult(overall_score=0.85)

  def _sync_wrapper(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> Awaitable[EvaluationResult]:
    del (
        eval_metric,
        actual_invocations,
        expected_invocations,
        conversation_scenario,
    )
    return _inner()

  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_sync_wrapper,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=EvalMetric(metric_name="wrapper_metric", threshold=0.5),
        custom_function_path="test_module.sync_wrapper",
    )

  result = await evaluator.evaluate_invocations([], None)

  assert result.overall_score == 0.85
