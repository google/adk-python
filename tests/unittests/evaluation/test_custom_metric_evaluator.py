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
import functools
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


async def test_custom_metric_evaluator_blocking_sync_functions_run_concurrently_without_blocking_event_loop():
  """Blocking sync custom metrics run in parallel worker threads without stalling asyncio.

  Setup: A synchronous custom metric callable that synchronizes across a 2-party
    threading.Barrier and waits for an asyncio heartbeat task to release it.
  Act: Run two evaluate_invocations calls concurrently alongside a heartbeat.
  Assert: Both calls execute on distinct worker threads, the heartbeat ticks
    while both workers are blocked, and threshold is cleared only on the copy.
  """
  main_thread_id = threading.get_ident()
  worker_thread_ids: list[int] = []
  received_thresholds: list[float | None] = []
  both_workers_entered = threading.Event()
  release_workers = threading.Event()
  barrier = threading.Barrier(2, action=both_workers_entered.set, timeout=2.0)

  def _blocking_sync_metric(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> EvaluationResult:
    del actual_invocations, expected_invocations, conversation_scenario
    worker_thread_ids.append(threading.get_ident())
    received_thresholds.append(eval_metric.threshold)
    barrier.wait()
    if not release_workers.wait(timeout=2.0):
      raise TimeoutError(
          "Event loop heartbeat did not run while workers blocked."
      )
    return EvaluationResult(overall_score=1.0)

  eval_metric = EvalMetric(metric_name="blocking_sync_metric", threshold=0.8)
  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_blocking_sync_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=eval_metric,
        custom_function_path="test_module.blocking_sync_metric",
    )

  heartbeat_ticks_while_blocked = 0

  async def _heartbeat() -> None:
    nonlocal heartbeat_ticks_while_blocked
    while not both_workers_entered.is_set():
      await asyncio.sleep(0.001)
    for _ in range(3):
      await asyncio.sleep(0)
      heartbeat_ticks_while_blocked += 1
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
  assert heartbeat_ticks_while_blocked == 3
  assert main_thread_id not in worker_thread_ids
  assert len(set(worker_thread_ids)) == 2
  assert received_thresholds == [None, None]
  assert eval_metric.threshold == 0.8


async def test_custom_metric_evaluator_sync_function_preserves_eval_client_label_in_worker_thread():
  """Synchronous custom metrics offloaded via to_thread retain EVAL_CLIENT_LABEL."""
  main_thread_id = threading.get_ident()
  observed_threads: list[int] = []
  observed_labels: list[list[str]] = []

  def _label_capturing_metric(
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
    observed_threads.append(threading.get_ident())
    observed_labels.append(get_client_labels())
    return EvaluationResult(overall_score=0.9)

  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_label_capturing_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=EvalMetric(metric_name="label_sync_metric", threshold=0.5),
        custom_function_path="test_module.label_capturing_metric",
    )

  with client_label_context(EVAL_CLIENT_LABEL):
    result = await evaluator.evaluate_invocations([], None)

  assert result.overall_score == 0.9
  assert len(observed_threads) == 1
  assert observed_threads[0] != main_thread_id
  assert len(observed_labels) == 1
  assert EVAL_CLIENT_LABEL in observed_labels[0]


async def test_custom_metric_evaluator_async_function_runs_on_event_loop_thread():
  """Asynchronous custom metrics are awaited directly on the event loop thread."""
  main_thread_id = threading.get_ident()
  observed_threads: list[int] = []
  received_thresholds: list[float | None] = []

  async def _async_metric(
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None,
      conversation_scenario: ConversationScenario | None,
  ) -> EvaluationResult:
    del actual_invocations, expected_invocations, conversation_scenario
    await asyncio.sleep(0)
    observed_threads.append(threading.get_ident())
    received_thresholds.append(eval_metric.threshold)
    return EvaluationResult(overall_score=0.75)

  eval_metric = EvalMetric(metric_name="async_metric", threshold=0.6)
  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=_async_metric,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=eval_metric,
        custom_function_path="test_module.async_metric",
    )

  result = await evaluator.evaluate_invocations([], None)

  assert result.overall_score == 0.75
  assert observed_threads == [main_thread_id]
  assert received_thresholds == [None]
  assert eval_metric.threshold == 0.6


async def test_custom_metric_evaluator_sync_wrapper_returning_awaitable_awaits_result():
  """Synchronous wrappers returning an awaitable still resolve to EvaluationResult."""

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


@pytest.mark.parametrize("wrap_with_partial", [False, True])
async def test_custom_metric_evaluator_async_callable_object_runs_on_event_loop_thread(
    wrap_with_partial: bool,
):
  """Callable class instances with async __call__ (and partials) run on the event loop."""
  main_thread_id = threading.get_ident()
  observed_threads: list[int] = []

  class _AsyncCallableMetric:

    async def __call__(
        self,
        eval_metric: EvalMetric,
        actual_invocations: list[Invocation],
        expected_invocations: list[Invocation] | None,
        conversation_scenario: ConversationScenario | None,
        *,
        score: float = 0.95,
    ) -> EvaluationResult:
      del (
          eval_metric,
          actual_invocations,
          expected_invocations,
          conversation_scenario,
      )
      await asyncio.sleep(0)
      observed_threads.append(threading.get_ident())
      return EvaluationResult(overall_score=score)

  metric_callable: object = _AsyncCallableMetric()
  if wrap_with_partial:
    metric_callable = functools.partial(metric_callable, score=0.95)

  with mock.patch(
      "google.adk.evaluation.custom_metric_evaluator._get_metric_function",
      return_value=metric_callable,
  ):
    evaluator = _CustomMetricEvaluator(
        eval_metric=EvalMetric(
            metric_name="callable_obj_metric", threshold=0.5
        ),
        custom_function_path="test_module.callable_obj_metric",
    )

  result = await evaluator.evaluate_invocations([], None)

  assert result.overall_score == 0.95
  assert observed_threads == [main_thread_id]
