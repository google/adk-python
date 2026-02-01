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
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
import pytest


@pytest.fixture(autouse=True)
def restore_metric_registry():
  original_registry = MetricEvaluatorRegistry._registry.copy()
  yield
  MetricEvaluatorRegistry._registry = original_registry


def fake_custom_metric(*_args, **_kwargs):
  return None


@pytest.mark.asyncio
async def test_evaluate_eval_set_registers_custom_metric(monkeypatch):
  eval_config = EvalConfig(
      criteria={"my_custom_metric": 0.5},
      custom_metrics={
          "my_custom_metric": {
              "code_config": {
                  "name": (
                      "tests.unittests.evaluation."
                      "test_agent_evaluator.fake_custom_metric"
                  ),
              },
          },
      },
  )
  eval_set = EvalSet(
      eval_set_id="eval_set",
      name="eval_set",
      eval_cases=[],
  )

  async def fake_get_agent_for_eval(*_args, **_kwargs):
    return object()

  async def fake_get_eval_results_by_eval_id(
      *_args, metric_evaluator_registry, **_kwargs
  ):
    eval_metric = EvalMetric(
        metric_name="my_custom_metric",
        threshold=0.5,
        criterion=BaseCriterion(threshold=0.5),
        custom_function_path=(
            "tests.unittests.evaluation.test_agent_evaluator.fake_custom_metric"
        ),
    )
    evaluator = metric_evaluator_registry.get_evaluator(eval_metric)
    assert isinstance(evaluator, _CustomMetricEvaluator)
    return {}

  monkeypatch.setattr(
      AgentEvaluator, "_get_agent_for_eval", fake_get_agent_for_eval
  )
  monkeypatch.setattr(
      AgentEvaluator,
      "_get_eval_results_by_eval_id",
      fake_get_eval_results_by_eval_id,
  )

  await AgentEvaluator.evaluate_eval_set(
      agent_module="dummy.module",
      eval_set=eval_set,
      eval_config=eval_config,
      num_runs=1,
      print_detailed_results=False,
  )


@pytest.mark.asyncio
async def test_evaluate_eval_set_does_not_register_without_custom_metrics(
    monkeypatch,
):
  eval_config = EvalConfig(criteria={"response_match_score": 0.8})
  eval_set = EvalSet(
      eval_set_id="eval_set",
      name="eval_set",
      eval_cases=[],
  )

  async def fake_get_agent_for_eval(*_args, **_kwargs):
    return object()

  async def fake_get_eval_results_by_eval_id(
      *_args, metric_evaluator_registry, **_kwargs
  ):
    eval_metric = EvalMetric(
        metric_name="my_custom_metric",
        threshold=0.5,
        criterion=BaseCriterion(threshold=0.5),
    )
    with pytest.raises(NotFoundError):
      metric_evaluator_registry.get_evaluator(eval_metric)
    return {}

  monkeypatch.setattr(
      AgentEvaluator, "_get_agent_for_eval", fake_get_agent_for_eval
  )
  monkeypatch.setattr(
      AgentEvaluator,
      "_get_eval_results_by_eval_id",
      fake_get_eval_results_by_eval_id,
  )

  await AgentEvaluator.evaluate_eval_set(
      agent_module="dummy.module",
      eval_set=eval_set,
      eval_config=eval_config,
      num_runs=1,
      print_detailed_results=False,
  )
