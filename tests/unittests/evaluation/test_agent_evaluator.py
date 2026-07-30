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

"""Tests for AgentEvaluator."""

from __future__ import annotations

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.simulation.user_simulator_provider import UserSimulatorProvider
import pytest


def _make_eval_set() -> EvalSet:
  return EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[EvalCase(eval_id="case1", conversation=[])],
  )


async def _empty_async_gen(*args, **kwargs):
  """An async generator that yields nothing (mocks perform_inference/evaluate)."""
  return
  yield  # pragma: no cover - makes this a generator.


from google.adk.evaluation.eval_config import LiveModelConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "live_model_config, expected_use_live",
    [
        (LiveModelConfig(timeout_seconds=600), True),
        (None, False),
    ],
)
async def test_get_eval_results_by_eval_id_threads_live_model_config(
    live_model_config, expected_use_live, mocker
):
  """`live_model_config` is forwarded to the InferenceRequest's InferenceConfig."""
  mock_service = mocker.MagicMock()
  mock_service.perform_inference = mocker.MagicMock(
      side_effect=_empty_async_gen
  )
  mock_service.evaluate = mocker.MagicMock(side_effect=_empty_async_gen)
  mocker.patch(
      "google.adk.evaluation.local_eval_service.LocalEvalService",
      return_value=mock_service,
  )

  await AgentEvaluator._get_eval_results_by_eval_id(
      agent_for_eval=mocker.MagicMock(),
      eval_set=_make_eval_set(),
      eval_metrics=[],
      num_runs=1,
      user_simulator_provider=UserSimulatorProvider(),
      live_model_config=live_model_config,
  )

  # A single inference request should be issued carrying the live flag.
  mock_service.perform_inference.assert_called_once()
  inference_request = mock_service.perform_inference.call_args.kwargs[
      "inference_request"
  ]
  assert inference_request.inference_config.use_live is expected_use_live
  if live_model_config:
    assert inference_request.inference_config.live_timeout_seconds == 600


@pytest.mark.asyncio
async def test_evaluate_eval_set_threads_artifact_service(mocker):
  """The artifact_service passed to evaluate_eval_set reaches LocalEvalService."""
  my_service = InMemoryArtifactService()

  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=mocker.AsyncMock(return_value=mocker.MagicMock()),
  )

  # LocalEvalService is imported lazily inside _get_eval_results_by_eval_id, so
  # the patch target is its defining module.
  mock_local_eval_service_cls = mocker.patch(
      "google.adk.evaluation.local_eval_service.LocalEvalService"
  )

  async def _empty(*args, **kwargs):
    return
    yield  # Makes this an (empty) async generator.

  instance = mock_local_eval_service_cls.return_value
  instance.perform_inference = _empty
  instance.evaluate = _empty

  await AgentEvaluator.evaluate_eval_set(
      agent_module="my.agent.module",
      eval_set=EvalSet(eval_set_id="es1", eval_cases=[]),
      eval_config=EvalConfig(),
      num_runs=1,
      artifact_service=my_service,
  )

  assert (
      mock_local_eval_service_cls.call_args.kwargs["artifact_service"]
      is my_service
  )
