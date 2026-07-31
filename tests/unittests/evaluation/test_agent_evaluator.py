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

from types import SimpleNamespace

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
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
      new=mocker.AsyncMock(return_value=(mocker.MagicMock(), None)),
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


class TestGetAgentForEval:
  """Resolution of the wrapping App alongside the agent to evaluate."""

  @pytest.mark.asyncio
  async def test_resolves_app_when_module_exposes_one(self, mocker):
    """When the module's `agent` exposes an `app`, it is returned too."""
    root_agent = BaseAgent(name="root_agent")
    app = App(name="my_app", root_agent=root_agent)
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app=app)
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is app

  @pytest.mark.asyncio
  async def test_returns_none_app_when_module_has_no_app(self, mocker):
    """When only `root_agent` is exposed, app is None."""
    root_agent = BaseAgent(name="root_agent")
    fake_module = SimpleNamespace(agent=SimpleNamespace(root_agent=root_agent))
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is None

  @pytest.mark.asyncio
  async def test_ignores_app_attribute_that_is_not_an_app(self, mocker):
    """A non-App `app` attribute is ignored and app resolves to None."""
    root_agent = BaseAgent(name="root_agent")
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app="not-an-app")
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is None

  @pytest.mark.asyncio
  async def test_surfaces_app_even_when_selecting_sub_agent(self, mocker):
    """A sub-agent is returned for eval, but the wrapping App is still surfaced."""
    sub_agent = BaseAgent(name="sub_agent")
    root_agent = BaseAgent(name="root_agent", sub_agents=[sub_agent])
    app = App(name="my_app", root_agent=root_agent)
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app=app)
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module", agent_name="sub_agent"
    )

    assert resolved_agent is sub_agent
    assert resolved_app is app


class TestGetEvalResultsByEvalId:
  """The pytest-gate path forwards the App into LocalEvalService."""

  @staticmethod
  def _empty_async_gen_factory():
    async def _agen(*args, **kwargs):
      return
      yield  # pragma: no cover - marks this as an async generator

    return _agen

  @pytest.mark.asyncio
  async def test_app_is_forwarded_to_local_eval_service(self, mocker):
    """`_get_eval_results_by_eval_id` passes `app=` into LocalEvalService."""
    root_agent = BaseAgent(name="root_agent")
    app = App(name="my_app", root_agent=root_agent)

    mock_service_cls = mocker.patch(
        "google.adk.evaluation.local_eval_service.LocalEvalService"
    )
    mock_service = mock_service_cls.return_value
    mock_service.perform_inference = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )
    mock_service.evaluate = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )

    await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=root_agent,
        eval_set=EvalSet(eval_set_id="set-1", eval_cases=[]),
        eval_metrics=[],
        num_runs=1,
        user_simulator_provider=UserSimulatorProvider(),
        app=app,
    )

    assert mock_service_cls.call_args.kwargs["app"] is app

  @pytest.mark.asyncio
  async def test_none_app_is_forwarded_by_default(self, mocker):
    """When no App is provided, LocalEvalService receives app=None."""
    root_agent = BaseAgent(name="root_agent")

    mock_service_cls = mocker.patch(
        "google.adk.evaluation.local_eval_service.LocalEvalService"
    )
    mock_service = mock_service_cls.return_value
    mock_service.perform_inference = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )
    mock_service.evaluate = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )

    await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=root_agent,
        eval_set=EvalSet(eval_set_id="set-1", eval_cases=[]),
        eval_metrics=[],
        num_runs=1,
        user_simulator_provider=UserSimulatorProvider(),
    )

    assert mock_service_cls.call_args.kwargs["app"] is None
