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

"""Unit tests for the App-aware threading in AgentEvaluator."""

from __future__ import annotations

from types import SimpleNamespace

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.simulation.user_simulator_provider import UserSimulatorProvider
import pytest


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
