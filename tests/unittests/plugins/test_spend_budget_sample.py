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

"""Tests for the SpendBudgetPlugin in contributing/samples/plugins/spend_budget."""

from __future__ import annotations

import asyncio
from collections import Counter
import importlib.util
from pathlib import Path
from typing import Any
from typing import AsyncGenerator

from google.adk import Agent
from google.adk import Workflow
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import JoinNode
from google.genai import types
import pytest

pytest.importorskip("agent_budget")

_PLUGIN_PATH = (
    Path(__file__).parents[3]
    / "contributing"
    / "samples"
    / "plugins"
    / "spend_budget"
    / "spend_budget_plugin.py"
)
_spec = importlib.util.spec_from_file_location(
    "spend_budget_plugin", _PLUGIN_PATH
)
spend_budget_plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spend_budget_plugin)
SpendBudgetPlugin = spend_budget_plugin.SpendBudgetPlugin
SPEND_BUDGET_EXCEEDED = spend_budget_plugin.SPEND_BUDGET_EXCEEDED

# Every scripted call reports 2,000 input and 200 output tokens. At $1 per
# million input and $10 per million output tokens that is $0.004 a call.
CALL_USD = 0.004


class _ScriptedLlm(BaseLlm):
  """Answers at once, asks for `ping` on every call, or raises."""

  model: str = "scripted"
  agent_name: str = ""
  behavior: str = "answer"
  calls: Any  # A Counter shared with the test; Any keeps pydantic from copying.

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.calls[self.agent_name] += 1
    await asyncio.sleep(0)
    if self.behavior == "raise":
      raise RuntimeError("model unavailable")
    if self.behavior == "loop":
      part = types.Part(function_call=types.FunctionCall(name="ping", args={}))
    else:
      part = types.Part(text="done")
    yield LlmResponse(
        content=types.Content(role="model", parts=[part]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=2_000, candidates_token_count=200
        ),
    )


def _ping() -> str:
  """Returns pong."""
  return "pong"


def _plugin(**kwargs) -> SpendBudgetPlugin:
  kwargs.setdefault("input_usd_per_million", 1.0)
  kwargs.setdefault("output_usd_per_million", 10.0)
  # Reserve 500 output tokens per call: an estimate of about $0.005.
  kwargs.setdefault("default_output_tokens", 500)
  return SpendBudgetPlugin(**kwargs)


def _agent(
    name: str, behavior: str, calls: Counter[str], mode: str = "single_turn"
) -> LlmAgent:
  return Agent(
      name=name,
      model=_ScriptedLlm(agent_name=name, behavior=behavior, calls=calls),
      tools=[_ping],
      mode=mode,
  )


def _fan_out(agents: list[LlmAgent]) -> Workflow:
  return Workflow(
      name="root_agent",
      edges=[("START", tuple(agents), JoinNode(name="join_for_results"))],
  )


async def _run(node, plugin: SpendBudgetPlugin) -> list:
  runner = InMemoryRunner(
      app=App(name="spend_budget_test", root_agent=node, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id="user"
  )
  return [
      event
      async for event in runner.run_async(
          user_id="user",
          session_id=session.id,
          new_message=types.UserContent("go"),
      )
  ]


def _assert_released(plugin: SpendBudgetPlugin) -> None:
  assert not plugin._runs
  assert not plugin._scopes
  assert not plugin._in_flight


@pytest.mark.asyncio
async def test_agent_cap_stops_a_looping_agent_and_the_run_finishes():
  calls: Counter[str] = Counter()
  plugin = _plugin(run_usd=1.0, agent_usd=0.03)
  workflow = _fan_out([
      _agent("looper", "loop", calls),
      _agent("writer", "answer", calls),
      _agent("checker", "answer", calls),
  ])

  events = await _run(workflow, plugin)

  # Each call reserves about $0.005 and books $0.004. With $0.024 booked
  # after 6 calls, a 7th still fits in $0.03 and an 8th does not.
  assert calls == Counter(looper=7, writer=1, checker=1)
  refused = [e for e in events if e.error_code == SPEND_BUDGET_EXCEEDED]
  assert len(refused) == 1
  assert refused[0].author == "looper"
  assert "Agent 'looper' stopped" in refused[0].error_message
  _assert_released(plugin)


@pytest.mark.asyncio
async def test_each_node_run_of_one_agent_name_has_its_own_cap():
  calls: Counter[str] = Counter()
  plugin = _plugin(run_usd=1.0, agent_usd=0.03)
  workflow = Workflow(
      name="root_agent",
      edges=[(
          "START",
          (
              _agent("looper", "loop", calls),
              Workflow(
                  name="inner",
                  edges=[("START", _agent("looper", "loop", calls))],
              ),
          ),
          JoinNode(name="join_for_results"),
      )],
  )

  await _run(workflow, plugin)

  assert calls == Counter(looper=14)
  _assert_released(plugin)


@pytest.mark.asyncio
async def test_parallel_agents_cannot_start_more_than_the_run_budget():
  calls: Counter[str] = Counter()
  plugin = _plugin(run_usd=0.05)
  workflow = _fan_out([_agent(f"looper_{i}", "loop", calls) for i in range(4)])

  await _run(workflow, plugin)

  total = sum(calls.values())
  assert total == 12
  assert total * CALL_USD <= 0.05
  _assert_released(plugin)


@pytest.mark.asyncio
async def test_agent_overrides_take_precedence_over_agent_usd():
  calls: Counter[str] = Counter()
  plugin = _plugin(
      run_usd=1.0, agent_usd=0.03, agent_usd_overrides={"careful": 0.01}
  )
  workflow = _fan_out([
      _agent("looper", "loop", calls),
      _agent("careful", "loop", calls),
  ])

  await _run(workflow, plugin)

  assert calls == Counter(looper=7, careful=2)
  _assert_released(plugin)


@pytest.mark.asyncio
async def test_a_call_whose_estimate_does_not_fit_is_never_made():
  calls: Counter[str] = Counter()
  plugin = _plugin(run_usd=1.0, agent_usd=0.001)
  events = await _run(_agent("writer", "answer", calls, mode="chat"), plugin)

  assert calls == Counter()
  assert [e.error_code for e in events if e.error_code] == [
      SPEND_BUDGET_EXCEEDED
  ]
  _assert_released(plugin)


@pytest.mark.asyncio
async def test_failed_model_call_releases_its_reservation():
  calls: Counter[str] = Counter()
  plugin = _plugin(run_usd=1.0, agent_usd=0.03)

  with pytest.raises(RuntimeError, match="model unavailable"):
    await _run(_agent("flaky", "raise", calls, mode="chat"), plugin)

  assert calls == Counter(flaky=1)
  _assert_released(plugin)


def test_cost_and_estimate_use_the_configured_prices():
  plugin = _plugin(run_usd=1.0)
  assert plugin.cost_usd(2_000, 200) == pytest.approx(CALL_USD)
  assert plugin.estimate_usd(
      LlmRequest(
          contents=[types.UserContent("x" * 400)],
          config=types.GenerateContentConfig(max_output_tokens=100),
      )
  ) == pytest.approx((101 * 1.0 + 100 * 10.0) / 1_000_000)
