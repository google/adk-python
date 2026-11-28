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

import asyncio
from collections.abc import AsyncGenerator
import copy
import json
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_engine import EnvironmentSimulationEngine
from google.adk.tools.environment_simulation.environment_simulation_factory import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_plugin import EnvironmentSimulationPlugin
from google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy import ToolSpecMockStrategy
from google.genai import types
import pytest
from typing_extensions import override

from ... import testing_utils


@pytest.mark.asyncio
class TestEnvironmentSimulationPlugin:
  """Test cases for the EnvironmentSimulationPlugin."""

  @pytest.fixture
  def mock_simulator_engine(self):
    """Fixture for a mock EnvironmentSimulationEngine."""
    engine = MagicMock()
    engine.simulate = AsyncMock()
    return engine

  async def test_before_tool_callback(self, mock_simulator_engine):
    """Test that the before_tool_callback calls the engine's simulate method."""
    plugin = EnvironmentSimulationPlugin(mock_simulator_engine)

    mock_tool = MagicMock()
    mock_args = {}
    mock_context = MagicMock()

    await plugin.before_tool_callback(mock_tool, mock_args, mock_context)

    mock_simulator_engine.simulate.assert_awaited_once_with(
        mock_tool, mock_args, mock_context
    )

  @pytest.mark.parametrize("response", [{}, {"status": "ok"}])
  async def test_injected_response_skips_tool_execution(self, response):
    """An injected response, even {}, is returned without running the tool."""
    tool_calls = []

    def fetch_data() -> dict[str, bool]:
      tool_calls.append(True)
      return {"live": True}

    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="fetch_data",
                injection_configs=[InjectionConfig(injected_response=response)],
            )
        ]
    )
    model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name="fetch_data", args={}),
            types.Part.from_text(text="done"),
        ]
    )
    agent = LlmAgent(name="test_agent", model=model, tools=[fetch_data])
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=agent,
            plugins=[
                EnvironmentSimulationPlugin(EnvironmentSimulationEngine(config))
            ],
        )
    )

    events = await runner.run_async("fetch data")

    tool_responses = [
        result for event in events for result in event.get_function_responses()
    ]
    assert len(tool_responses) == 1
    assert tool_responses[0].response == response
    assert not tool_calls


class _FakeSimulationLlm(BaseLlm):
  """Answers the tool connection analyzer and the tool spec mock strategy."""

  created_tickets: ClassVar[int] = 0

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    prompt = llm_request.contents[-1].parts[0].text
    if "stateful dependencies" in prompt:
      body = {
          "stateful_parameters": [{
              "parameter_name": "ticket_id",
              "creating_tools": ["create_ticket"],
              "consuming_tools": [],
          }]
      }
    else:
      _FakeSimulationLlm.created_tickets += 1
      body = {"ticket_id": f"T-{_FakeSimulationLlm.created_tickets}"}
    yield LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(text=json.dumps(body))]
        )
    )


@pytest.mark.asyncio
async def test_eval_runs_do_not_share_simulated_state(mocker):
  """Each eval case run starts from an empty simulated state."""
  mocker.patch.object(LLMRegistry, "resolve", return_value=_FakeSimulationLlm)
  seen_state = []
  original_mock = ToolSpecMockStrategy.mock

  async def recording_mock(self, tool, args, tool_context, *rest):
    connection_map, state_store, *rest = rest
    seen_state.append(copy.deepcopy(state_store))
    return await original_mock(
        self, tool, args, tool_context, connection_map, state_store, *rest
    )

  mocker.patch.object(ToolSpecMockStrategy, "mock", recording_mock)

  def create_ticket(title: str) -> dict[str, str]:
    """Creates a ticket and returns its ticket_id."""
    raise AssertionError("The plugin simulates this tool.")

  num_cases, num_runs = 2, 2
  model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="create_ticket", args={"title": "t"}
          ),
          types.Part.from_text(text="done"),
      ]
      * (num_cases * num_runs)
  )
  agent = LlmAgent(name="test_agent", model=model, tools=[create_ticket])
  plugin = EnvironmentSimulationFactory.create_plugin(
      EnvironmentSimulationConfig(
          simulation_model="fake-simulation-model",
          tool_simulation_configs=[
              ToolSimulationConfig(
                  tool_name="create_ticket",
                  mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
              )
          ],
      )
  )
  app = App(name="test_app", root_agent=agent, plugins=[plugin])
  eval_sets_manager = InMemoryEvalSetsManager()
  eval_sets_manager.create_eval_set(app_name="test_app", eval_set_id="set")
  for case_index in range(num_cases):
    eval_sets_manager.add_eval_case(
        app_name="test_app",
        eval_set_id="set",
        eval_case=EvalCase(
            eval_id=f"case_{case_index}",
            conversation=[
                Invocation(
                    invocation_id=f"inv_{case_index}",
                    user_content=types.Content(
                        role="user", parts=[types.Part(text="new ticket")]
                    ),
                )
            ],
        ),
    )
  eval_service = LocalEvalService(
      root_agent=agent, eval_sets_manager=eval_sets_manager, app=app
  )
  results = [
      result
      async for result in eval_service.perform_inference(
          InferenceRequest(
              app_name="test_app",
              eval_set_id="set",
              inference_config=InferenceConfig(
                  parallelism=1, num_runs=num_runs
              ),
          )
      )
  ]

  assert [r.status for r in results] == [InferenceStatus.SUCCESS] * len(results)
  assert len(seen_state) == num_cases * num_runs
  assert seen_state == [{}] * len(seen_state)


class _SeededFaultModel(BaseLlm):
  """Makes the requested number of calls, then reports each tool outcome.

  Reads only the current turn's messages, so one model can serve interleaved
  case runs without sharing response counters or consulting the eval context.
  """

  model: str = "seeded-fault-test"

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    # Give concurrent eval runs a chance to interleave their matching calls.
    await asyncio.sleep(0)
    outcomes = []
    for content in reversed(llm_request.contents):
      for part in content.parts or []:
        if part.function_response:
          response = part.function_response.response
          outcomes.append("FAILED" if "error_code" in response else "OK")
      if content.role == "user" and content.parts and content.parts[0].text:
        num_calls = int(content.parts[0].text)
        break
    if len(outcomes) < num_calls:
      content = types.ModelContent(
          [types.Part.from_function_call(name="charge_card", args={})]
      )
    else:
      content = types.ModelContent(",".join(reversed(outcomes)))
    yield LlmResponse(content=content)


def _seeded_fault_eval_service(
    case_ids=("case_a", "case_b"), calls_per_turn=(8,)
) -> LocalEvalService:
  """Builds a real eval service with an offline model and seeded injection."""

  def charge_card() -> dict[str, str]:
    """Charges the card when the plugin does not inject an error."""
    return {"status": "charged"}

  agent = LlmAgent(name="agent", model=_SeededFaultModel(), tools=[charge_card])
  plugin = EnvironmentSimulationFactory.create_plugin(
      EnvironmentSimulationConfig(
          tool_simulation_configs=[
              ToolSimulationConfig(
                  tool_name="charge_card",
                  injection_configs=[
                      InjectionConfig(
                          injection_probability=0.5,
                          random_seed=42,
                          injected_error=InjectedError(
                              injected_http_error_code=503,
                              error_message="unavailable",
                          ),
                      )
                  ],
              )
          ]
      )
  )
  app = App(name="app", root_agent=agent, plugins=[plugin])
  manager = InMemoryEvalSetsManager()
  manager.create_eval_set("app", "set")
  for case_id in case_ids:
    manager.add_eval_case(
        "app",
        "set",
        EvalCase(
            eval_id=case_id,
            conversation=[
                Invocation(user_content=types.UserContent(str(num_calls)))
                for num_calls in calls_per_turn
            ],
        ),
    )
  return LocalEvalService(root_agent=agent, eval_sets_manager=manager, app=app)


async def _seeded_fault_patterns(
    service: LocalEvalService, *, case_ids=None, parallelism=1, num_runs=3
) -> dict[tuple[str, int], tuple[str, ...]]:
  """Reads tool outcomes from actual eval results, keyed by case and run."""
  patterns = {}
  async for result in service.perform_inference(
      InferenceRequest(
          app_name="app",
          eval_set_id="set",
          eval_case_ids=case_ids,
          inference_config=InferenceConfig(
              parallelism=parallelism, num_runs=num_runs
          ),
      )
  ):
    assert result.status == InferenceStatus.SUCCESS, result.error_message
    pattern = tuple(
        outcome
        for invocation in result.inferences
        for outcome in invocation.final_response.parts[0].text.split(",")
    )
    patterns[(result.eval_case_id, result.repetition_index)] = pattern
  assert patterns
  observed_case_ids = {case_id for case_id, _ in patterns}
  assert set(patterns) == {
      (case_id, repetition_index)
      for case_id in case_ids or observed_case_ids
      for repetition_index in range(num_runs)
  }
  return patterns


@pytest.mark.parametrize("case_ids", [None, ["case_a", "case_b"], ["case_a"]])
@pytest.mark.parametrize("parallelism", [1, 4])
async def test_seeded_faults_ignore_eval_case_selection(case_ids, parallelism):
  """Selecting other cases leaves each repetition's fault pattern unchanged."""
  baseline_service = _seeded_fault_eval_service()
  selected_service = _seeded_fault_eval_service()

  baseline = await _seeded_fault_patterns(baseline_service, case_ids=["case_a"])
  selected = await _seeded_fault_patterns(
      selected_service, case_ids=case_ids, parallelism=parallelism
  )

  for key, pattern in baseline.items():
    assert selected[key] == pattern


@pytest.mark.parametrize("parallelism", [1, 4])
async def test_seeded_faults_ignore_eval_case_order(parallelism):
  """Reordering stored cases leaves every case run's fault pattern unchanged."""
  forward_service = _seeded_fault_eval_service(("case_a", "case_b"))
  reverse_service = _seeded_fault_eval_service(("case_b", "case_a"))

  forward = await _seeded_fault_patterns(
      forward_service, parallelism=parallelism
  )
  reverse = await _seeded_fault_patterns(
      reverse_service, parallelism=parallelism
  )

  assert reverse == forward


@pytest.mark.parametrize("parallelism", [1, 4])
async def test_seeded_faults_restart_when_eval_reuses_plugin(parallelism):
  """Evaluating again with the same plugin restarts each case run's stream."""
  service = _seeded_fault_eval_service()

  first = await _seeded_fault_patterns(service, parallelism=parallelism)
  second = await _seeded_fault_patterns(service, parallelism=parallelism)

  assert second == first


async def test_seeded_fault_stream_continues_across_eval_turns():
  """Splitting matching calls across turns preserves the case run's stream."""
  single_turn_service = _seeded_fault_eval_service(calls_per_turn=(8,))
  multi_turn_service = _seeded_fault_eval_service(calls_per_turn=(2, 2, 2, 2))

  single_turn = await _seeded_fault_patterns(single_turn_service, parallelism=4)
  multi_turn = await _seeded_fault_patterns(multi_turn_service, parallelism=4)

  assert multi_turn == single_turn
  assert all(len(pattern) == 8 for pattern in multi_turn.values())


async def test_seeded_faults_sample_different_eval_repetitions():
  """Repeating a case samples distinct patterns instead of replaying one."""
  service = _seeded_fault_eval_service(("case_a",))

  patterns = await _seeded_fault_patterns(service, parallelism=4)

  assert len(set(patterns.values())) > 1
