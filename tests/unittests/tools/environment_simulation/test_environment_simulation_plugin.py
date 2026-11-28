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
