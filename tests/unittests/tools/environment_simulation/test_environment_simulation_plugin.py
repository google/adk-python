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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation import UnmockedToolCallError
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_engine import EnvironmentSimulationEngine
from google.adk.tools.environment_simulation.environment_simulation_plugin import EnvironmentSimulationPlugin
from google.genai import types
import pytest

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
  @pytest.mark.parametrize("policy", ["passthrough", "error"])
  async def test_injected_response_skips_tool_execution(self, response, policy):
    """An injected response, even {}, is returned without running the tool."""
    tool_calls = []

    def fetch_data() -> dict[str, bool]:
      tool_calls.append(True)
      return {"live": True}

    config = EnvironmentSimulationConfig(
        unmocked_tool_policy=policy,
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="fetch_data",
                injection_configs=[InjectionConfig(injected_response=response)],
            )
        ],
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


def _make_runner(config, tools, calls, *, attachment="plugin", sub_agent=False):
  """Runs scripted calls through the real before-tool pipeline, without an API."""
  agent = LlmAgent(
      name="worker",
      model=testing_utils.MockModel.create(
          responses=[calls, [types.Part.from_text(text="done")]]
      ),
      tools=tools,
  )
  plugins = []
  if attachment == "callback":
    agent.before_tool_callback = EnvironmentSimulationFactory.create_callback(
        config
    )
  else:
    plugins = [EnvironmentSimulationFactory.create_plugin(config)]
  root = (
      SequentialAgent(name="root", sub_agents=[agent]) if sub_agent else agent
  )
  return testing_utils.InMemoryRunner(
      app=App(name="strict_simulation", root_agent=root, plugins=plugins)
  )


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", ["callback", "plugin"])
@pytest.mark.parametrize("gap", ["unconfigured", "mismatch", "probability"])
@pytest.mark.parametrize("policy", ["passthrough", "error"])
async def test_unmocked_tool_execution_policy(attachment, gap, policy):
  real_calls = []

  def fetch_data(key: str) -> dict:
    real_calls.append(key)
    return {"live": True}

  config = EnvironmentSimulationConfig(
      unmocked_tool_policy=policy,
      tool_simulation_configs=[
          ToolSimulationConfig(
              tool_name="other_tool" if gap == "unconfigured" else "fetch_data",
              injection_configs=[
                  InjectionConfig(
                      match_args={"key": "fixture"},
                      injection_probability=0.0
                      if gap == "probability"
                      else 1.0,
                      injected_response={},
                  )
              ],
          )
      ],
  )
  key = "unmatched-sensitive-value" if gap == "mismatch" else "fixture"
  runner = _make_runner(
      config,
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={"key": key})],
      attachment=attachment,
  )

  if policy == "error":
    expected_error = (
        RuntimeError if attachment == "plugin" else UnmockedToolCallError
    )
    with pytest.raises(expected_error, match="fetch_data.*blocked") as exc:
      await runner.run_async("fetch data")
    if attachment == "plugin":
      assert isinstance(exc.value.__cause__, UnmockedToolCallError)
    reason = (
        "no tool simulation configuration"
        if gap == "unconfigured"
        else "no injection applied and no fallback mock strategy"
    )
    assert reason in str(exc.value)
    assert key not in str(exc.value)
    assert real_calls == []
  else:
    events = await runner.run_async("fetch data")
    assert real_calls == [key]
    assert [
        response.response
        for event in events
        for response in event.get_function_responses()
    ] == [{"live": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_name", ["fetch_data", "FETCH_DATA", "fetch_*"]
)
@pytest.mark.parametrize("configured", [False, True])
async def test_allowlist_requires_exact_name(allowed_name, configured):
  real_calls = []

  def fetch_data() -> dict:
    real_calls.append(True)
    return {"live": True}

  config = EnvironmentSimulationConfig(
      unmocked_tool_policy="error", allowed_real_tools=[allowed_name]
  )
  if configured:
    config.tool_simulation_configs = [
        ToolSimulationConfig(
            tool_name="fetch_data",
            injection_configs=[
                InjectionConfig(injection_probability=0.0, injected_response={})
            ],
        )
    ]
  runner = _make_runner(
      config,
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={})],
  )

  if allowed_name == "fetch_data":
    await runner.run_async("fetch data")
    assert real_calls == [True]
  else:
    with pytest.raises(RuntimeError, match="fetch_data.*blocked"):
      await runner.run_async("fetch data")
    assert real_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("inject_error", [False, True])
async def test_injection_precedes_allowlist(inject_error):
  real_calls = []

  def fetch_data() -> dict:
    real_calls.append(True)
    return {"live": True}

  injection = (
      InjectionConfig(
          injected_error=InjectedError(
              injected_http_error_code=503, error_message="simulated outage"
          )
      )
      if inject_error
      else InjectionConfig(injected_response={})
  )
  config = EnvironmentSimulationConfig(
      unmocked_tool_policy="error",
      allowed_real_tools=["fetch_data"],
      tool_simulation_configs=[
          ToolSimulationConfig(
              tool_name="fetch_data", injection_configs=[injection]
          )
      ],
  )
  runner = _make_runner(
      config,
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={})],
  )

  events = await runner.run_async("fetch data")

  assert real_calls == []
  responses = [r.response for e in events for r in e.get_function_responses()]
  assert responses == (
      [{"error_code": 503, "error_message": "simulated outage"}]
      if inject_error
      else [{}]
  )


@pytest.mark.asyncio
async def test_strict_plugin_blocks_parallel_sub_agent_tools():
  real_calls = []

  def first_tool() -> dict:
    real_calls.append("first")
    return {}

  def second_tool() -> dict:
    real_calls.append("second")
    return {}

  runner = _make_runner(
      EnvironmentSimulationConfig(unmocked_tool_policy="error"),
      [first_tool, second_tool],
      [
          types.Part.from_function_call(name=name, args={})
          for name in ("first_tool", "second_tool")
      ],
      sub_agent=True,
  )

  with pytest.raises(
      RuntimeError, match="blocked by strict environment simulation"
  ):
    await runner.run_async("run tools")
  assert real_calls == []


@pytest.mark.asyncio
async def test_strict_plugin_precedes_later_overrides():
  calls = []

  class OverridePlugin(BasePlugin):

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
      calls.append("override")
      return {"override": True}

  def fetch_data() -> dict:
    calls.append("real tool")
    return {}

  runner = _make_runner(
      EnvironmentSimulationConfig(unmocked_tool_policy="error"),
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={})],
  )
  runner.runner.plugin_manager.plugins.append(OverridePlugin(name="override"))

  with pytest.raises(RuntimeError, match="fetch_data.*blocked"):
    await runner.run_async("fetch data")
  assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize(
    "outcome",
    [
        {},
        {"mocked": True},
        None,
        [],
        "invalid",
        RuntimeError("generator unavailable"),
    ],
)
async def test_strict_mock_generation_never_falls_through(allowed, outcome):
  real_calls = []

  def fetch_data() -> dict:
    real_calls.append(True)
    return {"live": True}

  config = EnvironmentSimulationConfig(
      unmocked_tool_policy="error",
      allowed_real_tools=["fetch_data"] if allowed else [],
      tool_simulation_configs=[
          ToolSimulationConfig(
              tool_name="fetch_data",
              mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
          )
      ],
  )
  runner = _make_runner(
      config,
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={})],
  )
  # Only the model-backed analysis and generation are substituted. The Runner,
  # plugin, engine and counting tool exercise the actual execution boundary.
  with (
      patch(
          "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer.analyze",
          new=AsyncMock(return_value=None),
      ),
      patch(
          "google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy.ToolSpecMockStrategy.mock",
          new=AsyncMock(**(
              {"side_effect": outcome}
              if isinstance(outcome, Exception)
              else {"return_value": outcome}
          )),
      ),
  ):
    if isinstance(outcome, dict):
      events = await runner.run_async("fetch data")
      assert [
          r.response for e in events for r in e.get_function_responses()
      ] == [outcome]
    else:
      reason = (
          "generator unavailable"
          if isinstance(outcome, Exception)
          else "must return a dictionary"
      )
      with pytest.raises(RuntimeError, match=reason):
        await runner.run_async("fetch data")
  assert real_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_response", ["sensitive invalid output", "[]", "null"]
)
async def test_strict_invalid_generated_json_blocks_allowlisted_tool(
    raw_response,
):
  real_calls = []

  def fetch_data() -> dict:
    real_calls.append(True)
    return {}

  config = EnvironmentSimulationConfig(
      unmocked_tool_policy="error",
      allowed_real_tools=["fetch_data"],
      tool_simulation_configs=[
          ToolSimulationConfig(
              tool_name="fetch_data",
              mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
          )
      ],
  )
  runner = _make_runner(
      config,
      [fetch_data],
      [types.Part.from_function_call(name="fetch_data", args={})],
  )
  with (
      patch(
          "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer.analyze",
          new=AsyncMock(return_value=None),
      ),
      patch(
          "google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy.LLMRegistry",
      ) as registry,
  ):
    registry.return_value.resolve.return_value = MagicMock(
        return_value=testing_utils.MockModel.create(responses=[raw_response])
    )
    with pytest.raises(
        RuntimeError, match="Mock strategy for tool 'fetch_data' failed"
    ) as exc:
      await runner.run_async("fetch data")

  assert real_calls == []
  assert raw_response not in str(exc.value)
