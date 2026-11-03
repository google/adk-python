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
import logging
import re
from typing import Any
from typing import ClassVar
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
from google.adk.tools.environment_simulation.environment_simulation_config import InjectionConfig
from google.adk.tools.environment_simulation.environment_simulation_config import MockStrategy
from google.adk.tools.environment_simulation.environment_simulation_config import ToolSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_engine import _MAX_CONCURRENT_ARG_MATCHES
from google.adk.tools.environment_simulation.environment_simulation_engine import EnvironmentSimulationEngine
from google.adk.tools.tool_args_matcher import _semantic_arg_matcher
from google.adk.tools.tool_args_matcher import ArgMatchResult
from google.adk.tools.tool_args_matcher import BaseArgMatcher
from google.adk.tools.tool_args_matcher import SemanticArgMatcher
from google.adk.tools.tool_args_matcher import ToolArgsMatcher
from google.genai import types as genai_types
from pydantic import PrivateAttr
import pytest


@patch(
    "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer"
)
@patch(
    "google.adk.tools.environment_simulation.environment_simulation_engine._create_mock_strategy"
)
@pytest.mark.asyncio
class TestEnvironmentSimulationEngineSimulate:
  """Test cases for the simulate method of EnvironmentSimulationEngine."""

  async def test_simulate_no_op_for_unconfigured_tool(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that simulate returns None for a tool not in the config."""
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="configured_tool",
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = EnvironmentSimulationEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "unconfigured_tool"
    result = await engine.simulate(mock_tool, {}, MagicMock())
    assert result is None

  @pytest.mark.parametrize("response", [{}, {"injected": True}])
  @pytest.mark.parametrize(
      "mock_strategy_type",
      [
          MockStrategy.MOCK_STRATEGY_UNSPECIFIED,
          MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
      ],
  )
  async def test_injection_with_matching_args(
      self, mock_create_strategy, mock_analyzer, response, mock_strategy_type
  ):
    """Test that an injection is applied when match_args match."""
    mock_create_strategy.return_value.mock = AsyncMock(
        return_value={"mocked": True}
    )
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response=response,
                    )
                ],
                mock_strategy_type=mock_strategy_type,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = EnvironmentSimulationEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    result = await engine.simulate(mock_tool, {"param": "value"}, MagicMock())
    assert result == response
    mock_create_strategy.assert_not_called()

  async def test_injection_not_applied_with_mismatched_args(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that an injection is not applied when match_args do not match."""
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.mock = AsyncMock(return_value={"mocked": True})
    mock_create_strategy.return_value = mock_strategy_instance
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = EnvironmentSimulationEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    result = await engine.simulate(
        mock_tool, {"param": "different_value"}, MagicMock()
    )
    assert result == {"mocked": True}
    mock_create_strategy.assert_called_once_with(
        config.tool_simulation_configs[0].mock_strategy_type,
        config.simulation_model,
        config.simulation_model_configuration,
    )
    mock_strategy_instance.mock.assert_awaited_once()

  async def test_no_op_when_no_injection_hit_and_unspecified_strategy(
      self, mock_create_strategy, mock_analyzer, caplog
  ):
    """Test for no-op and warning when no injection hits and mock strategy is unspecified."""
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        match_args={"param": "value"},
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_UNSPECIFIED,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = EnvironmentSimulationEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    caplog.set_level(logging.WARNING, logger="environment_simulation_logger")
    with caplog.at_level(
        logging.WARNING, logger="environment_simulation_logger"
    ):
      result = await engine.simulate(
          mock_tool, {"param": "different_value"}, MagicMock()
      )
      assert result is None
      assert (
          "did not hit any injection config and has no mock strategy"
          in caplog.text
      )
    mock_create_strategy.assert_not_called()

  async def test_injection_with_random_seed_is_deterministic(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that an injection with a random_seed is deterministic."""
    # With seed=42, random.random() is > 0.5, so this will NOT be injected
    # and should fall back to the mock strategy.
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.mock = AsyncMock(return_value={"mocked": True})
    mock_create_strategy.return_value = mock_strategy_instance
    config_mocked = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injection_probability=0.5,
                        random_seed=42,  # A fixed seed
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine_mocked = EnvironmentSimulationEngine(config_mocked)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    result1 = await engine_mocked.simulate(mock_tool, {}, MagicMock())
    assert result1 == {"mocked": True}
    mock_create_strategy.assert_called_once_with(
        config_mocked.tool_simulation_configs[0].mock_strategy_type,
        config_mocked.simulation_model,
        config_mocked.simulation_model_configuration,
    )
    mock_strategy_instance.mock.assert_awaited_once()

    mock_create_strategy.reset_mock()
    mock_strategy_instance.mock.reset_mock()

    # With seed=100, random.random() is < 0.5, so this WILL be injected.
    config_injected = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injection_probability=0.5,
                        random_seed=100,  # A different fixed seed
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine_injected = EnvironmentSimulationEngine(config_injected)
    result2 = await engine_injected.simulate(mock_tool, {}, MagicMock())
    assert result2 == {"injected": True}
    mock_create_strategy.assert_not_called()

  async def test_seeded_injection_draws_a_reproducible_sequence(
      self, mock_create_strategy, mock_analyzer
  ):
    """Test that a seeded injection draws a new value on each matching call."""
    del mock_analyzer
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.mock = AsyncMock(return_value={"mocked": True})
    mock_create_strategy.return_value = mock_strategy_instance
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injection_probability=0.3,
                        random_seed=42,
                        match_args={"target": True},
                        injected_response={"injected": True},
                    )
                ],
                mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    async def run_pattern(interleave_non_matching):
      engine = EnvironmentSimulationEngine(config)
      pattern = []
      for _ in range(50):
        if interleave_non_matching:
          await engine.simulate(mock_tool, {"target": False}, MagicMock())
        result = await engine.simulate(mock_tool, {"target": True}, MagicMock())
        pattern.append(result == {"injected": True})
      return pattern

    pattern = await run_pattern(interleave_non_matching=False)
    assert any(pattern)
    assert not all(pattern)
    # A call that does not match draws nothing, so interleaving one leaves the
    # matching calls on the same sequence.
    assert await run_pattern(interleave_non_matching=True) == pattern

  async def test_injected_latency_awaits_asyncio_sleep(
      self, mock_create_strategy, mock_analyzer
  ):
    """Regression guard against blocking time.sleep in the async path."""
    del mock_create_strategy, mock_analyzer
    latency = 0.2
    config = EnvironmentSimulationConfig(
        tool_simulation_configs=[
            ToolSimulationConfig(
                tool_name="test_tool",
                injection_configs=[
                    InjectionConfig(
                        injected_latency_seconds=latency,
                        injected_response={"injected": True},
                    )
                ],
            )
        ],
        simulation_model="test-model",
        simulation_model_configuration=genai_types.GenerateContentConfig(),
    )
    engine = EnvironmentSimulationEngine(config)
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    with patch(
        "google.adk.tools.environment_simulation.environment_simulation_engine.asyncio.sleep",
        new_callable=AsyncMock,
    ) as mock_sleep:
      result = await engine.simulate(mock_tool, {}, MagicMock())

    mock_sleep.assert_awaited_once_with(latency)
    assert result == {"injected": True}


class _ScriptedMatcher(BaseArgMatcher):
  """A non-deterministic matcher whose answer is set per expected value."""

  type: Literal["scripted"] = "scripted"
  is_deterministic: ClassVar[bool] = False

  answers: dict[str, Any]
  """The result for each expected value: a bool, or an exception to raise."""
  block_on: str | None = None

  _calls: list[str] = PrivateAttr(default_factory=list)
  _cancelled: list[str] = PrivateAttr(default_factory=list)

  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    self._calls.append(str(expected))
    if expected == self.block_on:
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        self._cancelled.append(str(expected))
        raise
    if isinstance(self.answers.get(str(expected)), Exception):
      raise self.answers[str(expected)]
    return ArgMatchResult(matched=self.answers[str(expected)])


def _semantic_config(
    injection_configs: list[InjectionConfig],
    mock_strategy_type: MockStrategy = MockStrategy.MOCK_STRATEGY_UNSPECIFIED,
    simulation_model: str = "test-model",
) -> EnvironmentSimulationConfig:
  return EnvironmentSimulationConfig(
      tool_simulation_configs=[
          ToolSimulationConfig(
              tool_name="search",
              injection_configs=injection_configs,
              mock_strategy_type=mock_strategy_type,
          )
      ],
      simulation_model=simulation_model,
      simulation_model_configuration=genai_types.GenerateContentConfig(),
  )


def _search_tool() -> MagicMock:
  tool = MagicMock()
  tool.name = "search"
  return tool


class _RecordingJudge:
  """Answers YES to every comparison, recording the model and argument."""

  def __init__(self):
    self.calls: list[tuple[str, str]] = []

  async def generate_content_async(self, llm_request, stream=False):
    del stream
    prompt = llm_request.contents[0].parts[0].text
    arg_name = re.search(r"Argument: (\S+)", prompt).group(1)
    self.calls.append((llm_request.model, arg_name))
    yield LlmResponse(
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text="YES")]
        )
    )


@patch(
    "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer"
)
@pytest.mark.asyncio
class TestEnvironmentSimulationEngineArgsMatcher:
  """Tests for InjectionConfig.args_matcher in simulate."""

  async def test_semantic_matcher_applies_rephrased_call(
      self, mock_analyzer, monkeypatch
  ):
    """A semantically equivalent query hits the recorded injection."""
    judge = MagicMock()

    async def _generate(llm_request, stream=False):
      yield LlmResponse(
          content=genai_types.Content(
              role="model", parts=[genai_types.Part(text="YES")]
          )
      )

    judge.generate_content_async = _generate
    requested_models = []

    def _new_llm(model):
      requested_models.append(model)
      return judge

    monkeypatch.setattr(
        _semantic_arg_matcher.LLMRegistry,
        "new_llm",
        staticmethod(_new_llm),
    )
    config = InjectionConfig.model_validate({
        "match_args": {"query": "employees in NYC", "limit": 10},
        "args_matcher": {"per_arg_matchers": {"query": {"type": "semantic"}}},
        "injected_response": {"rows": 3},
    })
    engine = EnvironmentSimulationEngine(_semantic_config([config]))

    result = await engine.simulate(
        _search_tool(),
        {"query": "New York City workforce", "limit": 10},
        MagicMock(),
    )

    assert result == {"rows": 3}
    assert requested_models == ["test-model"]

  async def test_semantic_matcher_without_model_uses_each_engine_model(
      self, mock_analyzer, monkeypatch
  ):
    """Each engine fills in its own simulation_model and keeps its own cache.

    Setup: one config, shared by two engines with different simulation models,
      whose default semantic matcher names no model and whose "other"
      argument pins one.
    Act: send the same call through both engines.
    Assert: each engine asks its own model about "query" (so the second engine
      did not reuse the first engine's cached verdict), "other" asks the
      pinned model, and the shared config is left without a model.
    """
    judge = _RecordingJudge()
    monkeypatch.setattr(
        _semantic_arg_matcher.LLMRegistry,
        "new_llm",
        staticmethod(lambda model: judge),
    )
    config = InjectionConfig(
        match_args={"query": "a", "other": "b"},
        args_matcher=ToolArgsMatcher(
            default_matcher=SemanticArgMatcher(),
            per_arg_matchers={"other": SemanticArgMatcher(model="pinned")},
        ),
        injected_response={"injected": True},
    )
    engine_one = EnvironmentSimulationEngine(
        _semantic_config([config], simulation_model="model-one")
    )
    engine_two = EnvironmentSimulationEngine(
        _semantic_config([config], simulation_model="model-two")
    )
    args = {"query": "x", "other": "y"}

    result_one = await engine_one.simulate(_search_tool(), args, MagicMock())
    result_two = await engine_two.simulate(_search_tool(), args, MagicMock())

    assert result_one == result_two == {"injected": True}
    assert ("model-one", "query") in judge.calls
    assert ("model-two", "query") in judge.calls
    assert judge.calls.count(("pinned", "other")) == 2
    assert config.args_matcher.default_matcher.model is None

  async def test_unmatched_args_matcher_falls_back_to_mock_strategy(
      self, mock_analyzer
  ):
    """A mismatch through args_matcher falls through like an exact miss."""
    matcher = _ScriptedMatcher(answers={"a": False})
    config = InjectionConfig(
        match_args={"query": "a"},
        args_matcher=ToolArgsMatcher(default_matcher=matcher),
        injected_response={"injected": True},
    )
    engine = EnvironmentSimulationEngine(
        _semantic_config(
            [config], mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC
        )
    )
    strategy = MagicMock()
    strategy.mock = AsyncMock(return_value={"mocked": True})

    with patch(
        "google.adk.tools.environment_simulation.environment_simulation_engine._create_mock_strategy",
        return_value=strategy,
    ):
      result = await engine.simulate(
          _search_tool(), {"query": "b"}, MagicMock()
      )

    assert result == {"mocked": True}

  async def test_first_matching_config_wins(self, mock_analyzer):
    """Priority order is kept even though comparisons run concurrently."""
    matcher = _ScriptedMatcher(answers={"a": False, "b": True, "c": True})
    configs = [
        InjectionConfig(
            match_args={"query": value},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": value},
        )
        for value in ("a", "b", "c")
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert result == {"from": "b"}

  async def test_exact_config_before_args_matcher_wins(self, mock_analyzer):
    """A config without args_matcher still competes in priority order."""
    matcher = _ScriptedMatcher(answers={"x": True}, block_on="x")
    configs = [
        InjectionConfig(
            match_args={"query": "x"},
            injected_response={"from": "exact"},
        ),
        InjectionConfig(
            match_args={"query": "x"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": "matcher"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await asyncio.wait_for(
        engine.simulate(_search_tool(), {"query": "x"}, MagicMock()),
        timeout=5,
    )

    assert result == {"from": "exact"}

  async def test_later_exact_match_beats_earlier_semantic_match(
      self, mock_analyzer
  ):
    """A call equal to a later config's match_args skips earlier matchers.

    The earlier config's matcher would answer YES, but an exact hit does not
    depend on a model's verdict and costs no comparison.
    """
    matcher = _ScriptedMatcher(
        answers={"python jobs": True, "python developer jobs in NYC": True}
    )
    configs = [
        InjectionConfig(
            match_args={"query": value},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": value},
        )
        for value in ("python jobs", "python developer jobs in NYC")
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(
        _search_tool(),
        {"query": "python developer jobs in NYC"},
        MagicMock(),
    )

    assert result == {"from": "python developer jobs in NYC"}
    assert matcher._calls == []

  async def test_later_plain_exact_config_beats_earlier_semantic_match(
      self, mock_analyzer
  ):
    """A later config without args_matcher also wins on an exact hit."""
    matcher = _ScriptedMatcher(answers={"broad": True})
    configs = [
        InjectionConfig(
            match_args={"query": "broad"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": "semantic"},
        ),
        InjectionConfig(
            match_args={"query": "narrow"},
            injected_response={"from": "exact"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(
        _search_tool(), {"query": "narrow"}, MagicMock()
    )

    assert result == {"from": "exact"}
    assert matcher._calls == []

  async def test_exact_check_honors_disallowed_extra_args(self, mock_analyzer):
    """A config that disallows extra arguments is no exact hit with extras."""
    matcher = _ScriptedMatcher(answers={"x": True})
    configs = [
        InjectionConfig(
            match_args={"query": "x"},
            args_matcher=ToolArgsMatcher(
                default_matcher=matcher, allow_extra_args=False
            ),
            injected_response={"from": "strict"},
        ),
        InjectionConfig(
            match_args={"query": "x"},
            injected_response={"from": "plain"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(
        _search_tool(), {"query": "x", "limit": 5}, MagicMock()
    )

    assert result == {"from": "plain"}

  async def test_exact_match_that_loses_its_draw_is_drawn_once(
      self, mock_analyzer
  ):
    """An exact hit that loses its draw is neither compared nor drawn again."""
    matcher = _ScriptedMatcher(answers={"x": True, "y": True})
    configs = [
        InjectionConfig(
            match_args={"query": "x"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injection_probability=0.5,
            injected_response={"from": "x"},
        ),
        InjectionConfig(
            match_args={"query": "y"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": "y"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))
    losing_draw = MagicMock()
    losing_draw.random.return_value = 0.9
    engine._seeded_generators[("search", 0)] = losing_draw

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert result == {"from": "y"}
    losing_draw.random.assert_called_once_with()
    assert "x" not in matcher._calls

  async def test_catch_all_before_exact_config_keeps_priority(
      self, mock_analyzer
  ):
    """A config without match_args still applies before a later exact one."""
    configs = [
        InjectionConfig(injected_response={"from": "catch-all"}),
        InjectionConfig(
            match_args={"query": "x"},
            injected_response={"from": "exact"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert result == {"from": "catch-all"}

  async def test_draws_follow_config_order_without_args_matcher(
      self, mock_analyzer
  ):
    """Without args_matcher, configs are drawn in order, each at most once.

    Setup: a non-matching config, a catch-all that loses its draw, and an
      exact config that wins its draw.
    Assert: the catch-all is drawn before the exact config, each once, and
      the non-matching config is never drawn.
    """
    configs = [
        InjectionConfig(
            match_args={"query": "other"},
            injection_probability=0.5,
            injected_response={"from": "other"},
        ),
        InjectionConfig(
            injection_probability=0.5,
            injected_response={"from": "catch-all"},
        ),
        InjectionConfig(
            match_args={"query": "x"},
            injection_probability=0.5,
            injected_response={"from": "exact"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))
    draws = []
    for index, value in ((0, 0.1), (1, 0.9), (2, 0.1)):
      generator = MagicMock()
      generator.random.side_effect = (
          lambda index=index, value=value: draws.append(index) or value
      )
      engine._seeded_generators[("search", index)] = generator

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert result == {"from": "exact"}
    assert draws == [1, 2]

  async def test_catch_all_between_semantic_and_exact_keeps_order(
      self, mock_analyzer
  ):
    """The exact check stops at a catch-all, so earlier configs keep priority.

    The call equals the last config's match_args, but a catch-all sits in
    between, so the configs are tried in order as before: the semantic config
    is compared and applies first.
    """
    matcher = _ScriptedMatcher(answers={"broad": True})
    configs = [
        InjectionConfig(
            match_args={"query": "broad"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": "semantic"},
        ),
        InjectionConfig(injected_response={"from": "catch-all"}),
        InjectionConfig(
            match_args={"query": "narrow"},
            injected_response={"from": "exact"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await engine.simulate(
        _search_tool(), {"query": "narrow"}, MagicMock()
    )

    assert result == {"from": "semantic"}
    assert matcher._calls == ["broad"]

  async def test_waiting_comparisons_are_cancelled_after_a_hit(
      self, mock_analyzer
  ):
    """Comparisons for lower-priority configs stop once one applies."""
    matcher = _ScriptedMatcher(answers={"a": True}, block_on="b")
    configs = [
        InjectionConfig(
            match_args={"query": value},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": value},
        )
        for value in ("a", "b")
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))

    result = await asyncio.wait_for(
        engine.simulate(_search_tool(), {"query": "x"}, MagicMock()),
        timeout=5,
    )

    assert result == {"from": "a"}
    assert matcher._cancelled == ["b"]

  async def test_waiting_comparisons_are_cancelled_before_injected_latency(
      self, mock_analyzer
  ):
    """Lower-priority comparisons are cancelled before latency is awaited."""
    matcher = _ScriptedMatcher(answers={"a": True}, block_on="b")
    configs = [
        InjectionConfig(
            match_args={"query": "a"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_latency_seconds=1.5,
            injected_response={"from": "a"},
        ),
        InjectionConfig(
            match_args={"query": "b"},
            args_matcher=ToolArgsMatcher(default_matcher=matcher),
            injected_response={"from": "b"},
        ),
    ]
    engine = EnvironmentSimulationEngine(_semantic_config(configs))
    cancelled_at_sleep = []

    async def _record_sleep(delay):
      cancelled_at_sleep.append((delay, list(matcher._cancelled)))

    with patch(
        "google.adk.tools.environment_simulation.environment_simulation_engine.asyncio.sleep",
        side_effect=_record_sleep,
    ):
      result = await asyncio.wait_for(
          engine.simulate(_search_tool(), {"query": "x"}, MagicMock()),
          timeout=5,
      )

    assert result == {"from": "a"}
    assert cancelled_at_sleep == [(1.5, ["b"])]

  async def test_matcher_error_propagates(self, mock_analyzer):
    """A failing comparison surfaces instead of silently missing."""
    matcher = _ScriptedMatcher(answers={"a": RuntimeError("judge down")})
    config = InjectionConfig(
        match_args={"query": "a"},
        args_matcher=ToolArgsMatcher(default_matcher=matcher),
        injected_response={"injected": True},
    )
    engine = EnvironmentSimulationEngine(_semantic_config([config]))

    with pytest.raises(RuntimeError, match="judge down"):
      await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())


class _GatedMatcher(BaseArgMatcher):
  """A non-deterministic matcher that records how comparisons overlap.

  One instance is shared by every injection config in a test, so its counters
  see all comparisons the engine runs for a tool call. The answer is looked up
  by "expected->actual" first, then by "expected", and defaults to no match. A
  delay of None blocks until the comparison is cancelled.
  """

  type: Literal["gated"] = "gated"
  is_deterministic: ClassVar[bool] = False

  answers: dict[str, bool] = {}
  delays: dict[str, float | None] = {}
  default_delay: float = 0.05
  wait_for: dict[str, str] = {}
  """Comparisons that start finishing only after the named one has finished."""
  barrier: int | None = None
  """When set, every comparison waits until this many are in flight at once."""

  _finished_events: dict[str, asyncio.Event] = PrivateAttr(default_factory=dict)
  _barrier_reached: asyncio.Event | None = PrivateAttr(default=None)
  _in_flight: int = PrivateAttr(default=0)
  _max_in_flight: int = PrivateAttr(default=0)
  _started: list[str] = PrivateAttr(default_factory=list)
  _finished: list[str] = PrivateAttr(default_factory=list)
  _cancelled: list[str] = PrivateAttr(default_factory=list)

  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    key = str(expected)
    self._started.append(key)
    self._in_flight += 1
    self._max_in_flight = max(self._max_in_flight, self._in_flight)
    try:
      if self.barrier is not None:
        if self._barrier_reached is None:
          self._barrier_reached = asyncio.Event()
        if self._in_flight >= self.barrier:
          self._barrier_reached.set()
        await self._barrier_reached.wait()
      if key in self.wait_for:
        await self._finished_event(self.wait_for[key]).wait()
      delay = self.delays.get(key, self.default_delay)
      if delay is None:
        await asyncio.Event().wait()
      else:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
      self._cancelled.append(key)
      raise
    finally:
      self._in_flight -= 1
    self._finished.append(key)
    self._finished_event(key).set()
    matched = self.answers.get(
        f"{expected}->{actual}", self.answers.get(key, False)
    )
    return ArgMatchResult(matched=matched)

  def _finished_event(self, key: str) -> asyncio.Event:
    return self._finished_events.setdefault(key, asyncio.Event())


def _gated_configs(
    matcher: _GatedMatcher, values: list[str]
) -> list[InjectionConfig]:
  """One injection config per value, all compared through `matcher`."""
  return [
      InjectionConfig(
          match_args={"query": value},
          args_matcher=ToolArgsMatcher(default_matcher=matcher),
          injected_response={"from": value},
      )
      for value in values
  ]


@patch(
    "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer"
)
@pytest.mark.asyncio
class TestEnvironmentSimulationEngineParallelism:
  """Tests for how simulate runs args_matcher comparisons concurrently."""

  async def test_configs_are_compared_concurrently(self, mock_analyzer):
    """Comparisons for different injection configs overlap in time."""
    matcher = _GatedMatcher()
    values = ["a", "b", "c", "d"]
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert result is None  # Nothing matched and there is no mock strategy.
    assert matcher._max_in_flight == len(values)

  async def test_concurrency_is_bounded(self, mock_analyzer):
    """No more than _MAX_CONCURRENT_ARG_MATCHES comparisons run at once."""
    matcher = _GatedMatcher(default_delay=0.01)
    values = [f"v{i}" for i in range(3 * _MAX_CONCURRENT_ARG_MATCHES)]
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert matcher._max_in_flight == _MAX_CONCURRENT_ARG_MATCHES
    assert sorted(matcher._finished) == sorted(values)

  async def test_comparisons_start_in_priority_order(self, mock_analyzer):
    """The semaphore admits comparisons in injection-config order."""
    matcher = _GatedMatcher(default_delay=0.01)
    values = [f"v{i:02d}" for i in range(3 * _MAX_CONCURRENT_ARG_MATCHES)]
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert matcher._started == values

  async def test_earlier_slow_match_beats_later_fast_match(self, mock_analyzer):
    """Priority order wins even when a later config finishes first."""
    # "slow" cannot finish until "fast" has, so completion order is fixed
    # without relying on wall-clock delays.
    matcher = _GatedMatcher(
        answers={"slow": True, "fast": True},
        default_delay=0.0,
        wait_for={"slow": "fast"},
    )
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, ["slow", "fast"]))
    )

    result = await engine.simulate(_search_tool(), {"query": "x"}, MagicMock())

    assert matcher._finished == ["fast", "slow"]
    assert result == {"from": "slow"}

  async def test_admitted_comparisons_all_run_at_once(self, mock_analyzer):
    """Admitted comparisons overlap, so none waits for another to finish.

    Each comparison blocks until all of them are in flight, so comparing the
    configs one at a time would never finish.
    """
    values = [f"v{i}" for i in range(_MAX_CONCURRENT_ARG_MATCHES)]
    matcher = _GatedMatcher(default_delay=0.0, barrier=len(values))
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    result = await asyncio.wait_for(
        engine.simulate(_search_tool(), {"query": "x"}, MagicMock()),
        timeout=5,
    )

    assert result is None  # Nothing matched and there is no mock strategy.
    assert sorted(matcher._finished) == sorted(values)

  async def test_hit_cancels_running_and_queued_comparisons(
      self, mock_analyzer
  ):
    """Once a config applies, lower-priority comparisons stop or never start."""
    values = [f"v{i:02d}" for i in range(3 * _MAX_CONCURRENT_ARG_MATCHES)]
    matcher = _GatedMatcher(
        answers={"v00": True},
        # Every comparison except the first blocks until cancelled.
        delays={value: None for value in values[1:]} | {"v00": 0.01},
    )
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    result = await asyncio.wait_for(
        engine.simulate(_search_tool(), {"query": "x"}, MagicMock()),
        timeout=5,
    )

    assert result == {"from": "v00"}
    # Only those admitted by the semaphore ever started. The hit frees one
    # slot just before simulate resumes, so at most one more may slip in.
    assert len(matcher._started) <= _MAX_CONCURRENT_ARG_MATCHES + 1
    assert values[-1] not in matcher._started
    # Every started comparison except the hit was cancelled, not abandoned.
    assert sorted(matcher._cancelled) == sorted(matcher._started[1:])
    assert matcher._in_flight == 0

  async def test_parallel_tool_calls_on_one_engine(self, mock_analyzer):
    """Concurrent simulate calls overlap and share the engine semaphore."""
    matcher = _GatedMatcher(answers={"a->call-a": True, "b->call-b": True})
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, ["a", "b"]))
    )

    result_a, result_b = await asyncio.gather(
        engine.simulate(_search_tool(), {"query": "call-a"}, MagicMock()),
        engine.simulate(_search_tool(), {"query": "call-b"}, MagicMock()),
    )

    assert result_a == {"from": "a"}
    assert result_b == {"from": "b"}
    # Two configs per call, two calls, all started before any finished.
    assert matcher._max_in_flight == 4

  async def test_parallel_tool_calls_share_concurrency_limit(
      self, mock_analyzer
  ):
    """Concurrent simulate calls share _MAX_CONCURRENT_ARG_MATCHES."""
    matcher = _GatedMatcher(default_delay=0.01)
    values = [f"v{i}" for i in range(2 * _MAX_CONCURRENT_ARG_MATCHES)]
    engine = EnvironmentSimulationEngine(
        _semantic_config(_gated_configs(matcher, values))
    )

    await asyncio.gather(
        engine.simulate(_search_tool(), {"query": "call-a"}, MagicMock()),
        engine.simulate(_search_tool(), {"query": "call-b"}, MagicMock()),
    )

    assert matcher._max_in_flight == _MAX_CONCURRENT_ARG_MATCHES


@patch(
    "google.adk.tools.environment_simulation.environment_simulation_engine.ToolConnectionAnalyzer"
)
def test_engine_reused_across_event_loops(mock_analyzer):
  """An engine with > _MAX_CONCURRENT_ARG_MATCHES configs works across loops."""
  values = [f"v{i}" for i in range(_MAX_CONCURRENT_ARG_MATCHES + 4)]
  matcher = _GatedMatcher(answers={values[-1]: True}, default_delay=0.01)
  engine = EnvironmentSimulationEngine(
      _semantic_config(_gated_configs(matcher, values))
  )

  result_1 = asyncio.run(
      engine.simulate(_search_tool(), {"query": "x"}, MagicMock())
  )
  result_2 = asyncio.run(
      engine.simulate(_search_tool(), {"query": "x"}, MagicMock())
  )

  assert result_1 == {"from": values[-1]}
  assert result_2 == {"from": values[-1]}
  assert matcher._max_in_flight == _MAX_CONCURRENT_ARG_MATCHES
