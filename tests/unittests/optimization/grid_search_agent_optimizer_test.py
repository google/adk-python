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

"""Tests for GridSearchAgentOptimizer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import logging
from typing import Any
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.optimization._grid_search_agent_optimizer import GridSearchAgentOptimizer
from google.adk.optimization._grid_search_agent_optimizer import GridSearchAgentOptimizerConfig
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.local_eval_sampler import LocalEvalSampler
from google.adk.optimization.local_eval_sampler import LocalEvalSamplerConfig
from google.adk.optimization.sampler import _ExampleSet
from google.adk.optimization.sampler import Sampler
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import ValidationError
import pytest


@pytest.fixture(autouse=True)
def _suppress_experimental_warning(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppresses experimental feature warning during unit tests."""
  monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "true")


class _FakeEvalSampler(Sampler[UnstructuredSamplingResult]):
  """In-memory Sampler returning LocalEvalSampler-shaped evaluation data."""

  def __init__(
      self,
      example_ids: Sequence[str] | None = None,
      *,
      models_without_tokens: Sequence[str] = (),
  ) -> None:
    self._example_ids = list(
        example_ids or ["case_1", "case_2", "case_3", "case_4"]
    )
    self._models_without_tokens = set(models_without_tokens)
    self.recorded_batches: list[tuple[str, list[str]]] = []

  def get_train_example_ids(self) -> list[str]:
    return list(self._example_ids)

  def get_validation_example_ids(self) -> list[str]:
    return list(self._example_ids)

  async def sample_and_score(
      self,
      candidate: BaseAgent,  # type: ignore[override]
      example_set: _ExampleSet = Sampler.VALIDATION_SET,
      batch: list[str] | None = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    """Returns deterministic scores and telemetry based on candidate config."""
    del example_set
    eval_ids = batch if batch is not None else self._example_ids
    model_str = str(getattr(candidate, "model", None) or "gemini-2.5-flash")
    self.recorded_batches.append((model_str, list(eval_ids)))
    gen_cfg = getattr(candidate, "generate_content_config", None)
    temperature = float(getattr(gen_cfg, "temperature", None) or 0.0)

    if "pro" in model_str:
      base_score = 0.95
      latency_s = 2.0
      tokens = 500
    elif "flash-lite" in model_str:
      base_score = 0.75
      latency_s = 0.5
      tokens = 80
    else:
      base_score = 0.85
      latency_s = 1.0
      tokens = 200

    quality = base_score - temperature * 0.05
    f1_score = quality * 0.9
    if not capture_full_eval_data:
      return UnstructuredSamplingResult(
          scores={uid: quality for uid in eval_ids}
      )
    scores = {uid: 1.0 for uid in eval_ids}

    metric_results: list[dict[str, Any]] = [
        {"metric_name": "quality", "score": quality, "eval_status": "PASSED"},
        {"metric_name": "f1", "score": f1_score, "eval_status": "PASSED"},
        {
            "metric_name": "invocation_duration_v1",
            "score": latency_s,
            "eval_status": "NOT_EVALUATED",
        },
    ]
    if model_str not in self._models_without_tokens:
      metric_results.append({
          "metric_name": "token_usage_v1",
          "score": float(tokens * 2),
          "eval_status": "NOT_EVALUATED",
          "details": {
              "token_usage_details": {
                  "input_tokens": tokens,
                  "output_tokens": tokens,
                  "cached_tokens": 0,
                  "reasoning_tokens": 0,
              }
          },
      })
    data = {
        uid: {"invocations": [{"eval_metric_results": metric_results}]}
        for uid in eval_ids
    }
    return UnstructuredSamplingResult(scores=scores, data=data)


class _TokenResultSampler(Sampler[UnstructuredSamplingResult]):
  """Returns one fixed `token_usage_v1` result for every example."""

  def __init__(self, token_result: Mapping[str, Any]) -> None:
    self._token_result = dict(token_result)

  def get_train_example_ids(self) -> list[str]:
    return ["c1", "c2"]

  def get_validation_example_ids(self) -> list[str]:
    return ["c1", "c2"]

  async def sample_and_score(
      self,
      candidate: BaseAgent,  # type: ignore[override]
      example_set: _ExampleSet = Sampler.VALIDATION_SET,
      batch: list[str] | None = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    del candidate, example_set, capture_full_eval_data
    eval_ids = batch if batch is not None else ["c1", "c2"]
    metric_results = [{
        "metric_name": "token_usage_v1",
        "eval_status": "NOT_EVALUATED",
        **self._token_result,
    }]
    return UnstructuredSamplingResult(
        scores={uid: 1.0 for uid in eval_ids},
        data={
            uid: {"invocations": [{"eval_metric_results": metric_results}]}
            for uid in eval_ids
        },
    )


class _ScriptedSampler(Sampler[UnstructuredSamplingResult]):
  """Returns the metric results a script picks for each model and example."""

  def __init__(
      self,
      script: Callable[[str, str], list[dict[str, Any]] | None],
      *,
      train_ids: Sequence[str] = ("c1", "c2"),
      validation_ids: Sequence[str] = ("v1", "v2"),
      delay_sec: float = 0.0,
  ) -> None:
    self._script = script
    self._train_ids = list(train_ids)
    self._validation_ids = list(validation_ids)
    self._delay_sec = delay_sec
    self.calls: list[tuple[str, str, list[str]]] = []
    self.candidates: list[BaseAgent] = []
    self.in_flight = 0
    self.peak_in_flight = 0

  def get_train_example_ids(self) -> list[str]:
    return list(self._train_ids)

  def get_validation_example_ids(self) -> list[str]:
    return list(self._validation_ids)

  async def sample_and_score(
      self,
      candidate: BaseAgent,  # type: ignore[override]
      example_set: _ExampleSet = Sampler.TRAIN_SET,
      batch: list[str] | None = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    del capture_full_eval_data
    model = str(getattr(candidate, "model", ""))
    default_ids = (
        self._train_ids
        if example_set == Sampler.TRAIN_SET
        else self._validation_ids
    )
    eval_ids = list(batch if batch is not None else default_ids)
    self.calls.append((model, example_set, eval_ids))
    self.candidates.append(candidate)
    self.in_flight += 1
    self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
    try:
      await asyncio.sleep(self._delay_sec)
    finally:
      self.in_flight -= 1
    scores: dict[str, float] = {}
    data: dict[str, Any] = {}
    for uid in eval_ids:
      results = self._script(model, uid)
      scores[uid] = 0.0 if results is None else 1.0
      invocations = (
          [] if results is None else [{"eval_metric_results": results}]
      )
      data[uid] = {"invocations": invocations}
    return UnstructuredSamplingResult(scores=scores, data=data)


def _metric(name: str, score: float | None) -> dict[str, Any]:
  """Returns one eval metric result in LocalEvalSampler's shape."""
  return {"metric_name": name, "score": score, "eval_status": "PASSED"}


def _models_on_front(result: Any) -> list[str]:
  """Returns the model of each agent on the Pareto front, best first."""
  return [
      getattr(item.optimized_agent, "model", "")
      for item in result.optimized_agents
  ]


def _model_of(agent: BaseAgent) -> Any:
  """Asserts the agent is an LlmAgent and returns its model."""
  assert isinstance(agent, LlmAgent)
  return agent.model


async def test_optimize_evaluates_candidates_and_returns_pareto_front() -> None:
  """Every candidate is scored, and the non-dominated ones are returned."""
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "model": [
              "gemini-2.5-flash-lite",
              "gemini-2.5-flash",
              "gemini-2.5-pro",
          ],
          "generate_content_config.temperature": [0.0, 0.5],
      },
  )
  optimizer = GridSearchAgentOptimizer(config)
  initial_agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(
      example_ids=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
  )

  result = await optimizer.optimize(initial_agent, sampler)

  models_on_front = {
      getattr(item.optimized_agent, "model", "")
      for item in result.optimized_agents
  }
  assert models_on_front == {
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash",
      "gemini-2.5-pro",
  }
  assert len(result.optimized_agents) == 3
  expected_pro_mean = (0.95 + 0.95 * 0.9) / 2.0
  assert result.optimized_agents[0].overall_score == pytest.approx(
      expected_pro_mean
  )
  assert result.optimized_agents[0].token_usage["total_tokens"] == 1000.0
  assert result.optimized_agents[0].latency_sec == pytest.approx(2.0)


async def test_every_candidate_is_scored_on_the_same_full_example_set() -> None:
  """No candidate is judged on fewer or different examples than another."""
  example_ids = [f"ex_{i}" for i in range(9)]
  sampler = _FakeEvalSampler(example_ids=example_ids)
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "model": [
              "gemini-2.5-flash-lite",
              "gemini-2.5-flash",
              "gemini-2.5-pro",
          ],
          "generate_content_config.temperature": [0.0, 0.5],
      },
      max_concurrent_candidates=2,
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert len(sampler.recorded_batches) == 6
  assert all(batch == example_ids for _, batch in sampler.recorded_batches)
  assert result.optimized_agents
  assert all(
      item.metadata["num_cases_evaluated"] == len(example_ids)
      for item in result.optimized_agents
  )


async def test_optimize_clones_sequential_and_parallel_without_mutation() -> (
    None
):
  """Canonical sub_agents paths update nested workflow leaves cleanly."""
  weather = LlmAgent(name="WeatherSpecialist", model="gemini-2.5-flash")
  traffic = LlmAgent(name="TrafficSpecialist", model="gemini-2.5-flash")
  fan_out = ParallelAgent(name="FanOut", sub_agents=[weather, traffic])
  aggregator = LlmAgent(name="Aggregator", model="gemini-2.5-flash")
  pipeline = SequentialAgent(
      name="ParallelPipeline", sub_agents=[fan_out, aggregator]
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "sub_agents.FanOut.sub_agents.WeatherSpecialist.model": [
              "gemini-2.5-flash-lite"
          ],
          "sub_agents.FanOut.sub_agents.TrafficSpecialist.model": [
              "gemini-2.5-pro"
          ],
          "sub_agents.Aggregator.model": ["gemini-2.5-pro"],
      }
  )
  optimizer = GridSearchAgentOptimizer(config)
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])

  result = await optimizer.optimize(pipeline, sampler)

  cloned_pipeline = result.optimized_agents[0].optimized_agent
  cloned_fan_out = cloned_pipeline.sub_agents[0]
  assert _model_of(cloned_fan_out.sub_agents[0]) == "gemini-2.5-flash-lite"
  assert _model_of(cloned_fan_out.sub_agents[1]) == "gemini-2.5-pro"
  assert _model_of(cloned_pipeline.sub_agents[1]) == "gemini-2.5-pro"
  assert _model_of(pipeline.sub_agents[0].sub_agents[0]) == "gemini-2.5-flash"
  assert _model_of(pipeline.sub_agents[0].sub_agents[1]) == "gemini-2.5-flash"
  assert _model_of(pipeline.sub_agents[1]) == "gemini-2.5-flash"


async def test_optimize_clones_agent_tool_children_without_mutation() -> None:
  """AgentTool-wrapped agents are cloned via tools.<Name> paths."""
  analyst = LlmAgent(
      name="SqlAnalystAgent",
      instruction="Analyze SQL queries.",
      model="gemini-2.5-flash",
  )
  tool = AgentTool(agent=analyst)
  manager = LlmAgent(
      name="ManagerAgent",
      instruction="Coordinate queries.",
      model="gemini-2.5-flash",
      tools=[tool],
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "tools.SqlAnalystAgent.model": ["gemini-2.5-pro"],
      }
  )
  optimizer = GridSearchAgentOptimizer(config)
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])

  result = await optimizer.optimize(manager, sampler)

  cloned_manager = result.optimized_agents[0].optimized_agent
  assert isinstance(cloned_manager, LlmAgent)
  cloned_tool = cloned_manager.tools[0]
  assert isinstance(cloned_tool, AgentTool)
  assert cloned_tool is not tool
  assert cloned_tool.agent is not analyst
  assert cloned_tool.agent.parent_agent is None
  assert _model_of(cloned_tool.agent) == "gemini-2.5-pro"
  assert analyst.model == "gemini-2.5-flash"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("sub_agents.Nonexistent.model", "has no sub-agent named"),
        ("tools.Nonexistent.model", "has no AgentTool agent named"),
        ("sub_agents.TrafficSpecialist.model", "has no sub-agent named"),
        ("TrafficSpecialist.model", "Prefix child agents"),
        ("temperature", "generate_content_config.temperature"),
        ("thinking_config.thinking_budget", "Prefix child agents"),
    ],
)
async def test_unknown_or_unprefixed_path_raises_before_any_model_call(
    path: str,
    message: str,
) -> None:
  """Invalid or shorthand paths fail up front before running evaluations."""
  traffic = LlmAgent(name="TrafficSpecialist", model="gemini-2.5-flash")
  fan_out = ParallelAgent(name="FanOut", sub_agents=[traffic])
  pipeline = SequentialAgent(name="Pipeline", sub_agents=[fan_out])
  sampler = _FakeEvalSampler(example_ids=["c1"])
  config = GridSearchAgentOptimizerConfig(param_grid={path: ["gemini-2.5-pro"]})

  with pytest.raises(ValueError, match=message):
    await GridSearchAgentOptimizer(config).optimize(pipeline, sampler)
  assert not sampler.recorded_batches


async def test_optimize_supports_string_quality_metric() -> None:
  """A string quality_metric selects that metric from per-example results."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      quality_metric="f1",
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  assert result.optimized_agents[0].overall_score == pytest.approx(0.765)


async def test_optimize_supports_weighted_mapping_quality_metric() -> None:
  """A mapping quality_metric computes the weighted sum of named metrics."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      quality_metric={"quality": 0.5, "f1": 0.5},
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  expected = 0.5 * 0.85 + 0.5 * 0.765
  assert result.optimized_agents[0].overall_score == pytest.approx(expected)


async def test_optimize_supports_callable_quality_metric() -> None:
  """A callable quality_metric receives the per-example metric mapping."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])

  def double_quality(scores: Mapping[str, float]) -> float:
    return float(scores.get("quality", 0.0)) * 2.0

  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      quality_metric=double_quality,
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  assert result.optimized_agents[0].overall_score == pytest.approx(1.70)


async def test_empty_quality_metric_mapping_averages_quality_metrics() -> None:
  """An empty mapping quality_metric averages all non-efficiency metrics."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      quality_metric={},
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  expected = (0.85 + 0.765) / 2.0
  assert result.optimized_agents[0].overall_score == pytest.approx(expected)


@pytest.mark.parametrize(
    "quality_metric", ["f_1", {"quality": 0.5, "f_1": 0.5}]
)
async def test_misnamed_quality_metric_raises(
    quality_metric: str | Mapping[str, float],
) -> None:
  """A quality_metric name the sampler never reports is a typo, not a zero."""
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      quality_metric=quality_metric,
  )

  with pytest.raises(ValueError, match="did not report"):
    await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"),
        _FakeEvalSampler(example_ids=["c1"]),
    )


async def test_optimize_filters_infeasible_candidates_by_thresholds() -> None:
  """Candidates violating quality, latency, or token gates are excluded."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "model": [
              "gemini-2.5-flash-lite",
              "gemini-2.5-flash",
              "gemini-2.5-pro",
          ],
      },
      min_quality_score=0.80,
      max_latency_sec=1.5,
      max_total_tokens=600,
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  assert len(result.optimized_agents) == 1
  winner = result.optimized_agents[0].optimized_agent
  assert _model_of(winner) == "gemini-2.5-flash"


async def test_optimize_supports_conditional_sub_grids() -> None:
  """A sequence of sub-grid dicts expands each conditional sub-grid cleanly."""
  agent = LlmAgent(name="Worker", model="gemini-2.5-flash")
  sampler = _FakeEvalSampler(example_ids=["c1", "c2"])
  budget_key = "generate_content_config.thinking_config.thinking_budget"
  config = GridSearchAgentOptimizerConfig(
      param_grid=[
          {"model": ["gemini-2.5-flash-lite"], budget_key: [0]},
          {"model": ["gemini-2.5-pro"], budget_key: [512]},
      ],
  )

  result = await GridSearchAgentOptimizer(config).optimize(agent, sampler)

  models = {
      getattr(item.optimized_agent, "model", "")
      for item in result.optimized_agents
  }
  assert models == {"gemini-2.5-flash-lite", "gemini-2.5-pro"}


async def test_token_total_counts_reasoning_tokens_once() -> None:
  """ADK's output_tokens already include reasoning, so they are not re-added."""
  sampler = _TokenResultSampler({
      "score": 1500.0,
      "details": {
          "token_usage_details": {
              "input_tokens": 1000.0,
              "output_tokens": 500.0,
              "reasoning_tokens": 400.0,
          }
      },
  })
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]}
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert result.optimized_agents[0].token_usage == {
      "total_tokens": 1500.0,
      "input_tokens": 1000.0,
      "output_tokens": 500.0,
      "reasoning_tokens": 400.0,
  }


async def test_score_only_token_usage_is_reported_as_total_only() -> None:
  """LocalEvalSampler forwards only the score, which is a total, not input."""
  sampler = _TokenResultSampler({"score": 1500.0})
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]}
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert result.optimized_agents[0].token_usage == {"total_tokens": 1500.0}


async def test_unreported_token_usage_is_unknown_not_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """A model whose tokens were not reported must not look free, and we warn."""
  sampler = _FakeEvalSampler(
      example_ids=["c1", "c2"], models_without_tokens=["gemini-2.5-flash-lite"]
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash-lite", "gemini-2.5-flash"]}
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  usage_by_model = {
      getattr(item.optimized_agent, "model", ""): item.token_usage
      for item in result.optimized_agents
  }
  assert usage_by_model["gemini-2.5-flash-lite"] == {}
  assert usage_by_model["gemini-2.5-flash"]["total_tokens"] == 400.0
  assert "1 of 2 scored candidates reported no token usage" in caplog.text


async def test_max_total_tokens_excludes_candidates_without_token_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """A token ceiling is not met when the tokens were not reported."""
  sampler = _FakeEvalSampler(
      example_ids=["c1", "c2"], models_without_tokens=["gemini-2.5-flash-lite"]
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash-lite", "gemini-2.5-flash"]},
      max_total_tokens=1000,
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert _models_on_front(result) == ["gemini-2.5-flash"]
  assert "fails max_total_tokens if set" in caplog.text


async def test_no_candidate_reporting_tokens_still_returns_the_front(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """If no candidate reports tokens, the search warns and still returns."""
  models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
  sampler = _FakeEvalSampler(
      example_ids=["c1", "c2"], models_without_tokens=models
  )
  config = GridSearchAgentOptimizerConfig(param_grid={"model": models})

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert len(result.optimized_agents) == 2
  assert "2 of 2 scored candidates reported no token usage" in caplog.text


async def test_token_ceiling_without_token_usage_excludes_every_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """With no token usage at all, max_total_tokens leaves nothing feasible."""
  models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
  sampler = _FakeEvalSampler(
      example_ids=["c1", "c2"], models_without_tokens=models
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": models}, max_total_tokens=1000
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert not result.optimized_agents
  assert "all 2 scored candidates failed feasibility thresholds" in caplog.text


@pytest.mark.parametrize(
    ("param_grid", "message"),
    [
        ({"model": "gemini-2.5-pro"}, "must be a list of values"),
        ({"model": []}, "has no values"),
        ({}, "empty grid"),
        ([], "no sub-grids"),
        ([{"model": ["gemini-2.5-pro"]}, {}], "empty grid"),
    ],
    ids=["bare_string", "no_values", "empty_grid", "no_sub_grids", "empty_sub"],
)
def test_param_grid_rejects_values_that_cannot_be_swept(
    param_grid: Any, message: str
) -> None:
  """A bare string, an empty value list, or an empty grid is an error."""
  with pytest.raises(ValueError, match=message):
    GridSearchAgentOptimizerConfig(param_grid=param_grid)


def test_param_grid_is_required() -> None:
  """A config without a grid would evaluate nothing, so it is rejected."""
  with pytest.raises(ValueError, match="param_grid"):
    GridSearchAgentOptimizerConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "quality_metric",
    ["token_usage_v1", {"quality": 1.0, "tool_call_count_v1": 0.1}],
)
def test_quality_metric_rejects_efficiency_metrics(
    quality_metric: str | Mapping[str, float],
) -> None:
  """Efficiency metrics measure cost, so they are not quality inputs."""
  with pytest.raises(ValueError, match="efficiency metrics"):
    GridSearchAgentOptimizerConfig(
        param_grid={"model": ["gemini-2.5-flash"]},
        quality_metric=quality_metric,
    )


@pytest.mark.parametrize(
    "weight", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "inf"]
)
def test_quality_metric_rejects_invalid_weights(weight: float) -> None:
  """Weights must keep an unscored example's 0 the lowest quality."""
  with pytest.raises(ValueError, match="invalid weights"):
    GridSearchAgentOptimizerConfig(
        param_grid={"model": ["gemini-2.5-flash"]},
        quality_metric={"quality": 1.0, "penalty": weight},
    )


async def test_unscored_examples_count_as_zero_quality(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """A candidate that fails some examples is not judged on fewer examples."""

  def script(model: str, uid: str) -> list[dict[str, Any]] | None:
    if model == "gemini-2.5-flash-lite" and uid != "c1":
      return None
    score = 1.0 if model == "gemini-2.5-flash-lite" else 0.8
    return [_metric("judge", score)]

  sampler = _ScriptedSampler(script, train_ids=["c1", "c2", "c3", "c4"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]},
      quality_metric="judge",
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert _models_on_front(result) == ["gemini-2.5-flash"]
  assert result.optimized_agents[0].overall_score == pytest.approx(0.8)
  assert "1 of 2 candidates could not be scored on every example" in (
      caplog.text
  )


async def test_quality_metric_without_a_score_does_not_stop_the_search(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """A judge that could not score one candidate is missing data, not a typo."""

  def script(model: str, uid: str) -> list[dict[str, Any]]:
    del uid
    judge = 0.9 if model == "gemini-2.5-flash" else None
    return [_metric("judge", judge), _metric("tool_call_count_v1", 1.0)]

  sampler = _ScriptedSampler(script)
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]},
      quality_metric="judge",
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert _models_on_front(result) == ["gemini-2.5-flash"]
  assert "1 of 2 candidates produced no scored examples" in caplog.text


async def test_non_finite_metric_values_count_as_unreported() -> None:
  """NaN quality, latency, or tokens never pass a gate or look free."""
  nan = float("nan")
  results_by_model = {
      "gemini-2.5-flash": [0.9, 1.0, 100.0],
      "nan-tokens": [0.8, 0.5, nan],
      "nan-latency": [0.95, nan, 50.0],
      "nan-quality": [nan, 0.5, 50.0],
  }

  def script(model: str, uid: str) -> list[dict[str, Any]]:
    del uid
    quality, latency, tokens = results_by_model[model]
    return [
        _metric("quality", quality),
        _metric("invocation_duration_v1", latency),
        _metric("token_usage_v1", tokens),
    ]

  sampler = _ScriptedSampler(script)
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": list(results_by_model)},
      quality_metric="quality",
      min_quality_score=0.5,
      max_latency_sec=2.0,
      max_total_tokens=1000,
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert _models_on_front(result) == ["gemini-2.5-flash"]


async def test_unreported_latency_fails_max_latency_sec() -> None:
  """A latency ceiling is not met when the latency was not reported."""

  def script(model: str, uid: str) -> list[dict[str, Any]]:
    del uid
    if model == "gemini-2.5-flash":
      return [_metric("quality", 0.9), _metric("invocation_duration_v1", 1.0)]
    return [_metric("quality", 0.95)]

  sampler = _ScriptedSampler(script)
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]},
      quality_metric="quality",
      max_latency_sec=5.0,
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert _models_on_front(result) == ["gemini-2.5-flash"]


async def test_max_concurrent_candidates_bounds_in_flight_evaluations() -> None:
  """No more than max_concurrent_candidates evaluations run at once."""
  sampler = _ScriptedSampler(
      lambda model, uid: [_metric("quality", 1.0)], delay_sec=0.01
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"generate_content_config.temperature": [0.0, 0.1, 0.2, 0.3]},
      max_concurrent_candidates=2,
  )

  await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert len(sampler.calls) == 4
  assert sampler.peak_in_flight == 2


async def test_failing_candidate_stops_the_remaining_evaluations() -> None:
  """Once one evaluation raises, the queued evaluations never start."""

  class _FailingSampler(_ScriptedSampler):

    async def sample_and_score(
        self, candidate: BaseAgent, **kwargs: Any
    ) -> UnstructuredSamplingResult:
      if getattr(candidate, "model", "") == "gemini-2.5-pro":
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
      return await super().sample_and_score(candidate, **kwargs)

  sampler = _FailingSampler(
      lambda model, uid: [_metric("quality", 1.0)], delay_sec=0.01
  )
  models = ["gemini-2.5-pro"] + [f"gemini-2.5-flash-{i}" for i in range(4)]
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": models}, max_concurrent_candidates=1
  )

  with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
    await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert len(sampler.calls) <= 1


@pytest.mark.parametrize(
    ("requested_set", "train_ids", "validation_ids", "expected_set", "ids"),
    [
        ("train", [], ["v1", "v2"], "validation", ["v1", "v2"]),
        ("validation", ["t1", "t2"], [], "train", ["t1", "t2"]),
    ],
)
async def test_empty_example_set_falls_back_to_other_split_with_warning(
    caplog: pytest.LogCaptureFixture,
    requested_set: _ExampleSet,
    train_ids: list[str],
    validation_ids: list[str],
    expected_set: str,
    ids: list[str],
) -> None:
  """An empty requested split falls back to the non-empty split and warns."""
  sampler = _ScriptedSampler(
      lambda model, uid: [_metric("quality", 1.0)],
      train_ids=train_ids,
      validation_ids=validation_ids,
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]},
      example_set=requested_set,
  )

  with caplog.at_level(logging.WARNING):
    await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert sampler.calls == [("gemini-2.5-flash", expected_set, ids)]
  assert (
      f"sampler has no {requested_set} examples; falling back to 2"
      f" {expected_set} examples."
      in caplog.text
  )


async def test_duplicate_example_ids_are_scored_once() -> None:
  """Repeated example IDs are sent and counted once."""
  sampler = _ScriptedSampler(
      lambda model, uid: [_metric("quality", 1.0)],
      train_ids=["a", "a", "b"],
  )
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]}
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert sampler.calls[0][2] == ["a", "b"]
  assert result.optimized_agents[0].metadata["num_cases_evaluated"] == 2


async def test_partial_failure_does_not_win_pareto_on_tokens_or_latency() -> (
    None
):
  """Crashing on hard examples cannot make a candidate look faster/cheaper."""

  def script(model: str, uid: str) -> list[dict[str, Any]] | None:
    if model == "gemini-2.5-flash-lite" and uid != "easy":
      return None
    tokens = 50.0 if uid == "easy" else 950.0
    latency = 0.2 if uid == "easy" else 3.8
    quality = 0.8 if model == "gemini-2.5-flash-lite" else 0.9
    return [
        _metric("judge", quality),
        _metric("token_usage_v1", tokens),
        _metric("invocation_duration_v1", latency),
    ]

  sampler = _ScriptedSampler(script, train_ids=["easy", "hard"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]},
      quality_metric="judge",
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
  )

  assert _models_on_front(result) == ["gemini-2.5-flash"]


def _make_local_eval_sampler(
    monkeypatch: pytest.MonkeyPatch,
    seen_sets: list[str],
) -> LocalEvalSampler:
  """Builds a real LocalEvalSampler with canned per-model EvalCaseResults."""
  eval_sets_manager = mock.MagicMock(spec=EvalSetsManager)
  eval_sets_manager.get_eval_case.return_value = None
  sampler = LocalEvalSampler(
      LocalEvalSamplerConfig(
          eval_config=EvalConfig(),
          app_name="test_app",
          train_eval_set="train_set",
          train_eval_case_ids=["t1", "t2"],
          validation_eval_set="val_set",
          validation_eval_case_ids=["v1"],
      ),
      eval_sets_manager,
  )

  def make_case_result(uid: str, model_name: str) -> EvalCaseResult:
    case = mock.MagicMock(spec=EvalCaseResult)
    case.eval_id = uid
    if model_name == "broken-model":
      case.final_eval_status = EvalStatus.FAILED
      case.eval_metric_result_per_invocation = []
      return case
    score = 0.954 if "pro" in model_name else 0.812
    tokens = 800.0 if "pro" in model_name else 300.0
    latency = 1.5 if "pro" in model_name else 0.6
    per_inv = mock.MagicMock(spec=EvalMetricResultPerInvocation)
    per_inv.actual_invocation = Invocation(
        user_content=types.Content(parts=[types.Part(text="hi")]),
        final_response=types.Content(parts=[types.Part(text="ok")]),
    )
    per_inv.expected_invocation = None
    per_inv.eval_metric_results = [
        EvalMetricResult(
            metric_name="response_match_score",
            score=score,
            eval_status=EvalStatus.PASSED,
        ),
        EvalMetricResult(
            metric_name="token_usage_v1",
            score=tokens,
            eval_status=EvalStatus.NOT_EVALUATED,
        ),
        EvalMetricResult(
            metric_name="invocation_duration_v1",
            score=latency,
            eval_status=EvalStatus.NOT_EVALUATED,
        ),
    ]
    case.final_eval_status = EvalStatus.PASSED
    case.eval_metric_result_per_invocation = [per_inv]
    return case

  async def fake_evaluate_agent(
      agent: BaseAgent, eval_set_id: str, eval_case_ids: list[str]
  ) -> list[EvalCaseResult]:
    seen_sets.append(eval_set_id)
    model_name = str(getattr(agent, "model", ""))
    return [make_case_result(uid, model_name) for uid in eval_case_ids]

  monkeypatch.setattr(sampler, "_evaluate_agent", fake_evaluate_agent)
  return sampler


async def test_real_local_eval_sampler_ranks_metrics_and_drops_crashes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Real LocalEvalSampler continuous scores rank and crashed runs are cut."""
  seen_sets: list[str] = []
  sampler = _make_local_eval_sampler(monkeypatch, seen_sets)
  models = ["gemini-2.5-pro", "gemini-2.5-flash", "broken-model"]
  config = GridSearchAgentOptimizerConfig(param_grid={"model": models})

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert seen_sets == ["train_set", "train_set", "train_set"]
  assert _models_on_front(result) == ["gemini-2.5-pro", "gemini-2.5-flash"]
  assert result.optimized_agents[0].overall_score == pytest.approx(0.95)
  assert result.optimized_agents[1].overall_score == pytest.approx(0.81)
  assert "1 of 3 candidates produced no scored examples" in caplog.text


async def test_all_scored_candidates_failing_thresholds_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """An empty front caused by strict thresholds logs the best quality seen."""
  sampler = _make_local_eval_sampler(monkeypatch, [])
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-pro", "gemini-2.5-flash"]},
      min_quality_score=0.99,
  )

  with caplog.at_level(logging.WARNING):
    empty = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert not empty.optimized_agents
  assert "all 2 scored candidates failed feasibility thresholds" in caplog.text


@pytest.mark.parametrize("quality_metric", [None, {}])
async def test_unscored_judge_metric_skips_overall_score_fallback(
    quality_metric: dict[str, float] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Reported quality metrics with score=None never fall back to overall_score."""

  def script(model: str, uid: str) -> list[dict[str, Any]]:
    if model == "all-unscored":
      return [
          _metric("final_response_match_v2", None),
          _metric("token_usage_v1", 50.0),
          _metric("invocation_duration_v1", 0.1),
      ]
    if model == "half-unscored" and uid == "c2":
      return [
          _metric("final_response_match_v2", None),
          _metric("token_usage_v1", 200.0),
          _metric("invocation_duration_v1", 0.4),
      ]
    score = 0.8 if model == "half-unscored" else 0.9
    tokens = 200.0 if model == "half-unscored" else 500.0
    latency = 0.4 if model == "half-unscored" else 1.0
    return [
        _metric("final_response_match_v2", score),
        _metric("token_usage_v1", tokens),
        _metric("invocation_duration_v1", latency),
    ]

  sampler = _ScriptedSampler(script, train_ids=["c1", "c2"])
  config = GridSearchAgentOptimizerConfig(
      param_grid={
          "model": ["gemini-2.5-pro", "half-unscored", "all-unscored"],
      },
      quality_metric=quality_metric,
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"), sampler
    )

  assert _models_on_front(result) == ["gemini-2.5-pro", "half-unscored"]
  partial = result.optimized_agents[1]
  assert partial.overall_score == pytest.approx(0.4)
  assert partial.eval_case_scores == {"c1": pytest.approx(0.8)}
  assert partial.metadata["num_cases_evaluated"] == 1
  assert "1 of 3 candidates produced no scored examples" in caplog.text
  assert "1 of 3 candidates could not be scored on every example" in caplog.text


class _MultiTurnSampler(Sampler[UnstructuredSamplingResult]):
  """Returns caller-specified multi-turn invocation lists per example."""

  def __init__(self, cases: dict[str, list[dict[str, Any]]]) -> None:
    self._cases = cases

  def get_train_example_ids(self) -> list[str]:
    return list(self._cases)

  def get_validation_example_ids(self) -> list[str]:
    return []

  async def sample_and_score(
      self,
      candidate: BaseAgent,  # type: ignore[override]
      example_set: _ExampleSet = Sampler.TRAIN_SET,
      batch: list[str] | None = None,
      capture_full_eval_data: bool = False,
  ) -> UnstructuredSamplingResult:
    del candidate, example_set, capture_full_eval_data
    ids = list(batch if batch is not None else self._cases)
    return UnstructuredSamplingResult(
        scores={uid: 1.0 for uid in ids},
        data={uid: {"invocations": self._cases[uid]} for uid in ids},
    )


def _detailed_token_metric(
    input_tokens: float,
    output_tokens: float,
    *,
    cached_tokens: float | None = None,
    reasoning_tokens: float | None = None,
) -> dict[str, Any]:
  """Builds a token_usage_v1 metric entry with optional None subset counts."""
  total = input_tokens + output_tokens
  return {
      "metric_name": "token_usage_v1",
      "score": total,
      "eval_status": "NOT_EVALUATED",
      "details": {
          "token_usage_details": {
              "total_tokens": total,
              "input_tokens": input_tokens,
              "output_tokens": output_tokens,
              "cached_tokens": cached_tokens,
              "reasoning_tokens": reasoning_tokens,
          }
      },
  }


async def test_multi_turn_sums_tokens_and_zero_fills_partial_optionals() -> (
    None
):
  """Multi-turn examples sum tokens, average latency/quality, and keep cache."""
  cases = {
      "c1": [
          {
              "eval_metric_results": [
                  _metric("turn_quality", 0.6),
                  _metric("rubric", None),
                  _metric("invocation_duration_v1", 1.0),
                  _detailed_token_metric(800.0, 200.0, cached_tokens=None),
              ]
          },
          {
              "eval_metric_results": [
                  _metric("turn_quality", 1.0),
                  _metric("rubric", 0.9),
                  _metric("invocation_duration_v1", 2.0),
                  _detailed_token_metric(1000.0, 200.0, cached_tokens=600.0),
              ]
          },
      ],
      "c2": [
          {
              "eval_metric_results": [
                  _metric("turn_quality", 0.5),
                  _metric("rubric", 0.5),
                  _metric("invocation_duration_v1", 0.5),
                  _detailed_token_metric(400.0, 100.0, cached_tokens=None),
              ]
          },
      ],
  }
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]}
  )

  result = await GridSearchAgentOptimizer(config).optimize(
      LlmAgent(name="Worker", model="gemini-2.5-flash"),
      _MultiTurnSampler(cases),
  )

  winner = result.optimized_agents[0]
  # c1 quality = mean(mean(0.6, 1.0)=0.8, 0.9) = 0.85; c2 quality = 0.5.
  assert winner.overall_score == pytest.approx(0.675)
  # c1 latency = mean(1.0, 2.0) = 1.5; c2 latency = 0.5 -> mean = 1.0.
  assert winner.latency_sec == pytest.approx(1.0)
  # c1 tokens: total=2200, in=1800, out=400, cached=0+600=600;
  # c2 tokens: total=500, in=400, out=100, cached=None (counts as 0);
  # reasoning_tokens is None everywhere so it stays omitted.
  assert winner.token_usage == pytest.approx({
      "total_tokens": 1350.0,
      "input_tokens": 1100.0,
      "output_tokens": 250.0,
      "cached_tokens": 300.0,
  })
  assert "reasoning_tokens" not in winner.token_usage


async def test_multi_turn_missing_token_turn_clears_candidate_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Omitting token_usage_v1 on any turn marks that example's tokens unknown."""
  cases = {
      "c1": [
          {
              "eval_metric_results": [
                  _metric("quality", 0.8),
                  _metric("invocation_duration_v1", 1.0),
                  _metric("token_usage_v1", 400.0),
              ]
          },
          {
              "eval_metric_results": [
                  _metric("quality", 1.0),
                  _metric("invocation_duration_v1", 2.0),
              ]
          },
      ]
  }
  config = GridSearchAgentOptimizerConfig(
      param_grid={"model": ["gemini-2.5-flash"]}
  )

  with caplog.at_level(logging.WARNING):
    result = await GridSearchAgentOptimizer(config).optimize(
        LlmAgent(name="Worker", model="gemini-2.5-flash"),
        _MultiTurnSampler(cases),
    )

  winner = result.optimized_agents[0]
  assert winner.overall_score == pytest.approx(0.9)
  assert winner.latency_sec == pytest.approx(1.5)
  assert winner.token_usage == {}
  assert "reported no token usage" in caplog.text


@pytest.mark.parametrize("field_name", ["min_quality_score", "max_latency_sec"])
@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_feasibility_thresholds_are_rejected(
    field_name: str, bad_value: float
) -> None:
  """NaN and infinite feasibility thresholds fail Pydantic validation."""
  with pytest.raises(ValidationError):
    GridSearchAgentOptimizerConfig(
        param_grid={"model": ["gemini-2.5-flash"]},
        **{field_name: bad_value},  # type: ignore[arg-type]
    )
