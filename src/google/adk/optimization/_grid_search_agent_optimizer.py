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

"""Multi-objective grid search over agent settings."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import itertools
import logging
import math
import statistics
from typing import Any
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from ..agents.base_agent import BaseAgent
from ..features import experimental
from ..features import FeatureName
from ._agent_tree_cloner import clone_agent_with_params
from .agent_optimizer import AgentOptimizer
from .data_types import AgentWithScores
from .data_types import OptimizerResult
from .data_types import UnstructuredSamplingResult
from .sampler import _ExampleSet
from .sampler import Sampler

logger = logging.getLogger("google_adk." + __name__)

# Must match evaluation.eval_metrics.PrebuiltMetrics and the efficiency
# metrics in evaluation.eval_config. Not imported, to avoid loading the
# evaluation package.
_TOKEN_METRIC = "token_usage_v1"
_DURATION_METRIC = "invocation_duration_v1"
_EFFICIENCY_METRICS: frozenset[str] = frozenset({
    _DURATION_METRIC,
    _TOKEN_METRIC,
    "tool_call_count_v1",
    "inference_call_count_v1",
})
# Field names follow ADK's TokenUsageDetails: input_tokens includes
# cached_tokens, output_tokens includes reasoning_tokens, and total_tokens is
# input_tokens + output_tokens.
_TOKEN_FIELDS: tuple[str, ...] = (
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
)
_OPTIONAL_TOKEN_FIELDS: frozenset[str] = frozenset({
    "cached_tokens",
    "reasoning_tokens",
})
_MAX_LOGGED_PARAMS_CHARS = 120

_ParamGrid: TypeAlias = (
    Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]]
)
_QualityMetric: TypeAlias = (
    str | Mapping[str, float] | Callable[[Mapping[str, float]], float] | None
)


@dataclass(frozen=True, slots=True)
class _TokenCounts:
  """Token counts using ADK's TokenUsageDetails names; None means unreported."""

  total_tokens: float | None = None
  input_tokens: float | None = None
  output_tokens: float | None = None
  cached_tokens: float | None = None
  reasoning_tokens: float | None = None


@dataclass(frozen=True, slots=True)
class _ExampleTelemetry:
  """What the sampler reported for one example."""

  metric_scores: dict[str, float]
  metric_names: frozenset[str]
  latency_sec: float | None
  tokens: _TokenCounts | None
  has_invocations: bool = True


class GridSearchAgentWithScores(AgentWithScores):
  """A Pareto-optimal candidate agent with its quality and efficiency."""

  # AgentWithScores types this as Agent; grid search accepts any BaseAgent.
  optimized_agent: BaseAgent = Field(  # type: ignore[assignment]
      description="The optimized agent hierarchy.",
  )
  overall_score: float = Field(
      default=0.0,
      description=(
          "Mean quality score over all evaluated examples. An example that"
          " could not be scored, for example because inference failed,"
          " counts as 0."
      ),
  )
  eval_case_scores: dict[str, float] = Field(
      default_factory=dict,
      description="Per-example quality scores keyed by example UID.",
  )
  token_usage: dict[str, float] = Field(
      default_factory=dict,
      description=(
          "Mean token counts per example when every example reported token"
          " usage, keyed by TokenUsageDetails field names (total_tokens,"
          " input_tokens, output_tokens, cached_tokens, reasoning_tokens)."
          " Empty when any example did not report token usage."
      ),
  )
  latency_sec: float | None = Field(
      default=None,
      description="Mean turn duration in seconds, or None if not reported.",
  )
  metadata: dict[str, Any] = Field(
      default_factory=dict,
      description="Candidate hyperparameters and evaluation metadata.",
  )


class GridSearchAgentOptimizerConfig(BaseModel):
  """Configuration for GridSearchAgentOptimizer."""

  model_config = ConfigDict(extra="forbid")

  param_grid: _ParamGrid = Field(
      description=(
          "Mapping of parameter paths to lists of candidate values, or a"
          " sequence of such mappings (conditional sub-grids). Paths name an"
          " agent field (e.g. `model`, `instruction`), a"
          " `generate_content_config.<field>` setting (e.g."
          " `generate_content_config.temperature`,"
          " `generate_content_config.thinking_config.thinking_budget`), or a"
          " child path prefixed with `sub_agents.<Name>.` or `tools.<Name>.`"
          " at each level (e.g."
          " `sub_agents.Pipeline.sub_agents.Researcher.model`)."
      ),
  )
  min_quality_score: float | None = Field(
      default=None,
      allow_inf_nan=False,
      description="Optional minimum mean quality score for feasibility.",
  )
  max_latency_sec: float | None = Field(
      default=None,
      ge=0.0,
      allow_inf_nan=False,
      description=(
          "Optional maximum latency for feasibility, in seconds per turn,"
          " averaged over turns and examples."
      ),
  )
  max_total_tokens: int | None = Field(
      default=None,
      ge=0,
      description=(
          "Optional maximum total tokens for feasibility, per example (summed"
          " over the conversation's turns), averaged over examples."
      ),
  )
  quality_metric: _QualityMetric = Field(
      default=None,
      description=(
          "How each example's quality score is computed. None: the mean of"
          " the example's quality metrics, falling back to the sampler's"
          " per-example score when no quality metrics are reported. A metric"
          " name: that metric. A mapping: the sum of the named metrics times"
          " their non-negative weights; an empty mapping means the mean of all"
          " quality metrics, falling back to the sampler's score. A callable:"
          " receives each quality metric's score plus `overall_score` (the"
          " sampler's score) and returns the quality. Quality is"
          " higher-is-better with 0 as the floor, as for ADK's built-in"
          " metrics: an example that could not be scored counts as 0."
          " Efficiency metrics such as token_usage_v1 are not quality metrics."
      ),
  )
  example_set: _ExampleSet = Field(
      default="train",
      description=(
          'Example split ("train" or "validation") used for scoring. Falls'
          " back to the other split with a warning if the requested split has"
          " no examples."
      ),
  )
  max_concurrent_candidates: int = Field(
      default=4,
      ge=1,
      description="Maximum number of candidates evaluated concurrently.",
  )

  @field_validator("param_grid")
  @classmethod
  def _check_param_grid(cls, param_grid: _ParamGrid) -> _ParamGrid:
    """Rejects grids that would expand to nothing or to single characters."""
    sub_grids = (
        [param_grid] if isinstance(param_grid, Mapping) else list(param_grid)
    )
    if not sub_grids:
      raise ValueError("param_grid has no sub-grids.")
    for grid in sub_grids:
      if not grid:
        raise ValueError("param_grid contains an empty grid.")
      for path, values in grid.items():
        if isinstance(values, (str, bytes)):
          raise ValueError(
              f"param_grid[{path!r}] must be a list of values, not the string"
              f" {values!r}. Write [{values!r}]."
          )
        if not values:
          raise ValueError(f"param_grid[{path!r}] has no values.")
    return param_grid

  @field_validator("quality_metric")
  @classmethod
  def _check_quality_metric(
      cls, quality_metric: _QualityMetric
  ) -> _QualityMetric:
    """Rejects efficiency metrics and negative weights."""
    if isinstance(quality_metric, str):
      names = {quality_metric}
    elif isinstance(quality_metric, Mapping):
      names = set(quality_metric)
      invalid = sorted(
          name
          for name, weight in quality_metric.items()
          if not (math.isfinite(weight) and weight >= 0)
      )
      if invalid:
        raise ValueError(
            f"quality_metric gives {invalid} invalid weights. Weights must be"
            " finite and non-negative so that 0, the score of an example that"
            " could not be scored, stays the lowest quality."
        )
    else:
      return quality_metric
    efficiency = sorted(names & _EFFICIENCY_METRICS)
    if efficiency:
      raise ValueError(
          f"quality_metric names efficiency metrics {efficiency}, which"
          " measure cost rather than quality. Token usage and latency are"
          " already Pareto objectives; limit them with max_total_tokens or"
          " max_latency_sec."
      )
    return quality_metric


@dataclass
class _TrialResult:
  """Aggregated metrics for one candidate configuration."""

  params: dict[str, Any]
  agent: BaseAgent
  quality_score: float = 0.0
  eval_case_scores: dict[str, float] = field(default_factory=dict)
  latency_sec: float | None = None
  tokens: _TokenCounts | None = None
  num_cases_evaluated: int = 0
  is_feasible: bool = False


def _expand_param_grid(param_grid: _ParamGrid) -> list[dict[str, Any]]:
  """Expands a parameter grid or sequence of sub-grids into combinations."""
  sub_grids = [param_grid] if isinstance(param_grid, Mapping) else param_grid
  candidates: list[dict[str, Any]] = []
  for grid in sub_grids:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
      candidates.append(dict(zip(keys, values, strict=True)))
  return candidates


def _finite(value: Any) -> float | None:
  """Returns `value` as a float, or None if it is missing, NaN, or infinite."""
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _parse_token_counts(details: Any, score: Any) -> _TokenCounts | None:
  """Reads one `token_usage_v1` result."""
  if isinstance(details, BaseModel):
    details = details.model_dump()
  counts: dict[str, float] = {}
  if isinstance(details, Mapping):
    for name in _TOKEN_FIELDS:
      value = _finite(details.get(name))
      if value is not None:
        counts[name] = max(0.0, value)
  if "total_tokens" not in counts:
    total = _finite(score)
    if "input_tokens" in counts and "output_tokens" in counts:
      counts["total_tokens"] = counts["input_tokens"] + counts["output_tokens"]
    elif total is not None:
      counts["total_tokens"] = max(0.0, total)
  return _TokenCounts(**counts) if counts else None


def _combine_token_counts(
    items: Sequence[_TokenCounts],
    *,
    combine: Callable[[list[float]], float],
) -> _TokenCounts | None:
  """Combines token counts across turns or examples.

  Backends omit optional subset counts (`cached_tokens`, `reasoning_tokens`) on
  calls that did not spend them, so missing entries count as 0 when at least one
  item reported the field and stay None when none did.
  """
  combined: dict[str, float] = {}
  for name in _TOKEN_FIELDS:
    values = [getattr(item, name) for item in items]
    known = [value for value in values if value is not None]
    if not known:
      continue
    if len(known) == len(values):
      combined[name] = combine(known)
    elif name in _OPTIONAL_TOKEN_FIELDS:
      combined[name] = combine([0.0 if v is None else v for v in values])
  return _TokenCounts(**combined) if combined else None


def _extract_example_telemetry(
    example_data: Mapping[str, Any] | None,
) -> _ExampleTelemetry:
  """Reads what the sampler reported for one example."""
  metric_values: dict[str, list[float]] = {}
  metric_names: set[str] = set()
  durations: list[float] = []
  turn_tokens: list[_TokenCounts] = []
  tokens_complete = True
  has_invocations_key = (
      isinstance(example_data, Mapping) and "invocations" in example_data
  )
  invocations = (
      (example_data.get("invocations") or [])
      if isinstance(example_data, Mapping)
      else []
  )
  has_invocations = bool(invocations) if has_invocations_key else True
  for invocation in invocations:
    if not isinstance(invocation, Mapping):
      continue
    turn_count: _TokenCounts | None = None
    for metric_result in invocation.get("eval_metric_results") or []:
      if not isinstance(metric_result, Mapping):
        continue
      name = str(metric_result.get("metric_name") or "")
      if name:
        metric_names.add(name)
      score = _finite(metric_result.get("score"))
      if name == _TOKEN_METRIC:
        details = metric_result.get("details")
        breakdown = (
            details.get("token_usage_details")
            if isinstance(details, Mapping)
            else None
        )
        turn_count = _parse_token_counts(breakdown, metric_result.get("score"))
      elif score is None:
        continue
      elif name == _DURATION_METRIC:
        durations.append(score)
      elif name:
        metric_values.setdefault(name, []).append(score)
    if turn_count is None:
      tokens_complete = False
    else:
      turn_tokens.append(turn_count)

  tokens = None
  if tokens_complete and turn_tokens:
    tokens = _combine_token_counts(turn_tokens, combine=math.fsum)
  return _ExampleTelemetry(
      metric_scores={
          name: statistics.fmean(values)
          for name, values in metric_values.items()
      },
      metric_names=frozenset(metric_names),
      latency_sec=statistics.fmean(durations) if durations else None,
      tokens=tokens,
      has_invocations=has_invocations,
  )


def _total_tokens(trial: _TrialResult) -> float | None:
  """Returns the trial's mean total tokens, or None if not reported."""
  return trial.tokens.total_tokens if trial.tokens else None


def _describe(trials: Sequence[_TrialResult], limit: int = 3) -> str:
  """Lists the first few trials' parameters, shortened, for a log message."""
  shown = []
  for trial in trials[:limit]:
    params = str(trial.params)
    if len(params) > _MAX_LOGGED_PARAMS_CHARS:
      params = params[: _MAX_LOGGED_PARAMS_CHARS - 3] + "..."
    shown.append(params)
  return "; ".join(shown) + ("; ..." if len(trials) > limit else "")


def _token_usage_dict(tokens: _TokenCounts | None) -> dict[str, float]:
  """Returns the reported token counts keyed by TokenUsageDetails names."""
  if tokens is None:
    return {}
  return {
      name: value for name, value in asdict(tokens).items() if value is not None
  }


def _objectives(trial: _TrialResult) -> tuple[float, float, float]:
  """Returns (-quality, tokens, latency); unreported values count as worst."""
  tokens = _total_tokens(trial)
  return (
      -trial.quality_score,
      math.inf if tokens is None else tokens,
      math.inf if trial.latency_sec is None else trial.latency_sec,
  )


def _dominates(a: _TrialResult, b: _TrialResult) -> bool:
  """Returns whether `a` Pareto-dominates `b`."""
  objectives_a, objectives_b = _objectives(a), _objectives(b)
  return objectives_a != objectives_b and all(
      x <= y for x, y in zip(objectives_a, objectives_b, strict=True)
  )


@experimental(FeatureName.GRID_SEARCH_AGENT_OPTIMIZER)
class GridSearchAgentOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, GridSearchAgentWithScores]
):
  """Evaluates every candidate in a parameter grid and returns the Pareto front.

  Every candidate is scored on the same examples, so the front compares like
  with like across quality, total tokens, and latency.
  """

  def __init__(self, config: GridSearchAgentOptimizerConfig) -> None:
    """Initializes the GridSearchAgentOptimizer."""
    self._config = config

  def _calculate_example_quality(
      self,
      *,
      overall_score: float | None,
      metric_scores: Mapping[str, float],
      metric_names: frozenset[str] = frozenset(),
  ) -> float | None:
    """Returns one example's quality score, or None if it is not computable."""
    quality_metric = self._config.quality_metric
    quality_scores = {
        name: value
        for name, value in metric_scores.items()
        if name not in _EFFICIENCY_METRICS
    }
    has_quality_metrics = bool(metric_names - _EFFICIENCY_METRICS)
    if quality_metric is None or (
        isinstance(quality_metric, Mapping) and not quality_metric
    ):
      if quality_scores:
        return statistics.fmean(quality_scores.values())
      if has_quality_metrics or overall_score is None:
        return None
      return float(overall_score)
    if isinstance(quality_metric, str):
      value = metric_scores.get(quality_metric)
      return None if value is None else float(value)
    if callable(quality_metric):
      if not quality_scores and overall_score is None:
        return None
      combined = dict(quality_scores)
      if overall_score is not None:
        combined.setdefault("overall_score", float(overall_score))
      return float(quality_metric(combined))
    if isinstance(quality_metric, Mapping):
      if any(name not in metric_scores for name in quality_metric):
        return None
      return math.fsum(
          float(metric_scores[name]) * float(weight)
          for name, weight in quality_metric.items()
      )
    raise TypeError(f"Unsupported quality_metric type: {type(quality_metric)}")

  def _check_quality_metric_reported(self, reported: set[str]) -> None:
    """Fails fast when the sampler never names a `quality_metric` metric."""
    quality_metric = self._config.quality_metric
    if isinstance(quality_metric, str):
      wanted = {quality_metric}
    elif isinstance(quality_metric, Mapping):
      wanted = set(quality_metric)
    else:
      return
    missing = wanted - reported
    if reported and missing:
      raise ValueError(
          f"quality_metric names {sorted(missing)}, which the sampler did not"
          f" report. Reported metrics: {sorted(reported)}."
      )

  def _score_candidate(
      self,
      *,
      params: dict[str, Any],
      agent: BaseAgent,
      example_ids: Sequence[str],
      sampling_result: UnstructuredSamplingResult,
  ) -> _TrialResult:
    """Aggregates one candidate's sampling result into a _TrialResult."""
    scores_map = sampling_result.scores or {}
    data_map = sampling_result.data or {}
    case_scores: dict[str, float] = {}
    latencies: list[float] = []
    example_tokens: list[_TokenCounts] = []
    reported_metrics: set[str] = set()
    for example_id in example_ids:
      telemetry = _extract_example_telemetry(data_map.get(example_id))
      reported_metrics.update(telemetry.metric_names)
      if telemetry.latency_sec is not None:
        latencies.append(telemetry.latency_sec)
      if telemetry.tokens is not None:
        example_tokens.append(telemetry.tokens)
      overall_score = (
          _finite(scores_map.get(example_id))
          if telemetry.has_invocations
          else None
      )
      quality = _finite(
          self._calculate_example_quality(
              overall_score=overall_score,
              metric_scores=telemetry.metric_scores,
              metric_names=telemetry.metric_names,
          )
      )
      if quality is not None:
        case_scores[example_id] = quality
    self._check_quality_metric_reported(reported_metrics)

    all_timed = len(latencies) == len(example_ids) and bool(latencies)
    all_counted = len(example_tokens) == len(example_ids) and bool(
        example_tokens
    )
    trial = _TrialResult(
        params=params,
        agent=agent,
        # Unscored examples count as 0 so every candidate is averaged over the
        # full example set.
        quality_score=math.fsum(case_scores.values()) / len(example_ids),
        eval_case_scores=case_scores,
        latency_sec=statistics.fmean(latencies) if all_timed else None,
        tokens=(
            _combine_token_counts(example_tokens, combine=statistics.fmean)
            if all_counted
            else None
        ),
        num_cases_evaluated=len(case_scores),
    )
    trial.is_feasible = self._is_trial_feasible(trial)
    return trial

  def _is_trial_feasible(self, trial: _TrialResult) -> bool:
    """Checks the quality floor and the latency and token ceilings."""
    min_quality = self._config.min_quality_score
    max_latency = self._config.max_latency_sec
    max_tokens = self._config.max_total_tokens
    total_tokens = _total_tokens(trial)
    if trial.num_cases_evaluated == 0:
      return False
    if min_quality is not None and trial.quality_score < min_quality:
      return False
    if max_latency is not None and (
        trial.latency_sec is None or trial.latency_sec > max_latency
    ):
      return False
    if max_tokens is not None and (
        total_tokens is None or total_tokens > max_tokens
    ):
      return False
    return True

  def _build_pareto_result(
      self,
      trials: Sequence[_TrialResult],
  ) -> OptimizerResult[GridSearchAgentWithScores]:
    """Returns the feasible trials no other feasible trial dominates."""
    feasible = [trial for trial in trials if trial.is_feasible]
    front = [
        trial
        for trial in feasible
        if not any(
            _dominates(other, trial) for other in feasible if other is not trial
        )
    ]
    front.sort(key=_objectives)
    return OptimizerResult(
        optimized_agents=[
            GridSearchAgentWithScores(
                optimized_agent=trial.agent,
                overall_score=trial.quality_score,
                eval_case_scores=trial.eval_case_scores,
                token_usage=_token_usage_dict(trial.tokens),
                latency_sec=trial.latency_sec,
                metadata={
                    "params": trial.params,
                    "num_cases_evaluated": trial.num_cases_evaluated,
                },
            )
            for trial in front
        ]
    )

  async def optimize(
      self,
      initial_agent: BaseAgent,
      sampler: Sampler[UnstructuredSamplingResult],
  ) -> OptimizerResult[GridSearchAgentWithScores]:
    """Evaluates every grid candidate on the same examples."""
    requested_set = self._config.example_set
    example_set = requested_set
    if requested_set == Sampler.TRAIN_SET:
      example_ids = list(sampler.get_train_example_ids())
      if not example_ids:
        example_set = Sampler.VALIDATION_SET
        example_ids = list(sampler.get_validation_example_ids())
    else:
      example_ids = list(sampler.get_validation_example_ids())
      if not example_ids:
        example_set = Sampler.TRAIN_SET
        example_ids = list(sampler.get_train_example_ids())
    example_ids = list(dict.fromkeys(example_ids))
    if not example_ids:
      logger.warning("Grid search: the sampler has no examples to score.")
      return OptimizerResult(optimized_agents=[])
    if example_set != requested_set:
      logger.warning(
          "Grid search: sampler has no %s examples; falling back to %d %s"
          " examples.",
          requested_set,
          len(example_ids),
          example_set,
      )

    candidates = _expand_param_grid(self._config.param_grid)
    agents = [
        clone_agent_with_params(initial_agent, params) for params in candidates
    ]
    logger.info(
        "Grid search: evaluating %d candidates on %d %s examples each.",
        len(candidates),
        len(example_ids),
        example_set,
    )
    semaphore = asyncio.Semaphore(self._config.max_concurrent_candidates)

    async def _evaluate(
        params: dict[str, Any], agent: BaseAgent
    ) -> _TrialResult:
      async with semaphore:
        sampling_result = await sampler.sample_and_score(
            agent,  # type: ignore[arg-type]  # Sampler is typed Agent.
            example_set=example_set,
            batch=list(example_ids),
            capture_full_eval_data=True,
        )
      return self._score_candidate(
          params=params,
          agent=agent,
          example_ids=example_ids,
          sampling_result=sampling_result,
      )

    tasks = [
        asyncio.create_task(_evaluate(params, agent))
        for params, agent in zip(candidates, agents, strict=True)
    ]
    try:
      trials = await asyncio.gather(*tasks)
    finally:
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
    self._warn_about_missing_data(trials, num_examples=len(example_ids))
    return self._build_pareto_result(trials)

  def _warn_about_missing_data(
      self, trials: Sequence[_TrialResult], *, num_examples: int
  ) -> None:
    """Logs warnings for unscored examples, failed gates, or missing data."""
    unscored = [trial for trial in trials if not trial.num_cases_evaluated]
    if unscored:
      logger.warning(
          "Grid search: %d of %d candidates produced no scored examples and"
          " are excluded: %s",
          len(unscored),
          len(trials),
          _describe(unscored),
      )
    scored = [trial for trial in trials if trial.num_cases_evaluated]
    if scored and not any(trial.is_feasible for trial in scored):
      logger.warning(
          "Grid search: all %d scored candidates failed feasibility thresholds"
          " (best quality=%.4f); returning an empty result.",
          len(scored),
          max(trial.quality_score for trial in scored),
      )
    partial = [
        trial for trial in scored if trial.num_cases_evaluated < num_examples
    ]
    if partial:
      logger.warning(
          "Grid search: %d of %d candidates could not be scored on every"
          " example (unscored examples count as 0 quality): %s",
          len(partial),
          len(trials),
          _describe(partial),
      )
    missing_tokens = [trial for trial in scored if _total_tokens(trial) is None]
    if missing_tokens:
      logger.warning(
          "Grid search: %d of %d scored candidates reported no token usage"
          " (ranked worst on tokens; fails max_total_tokens if set): %s",
          len(missing_tokens),
          len(scored),
          _describe(missing_tokens),
      )
    missing_latency = [trial for trial in scored if trial.latency_sec is None]
    if missing_latency:
      logger.warning(
          "Grid search: %d of %d scored candidates reported no latency"
          " (ranked worst on latency; fails max_latency_sec if set): %s",
          len(missing_latency),
          len(scored),
          _describe(missing_latency),
      )
