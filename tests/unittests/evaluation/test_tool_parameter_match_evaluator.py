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

"""Tests for the tool parameter match evaluator."""

import asyncio
from typing import ClassVar
from typing import Literal

from google.adk.evaluation._tool_parameter_match_evaluator import _MAX_CONCURRENT_CALL_SCORES
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import ToolParameterMatchCriterion
from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.metric_evaluator_registry import MetricEvaluatorRegistry
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.tool_args_matcher import ArgMatchResult
from google.adk.tools.tool_args_matcher import BaseArgMatcher
from google.adk.tools.tool_args_matcher import ToolArgsMatcher
from google.genai import types as genai_types
import pytest

_USER_CONTENT = genai_types.Content(parts=[genai_types.Part(text="test")])

_METRIC_NAME = PrebuiltMetrics.TOOL_PARAMETER_MATCH.value


def _invocation(*calls: genai_types.FunctionCall) -> Invocation:
  return Invocation(
      user_content=_USER_CONTENT,
      intermediate_data=IntermediateData(tool_uses=list(calls)),
  )


async def _evaluate(
    actual: list[Invocation],
    expected: list[Invocation] | None,
    eval_metric: EvalMetric | None = None,
) -> EvaluationResult:
  evaluator = MetricEvaluatorRegistry().get_evaluator(
      eval_metric or EvalMetric(metric_name=_METRIC_NAME, threshold=0.5)
  )
  return await evaluator.evaluate_invocations(actual, expected)


async def test_scores_expected_arguments_independently():
  """A partially correct tool call receives partial credit."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome", "rooms": 2})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome", "rooms": 1})
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 0.5
  assert result.overall_eval_status == EvalStatus.PASSED


async def test_call_without_expected_arguments_receives_full_credit():
  """A matched call with no expected arguments scores one."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  expected = _invocation(genai_types.FunctionCall(name="search", args={}))

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 1.0


async def test_aligns_repeated_calls_by_name_in_order():
  """Extra calls do not displace later expected calls with matching names."""
  actual = _invocation(
      genai_types.FunctionCall(name="log", args={}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 1.0


async def test_unmatched_expected_call_receives_no_credit():
  """An expected tool call absent from the actual trajectory scores zero."""
  actual = _invocation()
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 0.0
  assert result.overall_eval_status == EvalStatus.FAILED


async def test_unmatched_call_does_not_consume_later_match():
  """An absent expected call does not hide a later matching actual call."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="log", args={}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 0.5


async def test_empty_expected_trajectory_is_not_evaluated():
  """An invocation without expected tool calls is not evaluated."""
  invocation = _invocation()

  result = await _evaluate([invocation], [invocation])

  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
  )


async def test_missing_expected_invocations_raises_value_error():
  """Evaluating without reference invocations raises `ValueError`."""
  with pytest.raises(ValueError, match="expected_invocations is needed"):
    await _evaluate([_invocation()], None)


async def test_criterion_args_matcher_compares_arguments():
  """A criterion's argument matcher replaces exact equality."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "rome"})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=1.0,
          args_matcher=ToolArgsMatcher(
              per_arg_matchers={"city": {"type": "casefold"}}
          ),
      ),
  )

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 1.0


async def test_criterion_from_json_config_compares_arguments():
  """A camelCase JSON criterion configures the argument matcher."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"rooms": 2.05})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"rooms": 2})
  )
  eval_metric = EvalMetric.model_validate({
      "metricName": _METRIC_NAME,
      "criterion": {
          "threshold": 1.0,
          "argsMatcher": {
              "perArgMatchers": {"rooms": {"type": "numeric", "tolerance": 0.1}}
          },
      },
  })

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 1.0


async def test_criterion_accepts_unregistered_matcher_instance():
  """A matcher instance passed in code is used without registration."""

  class _AlwaysMatcher(BaseArgMatcher):
    type: Literal["always"] = "always"

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      return ArgMatchResult(matched=True)

  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Paris"})
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"})
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=1.0,
          args_matcher=ToolArgsMatcher(default_matcher=_AlwaysMatcher()),
      ),
  )

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 1.0


def test_invalid_criterion_is_rejected():
  """A criterion naming an unknown matcher type fails at evaluator creation."""
  eval_metric = EvalMetric.model_validate({
      "metricName": _METRIC_NAME,
      "criterion": {
          "threshold": 1.0,
          "argsMatcher": {"defaultMatcher": {"type": "unknown"}},
      },
  })

  with pytest.raises(ValueError, match="expects a criterion of type"):
    MetricEvaluatorRegistry().get_evaluator(eval_metric)


def _metric(match_type: ToolTrajectoryCriterion.MatchType) -> EvalMetric:
  return EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5, match_type=match_type
      ),
  )


async def test_in_order_does_not_pair_calls_made_out_of_order():
  """By default, an expected call cannot pair with an earlier actual call."""
  actual = _invocation(
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 0.5


async def test_in_order_pairs_first_same_name_call_regardless_of_arguments():
  """By default, a retried call is scored on its first attempt, not its best.

  Unlike `tool_trajectory_avg_score`, `IN_ORDER` pairs calls by name alone, so
  a later actual call with the expected arguments is left unpaired.
  """
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = await _evaluate([actual], [expected])

  assert result.overall_score == 0.0


async def test_exact_pairs_calls_by_position():
  """With `EXACT`, a call in a different position scores zero."""
  actual = _invocation(
      genai_types.FunctionCall(name="log", args={}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = await _evaluate(
      [actual], [expected], _metric(ToolTrajectoryCriterion.MatchType.EXACT)
  )

  assert result.overall_score == 0.0


async def test_exact_scores_matching_positions():
  """With `EXACT`, calls in the expected positions are scored normally."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Paris"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
  )

  result = await _evaluate(
      [actual], [expected], _metric(ToolTrajectoryCriterion.MatchType.EXACT)
  )

  assert result.overall_score == 0.5


async def test_any_order_pairs_calls_made_out_of_order():
  """With `ANY_ORDER`, calls pair by name regardless of their order."""
  actual = _invocation(
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
  )

  result = await _evaluate(
      [actual],
      [expected],
      _metric(ToolTrajectoryCriterion.MatchType.ANY_ORDER),
  )

  assert result.overall_score == 1.0


async def test_any_order_maximizes_total_score_of_repeated_calls():
  """Repeated calls pair for the best total, not the best first pair.

  Pairing the first expected call with its best match, the first actual call,
  would leave the second expected call with a 0.0 match, for 0.5 overall.
  """
  actual = _invocation(
      genai_types.FunctionCall(
          name="search", args={"x": 1, "y": 1, "z": 1, "w": 1}
      ),
      genai_types.FunctionCall(
          name="search", args={"x": 1, "y": 0, "z": 0, "w": 0}
      ),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"x": 1, "y": 1}),
      genai_types.FunctionCall(name="search", args={"z": 1, "w": 1}),
  )

  result = await _evaluate(
      [actual],
      [expected],
      _metric(ToolTrajectoryCriterion.MatchType.ANY_ORDER),
  )

  assert result.overall_score == 0.75


async def test_any_order_pairs_each_actual_call_once():
  """More expected calls than actual ones leaves the extras unpaired."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = await _evaluate(
      [actual],
      [expected],
      _metric(ToolTrajectoryCriterion.MatchType.ANY_ORDER),
  )

  assert result.overall_score == 0.5


async def test_any_order_pairs_expected_call_with_best_actual_call():
  """With more actual calls than expected, the best-matching one is paired."""
  actual = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Paris"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )

  result = await _evaluate(
      [actual],
      [expected],
      _metric(ToolTrajectoryCriterion.MatchType.ANY_ORDER),
  )

  assert result.overall_score == 1.0


async def test_match_type_is_configurable_from_json():
  """A camelCase JSON criterion names the match type as the trajectory one."""
  actual = _invocation(
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
  )
  eval_metric = EvalMetric.model_validate({
      "metricName": _METRIC_NAME,
      "criterion": {"threshold": 1.0, "matchType": "ANY_ORDER"},
  })

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 1.0


async def test_match_type_accepts_hyphenated_spelling():
  """The match type is read in the spellings the trajectory metric accepts."""
  actual = _invocation(
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
  )
  expected = _invocation(
      genai_types.FunctionCall(name="search", args={"city": "Rome"}),
      genai_types.FunctionCall(name="book", args={"city": "Rome"}),
  )
  eval_metric = EvalMetric.model_validate({
      "metricName": _METRIC_NAME,
      "criterion": {"threshold": 1.0, "matchType": "any-order"},
  })

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 1.0


async def test_natural_language_argument_uses_its_own_matcher():
  """A judge for one free-text argument leaves the others exact."""

  class _KeywordJudge(BaseArgMatcher):
    """Stands in for a model that judges whether two phrases mean the same."""

    type: Literal["keyword_judge"] = "keyword_judge"
    is_deterministic: ClassVar[bool] = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      shared = set(str(expected).lower().split()) & set(
          str(actual).lower().split()
      )
      return ArgMatchResult(matched=bool(shared), reason=f"Shared: {shared}")

  actual = _invocation(
      genai_types.FunctionCall(
          name="search", args={"query": "NYC workforce", "dept": "sales"}
      )
  )
  expected = _invocation(
      genai_types.FunctionCall(
          name="search", args={"query": "employees in NYC", "dept": "eng"}
      )
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5,
          args_matcher=ToolArgsMatcher(
              per_arg_matchers={"query": _KeywordJudge()}
          ),
      ),
  )

  result = await _evaluate([actual], [expected], eval_metric)

  assert result.overall_score == 0.5


async def test_paired_calls_are_scored_concurrently():
  """Each paired call is scored at the same time, not one after another."""
  barrier = asyncio.Barrier(2)

  class _BarrierJudge(BaseArgMatcher):
    type: Literal["barrier_judge"] = "barrier_judge"
    is_deterministic: ClassVar[bool] = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      await barrier.wait()
      return ArgMatchResult(matched=True)

  calls = _invocation(
      genai_types.FunctionCall(name="search", args={"query": "a"}),
      genai_types.FunctionCall(name="search", args={"query": "b"}),
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5,
          args_matcher=ToolArgsMatcher(default_matcher=_BarrierJudge()),
      ),
  )

  result = await asyncio.wait_for(
      _evaluate([calls], [calls], eval_metric), timeout=5
  )

  assert result.overall_score == 1.0


async def test_pair_scores_are_bounded_in_concurrency():
  """No more than the concurrency bound of pairs is scored at the same time."""
  in_flight = 0
  max_in_flight = 0

  class _CountingJudge(BaseArgMatcher):
    type: Literal["counting_judge"] = "counting_judge"
    is_deterministic: ClassVar[bool] = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      nonlocal in_flight, max_in_flight
      in_flight += 1
      max_in_flight = max(max_in_flight, in_flight)
      # Yields so that every pair the bound admits starts before any ends.
      for _ in range(3):
        await asyncio.sleep(0)
      in_flight -= 1
      return ArgMatchResult(matched=True)

  calls = _invocation(*(
      genai_types.FunctionCall(name="search", args={"query": str(i)})
      for i in range(_MAX_CONCURRENT_CALL_SCORES + 2)
  ))
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5,
          args_matcher=ToolArgsMatcher(default_matcher=_CountingJudge()),
      ),
  )

  await _evaluate([calls], [calls], eval_metric)

  assert max_in_flight == _MAX_CONCURRENT_CALL_SCORES


async def test_matcher_error_propagates():
  """An error raised by an argument matcher reaches the caller."""

  class _FailingJudge(BaseArgMatcher):
    type: Literal["failing_judge"] = "failing_judge"

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      raise RuntimeError("judge down")

  calls = _invocation(
      genai_types.FunctionCall(name="search", args={"query": "a"})
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5,
          args_matcher=ToolArgsMatcher(default_matcher=_FailingJudge()),
      ),
  )

  with pytest.raises(RuntimeError, match="judge down"):
    await _evaluate([calls], [calls], eval_metric)


async def test_failing_pair_cancels_other_pair_scores():
  """When one pair fails, the pairs still being scored are cancelled.

  Setup: the pair for query 'a' fails once the pair for 'b' has started; 'b'
    waits forever.
  Act: evaluate both pairs.
  Assert: the error reaches the caller and the 'b' pair was cancelled.
  """
  b_started = asyncio.Event()
  b_cancelled = asyncio.Event()

  class _FailFastJudge(BaseArgMatcher):
    type: Literal["fail_fast_judge"] = "fail_fast_judge"
    is_deterministic: ClassVar[bool] = False

    async def match(
        self,
        *,
        tool_name: str,
        arg_name: str,
        expected: object,
        actual: object,
    ) -> ArgMatchResult:
      if expected == "a":
        await b_started.wait()
        raise RuntimeError("judge down")
      b_started.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        b_cancelled.set()
        raise
      return ArgMatchResult(matched=True)

  calls = _invocation(
      genai_types.FunctionCall(name="search", args={"query": "a"}),
      genai_types.FunctionCall(name="search", args={"query": "b"}),
  )
  eval_metric = EvalMetric(
      metric_name=_METRIC_NAME,
      criterion=ToolParameterMatchCriterion(
          threshold=0.5,
          args_matcher=ToolArgsMatcher(default_matcher=_FailFastJudge()),
      ),
  )

  with pytest.raises(RuntimeError, match="judge down"):
    await asyncio.wait_for(_evaluate([calls], [calls], eval_metric), timeout=5)

  assert b_cancelled.is_set()


def test_evaluator_needs_tool_args_matcher_feature():
  """With the TOOL_ARGS_MATCHER feature off, the evaluator cannot be created."""
  eval_metric = EvalMetric(metric_name=_METRIC_NAME, threshold=0.5)

  with temporary_feature_override(FeatureName.TOOL_ARGS_MATCHER, False):
    with pytest.raises(RuntimeError, match="TOOL_ARGS_MATCHER is not enabled"):
      MetricEvaluatorRegistry().get_evaluator(eval_metric)
