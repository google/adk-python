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

from __future__ import annotations

import asyncio
import math
from typing import ClassVar

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ..tools.tool_args_matcher._tool_args_matcher import ToolArgsMatcher
from .eval_case import ConversationScenario
from .eval_case import get_all_tool_calls
from .eval_case import Invocation
from .eval_metrics import _get_metric_threshold
from .eval_metrics import EvalMetric
from .eval_metrics import ToolParameterMatchCriterion
from .eval_metrics import ToolTrajectoryCriterion
from .evaluator import _validate_invocation_lengths
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

_MatchType = ToolTrajectoryCriterion.MatchType

_MAX_CONCURRENT_CALL_SCORES = 8
"""How many pairs of tool calls of one invocation are scored at once.

A model-backed argument matcher runs once per non-deterministic argument, and
`ToolArgsMatcher.match` runs those arguments concurrently within each pair, so
this bounds concurrent pairs rather than total concurrent matcher calls.
"""


class _ToolParameterMatchEvaluator(Evaluator):
  """Scores the arguments of expected tool calls aligned with actual calls.

  Each expected call is paired with at most one actual call, as the criterion's
  `match_type` sets. A pair of calls with the same name scores the mean of the
  expected arguments' scores. A pair whose names differ, or an expected call
  left unpaired, scores 0.0.

  Arguments are compared with a `ToolArgsMatcher` even when the criterion sets
  none, so creating the evaluator raises `RuntimeError` while the
  `TOOL_ARGS_MATCHER` feature is turned off.
  """

  criterion_type: ClassVar[type[ToolParameterMatchCriterion]] = (
      ToolParameterMatchCriterion
  )

  def __init__(self, *, eval_metric: EvalMetric):
    self._threshold = _get_metric_threshold(eval_metric)
    criterion = _get_criterion(eval_metric)
    self._args_matcher = (
        criterion.args_matcher if criterion else None
    ) or ToolArgsMatcher()
    self._match_type = (
        criterion.match_type if criterion else _MatchType.IN_ORDER
    )

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation] | None = None,
      conversation_scenario: ConversationScenario | None = None,
  ) -> EvaluationResult:
    if expected_invocations is None:
      raise ValueError("expected_invocations is needed by this metric.")
    _validate_invocation_lengths(actual_invocations, expected_invocations)
    del conversation_scenario

    per_invocation_results = []
    evaluated_scores = []
    for actual, expected in zip(
        actual_invocations, expected_invocations, strict=True
    ):
      expected_calls = get_all_tool_calls(expected.intermediate_data)
      if not expected_calls:
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
            )
        )
        continue

      score = await self._score_invocation(actual, expected_calls)
      evaluated_scores.append(score)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
          )
      )

    if not evaluated_scores:
      return EvaluationResult(
          per_invocation_results=per_invocation_results,
      )

    overall_score = sum(evaluated_scores) / len(evaluated_scores)
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=self._get_eval_status(overall_score),
        per_invocation_results=per_invocation_results,
    )

  async def _score_invocation(
      self,
      actual_invocation: Invocation,
      expected_calls: list[genai_types.FunctionCall],
  ) -> float:
    actual_calls = get_all_tool_calls(actual_invocation.intermediate_data)
    if self._match_type == _MatchType.ANY_ORDER:
      call_scores = await self._score_calls_in_any_order(
          actual_calls, expected_calls
      )
    else:
      pair_calls = (
          _pair_calls_exactly
          if self._match_type == _MatchType.EXACT
          else _pair_calls_in_order
      )
      call_scores = await self._score_pairs(
          list(zip(pair_calls(actual_calls, expected_calls), expected_calls))
      )
    return sum(call_scores) / len(call_scores)

  async def _score_calls_in_any_order(
      self,
      actual_calls: list[genai_types.FunctionCall],
      expected_calls: list[genai_types.FunctionCall],
  ) -> list[float]:
    """Pairs same-name calls so that the total score is as high as possible.

    Finding the highest-scoring one-to-one assignment requires the score of
    every `(actual, expected)` pair of the same tool name, so a tool expected
    `n` times and called `m` times scores `n * m` pairs.
    """
    groups: list[tuple[list[int], list[int]]] = []
    for name in dict.fromkeys(call.name for call in expected_calls):
      expected_indexes = [
          i for i, call in enumerate(expected_calls) if call.name == name
      ]
      actual_indexes = [
          j for j, call in enumerate(actual_calls) if call.name == name
      ]
      if actual_indexes:
        groups.append((expected_indexes, actual_indexes))

    # Every same-name pair of every tool is scored in one batch, so that the
    # tools share the concurrency bound instead of waiting for each other.
    pair_scores = iter(
        await self._score_pairs([
            (actual_calls[j], expected_calls[i])
            for expected_indexes, actual_indexes in groups
            for i in expected_indexes
            for j in actual_indexes
        ])
    )
    call_scores = [0.0] * len(expected_calls)
    for expected_indexes, actual_indexes in groups:
      scores = [
          [next(pair_scores) for _ in actual_indexes] for _ in expected_indexes
      ]
      for row, column in _max_score_assignment(scores):
        call_scores[expected_indexes[row]] = scores[row][column]
    return call_scores

  async def _score_pairs(
      self,
      pairs: list[
          tuple[genai_types.FunctionCall | None, genai_types.FunctionCall]
      ],
  ) -> list[float]:
    """Scores pairs of actual and expected calls concurrently, in order.

    At most `_MAX_CONCURRENT_CALL_SCORES` pairs are scored at once. If scoring
    one pair raises, the others still running are cancelled before the
    exception propagates.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CALL_SCORES)

    async def _score_bounded(
        actual_call: genai_types.FunctionCall | None,
        expected_call: genai_types.FunctionCall,
    ) -> float:
      async with semaphore:
        return await self._score_call(actual_call, expected_call)

    tasks = [
        asyncio.create_task(_score_bounded(actual_call, expected_call))
        for actual_call, expected_call in pairs
    ]
    try:
      return list(await asyncio.gather(*tasks))
    finally:
      # `gather` returns at the first error and leaves the other tasks
      # running, which for a model-backed matcher means paying for calls whose
      # result nobody reads.
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  async def _score_call(
      self,
      actual_call: genai_types.FunctionCall | None,
      expected_call: genai_types.FunctionCall,
  ) -> float:
    if actual_call is None or actual_call.name != expected_call.name:
      return 0.0
    result = await self._args_matcher.match(
        tool_name=expected_call.name or "",
        expected=expected_call.args or {},
        actual=actual_call.args or {},
    )
    return result.score

  def _get_eval_status(self, score: float) -> EvalStatus:
    return EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED


def _pair_calls_exactly(
    actual_calls: list[genai_types.FunctionCall],
    expected_calls: list[genai_types.FunctionCall],
) -> list[genai_types.FunctionCall | None]:
  """Returns, for each expected call, the actual call at the same position."""
  return [
      actual_calls[index] if index < len(actual_calls) else None
      for index in range(len(expected_calls))
  ]


def _pair_calls_in_order(
    actual_calls: list[genai_types.FunctionCall],
    expected_calls: list[genai_types.FunctionCall],
) -> list[genai_types.FunctionCall | None]:
  """Returns, for each expected call, the next actual call of the same name.

  Calls are paired by name alone, not by arguments, so a retried call is paired
  with its first attempt even when a later attempt has the expected arguments.
  Skipping on arguments would need every same-name pair scored first, which for
  a model-backed argument matcher costs `n * m` matcher calls instead of `n`.

  An expected call with no such actual call is paired with None, and does not
  move past any actual call, so it cannot hide a later match.
  """
  pairs: list[genai_types.FunctionCall | None] = []
  actual_index = 0
  for expected_call in expected_calls:
    match_index = next(
        (
            index
            for index in range(actual_index, len(actual_calls))
            if actual_calls[index].name == expected_call.name
        ),
        None,
    )
    if match_index is None:
      pairs.append(None)
    else:
      pairs.append(actual_calls[match_index])
      actual_index = match_index + 1
  return pairs


def _get_criterion(
    eval_metric: EvalMetric,
) -> ToolParameterMatchCriterion | None:
  """Returns the metric's criterion as a `ToolParameterMatchCriterion`.

  Args:
    eval_metric: The metric to read the criterion from.

  Returns:
    The criterion, or None when the metric has no criterion.

  Raises:
    ValueError: If the criterion is not a valid `ToolParameterMatchCriterion`.
  """
  criterion = eval_metric.criterion
  if criterion is None or isinstance(criterion, ToolParameterMatchCriterion):
    return criterion
  # A criterion loaded from JSON arrives as a BaseCriterion that keeps the
  # metric-specific keys as extras, so it is validated again here.
  try:
    return ToolParameterMatchCriterion.model_validate(criterion.model_dump())
  except ValidationError as e:
    raise ValueError(
        f"`{eval_metric.metric_name}` metric expects a criterion of type"
        f" `{ToolParameterMatchCriterion.__name__}`."
    ) from e


def _max_score_assignment(
    scores: list[list[float]],
) -> list[tuple[int, int]]:
  """Pairs rows with columns one-to-one so that the summed score is maximal.

  This is the Hungarian algorithm. Every row is paired when there are at least
  as many columns as rows, and every column otherwise.

  Args:
    scores: A non-empty matrix where `scores[row][column]` is the score of
      pairing that row with that column. All rows have the same length.

  Returns:
    The `(row, column)` pairs, in row order.
  """
  if len(scores) > len(scores[0]):
    transposed = [list(column) for column in zip(*scores)]
    return sorted(
        (row, column) for column, row in _max_score_assignment(transposed)
    )

  # Minimizes the negated scores over a 1-based matrix with `rows <= columns`.
  # `owner[column]` is the row paired with that column, or 0 for none.
  rows, columns = len(scores), len(scores[0])
  row_potential = [0.0] * (rows + 1)
  column_potential = [0.0] * (columns + 1)
  owner = [0] * (columns + 1)
  previous = [0] * (columns + 1)
  for row in range(1, rows + 1):
    owner[0] = row
    free_column = 0
    slack = [math.inf] * (columns + 1)
    visited = [False] * (columns + 1)
    while owner[free_column] != 0:
      visited[free_column] = True
      current_row = owner[free_column]
      delta = math.inf
      next_column = 0
      for column in range(1, columns + 1):
        if visited[column]:
          continue
        reduced_cost = (
            -scores[current_row - 1][column - 1]
            - row_potential[current_row]
            - column_potential[column]
        )
        if reduced_cost < slack[column]:
          slack[column] = reduced_cost
          previous[column] = free_column
        if slack[column] < delta:
          delta = slack[column]
          next_column = column
      for column in range(columns + 1):
        if visited[column]:
          row_potential[owner[column]] += delta
          column_potential[column] -= delta
        else:
          slack[column] -= delta
      free_column = next_column
    while free_column != 0:
      previous_column = previous[free_column]
      owner[free_column] = owner[previous_column]
      free_column = previous_column

  return sorted(
      (owner[column] - 1, column - 1)
      for column in range(1, columns + 1)
      if owner[column] != 0
  )
