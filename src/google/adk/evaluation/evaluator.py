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

from abc import ABC
from collections.abc import Awaitable
from typing import ClassVar
from typing import Optional

from pydantic import BaseModel

from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_case import SessionState
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalStatus as EvalStatus
from .eval_metrics import TokenUsageDetails
from .eval_rubrics import RubricScore


def _validate_invocation_lengths(
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
) -> None:
  """Rejects invocation lists that cannot be paired without truncation."""
  if expected_invocations is not None and len(actual_invocations) != len(
      expected_invocations
  ):
    raise ValueError(
        "actual_invocations and expected_invocations must have the same"
        f" length; got {len(actual_invocations)} and"
        f" {len(expected_invocations)}."
    )


class PerInvocationResult(BaseModel):
  """Metric evaluation score per invocation."""

  actual_invocation: Invocation
  expected_invocation: Optional[Invocation] = None
  score: Optional[float] = None
  eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  rubric_scores: Optional[list[RubricScore]] = None
  token_usage_details: Optional[TokenUsageDetails] = None
  """Per-type token counts, reported by the token usage metric."""


class EvaluationResult(BaseModel):
  overall_score: Optional[float] = None
  """Overall score, based on each invocation."""

  overall_eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  """Overall status, based on each invocation."""

  per_invocation_results: list[PerInvocationResult] = []
  """Detailed results per invocation."""

  overall_rubric_scores: Optional[list[RubricScore]] = None
  """Overall rubric, based on each invocation."""

  overall_token_usage_details: Optional[TokenUsageDetails] = None
  """Per-type token counts, averaged over invocations."""


class EvaluationContext(BaseModel):
  """Session state for one eval case.

  LocalEvalService gives each metric its own copy. Actual states are snapshots
  from inference, so later session updates do not change the evaluation input.
  """

  initial_session_state: Optional[SessionState] = None
  """Actual state before the first turn, or None if no snapshot is available."""

  final_session_state: Optional[SessionState] = None
  """Actual state after the last turn, or None if no snapshot is available."""

  expected_final_session_state: Optional[SessionState] = None
  """Expected final state from EvalCase.final_session_state."""


class Evaluator(ABC):
  """A metrics evaluator interface."""

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

  def evaluate_with_context(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
      *,
      context: EvaluationContext,
  ) -> EvaluationResult | Awaitable[EvaluationResult]:
    """Returns a metric result with access to session state.

    Override this method for metrics that need session state. The default calls
    evaluate_invocations with its original arguments, so existing evaluators
    keep their current signatures. Both methods can return an awaitable.

    Args:
      actual_invocations: Invocations from the agent under test.
      expected_invocations: Optional reference invocations.
      conversation_scenario: Optional scenario for a multi-turn conversation.
      context: Actual state snapshots and the expected final state for this case.

    Returns:
      The evaluation result, or an awaitable that returns it.
    """
    return self.evaluate_invocations(
        actual_invocations=actual_invocations,
        expected_invocations=expected_invocations,
        conversation_scenario=conversation_scenario,
    )

  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult | Awaitable[EvaluationResult]:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations.

    Args:
      actual_invocations: These are the invocations that are obtained from the
        agent under test.
      expected_invocations: An optional list of invocations, if specified,
        usually act as a benchmark/golden response. If these are specified
        usually the expectation is that the length of this list and actual
        invocation is the same.
      conversation_scenario: An optional conversation scenario for multi-turn
        conversations.
    """
    raise NotImplementedError()
