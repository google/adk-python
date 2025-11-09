# Copyright 2025 Google LLC
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
from typing import ClassVar
from typing import Optional

from pydantic import BaseModel
from typing_extensions import TypeAlias

from .eval_case import Invocation
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalStatus
from .eval_rubrics import RubricScore

# Redefining the type here for backward compatibility.
EvalStatus: TypeAlias = EvalStatus


class PerInvocationResult(BaseModel):
  """Metric evaluation score per invocation."""

  actual_invocation: Invocation
  expected_invocation: Optional[Invocation] = None
  score: Optional[float] = None
  eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  rubric_scores: Optional[list[RubricScore]] = None


class EvaluationResult(BaseModel):
  overall_score: Optional[float] = None
  """Overall score, based on each invocation."""

  overall_eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  """Overall status, based on each invocation."""

  per_invocation_results: list[PerInvocationResult] = []
  """Detailed results per invocation."""

  overall_rubric_scores: Optional[list[RubricScore]] = None
  """Overall rubric, based on each invocation."""


class Evaluator(ABC):
  """A metrics evaluator interface."""

  criterion_type: ClassVar[type[BaseCriterion]] = BaseCriterion

  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ) -> EvaluationResult:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations.

    Args:
      actual_invocations: These are the invocations that are obtained from the
        agent under test.
      expected_invocations: An optional list of invocations, if specified,
        usually act as a benchmark/golden response. If these are specified
        usually the expectation is that the length of this list and actual
        invocation is the same.
    """
    raise NotImplementedError()
