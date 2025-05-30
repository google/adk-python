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

from abc import ABC
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .eval_case import Invocation


class EvalStatus(Enum):
  PASSED = 1
  FAILED = 2
  NOT_EVALUATED = 3


class PerInvocationResult(BaseModel):
  """Metric evaluation score per invocation."""

  actual_invocation: Invocation
  expected_invocation: Invocation
  score: Optional[float] = None
  eval_status: EvalStatus = EvalStatus.NOT_EVALUATED


class EvaluationResult(BaseModel):
  overall_score: Optional[float] = None
  """Overall score, based on each invocation."""

  overall_eval_status: EvalStatus = EvalStatus.NOT_EVALUATED
  """Overall status, based on each invocation."""

  per_invocation_results: list[PerInvocationResult] = []


class Evaluator(ABC):
  """A merics evaluator interface."""

  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    """Returns EvaluationResult after performing evaluations using actual and expected invocations."""
    raise NotImplementedError()
