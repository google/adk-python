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

"""A custom eval metric, used to exercise custom metrics end to end."""

from __future__ import annotations

from typing import Optional

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult


def tool_trajectory_length_match(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]] = None,
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Scores 1.0 per invocation when actual and expected tool counts match."""
  del eval_metric
  del conversation_scenario
  expected_invocations = expected_invocations or []

  per_invocation_results = []
  for idx, actual in enumerate(actual_invocations):
    expected = (
        expected_invocations[idx] if idx < len(expected_invocations) else None
    )
    actual_tools = get_all_tool_calls(actual.intermediate_data)
    expected_tools = (
        get_all_tool_calls(expected.intermediate_data) if expected else []
    )
    match = len(actual_tools) == len(expected_tools)
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=1.0 if match else 0.0,
            eval_status=EvalStatus.PASSED if match else EvalStatus.FAILED,
        )
    )

  overall_score = (
      sum(r.score for r in per_invocation_results) / len(per_invocation_results)
      if per_invocation_results
      else 0.0
  )
  overall_eval_status = (
      EvalStatus.PASSED if overall_score == 1.0 else EvalStatus.FAILED
  )
  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=overall_eval_status,
      per_invocation_results=per_invocation_results,
  )
