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

"""Custom metrics for SkillsBench evaluation.

These metrics follow the ADK custom metric function signature:

    def metric_fn(
        eval_metric: EvalMetric,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]],
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult

They can be referenced in eval configs via their dotted path, e.g.:
    "benchmarks.skillsbench.metrics.skill_discovery_score"
"""

from __future__ import annotations

from typing import Optional

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult


def _get_tool_names_from_invocations(
    invocations: list[Invocation],
) -> list[str]:
  """Extract all tool call names from a list of invocations."""
  names = []
  for inv in invocations:
    for tool_call in get_all_tool_calls(inv.intermediate_data):
      names.append(tool_call.name)
  return names


def skill_discovery_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score 1.0 if the agent called both list_skills and load_skill.

  This metric checks whether the agent properly discovered skills
  before attempting to use them, which is the expected workflow for
  SkillsBench tasks.
  """
  tool_names = _get_tool_names_from_invocations(actual_invocations)
  called_list = any(name == "list_skills" for name in tool_names)
  called_load = any(name == "load_skill" for name in tool_names)

  score = 1.0 if (called_list and called_load) else 0.0
  status = EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED

  per_invocation = []
  for i, actual in enumerate(actual_invocations):
    expected = None
    if expected_invocations and i < len(expected_invocations):
      expected = expected_invocations[i]
    per_invocation.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status,
        )
    )

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=per_invocation,
  )


def tool_usage_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Fraction of expected tool calls that were actually made.

  Compares the set of tool names in expected_invocations against
  actual_invocations. Score is |expected ∩ actual| / |expected|.
  Uses ANY_ORDER matching — only checks that expected tools were
  called, regardless of order or extra calls.
  """
  if not expected_invocations:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  expected_names = set(_get_tool_names_from_invocations(expected_invocations))
  actual_names = set(_get_tool_names_from_invocations(actual_invocations))

  if not expected_names:
    score = 1.0
  else:
    matched = expected_names & actual_names
    score = len(matched) / len(expected_names)

  status = EvalStatus.PASSED if score >= 0.5 else EvalStatus.FAILED

  per_invocation = []
  for i, actual in enumerate(actual_invocations):
    expected = None
    if expected_invocations and i < len(expected_invocations):
      expected = expected_invocations[i]
    per_invocation.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status,
        )
    )

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=per_invocation,
  )


def skillsbench_binary_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Binary pass/fail: 1.0 if final response contains expected text.

  Mirrors the SkillsBench binary scoring methodology. Checks whether
  key strings from the expected final response appear in the actual
  final response. The match is case-insensitive and checks for
  substring containment of each non-empty line in the reference.
  """
  if not expected_invocations or not actual_invocations:
    return EvaluationResult(
        overall_score=0.0,
        overall_eval_status=EvalStatus.NOT_EVALUATED,
    )

  # Get the last actual response text
  actual_text = ""
  for inv in reversed(actual_invocations):
    if inv.final_response and inv.final_response.parts:
      for part in inv.final_response.parts:
        if part.text:
          actual_text = part.text
          break
    if actual_text:
      break

  # Get the expected response text
  expected_text = ""
  for inv in reversed(expected_invocations):
    if inv.final_response and inv.final_response.parts:
      for part in inv.final_response.parts:
        if part.text:
          expected_text = part.text
          break
    if expected_text:
      break

  if not expected_text:
    return EvaluationResult(
        overall_score=0.0,
        overall_eval_status=EvalStatus.NOT_EVALUATED,
    )

  # Check that each non-empty reference line appears in the actual
  reference_lines = [
      line.strip() for line in expected_text.split("\n") if line.strip()
  ]
  actual_lower = actual_text.lower()
  matched = sum(1 for line in reference_lines if line.lower() in actual_lower)
  score = (
      1.0
      if matched == len(reference_lines)
      else matched / max(len(reference_lines), 1)
  )

  # Binary: pass only if all reference lines matched
  is_pass = matched == len(reference_lines) and len(reference_lines) > 0
  status = EvalStatus.PASSED if is_pass else EvalStatus.FAILED

  per_invocation = []
  for i, actual in enumerate(actual_invocations):
    expected = None
    if expected_invocations and i < len(expected_invocations):
      expected = expected_invocations[i]
    per_invocation.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=1.0 if is_pass else 0.0,
            eval_status=status,
        )
    )

  return EvaluationResult(
      overall_score=1.0 if is_pass else 0.0,
      overall_eval_status=status,
      per_invocation_results=per_invocation,
  )
