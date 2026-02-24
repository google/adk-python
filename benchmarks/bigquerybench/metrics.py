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

"""Custom metrics for BigQueryBench evaluation.

Three metrics following the ADK custom metric function signature:

    def metric_fn(
        eval_metric: EvalMetric,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]],
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult

Reference via dotted path in eval configs:
    "benchmarks.bigquerybench.metrics.schema_discovery_score"
    "benchmarks.bigquerybench.metrics.tool_usage_score"
    "benchmarks.bigquerybench.metrics.output_correctness_score"
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

# Tools that count as "schema exploration".
_SCHEMA_TOOLS = frozenset({
    "list_dataset_ids",
    "list_table_ids",
    "get_dataset_info",
    "get_table_info",
})


def _get_tool_names(invocations: list[Invocation]) -> list[str]:
  """Extract all tool call names from a list of invocations."""
  names = []
  for inv in invocations:
    for tool_call in get_all_tool_calls(inv.intermediate_data):
      names.append(tool_call.name)
  return names


def _make_per_invocation(
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    score: float,
    status: EvalStatus,
) -> list[PerInvocationResult]:
  """Build per-invocation results list."""
  results = []
  for i, actual in enumerate(actual_invocations):
    expected = None
    if expected_invocations and i < len(expected_invocations):
      expected = expected_invocations[i]
    results.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status,
        )
    )
  return results


def schema_discovery_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score 1.0 if the agent called at least one schema exploration tool.

  Schema tools: list_dataset_ids, list_table_ids, get_dataset_info,
  get_table_info.

  This metric enforces the "explore before query" pattern — agents
  should understand the schema before generating SQL or calling AI/ML
  tools.
  """
  tool_names = set(_get_tool_names(actual_invocations))
  called_schema = bool(tool_names & _SCHEMA_TOOLS)

  score = 1.0 if called_schema else 0.0
  status = EvalStatus.PASSED if called_schema else EvalStatus.FAILED

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=_make_per_invocation(
          actual_invocations,
          expected_invocations,
          score,
          status,
      ),
  )


def tool_usage_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Fraction of expected tool calls that were actually made.

  Score = |expected_tools ∩ actual_tools| / |expected_tools|.
  Uses set-based matching (any order). Passes at >= 0.5.

  Same semantics as SkillsBench tool_usage_score for consistency.
  """
  if not expected_invocations:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  expected_names = set(_get_tool_names(expected_invocations))
  actual_names = set(_get_tool_names(actual_invocations))

  if not expected_names:
    score = 1.0
  else:
    matched = expected_names & actual_names
    score = len(matched) / len(expected_names)

  status = EvalStatus.PASSED if score >= 0.5 else EvalStatus.FAILED

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=_make_per_invocation(
          actual_invocations,
          expected_invocations,
          score,
          status,
      ),
  )


def output_correctness_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Binary pass/fail: 1.0 if response contains all reference lines.

  Each non-empty line in the expected final_response is a reference
  phrase. The actual response must contain every phrase as a
  case-insensitive substring.

  Same semantics as SkillsBench skillsbench_binary_score for
  consistency.
  """
  if not expected_invocations or not actual_invocations:
    return EvaluationResult(
        overall_score=0.0,
        overall_eval_status=EvalStatus.NOT_EVALUATED,
    )

  # Get the last actual response text.
  actual_text = ""
  for inv in reversed(actual_invocations):
    if inv.final_response and inv.final_response.parts:
      for part in inv.final_response.parts:
        if part.text:
          actual_text = part.text
          break
    if actual_text:
      break

  # Get the expected response text.
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

  reference_lines = [
      line.strip() for line in expected_text.split("\n") if line.strip()
  ]
  actual_lower = actual_text.lower()
  matched = sum(1 for line in reference_lines if line.lower() in actual_lower)

  is_pass = matched == len(reference_lines) and len(reference_lines) > 0
  score = 1.0 if is_pass else 0.0
  status = EvalStatus.PASSED if is_pass else EvalStatus.FAILED

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
      per_invocation_results=_make_per_invocation(
          actual_invocations,
          expected_invocations,
          score,
          status,
      ),
  )
