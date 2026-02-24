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

"""Trace-based metrics for BigQueryBench evaluation.

Two metrics that check the agent's tool-call trace only (no response
text matching).  This makes evaluation deterministic and easy to
maintain: just specify which tools should be called and with which
key arguments.

    def metric_fn(
        eval_metric: EvalMetric,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]],
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult

Reference via dotted path in eval configs:
    "benchmarks.bigquerybench.metrics.tool_invocation_score"
    "benchmarks.bigquerybench.metrics.tool_args_score"
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


def _get_tool_calls(invocations: list[Invocation]):
  """Yield (name, args_dict) for every tool call in the trace."""
  for inv in invocations:
    for tc in get_all_tool_calls(inv.intermediate_data):
      yield tc.name, (tc.args or {})


def _make_per_invocation(
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    score: float,
    status: EvalStatus,
) -> list[PerInvocationResult]:
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


# ── Metric 1: correct tools invoked ──────────────────────────────


def tool_invocation_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score = fraction of expected tool names present in the trace.

  Checks that the agent called the right BigQuery functions (e.g.
  ``get_table_info``, ``execute_sql``, ``forecast``).  Order does
  not matter; extra tool calls are ignored.

  Score = |expected_names ∩ actual_names| / |expected_names|.
  Pass threshold: 1.0 (all expected tools must be called).
  """
  if not expected_invocations:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  expected_names = {name for name, _ in _get_tool_calls(expected_invocations)}
  actual_names = {name for name, _ in _get_tool_calls(actual_invocations)}

  if not expected_names:
    score = 1.0
  else:
    matched = expected_names & actual_names
    score = len(matched) / len(expected_names)

  status = EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED

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


# ── Metric 2: correct args on key tool calls ─────────────────────

# Args that identify the *target data* — these are what we check.
# We intentionally skip volatile args like ``query`` (the exact SQL
# the LLM generates will vary) and only verify that the agent
# pointed at the right dataset / table / project.
_KEY_ARGS = frozenset({
    "project_id",
    "dataset_id",
    "table_id",
})


def tool_args_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score = fraction of expected (tool, key-arg) pairs matched.

  For each expected tool call that has ``project_id``, ``dataset_id``,
  or ``table_id`` in its args, check that the agent made a call to
  the *same tool* with the *same value* for that arg.  This verifies
  the agent loaded the right reference data (correct dataset, correct
  table) without caring about the exact SQL or response text.

  Score = matched_pairs / expected_pairs.  Pass threshold: 1.0.
  If no key args exist in the expected trace, score is 1.0 (vacuous).
  """
  if not expected_invocations:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  # Build expected set: (tool_name, arg_key, arg_value).
  expected_pairs: set[tuple[str, str, str]] = set()
  for name, args in _get_tool_calls(expected_invocations):
    for key in _KEY_ARGS:
      if key in args:
        expected_pairs.add((name, key, str(args[key])))

  if not expected_pairs:
    # No key args to check — pass vacuously.
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=_make_per_invocation(
            actual_invocations,
            expected_invocations,
            1.0,
            EvalStatus.PASSED,
        ),
    )

  # Build actual set the same way.
  actual_pairs: set[tuple[str, str, str]] = set()
  for name, args in _get_tool_calls(actual_invocations):
    for key in _KEY_ARGS:
      if key in args:
        actual_pairs.add((name, key, str(args[key])))

  matched = expected_pairs & actual_pairs
  score = len(matched) / len(expected_pairs)
  status = EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED

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
