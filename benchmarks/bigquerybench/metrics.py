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

"""Metrics for BigQueryBench skill evaluation.

Three metrics:
1. tool_invocation_score — trace-based, checks tool names
2. tool_args_score — trace-based, checks key tool arguments
3. instruction_adherence_score — LLM-as-judge, checks rubrics

    def metric_fn(
        eval_metric: EvalMetric,
        actual_invocations: list[Invocation],
        expected_invocations: Optional[list[Invocation]],
        conversation_scenario: Optional[ConversationScenario] = None,
    ) -> EvaluationResult
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import google.genai

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult

logger = logging.getLogger(__name__)


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
  """Score = fraction of expected skill tool names present in the trace.

  Checks that the agent called the right skill functions (e.g.
  ``list_skills``, ``load_skill``, ``load_skill_resource``,
  ``run_skill_script``).  Order does not matter; extra tool calls
  are ignored.

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

# Args that identify the *target skill* — these are what we check.
# We only verify that the agent invoked the correct skill tools
# with the correct skill name, resource path, and script path.
_KEY_ARGS = frozenset({
    "name",  # load_skill(name=...)
    "skill_name",  # load_skill_resource / run_skill_script
    "path",  # load_skill_resource(path=...)
    "script_path",  # run_skill_script(script_path=...)
})


def tool_args_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Score = fraction of expected (tool, key-arg) pairs matched.

  For each expected skill tool call that has ``name``,
  ``skill_name``, ``path``, or ``script_path`` in its args, check
  that the agent made a call to the *same tool* with the *same
  value* for that arg.  This verifies the agent invoked the correct
  skill with the correct resources.

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


# ── Metric 3: instruction adherence (LLM-as-judge) ────────────────

_JUDGE_PROMPT = """\
You are evaluating an AI agent's behavior.

For EACH rubric listed below, respond in EXACTLY this format:

Property: <the rubric text>
Rationale: <one sentence explanation>
Verdict: yes or no

---

USER REQUEST:
{user_input}

TOOL CALLS:
{tool_trace}

AGENT FINAL RESPONSE:
{final_response}

RUBRICS TO EVALUATE:
{rubrics_text}
"""

_VERDICT_RE = re.compile(r"(?<=Verdict: )(.*)")


def _extract_text(content) -> str:
  """Extract text from a genai Content object."""
  if content and content.parts:
    return "\n".join(p.text for p in content.parts if p.text)
  return ""


def _format_tool_trace(invocations: list[Invocation]) -> str:
  """Format tool calls as readable text for the judge."""
  lines = []
  step = 1
  for inv in invocations:
    if not inv.intermediate_data:
      continue
    for tc in get_all_tool_calls(inv.intermediate_data):
      args_str = ", ".join(f"{k}={v!r}" for k, v in (tc.args or {}).items())
      lines.append(f"Step {step}: {tc.name}({args_str})")
      step += 1
  return "\n".join(lines) if lines else "No tool calls."


async def instruction_adherence_score(
    actual_invocations: list[Invocation],
    rubrics: Optional[list[Rubric]],
    judge_model: str = "gemini-2.5-flash",
) -> EvaluationResult:
  """LLM-as-judge: check each rubric against the agent's output.

  For each rubric, prompts a judge model:
    "Does the agent satisfy: <rubric>?"
  Judge answers yes (1.0) or no (0.0) per rubric.

  Score = fraction of rubrics with "yes" verdict.
  Pass threshold: 0.75.
  """
  if not rubrics:
    return EvaluationResult(
        overall_score=1.0,
        overall_eval_status=EvalStatus.PASSED,
    )

  # Extract conversation data from actual invocations.
  user_input = ""
  final_response = ""
  for inv in actual_invocations:
    text = _extract_text(inv.user_content)
    if text:
      user_input = text
    text = _extract_text(inv.final_response)
    if text:
      final_response = text

  tool_trace = _format_tool_trace(actual_invocations)

  rubrics_text = "\n".join(
      f"- {r.rubric_content.text_property}"
      for r in rubrics
      if r.rubric_content and r.rubric_content.text_property
  )

  prompt = _JUDGE_PROMPT.format(
      user_input=user_input or "(empty)",
      tool_trace=tool_trace,
      final_response=final_response or "(No response)",
      rubrics_text=rubrics_text,
  )

  try:
    client = google.genai.Client()
    response = await client.models.generate_content_async(
        model=judge_model,
        contents=prompt,
    )
    response_text = response.text or ""
  except Exception as e:
    logger.error("Judge LLM call failed: %s", e)
    return EvaluationResult(
        overall_score=0.0,
        overall_eval_status=EvalStatus.FAILED,
    )

  # Parse verdicts.
  verdicts = _VERDICT_RE.findall(response_text)
  yes_count = sum(1 for v in verdicts if v.strip().lower().startswith("yes"))
  total = len(rubrics)
  score = yes_count / total if total > 0 else 1.0

  status = EvalStatus.PASSED if score >= 0.75 else EvalStatus.FAILED

  return EvaluationResult(
      overall_score=score,
      overall_eval_status=status,
  )
