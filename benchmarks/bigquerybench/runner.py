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

"""BigQueryBench evaluation runner.

Evaluates skill invocation correctness (trace-based) and instruction
adherence (LLM-as-judge with rubrics).

Usage:
    python -m benchmarks.bigquerybench.runner
    python -m benchmarks.bigquerybench.runner --filter skill_load
    python -m benchmarks.bigquerybench.runner --num-runs 3
    python -m benchmarks.bigquerybench.runner --dry-run

Environment variables:
    GOOGLE_CLOUD_PROJECT        — GCP project for BigQuery API calls
    GOOGLE_API_KEY              — API key for Google AI Studio
    GOOGLE_GENAI_USE_VERTEXAI   — Set to 1 for Vertex AI backend
    BQ_EVAL_WRITE_MODE          — blocked / protected / allowed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time
from typing import Optional
import uuid

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.utils.context_utils import Aclosing

from .metrics import instruction_adherence_score
from .metrics import tool_args_score
from .metrics import tool_invocation_score

logger = logging.getLogger(__name__)

_BENCH_DIR = pathlib.Path(__file__).parent
_DEFAULT_EVAL_SET = _BENCH_DIR / "eval_sets" / "bigquerybench_eval.json"


def load_eval_set(path: pathlib.Path) -> EvalSet:
  with open(path) as f:
    data = json.load(f)
  return EvalSet.model_validate(data)


def print_header():
  print()
  print("=" * 72)
  print("  BigQueryBench — Skill Evaluation")
  print("=" * 72)
  print()


def _case_passed(scores: dict[str, float]) -> bool:
  trace_ok = (
      scores.get("tool_invocation", 0.0) >= 1.0
      and scores.get("tool_args", 0.0) >= 1.0
  )
  adherence_ok = scores.get("adherence", 1.0) >= 0.75
  return trace_ok and adherence_ok


def print_results_table(results: dict[str, dict[str, float]]):
  print(
      f"{'Eval Case':<34} {'Tools':>6} {'Args':>6} {'Adhere':>7} {'Result':>7}"
  )
  print("-" * 72)
  for case_id, scores in results.items():
    short_id = case_id[:33]
    tools = scores.get("tool_invocation", 0.0)
    args = scores.get("tool_args", 0.0)
    adhere = scores.get("adherence", 1.0)
    mark = "PASS" if _case_passed(scores) else "FAIL"
    print(
        f"{short_id:<34} {tools:>6.2f} {args:>6.2f} {adhere:>7.2f}   {mark:>4}"
    )
  print("-" * 72)


def print_summary(
    results: dict[str, dict[str, float]],
    num_cases: int,
    elapsed: float,
):
  n = max(len(results), 1)
  passed = sum(1 for s in results.values() if _case_passed(s))
  avg_tools = sum(s.get("tool_invocation", 0.0) for s in results.values()) / n
  avg_args = sum(s.get("tool_args", 0.0) for s in results.values()) / n
  avg_adhere = sum(s.get("adherence", 1.0) for s in results.values()) / n
  pct = (passed / max(num_cases, 1)) * 100

  print()
  print("=" * 72)
  print("  Summary")
  print("=" * 72)
  print(f"  Cases:              {passed}/{num_cases} ({pct:.1f}%)")
  print(f"  Avg Tool Match:     {avg_tools:.2f}")
  print(f"  Avg Args Match:     {avg_args:.2f}")
  print(f"  Avg Adherence:      {avg_adhere:.2f}")
  print(f"  Elapsed:            {elapsed:.1f}s")
  print("=" * 72)


async def run_single_eval_case(root_agent, eval_case) -> list[Invocation]:
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()

  app_name = "bigquerybench_eval"
  user_id = "eval_user"
  session_id = str(uuid.uuid4())

  await session_service.create_session(
      app_name=app_name,
      user_id=user_id,
      state={},
      session_id=session_id,
  )

  async with Runner(
      app_name=app_name,
      agent=root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
  ) as runner:
    events = []
    for invocation in eval_case.conversation:
      user_content = invocation.user_content

      async with Aclosing(
          runner.run_async(
              user_id=user_id,
              session_id=session_id,
              new_message=user_content,
          )
      ) as agen:
        invocation_id = None
        async for event in agen:
          if not invocation_id:
            invocation_id = event.invocation_id
            from google.adk.events.event import Event

            events.append(
                Event(
                    content=user_content,
                    author="user",
                    invocation_id=invocation_id,
                )
            )
          events.append(event)

    return EvaluationGenerator.convert_events_to_eval_invocations(events)


async def score_invocations(
    actual: list[Invocation],
    expected: Optional[list[Invocation]],
    rubrics=None,
) -> dict[str, float]:
  metric = EvalMetric(metric_name="bigquerybench")

  r1 = tool_invocation_score(metric, actual, expected)
  r2 = tool_args_score(metric, actual, expected)

  scores = {
      "tool_invocation": r1.overall_score or 0.0,
      "tool_args": r2.overall_score or 0.0,
  }

  if rubrics:
    r3 = await instruction_adherence_score(actual, rubrics)
    scores["adherence"] = r3.overall_score or 0.0

  return scores


_TRACE_ARGS = frozenset({
    "name",
    "skill_name",
    "path",
    "script_path",
})


def _print_trace(actual_invocations: list[Invocation]):
  """Print the tool-call trace for debugging."""
  from google.adk.evaluation.eval_case import get_all_tool_calls

  for inv in actual_invocations:
    for tc in get_all_tool_calls(inv.intermediate_data):
      args_summary = ", ".join(
          f"{k}={v!r}" for k, v in (tc.args or {}).items() if k in _TRACE_ARGS
      )
      print(f"    -> {tc.name}({args_summary})")


async def run_evaluation(
    eval_set_path: Optional[pathlib.Path] = None,
    num_runs: int = 1,
    filter_str: Optional[str] = None,
) -> dict[str, dict[str, float]]:
  path = eval_set_path or _DEFAULT_EVAL_SET
  eval_set = load_eval_set(path)

  from .agent import root_agent

  cases = eval_set.eval_cases
  if filter_str:
    cases = [c for c in cases if filter_str in c.eval_id]
    if not cases:
      print(f"  No eval cases matched filter: {filter_str!r}")
      return {}

  results: dict[str, dict[str, float]] = {}

  for idx, eval_case in enumerate(cases, 1):
    eval_id = eval_case.eval_id
    print(f"\n[{idx}/{len(cases)}] {eval_id}")

    run_scores: list[dict[str, float]] = []
    for run in range(num_runs):
      if num_runs > 1:
        print(f"  Run {run + 1}/{num_runs}...")

      try:
        actual = await run_single_eval_case(root_agent, eval_case)
        _print_trace(actual)

        scores = await score_invocations(
            actual, eval_case.conversation, eval_case.rubrics
        )
        run_scores.append(scores)

        tools = scores["tool_invocation"]
        args = scores["tool_args"]
        adhere = scores.get("adherence", 1.0)
        mark = "PASS" if _case_passed(scores) else "FAIL"
        parts = [f"tools={tools:.2f}", f"args={args:.2f}"]
        if "adherence" in scores:
          parts.append(f"adherence={adhere:.2f}")
        print(f"  {' '.join(parts)}  {mark}")

      except Exception as e:
        logger.error("Error running %s: %s", eval_id, e)
        print(f"  ERROR: {e}")
        run_scores.append({"tool_invocation": 0.0, "tool_args": 0.0})

    avg: dict[str, float] = {}
    for key in ("tool_invocation", "tool_args", "adherence"):
      vals = [s[key] for s in run_scores if key in s]
      if vals:
        avg[key] = sum(vals) / len(vals)
    results[eval_id] = avg

  return results


def main():
  parser = argparse.ArgumentParser(
      description="BigQueryBench trace-based evaluation runner",
  )
  parser.add_argument(
      "--eval-set",
      type=pathlib.Path,
      default=None,
  )
  parser.add_argument(
      "--num-runs",
      type=int,
      default=1,
  )
  parser.add_argument(
      "--filter",
      type=str,
      default=None,
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  if args.dry_run:
    path = args.eval_set or _DEFAULT_EVAL_SET
    es = load_eval_set(path)
    print(f"Eval set: {es.name}")
    print(f"Cases:    {len(es.eval_cases)}")
    for case in es.eval_cases:
      tools = [
          t.name for t in case.conversation[0].intermediate_data.tool_uses or []
      ]
      n_rubrics = len(case.rubrics) if case.rubrics else 0
      rubric_info = f" ({n_rubrics} rubrics)" if n_rubrics else ""
      print(f"  {case.eval_id}: {' -> '.join(tools)}{rubric_info}")
    print("\nJSON valid.")
    sys.exit(0)

  print_header()
  start = time.time()

  results = asyncio.run(
      run_evaluation(
          eval_set_path=args.eval_set,
          num_runs=args.num_runs,
          filter_str=args.filter,
      )
  )

  elapsed = time.time() - start
  num_cases = len(results)

  print()
  print_results_table(results)
  print_summary(results, num_cases, elapsed)

  all_pass = all(_case_passed(s) for s in results.values())
  sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
  main()
