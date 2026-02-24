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

"""Standalone BigQueryBench runner that produces leaderboard scores.

Usage:
    python -m benchmarks.bigquerybench.runner
    python -m benchmarks.bigquerybench.runner --num-runs 3
    python -m benchmarks.bigquerybench.runner --filter sql_shakespeare
    python -m benchmarks.bigquerybench.runner --dry-run

Environment variables:
    GOOGLE_CLOUD_PROJECT        — GCP project for BigQuery API calls
    GOOGLE_API_KEY              — API key for Google AI Studio
    GOOGLE_GENAI_USE_VERTEXAI   — Set to 1 for Vertex AI backend
    BQ_EVAL_WRITE_MODE          — Override write mode (blocked/protected)

This script:
1. Loads the BigQueryBench agent and eval set
2. Runs each case through the ADK Runner
3. Applies 3 custom metrics (schema_discovery, tool_usage, output)
4. Outputs per-case results and a leaderboard-format summary
5. Exits with code 0 if all pass, 1 if any fail
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

from .metrics import output_correctness_score
from .metrics import schema_discovery_score
from .metrics import tool_usage_score

logger = logging.getLogger(__name__)

_BENCH_DIR = pathlib.Path(__file__).parent
_DEFAULT_EVAL_SET = _BENCH_DIR / "eval_sets" / "bigquerybench_eval.json"


def load_eval_set(path: pathlib.Path) -> EvalSet:
  """Load an EvalSet from a JSON file."""
  with open(path) as f:
    data = json.load(f)
  return EvalSet.model_validate(data)


def print_header():
  print()
  print("=" * 65)
  print("  BigQueryBench Evaluation — ADK BigQueryToolset")
  print("=" * 65)
  print()


def print_results_table(results: dict[str, dict[str, float]]):
  """Print per-case results as a formatted table."""
  print(
      f"{'Eval Case':<40} {'Schema':>8} {'Tools':>7}"
      f" {'Output':>8} {'Result':>8}"
  )
  print("-" * 75)
  for case_id, scores in results.items():
    short_id = case_id[:39]
    schema = scores.get("schema_discovery", 0.0)
    tools = scores.get("tool_usage", 0.0)
    output = scores.get("output_correctness", 0.0)
    mark = "PASS" if output >= 1.0 else "FAIL"
    print(
        f"{short_id:<40} {schema:>7.1f} {tools:>7.2f}"
        f" {output:>7.1f}   {mark:>5}"
    )
  print("-" * 75)


def print_leaderboard_summary(
    results: dict[str, dict[str, float]],
    num_cases: int,
    elapsed: float,
):
  """Print a leaderboard-format summary."""
  passed = sum(
      1 for s in results.values() if s.get("output_correctness", 0.0) >= 1.0
  )
  avg_schema = sum(
      s.get("schema_discovery", 0.0) for s in results.values()
  ) / max(len(results), 1)
  avg_tools = sum(s.get("tool_usage", 0.0) for s in results.values()) / max(
      len(results), 1
  )
  pct = (passed / max(num_cases, 1)) * 100

  print()
  print("=" * 65)
  print("  Leaderboard Summary")
  print("=" * 65)
  print(f"  Framework:          ADK BigQueryToolset")
  print(f"  Cases:              {passed}/{num_cases} ({pct:.1f}%)")
  print(f"  Avg Schema Disc.:   {avg_schema:.2f}")
  print(f"  Avg Tool Usage:     {avg_tools:.2f}")
  print(f"  Elapsed:            {elapsed:.1f}s")
  print("=" * 65)


async def run_single_eval_case(
    root_agent,
    eval_case,
) -> list[Invocation]:
  """Run a single eval case through the Runner."""
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


def score_invocations(
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
) -> dict[str, float]:
  """Apply all 3 metrics and return scores."""
  metric = EvalMetric(metric_name="bigquerybench")
  scores = {}

  result = schema_discovery_score(
      metric,
      actual_invocations,
      expected_invocations,
  )
  scores["schema_discovery"] = result.overall_score or 0.0

  result = tool_usage_score(
      metric,
      actual_invocations,
      expected_invocations,
  )
  scores["tool_usage"] = result.overall_score or 0.0

  result = output_correctness_score(
      metric,
      actual_invocations,
      expected_invocations,
  )
  scores["output_correctness"] = result.overall_score or 0.0

  return scores


async def run_evaluation(
    eval_set_path: Optional[pathlib.Path] = None,
    num_runs: int = 1,
    filter_str: Optional[str] = None,
) -> dict[str, dict[str, float]]:
  """Run the BigQueryBench evaluation."""
  path = eval_set_path or _DEFAULT_EVAL_SET
  eval_set = load_eval_set(path)

  # Import agent (triggers BigQuery toolset setup).
  from .agent import root_agent

  cases = eval_set.eval_cases
  if filter_str:
    cases = [c for c in cases if filter_str in c.eval_id]
    if not cases:
      print(f"  No eval cases matched filter: {filter_str!r}")
      return {}

  results: dict[str, dict[str, float]] = {}
  total = len(cases)

  for idx, eval_case in enumerate(cases, 1):
    eval_id = eval_case.eval_id
    print(f"\n[{idx}/{total}] Running: {eval_id}")

    run_scores: list[dict[str, float]] = []
    for run in range(num_runs):
      if num_runs > 1:
        print(f"  Run {run + 1}/{num_runs}...")

      try:
        actual = await run_single_eval_case(root_agent, eval_case)

        # Print response preview.
        for inv in actual:
          if inv.final_response and inv.final_response.parts:
            for part in inv.final_response.parts:
              if part.text:
                preview = part.text[:200].replace("\n", " ")
                print(f"  Response: {preview}...")
                break

        expected = eval_case.conversation
        scores = score_invocations(actual, expected)
        run_scores.append(scores)

        schema = scores["schema_discovery"]
        tools = scores["tool_usage"]
        output = scores["output_correctness"]
        mark = "PASS" if output >= 1.0 else "FAIL"
        print(f"  Scores: schema={schema:.1f} tools={tools:.2f} output={mark}")

      except Exception as e:
        logger.error("Error running %s: %s", eval_id, e)
        print(f"  ERROR: {e}")
        run_scores.append({
            "schema_discovery": 0.0,
            "tool_usage": 0.0,
            "output_correctness": 0.0,
        })

    # Average scores across runs.
    avg: dict[str, float] = {}
    for key in ["schema_discovery", "tool_usage", "output_correctness"]:
      values = [s[key] for s in run_scores]
      avg[key] = sum(values) / len(values)
    results[eval_id] = avg

  return results


def main():
  parser = argparse.ArgumentParser(
      description="BigQueryBench evaluation runner for ADK",
  )
  parser.add_argument(
      "--eval-set",
      type=pathlib.Path,
      default=None,
      help="Path to eval set JSON (default: built-in)",
  )
  parser.add_argument(
      "--num-runs",
      type=int,
      default=1,
      help="Number of runs per case (default: 1)",
  )
  parser.add_argument(
      "--filter",
      type=str,
      default=None,
      help="Substring filter for eval_id (e.g., 'sql_shakespeare')",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Validate eval set JSON without running LLM inference",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  if args.dry_run:
    path = args.eval_set or _DEFAULT_EVAL_SET
    eval_set = load_eval_set(path)
    print(f"Eval set: {eval_set.name}")
    print(f"Cases:    {len(eval_set.eval_cases)}")
    for case in eval_set.eval_cases:
      print(f"  - {case.eval_id}")
    print("\nDry run: eval set JSON is valid.")
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
  eval_path = args.eval_set or _DEFAULT_EVAL_SET
  eval_set = load_eval_set(eval_path)

  num_cases = len(eval_set.eval_cases)
  if args.filter:
    num_cases = len(results)

  print()
  print_results_table(results)
  print_leaderboard_summary(results, num_cases, elapsed)

  # Exit code: 0 if all pass, 1 if any fail.
  all_pass = all(
      s.get("output_correctness", 0.0) >= 1.0 for s in results.values()
  )
  sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
  main()
