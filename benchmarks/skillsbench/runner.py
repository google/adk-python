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

"""Standalone SkillsBench runner that produces leaderboard-compatible scores.

Usage:
    python benchmarks/skillsbench/runner.py
    python benchmarks/skillsbench/runner.py --num-runs 3
    python benchmarks/skillsbench/runner.py --eval-set path/to/eval.json

Environment variables:
    GOOGLE_API_KEY              — API key for authentication
    GOOGLE_GENAI_USE_VERTEXAI   — Set to 1 for Vertex AI backend

This script:
1. Loads the SkillsBench agent and eval set
2. Runs each task through the ADK Runner directly
3. Applies all 3 custom metrics (discovery, tool usage, binary)
4. Outputs per-task results and a leaderboard-format summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time
import uuid
from typing import Optional

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types

from .metrics import (
    skill_discovery_score,
    skillsbench_binary_score,
    tool_usage_score,
)

logger = logging.getLogger(__name__)

_BENCHMARKS_DIR = pathlib.Path(__file__).parent
_DEFAULT_EVAL_SET = _BENCHMARKS_DIR / "eval_sets" / "skillsbench_eval.json"
_MODEL_NAME = "gemini-3-flash-preview"


def load_eval_set(path: pathlib.Path) -> EvalSet:
  """Load an EvalSet from a JSON file."""
  with open(path) as f:
    data = json.load(f)
  return EvalSet.model_validate(data)


def print_header():
  """Print the SkillsBench header."""
  print()
  print("=" * 60)
  print("  SkillsBench Evaluation — ADK SkillToolset")
  print("=" * 60)
  print()


def print_results_table(results: dict[str, dict[str, float]]):
  """Print per-task results as a formatted table."""
  print(f"{'Task':<45} {'Discovery':>9} {'Tools':>7} {'Pass':>6}")
  print("-" * 70)
  for task_id, scores in results.items():
    short_id = task_id[:44]
    disc = scores.get("skill_discovery", 0.0)
    tools = scores.get("tool_usage", 0.0)
    binary = scores.get("binary_pass", 0.0)
    mark = "PASS" if binary >= 1.0 else "FAIL"
    print(f"{short_id:<45} {disc:>8.1f} {tools:>7.2f} {mark:>6}")
  print("-" * 70)


def print_leaderboard_summary(
    results: dict[str, dict[str, float]],
    num_tasks: int,
    elapsed: float,
):
  """Print a leaderboard-format summary."""
  passed = sum(
      1 for scores in results.values() if scores.get("binary_pass", 0.0) >= 1.0
  )
  avg_discovery = sum(
      s.get("skill_discovery", 0.0) for s in results.values()
  ) / max(len(results), 1)
  avg_tools = sum(s.get("tool_usage", 0.0) for s in results.values()) / max(
      len(results), 1
  )
  pct = (passed / max(num_tasks, 1)) * 100

  print()
  print("=" * 60)
  print("  Leaderboard Summary")
  print("=" * 60)
  print(f"  Model:              {_MODEL_NAME}")
  print(f"  Framework:          ADK SkillToolset")
  print(f"  Tasks:              {passed}/{num_tasks} ({pct:.1f}%)")
  print(f"  Avg Discovery:      {avg_discovery:.2f}")
  print(f"  Avg Tool Usage:     {avg_tools:.2f}")
  print(f"  Elapsed:            {elapsed:.1f}s")
  print("=" * 60)


async def run_single_eval_case(
    root_agent,
    eval_case,
) -> list[Invocation]:
  """Run a single eval case through the Runner and return invocations."""
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()

  app_name = "skillsbench_eval"
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
  metric = EvalMetric(metric_name="skillsbench")
  scores = {}

  result = skill_discovery_score(
      metric, actual_invocations, expected_invocations
  )
  scores["skill_discovery"] = result.overall_score or 0.0

  result = tool_usage_score(metric, actual_invocations, expected_invocations)
  scores["tool_usage"] = result.overall_score or 0.0

  result = skillsbench_binary_score(
      metric, actual_invocations, expected_invocations
  )
  scores["binary_pass"] = result.overall_score or 0.0

  return scores


async def run_evaluation(
    eval_set_path: Optional[pathlib.Path] = None,
    num_runs: int = 1,
) -> dict[str, dict[str, float]]:
  """Run the full SkillsBench evaluation."""
  path = eval_set_path or _DEFAULT_EVAL_SET
  eval_set = load_eval_set(path)

  # Import agent (triggers skill loading)
  from .agent import root_agent

  results: dict[str, dict[str, float]] = {}
  total = len(eval_set.eval_cases)

  for idx, eval_case in enumerate(eval_set.eval_cases, 1):
    eval_id = eval_case.eval_id
    print(f"\n[{idx}/{total}] Running: {eval_id}")

    run_scores: list[dict[str, float]] = []
    for run in range(num_runs):
      if num_runs > 1:
        print(f"  Run {run + 1}/{num_runs}...")

      try:
        actual_invocations = await run_single_eval_case(root_agent, eval_case)

        # Print what the agent did
        for inv in actual_invocations:
          if inv.final_response and inv.final_response.parts:
            for part in inv.final_response.parts:
              if part.text:
                preview = part.text[:200].replace("\n", " ")
                print(f"  Response: {preview}...")
                break

        expected_invocations = eval_case.conversation
        scores = score_invocations(actual_invocations, expected_invocations)
        run_scores.append(scores)

        disc = scores["skill_discovery"]
        tools = scores["tool_usage"]
        binary = scores["binary_pass"]
        mark = "PASS" if binary >= 1.0 else "FAIL"
        print(f"  Scores: discovery={disc:.1f} tools={tools:.2f} binary={mark}")

      except Exception as e:
        logger.error("Error running %s: %s", eval_id, e)
        print(f"  ERROR: {e}")
        run_scores.append({
            "skill_discovery": 0.0,
            "tool_usage": 0.0,
            "binary_pass": 0.0,
        })

    # Average scores across runs
    avg_scores: dict[str, float] = {}
    for key in ["skill_discovery", "tool_usage", "binary_pass"]:
      values = [s[key] for s in run_scores]
      avg_scores[key] = sum(values) / len(values)
    results[eval_id] = avg_scores

  return results


def main():
  parser = argparse.ArgumentParser(
      description="SkillsBench evaluation runner for ADK"
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
      help="Number of runs per task (default: 1)",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  print_header()
  start = time.time()

  results = asyncio.run(
      run_evaluation(
          eval_set_path=args.eval_set,
          num_runs=args.num_runs,
      )
  )

  elapsed = time.time() - start
  eval_path = args.eval_set or _DEFAULT_EVAL_SET
  eval_set = load_eval_set(eval_path)

  print()
  print_results_table(results)
  print_leaderboard_summary(results, len(eval_set.eval_cases), elapsed)


if __name__ == "__main__":
  main()
