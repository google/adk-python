#!/usr/bin/env python3
"""
Extract and display all llm_query/llm_query_batched calls from an RLM run.

Usage:
    python scripts/show_llm_calls.py [LOG_FILE] [OPTIONS]

Examples:
    python scripts/show_llm_calls.py
    python scripts/show_llm_calls.py --depth 0
    python scripts/show_llm_calls.py --batched-only
    python scripts/show_llm_calls.py --stats
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


@dataclass
class LLMCall:
  iteration: int
  depth: int
  agent: str
  call_type: str  # "llm_query" or "llm_query_batched"
  prompt_preview: str
  has_context: bool
  recursive: bool | None
  model: str | None
  full_code: str


def load_log(log_path: Path) -> list[dict]:
  """Load JSONL log."""
  entries = []
  with open(log_path) as f:
    for line in f:
      line = line.strip()
      if line:
        try:
          entries.append(json.loads(line))
        except json.JSONDecodeError:
          pass
  return entries


def find_latest_log(log_dir: Path = Path("logs")) -> Path | None:
  """Find the most recent log file."""
  logs = sorted(log_dir.glob("rlm_*.jsonl"), key=lambda p: p.stat().st_mtime)
  return logs[-1] if logs else None


def extract_llm_calls(entries: list[dict]) -> list[LLMCall]:
  """Extract all llm_query calls from code blocks."""
  calls = []
  iterations = [e for e in entries if e.get("type") == "iteration"]

  for it in iterations:
    depth = it.get("depth", 0)
    iteration = it.get("iteration", 0)
    agent = it.get("agent_name", "")
    code_blocks = it.get("code_blocks", [])

    for block in code_blocks:
      code = block.get("code", "")

      # Find llm_query calls
      patterns = [
          (r"llm_query_batched\s*\(", "llm_query_batched"),
          (r"llm_query\s*\(", "llm_query"),
      ]

      for pattern, call_type in patterns:
        if re.search(pattern, code):
          # Extract prompt preview
          prompt_match = re.search(
              r'(?:prompt|prompts)\s*=\s*(?:f?["\'](.{20,80})|"""(.{20,80})|\[\s*f?["\'](.{20,80}))',
              code,
              re.DOTALL,
          )
          prompt_preview = ""
          if prompt_match:
            prompt_preview = next((g for g in prompt_match.groups() if g), "")[
                :60
            ]

          # Check for context parameter
          has_context = "context=" in code and "context=None" not in code

          # Check recursive parameter
          recursive = None
          if "recursive=True" in code:
            recursive = True
          elif "recursive=False" in code:
            recursive = False

          # Check model parameter
          model = None
          model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', code)
          if model_match:
            model = model_match.group(1)

          calls.append(
              LLMCall(
                  iteration=iteration,
                  depth=depth,
                  agent=agent,
                  call_type=call_type,
                  prompt_preview=prompt_preview.replace("\n", " ")[:60],
                  has_context=has_context,
                  recursive=recursive,
                  model=model,
                  full_code=code,
              )
          )

  return calls


def print_calls(
    calls: list[LLMCall],
    depth_filter: int | None = None,
    batched_only: bool = False,
    show_code: bool = False,
):
  """Print the extracted calls."""
  filtered = calls
  if depth_filter is not None:
    filtered = [c for c in filtered if c.depth == depth_filter]
  if batched_only:
    filtered = [c for c in filtered if c.call_type == "llm_query_batched"]

  print(f"\nFound {len(filtered)} llm_query calls:")
  print("=" * 80)

  for call in filtered:
    rec_str = ""
    if call.recursive is True:
      rec_str = " [RECURSIVE]"
    elif call.recursive is False:
      rec_str = " [simple]"

    ctx_str = " +ctx" if call.has_context else ""
    model_str = f" ({call.model})" if call.model else ""

    print(f"\n[Iter {call.iteration:3d}] depth={call.depth} {call.agent}")
    print(f"  {call.call_type}{rec_str}{ctx_str}{model_str}")
    if call.prompt_preview:
      print(f'  Prompt: "{call.prompt_preview}..."')

    if show_code:
      print(f"  Code:\n    " + call.full_code.replace("\n", "\n    "))


def print_stats(calls: list[LLMCall]):
  """Print statistics about llm_query usage."""
  print("\nLLM Call Statistics:")
  print("=" * 60)

  # By type
  by_type = defaultdict(int)
  for c in calls:
    by_type[c.call_type] += 1
  print("\nBy Call Type:")
  for t, count in sorted(by_type.items()):
    print(f"  {t}: {count}")

  # By depth
  by_depth = defaultdict(int)
  for c in calls:
    by_depth[c.depth] += 1
  print("\nBy Depth:")
  for d, count in sorted(by_depth.items()):
    print(f"  Depth {d}: {count}")

  # Recursive vs simple
  recursive_count = sum(1 for c in calls if c.recursive is True)
  simple_count = sum(1 for c in calls if c.recursive is False)
  unspecified = len(calls) - recursive_count - simple_count
  print("\nRecursive vs Simple:")
  print(f"  recursive=True:  {recursive_count}")
  print(f"  recursive=False: {simple_count}")
  print(f"  unspecified:     {unspecified}")

  # With context
  with_ctx = sum(1 for c in calls if c.has_context)
  print(f"\nWith context= parameter: {with_ctx}/{len(calls)}")

  # Model usage
  models = defaultdict(int)
  for c in calls:
    models[c.model or "(default)"] += 1
  print("\nModels Used:")
  for m, count in sorted(models.items(), key=lambda x: -x[1]):
    print(f"  {m}: {count}")

  # Recommendations
  print("\n" + "=" * 60)
  print("Observations:")

  if recursive_count > simple_count:
    print("  ⚠ More recursive calls than simple - may cause explosion")
    print("    Consider using recursive=False for extraction/summarization")

  if with_ctx < len(calls) * 0.5:
    print("  ⚠ Most calls don't use context= parameter")
    print("    Pass file objects via context= to properly delegate")

  batched = sum(1 for c in calls if c.call_type == "llm_query_batched")
  if batched == 0 and len(calls) > 10:
    print("  ℹ No batched calls found")
    print("    Consider llm_query_batched for parallel processing")


def main():
  parser = argparse.ArgumentParser(
      description="Show llm_query calls from RLM logs"
  )
  parser.add_argument("log_file", nargs="?", help="Path to log file")
  parser.add_argument("--depth", "-d", type=int, help="Filter by depth")
  parser.add_argument(
      "--batched-only",
      "-b",
      action="store_true",
      help="Show only batched calls",
  )
  parser.add_argument(
      "--stats", "-s", action="store_true", help="Show statistics only"
  )
  parser.add_argument(
      "--code", "-c", action="store_true", help="Show full code blocks"
  )
  args = parser.parse_args()

  if args.log_file:
    log_path = Path(args.log_file)
  else:
    log_path = find_latest_log()
    if not log_path:
      print("No log files found", file=sys.stderr)
      sys.exit(1)

  print(f"Analyzing: {log_path.name}")

  entries = load_log(log_path)
  calls = extract_llm_calls(entries)

  if args.stats:
    print_stats(calls)
  else:
    print_calls(calls, args.depth, args.batched_only, args.code)

    if len(calls) > 5:
      print("\n" + "-" * 40)
      print("Tip: Use --stats for usage statistics")


if __name__ == "__main__":
  main()
