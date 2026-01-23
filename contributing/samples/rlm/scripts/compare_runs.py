#!/usr/bin/env python3
"""
Compare multiple RLM runs side-by-side.

Usage:
    python scripts/compare_runs.py LOG1 LOG2 [LOG3 ...]
    python scripts/compare_runs.py --latest 5

Examples:
    python scripts/compare_runs.py logs/run1.jsonl logs/run2.jsonl
    python scripts/compare_runs.py --latest 3
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass
class RunStats:
  path: Path
  model: str
  total_iterations: int
  depth_distribution: dict[int, int]
  has_final: bool
  final_length: int
  total_time: float
  max_depth_used: int
  file_count: int | None


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


def find_latest_logs(log_dir: Path = Path("logs"), n: int = 5) -> list[Path]:
  """Find the n most recent log files."""
  logs = sorted(log_dir.glob("rlm_*.jsonl"), key=lambda p: p.stat().st_mtime)
  return logs[-n:] if logs else []


def analyze_run(log_path: Path) -> RunStats:
  """Analyze a single run."""
  entries = load_log(log_path)

  meta = next((e for e in entries if e.get("type") == "metadata"), {})
  iterations = [e for e in entries if e.get("type") == "iteration"]

  # Depth distribution
  depth_dist = defaultdict(int)
  for it in iterations:
    depth_dist[it.get("depth", 0)] += 1

  # Find final answer
  final = None
  for it in reversed(iterations):
    if it.get("final_answer"):
      final = it.get("final_answer")
      break

  # Total time
  times = [
      it.get("iteration_time", 0)
      for it in iterations
      if it.get("iteration_time")
  ]
  total_time = sum(times)

  # File count (from first iteration's context inspection if available)
  file_count = None
  for it in iterations:
    for block in it.get("code_blocks", []):
      output = block.get("output", "")
      if "file_count" in output or "files" in output:
        import re

        match = re.search(r"file_count['\"]?:\s*(\d+)", output)
        if match:
          file_count = int(match.group(1))
          break
        match = re.search(r"(\d+)\s*files?", output)
        if match:
          file_count = int(match.group(1))
          break
    if file_count:
      break

  return RunStats(
      path=log_path,
      model=meta.get("root_model", "unknown"),
      total_iterations=len(iterations),
      depth_distribution=dict(depth_dist),
      has_final=final is not None,
      final_length=len(final) if final else 0,
      total_time=total_time,
      max_depth_used=max(depth_dist.keys()) if depth_dist else 0,
      file_count=file_count,
  )


def print_comparison(runs: list[RunStats]):
  """Print comparison table."""
  print("\n" + "=" * 90)
  print("RLM Run Comparison")
  print("=" * 90)

  # Header
  col_width = max(20, max(len(r.path.stem[:20]) for r in runs) + 2)
  header = "Metric".ljust(25) + "".join(
      r.path.stem[:20].ljust(col_width) for r in runs
  )
  print(f"\n{header}")
  print("-" * len(header))

  # Model
  row = "Model".ljust(25)
  for r in runs:
    row += r.model[:18].ljust(col_width)
  print(row)

  # Total iterations
  row = "Total Iterations".ljust(25)
  for r in runs:
    row += str(r.total_iterations).ljust(col_width)
  print(row)

  # Max depth
  row = "Max Depth Used".ljust(25)
  for r in runs:
    row += str(r.max_depth_used).ljust(col_width)
  print(row)

  # Depth breakdown
  all_depths = set()
  for r in runs:
    all_depths.update(r.depth_distribution.keys())

  for depth in sorted(all_depths):
    row = f"  Depth {depth}".ljust(25)
    for r in runs:
      count = r.depth_distribution.get(depth, 0)
      row += str(count).ljust(col_width)
    print(row)

  # File count
  row = "Files Processed".ljust(25)
  for r in runs:
    val = str(r.file_count) if r.file_count else "?"
    row += val.ljust(col_width)
  print(row)

  # Total time
  row = "Total Time (s)".ljust(25)
  for r in runs:
    val = f"{r.total_time:.1f}" if r.total_time else "?"
    row += val.ljust(col_width)
  print(row)

  # Has final
  row = "Has Final Answer".ljust(25)
  for r in runs:
    val = "✓" if r.has_final else "✗"
    row += val.ljust(col_width)
  print(row)

  # Final length
  row = "Final Length (chars)".ljust(25)
  for r in runs:
    row += str(r.final_length).ljust(col_width)
  print(row)

  # Efficiency metrics
  print("\n" + "-" * len(header))
  print("Efficiency Metrics:")

  row = "Iters per File".ljust(25)
  for r in runs:
    if r.file_count and r.file_count > 0:
      val = f"{r.total_iterations / r.file_count:.1f}"
    else:
      val = "?"
    row += val.ljust(col_width)
  print(row)

  row = "Time per Iter (s)".ljust(25)
  for r in runs:
    if r.total_iterations > 0 and r.total_time > 0:
      val = f"{r.total_time / r.total_iterations:.1f}"
    else:
      val = "?"
    row += val.ljust(col_width)
  print(row)

  row = "Depth Ratio (d0/total)".ljust(25)
  for r in runs:
    d0 = r.depth_distribution.get(0, 0)
    if r.total_iterations > 0:
      val = f"{d0 / r.total_iterations:.2f}"
    else:
      val = "?"
    row += val.ljust(col_width)
  print(row)

  # Analysis
  print("\n" + "=" * 90)
  print("Analysis:")
  print("=" * 90)

  # Best/worst iterations
  if len(runs) > 1:
    sorted_by_iters = sorted(runs, key=lambda r: r.total_iterations)
    print(
        f"\nFewest iterations: {sorted_by_iters[0].path.stem}"
        f" ({sorted_by_iters[0].total_iterations})"
    )
    print(
        f"Most iterations:   {sorted_by_iters[-1].path.stem}"
        f" ({sorted_by_iters[-1].total_iterations})"
    )

    # Check for explosion
    for r in runs:
      if r.total_iterations > 100 and r.max_depth_used >= 3:
        print(
            f"\n⚠ {r.path.stem}: Possible iteration explosion (depth"
            f" {r.max_depth_used}, {r.total_iterations} iters)"
        )

    # Check for missing finals
    missing = [r for r in runs if not r.has_final]
    if missing:
      print(f"\n⚠ Runs without final answer: {[r.path.stem for r in missing]}")


def main():
  parser = argparse.ArgumentParser(description="Compare RLM runs")
  parser.add_argument("log_files", nargs="*", help="Log files to compare")
  parser.add_argument(
      "--latest", "-l", type=int, help="Compare N most recent runs"
  )
  args = parser.parse_args()

  if args.latest:
    log_paths = find_latest_logs(n=args.latest)
    if not log_paths:
      print("No log files found", file=sys.stderr)
      sys.exit(1)
  elif args.log_files:
    log_paths = [Path(f) for f in args.log_files]
  else:
    parser.print_help()
    sys.exit(1)

  # Analyze all runs
  runs = []
  for path in log_paths:
    if not path.exists():
      print(f"Warning: {path} not found, skipping", file=sys.stderr)
      continue
    try:
      runs.append(analyze_run(path))
    except Exception as e:
      print(f"Warning: Error analyzing {path}: {e}", file=sys.stderr)

  if not runs:
    print("No valid runs to compare", file=sys.stderr)
    sys.exit(1)

  print_comparison(runs)


if __name__ == "__main__":
  main()
