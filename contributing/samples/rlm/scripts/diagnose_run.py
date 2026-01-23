#!/usr/bin/env python3
"""
Diagnose issues in RLM runs - detect context problems, iteration explosions, etc.

Usage:
    python scripts/diagnose_run.py [LOG_FILE]

Examples:
    python scripts/diagnose_run.py
    python scripts/diagnose_run.py logs/rlm_2026-01-22_*.jsonl
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


@dataclass
class Issue:
  severity: str  # "error", "warning", "info"
  category: str
  message: str
  iteration: int | None = None
  depth: int | None = None
  agent: str | None = None


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


def check_context_issues(entries: list[dict]) -> list[Issue]:
  """Check for context propagation problems."""
  issues = []
  iterations = [e for e in entries if e.get("type") == "iteration"]

  for it in iterations:
    depth = it.get("depth", 0)
    iteration = it.get("iteration", 0)
    agent = it.get("agent_name", "")
    code_blocks = it.get("code_blocks", [])

    for block in code_blocks:
      code = block.get("code", "")
      output = block.get("output", "")

      # Check for small context warnings in output
      if output:
        # Pattern: context is very small but agent expected documents
        small_ctx_patterns = [
            r"Context length: (\d+)",
            r"total length \((\d+) characters\)",
            r"(\d+) chars",
        ]
        for pattern in small_ctx_patterns:
          match = re.search(pattern, output)
          if match:
            size = int(match.group(1))
            if size < 1000 and depth > 0:
              issues.append(
                  Issue(
                      severity="warning",
                      category="context_size",
                      message=(
                          f"Small context ({size} chars) at depth {depth} - may"
                          " have received filenames instead of content"
                      ),
                      iteration=iteration,
                      depth=depth,
                      agent=agent,
                  )
              )

      # Check for filename list patterns in context inspection
      if "context[:5]" in code or "context[:3]" in code:
        if output and ".md" in output and "Tegus" not in output:
          # Looks like filenames, not content
          issues.append(
              Issue(
                  severity="error",
                  category="context_type",
                  message=(
                      "Context appears to be filenames (strings) instead of"
                      " file objects"
                  ),
                  iteration=iteration,
                  depth=depth,
                  agent=agent,
              )
          )

      # Check for file not found errors
      if "No such file or directory" in str(
          output
      ) or "FileNotFoundError" in str(output):
        issues.append(
            Issue(
                severity="error",
                category="file_error",
                message="File not found error during execution",
                iteration=iteration,
                depth=depth,
                agent=agent,
            )
        )

  return issues


def check_iteration_explosion(entries: list[dict]) -> list[Issue]:
  """Check for iteration explosion patterns."""
  issues = []
  iterations = [e for e in entries if e.get("type") == "iteration"]
  meta = next((e for e in entries if e.get("type") == "metadata"), {})

  total = len(iterations)
  max_iterations = meta.get("max_iterations", 30)

  # Count by depth
  depth_counts = defaultdict(int)
  for it in iterations:
    depth_counts[it.get("depth", 0)] += 1

  # Check for explosion
  if total > max_iterations * 3:
    issues.append(
        Issue(
            severity="error",
            category="explosion",
            message=(
                f"Iteration explosion: {total} iterations (expected"
                f" ~{max_iterations})"
            ),
        )
    )

  # Check for deep recursion explosion
  for depth, count in depth_counts.items():
    if depth >= 2 and count > 50:
      issues.append(
          Issue(
              severity="warning",
              category="deep_recursion",
              message=(
                  f"High iteration count at depth {depth}: {count} iterations"
              ),
          )
      )

  # Check for ratio imbalance
  if depth_counts.get(0, 0) < 5 and sum(depth_counts.values()) > 100:
    issues.append(
        Issue(
            severity="warning",
            category="ratio_imbalance",
            message=(
                f"Root agent only had {depth_counts.get(0, 0)} iterations but"
                f" spawned {sum(depth_counts.values())} total - aggregation may"
                " be missing"
            ),
        )
    )

  return issues


def check_redundant_work(entries: list[dict]) -> list[Issue]:
  """Check for redundant/repeated work."""
  issues = []
  iterations = [e for e in entries if e.get("type") == "iteration"]

  # Track prompts by similarity
  prompt_hashes = defaultdict(list)

  for it in iterations:
    code_blocks = it.get("code_blocks", [])
    for block in code_blocks:
      code = block.get("code", "")
      # Look for llm_query calls
      if "llm_query" in code:
        # Extract prompt pattern (simplified)
        prompt_match = re.search(r'prompt\s*=\s*["\'](.{50,100})', code)
        if prompt_match:
          prompt_key = prompt_match.group(1)[:50]
          prompt_hashes[prompt_key].append(it.get("iteration", 0))

  # Find duplicates
  for prompt_key, iters in prompt_hashes.items():
    if len(iters) > 3:
      issues.append(
          Issue(
              severity="warning",
              category="redundant_work",
              message=(
                  f"Similar prompt pattern used {len(iters)} times:"
                  f" '{prompt_key[:40]}...'"
              ),
          )
      )

  return issues


def check_final_answer(entries: list[dict]) -> list[Issue]:
  """Check final answer quality."""
  issues = []
  iterations = [e for e in entries if e.get("type") == "iteration"]

  # Find final answer
  final = None
  final_depth = None
  for it in reversed(iterations):
    if it.get("final_answer"):
      final = it.get("final_answer")
      final_depth = it.get("depth", 0)
      break

  if not final:
    issues.append(
        Issue(
            severity="error",
            category="no_answer",
            message="No final answer found - run may have failed or timed out",
        )
    )
  else:
    # Check for error patterns in final answer
    error_patterns = [
        "No such file or directory",
        "was not included",
        "please paste",
        "I cannot",
        "error occurred",
    ]
    for pattern in error_patterns:
      if pattern.lower() in final.lower():
        issues.append(
            Issue(
                severity="error",
                category="answer_error",
                message=f"Final answer contains error pattern: '{pattern}'",
            )
        )

    # Check if final came from deep recursion
    if final_depth and final_depth >= 3:
      issues.append(
          Issue(
              severity="warning",
              category="deep_final",
              message=(
                  f"Final answer came from depth {final_depth} - may be"
                  " incomplete synthesis"
              ),
          )
      )

    # Check answer length
    if len(final) < 100:
      issues.append(
          Issue(
              severity="warning",
              category="short_answer",
              message=f"Final answer is very short ({len(final)} chars)",
          )
      )

  return issues


def print_diagnosis(issues: list[Issue], entries: list[dict], log_path: Path):
  """Print diagnosis report."""
  meta = next((e for e in entries if e.get("type") == "metadata"), {})
  iterations = [e for e in entries if e.get("type") == "iteration"]

  print("=" * 70)
  print(f"RLM Run Diagnosis: {log_path.name}")
  print("=" * 70)

  # Quick stats
  print(f"\nModel: {meta.get('root_model', 'unknown')}")
  print(f"Total Iterations: {len(iterations)}")

  depth_counts = defaultdict(int)
  for it in iterations:
    depth_counts[it.get("depth", 0)] += 1
  print(f"Depth Distribution: {dict(sorted(depth_counts.items()))}")

  # Issues
  errors = [i for i in issues if i.severity == "error"]
  warnings = [i for i in issues if i.severity == "warning"]
  infos = [i for i in issues if i.severity == "info"]

  print(f"\n{'='*70}")
  print(
      f"Issues Found: {len(errors)} errors, {len(warnings)} warnings,"
      f" {len(infos)} info"
  )
  print("=" * 70)

  if errors:
    print("\n[ERRORS]")
    for issue in errors:
      loc = ""
      if issue.iteration:
        loc = f" (iter {issue.iteration}, depth {issue.depth})"
      print(f"  ✗ [{issue.category}]{loc}: {issue.message}")

  if warnings:
    print("\n[WARNINGS]")
    for issue in warnings:
      loc = ""
      if issue.iteration:
        loc = f" (iter {issue.iteration}, depth {issue.depth})"
      print(f"  ⚠ [{issue.category}]{loc}: {issue.message}")

  if infos:
    print("\n[INFO]")
    for issue in infos:
      print(f"  ℹ [{issue.category}]: {issue.message}")

  if not issues:
    print("\n✓ No issues detected - run looks healthy!")

  # Recommendations
  if issues:
    print(f"\n{'='*70}")
    print("Recommendations:")
    print("=" * 70)

    categories = set(i.category for i in issues)

    if "context_type" in categories or "context_size" in categories:
      print(
          "  • Ensure file objects (not filenames) are passed via context="
          " parameter"
      )
      print(
          "  • Use: llm_query(prompt, context=file_obj) not llm_query(prompt +"
          " filename)"
      )

    if "explosion" in categories or "deep_recursion" in categories:
      print(
          "  • Use llm_query_batched with recursive=False for parallel"
          " extraction"
      )
      print("  • Aggregate results at calling level, don't spawn more children")

    if "ratio_imbalance" in categories:
      print(
          "  • Root agent should run more iterations to aggregate child results"
      )
      print(
          "  • Check that llm_query_batched results are being collected and"
          " synthesized"
      )

    if "redundant_work" in categories:
      print("  • Consider caching or deduplicating file analysis")
      print("  • Use batch queries instead of individual file queries")


def main():
  parser = argparse.ArgumentParser(description="Diagnose RLM run issues")
  parser.add_argument("log_file", nargs="?", help="Path to log file")
  args = parser.parse_args()

  if args.log_file:
    log_path = Path(args.log_file)
  else:
    log_path = find_latest_log()
    if not log_path:
      print("No log files found", file=sys.stderr)
      sys.exit(1)

  entries = load_log(log_path)

  # Run all checks
  issues = []
  issues.extend(check_context_issues(entries))
  issues.extend(check_iteration_explosion(entries))
  issues.extend(check_redundant_work(entries))
  issues.extend(check_final_answer(entries))

  print_diagnosis(issues, entries, log_path)


if __name__ == "__main__":
  main()
