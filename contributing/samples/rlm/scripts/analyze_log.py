#!/usr/bin/env python3
"""
Analyze RLM JSONL logs for quick insights.

Usage:
    python scripts/analyze_log.py [LOG_FILE] [OPTIONS]

Examples:
    # Analyze the most recent log
    python scripts/analyze_log.py

    # Analyze a specific log
    python scripts/analyze_log.py logs/rlm_2026-01-22_*.jsonl

    # Show only the summary
    python scripts/analyze_log.py --summary

    # Show the iteration tree
    python scripts/analyze_log.py --tree

    # Show all code blocks
    python scripts/analyze_log.py --code

    # Show final answer only
    python scripts/analyze_log.py --final

    # Filter by depth
    python scripts/analyze_log.py --depth 0

    # Show LLM responses (truncated)
    python scripts/analyze_log.py --responses

    # Show simple LLM calls (non-recursive llm_query calls)
    python scripts/analyze_log.py --simple

    # Show only failed simple LLM calls
    python scripts/analyze_log.py --simple --failed

    # Export to markdown
    python scripts/analyze_log.py --export report.md
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys


def load_log(log_path: Path) -> list[dict]:
  """Load JSONL log, skipping malformed lines."""
  entries = []
  with open(log_path) as f:
    for i, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      try:
        entries.append(json.loads(line))
      except json.JSONDecodeError as e:
        print(f"Warning: Skipping malformed line {i}: {e}", file=sys.stderr)
  return entries


def find_latest_log(log_dir: Path = Path("logs")) -> Path | None:
  """Find the most recent log file."""
  logs = sorted(log_dir.glob("rlm_*.jsonl"), key=lambda p: p.stat().st_mtime)
  return logs[-1] if logs else None


def get_metadata(entries: list[dict]) -> dict | None:
  """Extract metadata entry."""
  for e in entries:
    if e.get("type") == "metadata":
      return e
  return None


def get_iterations(entries: list[dict]) -> list[dict]:
  """Get all iteration entries."""
  return [e for e in entries if e.get("type") == "iteration"]


def get_simple_llm_calls(entries: list[dict]) -> list[dict]:
  """Get all simple_llm_call entries (non-recursive llm_query calls)."""
  return [e for e in entries if e.get("type") == "simple_llm_call"]


def print_summary(entries: list[dict], log_path: Path):
  """Print a summary of the run."""
  meta = get_metadata(entries)
  iters = get_iterations(entries)
  simple_calls = get_simple_llm_calls(entries)

  print("=" * 70)
  print(f"RLM Log Analysis: {log_path.name}")
  print("=" * 70)

  if meta:
    print(f"\nModel:          {meta.get('root_model', 'unknown')}")
    print(f"Max Iterations: {meta.get('max_iterations', 'unknown')}")
    print(f"Max Depth:      {meta.get('max_depth', 'unknown')}")
    print(f"Timestamp:      {meta.get('timestamp', 'unknown')}")

  print(f"\nTotal Iterations: {len(iters)}")

  # Count by depth
  depth_counts = defaultdict(int)
  for it in iters:
    depth_counts[it.get("depth", 0)] += 1

  print("\nIterations by Depth:")
  for depth in sorted(depth_counts.keys()):
    print(f"  Depth {depth}: {depth_counts[depth]} iterations")

  # Simple LLM calls summary
  if simple_calls:
    success_count = sum(1 for c in simple_calls if c.get("success", True))
    failed_count = len(simple_calls) - success_count
    total_time_ms = sum(c.get("execution_time_ms", 0) for c in simple_calls)

    print(f"\nSimple LLM Calls: {len(simple_calls)}")
    print(f"  Successful: {success_count}")
    if failed_count > 0:
      print(f"  Failed:     {failed_count}")
    print(f"  Total Time: {total_time_ms/1000:.1f}s")

    # Count by depth
    simple_depth_counts = defaultdict(int)
    for c in simple_calls:
      simple_depth_counts[c.get("depth", 0)] += 1

    if len(simple_depth_counts) > 1:
      print("  By Depth:")
      for depth in sorted(simple_depth_counts.keys()):
        print(f"    Depth {depth}: {simple_depth_counts[depth]} calls")

  # Find final answer
  final = None
  for it in reversed(iters):
    if it.get("final_answer"):
      final = it.get("final_answer")
      break

  if final:
    print(f"\nFinal Answer Found: Yes ({len(final)} chars)")
  else:
    print("\nFinal Answer Found: No (run may still be in progress)")

  # Total time
  times = [
      it.get("iteration_time", 0) for it in iters if it.get("iteration_time")
  ]
  if times:
    print(f"\nTotal Iteration Time: {sum(times):.1f}s")
    print(f"Avg Iteration Time:   {sum(times)/len(times):.1f}s")


def _print_simple_calls_summary(calls: list[dict], indent: str) -> None:
  """Print a grouped summary of simple LLM calls for an iteration.

  Args:
      calls: List of simple_llm_call entries for this iteration.
      indent: Indentation string to align with parent iteration.
  """
  total = len(calls)
  success = sum(1 for c in calls if c.get("success", True))
  failed = total - success
  total_time_ms = sum(c.get("execution_time_ms", 0) for c in calls)

  # Check if this is a batch
  batch_sizes = {c.get("batch_size") for c in calls if c.get("batch_size")}
  is_batch = len(batch_sizes) == 1 and batch_sizes.pop() == total

  # Build status string
  if failed > 0:
    status = f"{success} ok, {failed} failed"
  else:
    status = "ok" if total == 1 else f"{total} ok"

  # Build description
  if is_batch:
    desc = f"batch[{total}]"
  elif total == 1:
    desc = "llm_query"
  else:
    desc = f"llm_query x{total}"

  time_str = (
      f"{total_time_ms/1000:.1f}s"
      if total_time_ms >= 1000
      else f"{total_time_ms:.0f}ms"
  )

  # Print with arrow to show it's a sub-call
  print(f"{indent}    └─ {desc} ({time_str}) [{status}]")


def print_tree(entries: list[dict], show_simple: bool = True):
  """Print the iteration tree showing agent hierarchy.

  Args:
      entries: Log entries to display.
      show_simple: If True, show simple LLM calls grouped after each iteration.
  """
  iters = get_iterations(entries)
  simple_calls = get_simple_llm_calls(entries) if show_simple else []

  # Group simple calls by (depth, parent_iteration)
  simple_by_iter: dict[tuple[int, int], list[dict]] = defaultdict(list)
  for call in simple_calls:
    key = (call.get("depth", 0), call.get("parent_iteration", 0))
    simple_by_iter[key].append(call)

  print("\nIteration Tree:")
  print("-" * 50)

  for it in iters:
    depth = it.get("depth", 0)
    agent = it.get("agent_name", "unknown")
    iteration = it.get("iteration", 0)
    time_s = it.get("iteration_time") or 0
    has_code = bool(it.get("code_blocks"))
    has_final = bool(it.get("final_answer"))

    indent = "  " * depth
    code_marker = " [code]" if has_code else ""
    final_marker = " [FINAL]" if has_final else ""

    # Truncate agent name for display
    agent_short = agent.replace("rlm_agent", "rlm")

    time_str = f"({time_s:.1f}s)" if time_s else ""
    print(
        f"{indent}[{iteration:2d}] {agent_short}"
        f" {time_str}{code_marker}{final_marker}"
    )

    # Show simple LLM calls for this iteration
    if show_simple:
      key = (depth, iteration)
      calls = simple_by_iter.get(key, [])
      if calls:
        _print_simple_calls_summary(calls, indent)


def print_code_blocks(entries: list[dict], depth_filter: int | None = None):
  """Print all code blocks from the log."""
  iters = get_iterations(entries)

  print("\nCode Blocks:")
  print("=" * 70)

  for it in iters:
    depth = it.get("depth", 0)
    if depth_filter is not None and depth != depth_filter:
      continue

    code_blocks = it.get("code_blocks", [])
    if not code_blocks:
      continue

    iteration = it.get("iteration", "?")
    agent = it.get("agent_name", "unknown")

    for i, block in enumerate(code_blocks):
      print(
          f"\n--- Iteration {iteration} (depth={depth}, {agent}) Block"
          f" {i+1} ---"
      )
      print(f"Code:\n{block.get('code', '')}")
      output = block.get("output", "")
      if output:
        # Truncate long outputs
        if len(output) > 1000:
          output = (
              output[:1000] + f"\n... (truncated, {len(output)} chars total)"
          )
        print(f"\nOutput:\n{output}")
      error = block.get("error", "")
      if error:
        print(f"\nError:\n{error}")


def print_responses(
    entries: list[dict], depth_filter: int | None = None, max_len: int = 500
):
  """Print LLM responses (truncated)."""
  iters = get_iterations(entries)

  print("\nLLM Responses:")
  print("=" * 70)

  for it in iters:
    depth = it.get("depth", 0)
    if depth_filter is not None and depth != depth_filter:
      continue

    iteration = it.get("iteration", "?")
    agent = it.get("agent_name", "unknown")
    response = it.get("response", "")

    if not response:
      continue

    print(f"\n--- Iteration {iteration} (depth={depth}, {agent}) ---")
    if len(response) > max_len:
      print(
          response[:max_len] + f"\n... (truncated, {len(response)} chars total)"
      )
    else:
      print(response)


def print_final_answer(entries: list[dict]):
  """Print the final answer."""
  iters = get_iterations(entries)

  for it in reversed(iters):
    if it.get("final_answer"):
      print("\nFinal Answer:")
      print("=" * 70)
      print(it["final_answer"])
      return

  print("\nNo final answer found (run may still be in progress)")


def print_simple_llm_calls(
    entries: list[dict],
    depth_filter: int | None = None,
    max_len: int = 300,
    show_failed_only: bool = False,
):
  """Print simple LLM calls (non-recursive llm_query calls)."""
  simple_calls = get_simple_llm_calls(entries)

  if not simple_calls:
    print("\nNo simple LLM calls found.")
    return

  print("\nSimple LLM Calls (recursive=False):")
  print("=" * 70)

  for i, call in enumerate(simple_calls):
    depth = call.get("depth", 0)
    if depth_filter is not None and depth != depth_filter:
      continue

    success = call.get("success", True)
    if show_failed_only and success:
      continue

    agent = call.get("agent_name", "unknown")
    model = call.get("model", "unknown")
    time_ms = call.get("execution_time_ms", 0)
    parent_iter = call.get("parent_iteration", "?")
    parent_block = call.get("parent_block_index", "?")
    batch_idx = call.get("batch_index")
    batch_size = call.get("batch_size")

    # Header
    status = "OK" if success else "FAILED"
    batch_info = f" [batch {batch_idx+1}/{batch_size}]" if batch_size else ""
    print(
        f"\n--- Call {i+1} ({status}) depth={depth} iter={parent_iter}"
        f" block={parent_block}{batch_info} ---"
    )
    print(f"Agent: {agent} | Model: {model} | Time: {time_ms:.0f}ms")

    # Prompt
    prompt = call.get("prompt", call.get("prompt_full", ""))
    if prompt:
      if len(prompt) > max_len:
        prompt = (
            prompt[:max_len]
            + f"... ({len(call.get('prompt_full', prompt))} chars)"
        )
      print(f"\nPrompt:\n{prompt}")

    # Response or error
    if not success:
      error = call.get("error", "Unknown error")
      print(f"\nError: {error}")
    else:
      response = call.get("response", call.get("response_full", ""))
      if response:
        if len(response) > max_len:
          response = (
              response[:max_len]
              + f"... ({len(call.get('response_full', response))} chars)"
          )
        print(f"\nResponse:\n{response}")


def export_markdown(entries: list[dict], output_path: Path, log_path: Path):
  """Export the log to a markdown report."""
  meta = get_metadata(entries)
  iters = get_iterations(entries)
  simple_calls = get_simple_llm_calls(entries)

  lines = []
  lines.append(f"# RLM Run Report: {log_path.name}\n")

  # Metadata
  if meta:
    lines.append("## Configuration\n")
    lines.append(f"- **Model:** {meta.get('root_model', 'unknown')}")
    lines.append(
        f"- **Max Iterations:** {meta.get('max_iterations', 'unknown')}"
    )
    lines.append(f"- **Max Depth:** {meta.get('max_depth', 'unknown')}")
    lines.append(f"- **Timestamp:** {meta.get('timestamp', 'unknown')}")
    lines.append("")

  # Summary stats
  lines.append("## Summary\n")
  lines.append(f"- **Total Iterations:** {len(iters)}")

  depth_counts = defaultdict(int)
  for it in iters:
    depth_counts[it.get("depth", 0)] += 1

  for depth in sorted(depth_counts.keys()):
    lines.append(f"- **Depth {depth}:** {depth_counts[depth]} iterations")

  if simple_calls:
    success_count = sum(1 for c in simple_calls if c.get("success", True))
    lines.append(
        f"- **Simple LLM Calls:** {len(simple_calls)} ({success_count}"
        " successful)"
    )

  lines.append("")

  # Iterations
  lines.append("## Iterations\n")

  for it in iters:
    depth = it.get("depth", 0)
    iteration = it.get("iteration", "?")
    agent = it.get("agent_name", "unknown")
    response = it.get("response", "")
    code_blocks = it.get("code_blocks", [])
    final = it.get("final_answer")

    lines.append(f"### Iteration {iteration} (Depth {depth})\n")
    lines.append(f"**Agent:** `{agent}`\n")

    if response:
      lines.append("**Response:**\n")
      lines.append(
          f"```\n{response[:2000]}{'...' if len(response) > 2000 else ''}\n```\n"
      )

    for i, block in enumerate(code_blocks):
      lines.append(f"**Code Block {i+1}:**\n")
      lines.append(f"```python\n{block.get('code', '')}\n```\n")
      output = block.get("output", "")
      if output:
        lines.append(
            f"**Output:**\n```\n{output[:1000]}{'...' if len(output) > 1000 else ''}\n```\n"
        )

    if final:
      lines.append(f"**FINAL ANSWER:**\n\n{final}\n")

    lines.append("---\n")

  # Simple LLM Calls section
  if simple_calls:
    lines.append("## Simple LLM Calls\n")
    lines.append(
        "These are non-recursive `llm_query()` calls made during code"
        " execution.\n"
    )

    for i, call in enumerate(simple_calls):
      depth = call.get("depth", 0)
      success = call.get("success", True)
      agent = call.get("agent_name", "unknown")
      model = call.get("model", "unknown")
      time_ms = call.get("execution_time_ms", 0)
      parent_iter = call.get("parent_iteration", "?")
      batch_idx = call.get("batch_index")
      batch_size = call.get("batch_size")

      status = "OK" if success else "FAILED"
      batch_info = f" (batch {batch_idx+1}/{batch_size})" if batch_size else ""

      lines.append(f"### Call {i+1} - {status}{batch_info}\n")
      lines.append(f"- **Agent:** `{agent}`")
      lines.append(f"- **Model:** {model}")
      lines.append(f"- **Depth:** {depth}")
      lines.append(f"- **Parent Iteration:** {parent_iter}")
      lines.append(f"- **Time:** {time_ms:.0f}ms")
      lines.append("")

      prompt = call.get("prompt_full", call.get("prompt", ""))
      if prompt:
        lines.append("**Prompt:**\n")
        lines.append(
            f"```\n{prompt[:1000]}{'...' if len(prompt) > 1000 else ''}\n```\n"
        )

      if not success:
        error = call.get("error", "Unknown error")
        lines.append(f"**Error:** {error}\n")
      else:
        response = call.get("response_full", call.get("response", ""))
        if response:
          lines.append("**Response:**\n")
          lines.append(
              f"```\n{response[:1000]}{'...' if len(response) > 1000 else ''}\n```\n"
          )

      lines.append("---\n")

  with open(output_path, "w") as f:
    f.write("\n".join(lines))

  print(f"Exported report to {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Analyze RLM JSONL logs",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "log_file", nargs="?", help="Path to log file (default: latest)"
  )
  parser.add_argument(
      "--summary", "-s", action="store_true", help="Show summary only"
  )
  parser.add_argument(
      "--tree", "-t", action="store_true", help="Show iteration tree"
  )
  parser.add_argument(
      "--code", "-c", action="store_true", help="Show code blocks"
  )
  parser.add_argument(
      "--responses", "-r", action="store_true", help="Show LLM responses"
  )
  parser.add_argument(
      "--final", "-f", action="store_true", help="Show final answer only"
  )
  parser.add_argument(
      "--simple",
      action="store_true",
      help="Show simple LLM calls (recursive=False)",
  )
  parser.add_argument(
      "--failed",
      action="store_true",
      help="With --simple, show only failed calls",
  )
  parser.add_argument(
      "--no-simple-tree",
      action="store_true",
      help="Hide simple LLM calls from iteration tree",
  )
  parser.add_argument("--depth", "-d", type=int, help="Filter by depth")
  parser.add_argument(
      "--export", "-e", type=str, help="Export to markdown file"
  )
  parser.add_argument(
      "--list", "-l", action="store_true", help="List available log files"
  )

  args = parser.parse_args()

  # List logs
  if args.list:
    log_dir = Path("logs")
    logs = sorted(log_dir.glob("rlm_*.jsonl"), key=lambda p: p.stat().st_mtime)
    print("Available log files:")
    for log in logs[-10:]:  # Last 10
      size = log.stat().st_size
      size_str = (
          f"{size/1024:.1f}KB"
          if size < 1024 * 1024
          else f"{size/1024/1024:.1f}MB"
      )
      print(f"  {log.name} ({size_str})")
    return

  # Find log file
  if args.log_file:
    log_path = Path(args.log_file)
  else:
    log_path = find_latest_log()
    if not log_path:
      print("No log files found in logs/", file=sys.stderr)
      sys.exit(1)

  if not log_path.exists():
    print(f"Log file not found: {log_path}", file=sys.stderr)
    sys.exit(1)

  # Load entries
  entries = load_log(log_path)
  if not entries:
    print("No valid entries found in log", file=sys.stderr)
    sys.exit(1)

  # Export mode
  if args.export:
    export_markdown(entries, Path(args.export), log_path)
    return

  # Determine what to show
  show_all = not any([
      args.summary,
      args.tree,
      args.code,
      args.responses,
      args.final,
      args.simple,
  ])

  if show_all or args.summary:
    print_summary(entries, log_path)

  if show_all or args.tree:
    show_simple_in_tree = not getattr(args, "no_simple_tree", False)
    print_tree(entries, show_simple=show_simple_in_tree)

  if args.code:
    print_code_blocks(entries, args.depth)

  if args.responses:
    print_responses(entries, args.depth)

  if args.simple:
    print_simple_llm_calls(entries, args.depth, show_failed_only=args.failed)

  if args.final:
    print_final_answer(entries)


if __name__ == "__main__":
  main()
