#!/usr/bin/env python3
"""
Extract unique insights and findings from RLM run iterations.

This script finds substantive outputs from code executions and llm_query calls,
useful for understanding what the RLM actually learned from the data.

Usage:
    python scripts/extract_insights.py [LOG_FILE] [OPTIONS]

Examples:
    python scripts/extract_insights.py
    python scripts/extract_insights.py --min-length 200
    python scripts/extract_insights.py --depth 2 --format md
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


@dataclass
class Insight:
  iteration: int
  depth: int
  agent: str
  content: str
  source: str  # "code_output", "llm_response", "final_answer"
  topic: str | None  # extracted topic if identifiable


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


def is_substantive_output(text: str, min_length: int = 100) -> bool:
  """Check if output contains substantive content (not just debugging)."""
  if not text or len(text) < min_length:
    return False

  # Skip debugging output
  debug_patterns = [
      r"^Context type:",
      r"^dict_keys\(",
      r"^<LazyFile",
      r"^file_count:",
      r"^\d+ files",
      r"^print\(",
      r"^Error:",
      r"^Traceback",
  ]
  for pattern in debug_patterns:
    if re.match(pattern, text.strip()):
      return False

  # Look for substantive patterns
  substantive_patterns = [
      r"moat",
      r"risk",
      r"opportunit",
      r"competitor",
      r"strength",
      r"weakness",
      r"growth",
      r"market",
      r"customer",
      r"revenue",
      r"1\.",  # numbered lists
      r"\*\*",  # bold text
      r"summary",
      r"conclusion",
      r"recommend",
  ]
  text_lower = text.lower()
  for pattern in substantive_patterns:
    if re.search(pattern, text_lower):
      return True

  return False


def extract_topic(text: str) -> str | None:
  """Try to extract the main topic from text."""
  # Look for company names
  companies = [
      "Tyler Technologies",
      "Tyler",
      "Mark43",
      "Accela",
      "OpenGov",
      "CentralSquare",
      "Motorola",
      "Hexagon",
      "Axon",
      "Microsoft",
      "Oracle",
      "Workday",
      "Granicus",
      "CivicPlus",
  ]
  for company in companies:
    if company.lower() in text.lower():
      return company

  # Look for topic headers
  topic_patterns = [
      r"regarding\s+(\w+(?:\s+\w+){0,2})",
      r"about\s+(\w+(?:\s+\w+){0,2})",
      r"analyzing\s+(\w+(?:\s+\w+){0,2})",
  ]
  for pattern in topic_patterns:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
      return match.group(1)

  return None


def extract_insights(
    entries: list[dict], min_length: int = 100
) -> list[Insight]:
  """Extract substantive insights from the run."""
  insights = []
  iterations = [e for e in entries if e.get("type") == "iteration"]
  seen_content = set()  # Dedupe

  for it in iterations:
    depth = it.get("depth", 0)
    iteration = it.get("iteration", 0)
    agent = it.get("agent_name", "")

    # Check code block outputs
    for block in it.get("code_blocks", []):
      output = block.get("output", "")
      if output and is_substantive_output(output, min_length):
        # Hash for deduplication
        content_hash = hash(output[:200])
        if content_hash not in seen_content:
          seen_content.add(content_hash)
          insights.append(
              Insight(
                  iteration=iteration,
                  depth=depth,
                  agent=agent,
                  content=output,
                  source="code_output",
                  topic=extract_topic(output),
              )
          )

    # Check final answers
    final = it.get("final_answer")
    if final and len(final) >= min_length:
      content_hash = hash(final[:200])
      if content_hash not in seen_content:
        seen_content.add(content_hash)
        insights.append(
            Insight(
                iteration=iteration,
                depth=depth,
                agent=agent,
                content=final,
                source="final_answer",
                topic=extract_topic(final),
            )
        )

  return insights


def print_insights(
    insights: list[Insight],
    depth_filter: int | None = None,
    format: str = "text",
    max_content: int = 500,
):
  """Print extracted insights."""
  filtered = insights
  if depth_filter is not None:
    filtered = [i for i in filtered if i.depth == depth_filter]

  if format == "md":
    print_insights_markdown(filtered, max_content)
  else:
    print_insights_text(filtered, max_content)


def print_insights_text(insights: list[Insight], max_content: int):
  """Print insights in text format."""
  print(f"\nExtracted {len(insights)} substantive insights:")
  print("=" * 70)

  # Group by topic
  by_topic = defaultdict(list)
  for i in insights:
    topic = i.topic or "General"
    by_topic[topic].append(i)

  for topic, topic_insights in sorted(by_topic.items()):
    print(f"\n### {topic} ({len(topic_insights)} insights)")
    print("-" * 40)

    for insight in topic_insights[:5]:  # Limit per topic
      print(
          f"\n[Iter {insight.iteration}, depth {insight.depth}]"
          f" ({insight.source})"
      )
      content = insight.content
      if len(content) > max_content:
        content = content[:max_content] + "..."
      print(content)

    if len(topic_insights) > 5:
      print(f"\n  ... and {len(topic_insights) - 5} more")


def print_insights_markdown(insights: list[Insight], max_content: int):
  """Print insights in markdown format."""
  print("# Extracted Insights\n")

  by_topic = defaultdict(list)
  for i in insights:
    topic = i.topic or "General"
    by_topic[topic].append(i)

  for topic, topic_insights in sorted(by_topic.items()):
    print(f"## {topic}\n")

    for insight in topic_insights:
      print(f"### Iteration {insight.iteration} (depth {insight.depth})\n")
      print(f"*Source: {insight.source}*\n")
      content = insight.content
      if len(content) > max_content:
        content = content[:max_content] + "..."
      print(f"```\n{content}\n```\n")


def print_summary(insights: list[Insight]):
  """Print summary statistics."""
  print("\nInsights Summary:")
  print("=" * 50)

  # By source
  by_source = defaultdict(int)
  for i in insights:
    by_source[i.source] += 1
  print("\nBy Source:")
  for s, count in sorted(by_source.items()):
    print(f"  {s}: {count}")

  # By depth
  by_depth = defaultdict(int)
  for i in insights:
    by_depth[i.depth] += 1
  print("\nBy Depth:")
  for d, count in sorted(by_depth.items()):
    print(f"  Depth {d}: {count}")

  # By topic
  by_topic = defaultdict(int)
  for i in insights:
    by_topic[i.topic or "Unclassified"] += 1
  print("\nBy Topic:")
  for t, count in sorted(by_topic.items(), key=lambda x: -x[1])[:10]:
    print(f"  {t}: {count}")

  # Average content length
  avg_len = (
      sum(len(i.content) for i in insights) / len(insights) if insights else 0
  )
  print(f"\nAverage insight length: {avg_len:.0f} chars")


def main():
  parser = argparse.ArgumentParser(description="Extract insights from RLM logs")
  parser.add_argument("log_file", nargs="?", help="Path to log file")
  parser.add_argument(
      "--min-length",
      "-m",
      type=int,
      default=100,
      help="Minimum content length (default: 100)",
  )
  parser.add_argument("--depth", "-d", type=int, help="Filter by depth")
  parser.add_argument(
      "--format",
      "-f",
      choices=["text", "md"],
      default="text",
      help="Output format",
  )
  parser.add_argument(
      "--max-content",
      type=int,
      default=500,
      help="Max content to show per insight",
  )
  parser.add_argument(
      "--summary", "-s", action="store_true", help="Show summary only"
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
  insights = extract_insights(entries, args.min_length)

  if args.summary:
    print_summary(insights)
  else:
    print_insights(insights, args.depth, args.format, args.max_content)
    print_summary(insights)


if __name__ == "__main__":
  main()
