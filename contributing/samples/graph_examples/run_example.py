#!/usr/bin/env python3
"""Utility to run graph_examples with optional trace logging and LLM mode.

Usage:
    # Default mode (deterministic agents)
    python run_example.py 01_basic

    # LLM mode
    python run_example.py 01_basic --use-llm

    # With trace logs
    python run_example.py 01_basic --trace

    # Both
    python run_example.py 01_basic --use-llm --trace

    # List all examples
    python run_example.py --list
"""

import argparse
import importlib
import logging
from pathlib import Path
import sys


def setup_logging(trace: bool = False):
  """Configure logging based on trace flag."""
  level = logging.DEBUG if trace else logging.INFO
  format_str = (
      "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
      if trace
      else "%(message)s"
  )

  logging.basicConfig(
      level=level,
      format=format_str,
      datefmt="%H:%M:%S",
  )

  # Enable ADK trace logging
  if trace:
    logging.getLogger("google_adk").setLevel(logging.DEBUG)
    logging.getLogger("google.adk").setLevel(logging.DEBUG)


def list_examples():
  """List all available examples."""
  examples_dir = Path(__file__).parent
  examples = sorted([
      d.name
      for d in examples_dir.iterdir()
      if d.is_dir() and (d / "agent.py").exists() and not d.name.startswith("_")
  ])

  print("\n📚 Available graph_examples:\n")
  for ex in examples:
    agent_file = examples_dir / ex / "agent.py"
    # Read first docstring line
    with open(agent_file) as f:
      lines = f.readlines()
      desc = ""
      for line in lines:
        if line.strip().startswith('"""'):
          desc = line.strip('"""').strip()
          break
    print(f"   {ex:30s} - {desc}")

  print("\n")


def run_example(example_name: str, use_llm: bool = False, trace: bool = False):
  """Run a specific example."""
  setup_logging(trace=trace)

  # Set USE_LLM env var if needed
  if use_llm:
    import os

    os.environ["USE_LLM"] = "1"

  # Verify example exists
  example_dir = Path(__file__).parent / example_name
  if not example_dir.exists() or not (example_dir / "agent.py").exists():
    print(f"❌ Example '{example_name}' not found")
    print("\nRun with --list to see available examples")
    sys.exit(1)

  # Run via subprocess to handle module names starting with numbers
  import os
  import subprocess

  env = os.environ.copy()
  if use_llm:
    env["USE_LLM"] = "1"

  # Run from adk-python root
  adk_root = Path(__file__).parent.parent.parent.parent
  module_path = f"contributing.samples.graph_examples.{example_name}.agent"

  print(f"\n{'='*70}")
  print(f"Running: {example_name}")
  print(f"Mode: {'🤖 LLM' if use_llm else '🎭 Deterministic'}")
  print(f"Trace: {'✓ Enabled' if trace else '✗ Disabled'}")
  print(f"{'='*70}\n")

  try:
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=str(adk_root),
        env=env,
        capture_output=False,
        text=True,
    )
    sys.exit(result.returncode)
  except Exception as e:
    print(f"\n❌ Error running example: {e}")
    if trace:
      import traceback

      traceback.print_exc()
    sys.exit(1)


def main():
  parser = argparse.ArgumentParser(
      description="Run graph_examples with optional trace logging and LLM mode"
  )
  parser.add_argument(
      "example", nargs="?", help="Example name (e.g., 01_basic)"
  )
  parser.add_argument(
      "--use-llm",
      action="store_true",
      help="Use real LLM endpoints instead of deterministic agents",
  )
  parser.add_argument(
      "--trace", action="store_true", help="Enable detailed trace logging"
  )
  parser.add_argument("--list", action="store_true", help="List all examples")

  args = parser.parse_args()

  if args.list:
    list_examples()
    return

  if not args.example:
    parser.print_help()
    print("\nRun with --list to see available examples")
    sys.exit(1)

  run_example(args.example, use_llm=args.use_llm, trace=args.trace)


if __name__ == "__main__":
  main()
