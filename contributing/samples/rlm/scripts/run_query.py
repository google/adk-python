#!/usr/bin/env python3
"""
Run an RLM query and save results.

Usage:
    python scripts/run_query.py "Your question here" [OPTIONS]

Examples:
    # Simple query with files
    python scripts/run_query.py "Summarize the key themes" --files "./docs/**/*.md"

    # Query with verbose output
    python scripts/run_query.py "What are the main risks?" --files "./corpora/**/*.md" -v

    # Use a specific model
    python scripts/run_query.py "Analyze this" --files "*.txt" --model gemini-3-flash-preview

    # Save output to file
    python scripts/run_query.py "Compare X and Y" --files "./data/*.md" --output results.md

    # Run in background (for long queries)
    python scripts/run_query.py "Complex analysis" --files "./large_corpus/**/*.md" --background
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import time


def run_query(
    prompt: str,
    files: list[str],
    model: str = "gemini-3-pro-preview",
    max_iterations: int = 30,
    max_depth: int = 5,
    verbose: bool = False,
    output: str | None = None,
) -> str:
  """Run an RLM query and return the result."""
  from adk_rlm import completion

  start = time.perf_counter()

  result = completion(
      files=files,
      prompt=prompt,
      model=model,
      max_iterations=max_iterations,
      max_depth=max_depth,
      log_dir="./logs",
      verbose=verbose,
  )

  elapsed = time.perf_counter() - start

  # Build output text
  output_text = f"""# RLM Query Result

**Prompt:** {prompt}

**Files:** {', '.join(files)}

**Model:** {model}

**Execution Time:** {elapsed:.1f}s

---

## Answer

{result.response}

---

*Generated at {datetime.now().isoformat()}*
"""

  # Save if output path specified
  if output:
    Path(output).write_text(output_text)
    print(f"Results saved to {output}")

  return result.response


def main():
  parser = argparse.ArgumentParser(
      description="Run an RLM query",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument("prompt", help="The question/prompt to ask")
  parser.add_argument(
      "--files", "-f", nargs="+", required=True, help="File patterns to load"
  )
  parser.add_argument(
      "--model", "-m", default="gemini-3-pro-preview", help="Model to use"
  )
  parser.add_argument(
      "--max-iterations", "-i", type=int, default=30, help="Max iterations"
  )
  parser.add_argument(
      "--max-depth", "-d", type=int, default=5, help="Max recursion depth"
  )
  parser.add_argument(
      "--verbose", "-v", action="store_true", help="Show verbose output"
  )
  parser.add_argument("--output", "-o", help="Save results to file")
  parser.add_argument(
      "--background", "-b", action="store_true", help="Run in background"
  )

  args = parser.parse_args()

  if args.background:
    # Fork to background
    import subprocess

    cmd = [
        sys.executable,
        __file__,
        args.prompt,
        "--files",
        *args.files,
        "--model",
        args.model,
        "--max-iterations",
        str(args.max_iterations),
        "--max-depth",
        str(args.max_depth),
    ]
    if args.verbose:
      cmd.append("--verbose")

    # Generate output file if not specified
    output = (
        args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    cmd.extend(["--output", output])

    log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_file, "w") as f:
      proc = subprocess.Popen(
          cmd,
          stdout=f,
          stderr=subprocess.STDOUT,
          start_new_session=True,
      )

    print(f"Started background process (PID: {proc.pid})")
    print(f"Output will be saved to: {output}")
    print(f"Logs: {log_file}")
    print(f"\nMonitor with: tail -f {log_file}")
    print(f"Check results: cat {output}")
    return

  result = run_query(
      prompt=args.prompt,
      files=args.files,
      model=args.model,
      max_iterations=args.max_iterations,
      max_depth=args.max_depth,
      verbose=args.verbose,
      output=args.output,
  )

  if not args.output:
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
  main()
