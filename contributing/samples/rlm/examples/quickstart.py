#!/usr/bin/env python3
"""
ADK-RLM Quickstart - Minimal example to get started.

This is the simplest possible RLM example. It demonstrates:
1. Using the completion convenience function
2. Running a computation that uses the REPL
3. Getting the final answer

Run with: python examples/quickstart.py
"""

from adk_rlm import completion


def main():
  # Use the convenience function for simple synchronous completion
  result = completion(
      context="Calculate the sum of the first 100 positive integers.",
      prompt="What is the sum? Use the REPL to compute it.",
      model="gemini-3-flash-preview",
  )

  # The expected answer is 5050 (Gauss's formula: n*(n+1)/2 = 100*101/2 = 5050)
  print(f"Answer: {result.response}")
  print(f"Execution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
  main()
