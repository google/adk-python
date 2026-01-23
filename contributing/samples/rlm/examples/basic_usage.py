#!/usr/bin/env python3
"""
Basic usage example for ADK-RLM.

This example demonstrates how to use the completion function to analyze
a simple context and get a response.
"""

from adk_rlm import completion


def main():
  # Simple context with a question
  context = """
    The following is a list of famous mathematicians and their contributions:

    1. Euclid (c. 300 BC) - Known as the "Father of Geometry", wrote "Elements"
    2. Isaac Newton (1643-1727) - Developed calculus and laws of motion
    3. Carl Friedrich Gauss (1777-1855) - Made contributions to number theory, statistics
    4. Leonhard Euler (1707-1783) - Prolific mathematician, introduced notation like e and i
    5. Srinivasa Ramanujan (1887-1920) - Made groundbreaking contributions to mathematical analysis
    """

  # Use the convenience function with verbose output
  result = completion(
      context=context,
      prompt="Who is known as the Father of Geometry and when did they live?",
      model="gemini-3-flash-preview",
      max_iterations=10,
      verbose=True,
  )

  print("\n" + "=" * 50)
  print("FINAL RESULT:")
  print("=" * 50)
  print(result.response)
  print(f"\nExecution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
  main()
