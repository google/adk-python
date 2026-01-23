#!/usr/bin/env python3
"""
Multi-turn conversation example for ADK-RLM.

This example demonstrates how to use persistent mode for
multi-turn conversations where context accumulates.

For multi-turn, we use the RLM class directly with run_streaming()
since persistent mode requires maintaining state across calls.
"""

import asyncio

from adk_rlm import RLM
from adk_rlm import RLMEventType


async def run_query(rlm: RLM, context: str, prompt: str) -> str:
  """Run a query and return the final answer."""
  final_answer = None
  async for event in rlm.run_streaming(context, prompt):
    if event.custom_metadata:
      event_type = event.custom_metadata.get("event_type")
      if event_type == RLMEventType.FINAL_ANSWER.value:
        final_answer = event.custom_metadata.get("answer")
  return final_answer or ""


async def main():
  # Create an RLM instance with persistence enabled
  rlm = RLM(
      model="gemini-3-flash-preview",
      verbose=True,
      persistent=True,  # Enable multi-turn persistence
      max_iterations=10,
  )

  try:
    # First turn: introduce some data
    print("\n" + "=" * 50)
    print("TURN 1: Introduce data")
    print("=" * 50)

    result1 = await run_query(
        rlm,
        context="""
            Employee records:
            - Alice: Software Engineer, 5 years experience, salary $120,000
            - Bob: Data Scientist, 3 years experience, salary $110,000
            - Carol: Product Manager, 7 years experience, salary $140,000
            - Dave: DevOps Engineer, 2 years experience, salary $95,000
            """,
        prompt=(
            "Store this employee data in a variable called 'employees' as a"
            " list of dictionaries."
        ),
    )
    print(
        f"Result: {result1[:200]}..."
        if len(result1) > 200
        else f"Result: {result1}"
    )

    # Second turn: ask about the data
    print("\n" + "=" * 50)
    print("TURN 2: Query the data")
    print("=" * 50)

    result2 = await run_query(
        rlm,
        context="Calculate the average salary.",
        prompt="What is the average salary of all employees?",
    )
    print(f"Result: {result2}")

    # Third turn: more analysis
    print("\n" + "=" * 50)
    print("TURN 3: Further analysis")
    print("=" * 50)

    result3 = await run_query(
        rlm,
        context="Find the most experienced employee.",
        prompt="Who has the most years of experience and what is their role?",
    )
    print(f"Result: {result3}")

    print("\n" + "=" * 50)
    print("MULTI-TURN COMPLETE")
    print("=" * 50)

  finally:
    rlm.close()


if __name__ == "__main__":
  asyncio.run(main())
