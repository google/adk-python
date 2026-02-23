"""Example 11: Parallel Execution - WAIT_N (continue after N of M complete)

Demonstrates:
- WAIT_N join strategy: proceed after N out of M branches complete
- Timing: faster than WAIT_ALL (doesn't wait for slowest branch)
- Three branches with different latencies: fast (10ms), medium (30ms), slow (100ms)
- WAIT_N=2 means the workflow continues as soon as any 2 branches finish

Run modes:
- Default: python -m contributing.samples.graph_examples.11_parallel_wait_n.agent
- LLM: python -m contributing.samples.graph_examples.11_parallel_wait_n.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.11_parallel_wait_n.agent
"""

import asyncio
import time

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import JoinStrategy
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

WAIT_N = 2


# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class LatencyAgent(BaseAgent):
  """Simulates an agent with a configurable latency."""

  def __init__(self, name: str, label: str, delay_ms: int, **kwargs):
    super().__init__(name=name, **kwargs)
    self._label = label
    self._delay_ms = delay_ms

  async def _run_async_impl(self, ctx):
    await asyncio.sleep(self._delay_ms / 1000.0)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        f"Branch '{self._label}' completed ({self._delay_ms}ms)"
                    )
                )
            ]
        ),
    )


class SetupAgent(BaseAgent):
  """Initialises the workflow."""

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(text="Setup complete — launching parallel branches")
            ]
        ),
    )


class MergeAgent(BaseAgent):
  """Aggregates results from completed branches."""

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        "Merge complete: collected results from "
                        f"{WAIT_N}/3 branches (WAIT_N strategy)"
                    )
                )
            ]
        ),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents():
  """Create agents based on USE_LLM mode.

  Returns:
      tuple: (setup, fast, medium, slow, merge) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    setup = create_llm_agent(
        name="setup",
        instruction=(
            "Respond with 'Setup complete — launching parallel branches'"
            " exactly."
        ),
    )
    fast = create_llm_agent(
        name="fast",
        instruction=(
            "Respond with \"Branch 'fast' completed (10ms)\" exactly. Respond"
            " quickly without delays."
        ),
    )
    medium = create_llm_agent(
        name="medium",
        instruction=(
            "Respond with \"Branch 'medium' completed (30ms)\" exactly. Respond"
            " quickly without delays."
        ),
    )
    slow = create_llm_agent(
        name="slow",
        instruction=(
            "Respond with \"Branch 'slow' completed (100ms)\" exactly. Respond"
            " quickly without delays."
        ),
    )
    merge = create_llm_agent(
        name="merge",
        instruction=(
            f"Respond with 'Merge complete: collected results from {WAIT_N}/3"
            " branches (WAIT_N strategy)' exactly."
        ),
    )

    return setup, fast, medium, slow, merge
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    setup = SetupAgent(name="setup")
    fast = LatencyAgent(name="fast", label="fast", delay_ms=10)
    medium = LatencyAgent(name="medium", label="medium", delay_ms=30)
    slow = LatencyAgent(name="slow", label="slow", delay_ms=100)
    merge = MergeAgent(name="merge")

    return setup, fast, medium, slow, merge


async def main():
  print("\n" + "=" * 60)
  print("Example 11: Parallel Execution - WAIT_N")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  setup, fast, medium, slow, merge = create_agents()

  # Build graph
  graph = (
      GraphAgent(name="wait_n_workflow")
      .add_node("setup", agent=setup)
      .add_node("fast", agent=fast)
      .add_node("medium", agent=medium)
      .add_node("slow", agent=slow)
      .add_node("merge", agent=merge)
      # Parallel group with WAIT_N strategy
      .add_parallel_group(
          "branches",
          ParallelNodeGroup(
              nodes=["fast", "medium", "slow"],
              join_strategy=JoinStrategy.WAIT_N,
              wait_n=WAIT_N,
          ),
      )
      # Linear edges: setup -> (parallel group) -> merge
      .add_edge("setup", "fast")
      .add_edge("fast", "merge")
      .set_start("setup")
      .set_end("merge")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="wait_n_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print(f"Executing WAIT_N workflow (N={WAIT_N} of 3 branches)")
  print("   Branch latencies: fast=10ms, medium=30ms, slow=100ms")
  print(
      f"   Expected: ~30ms (waits for {WAIT_N} fastest, ignores slow=100ms)\n"
  )

  start_time = time.time()

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          elapsed = int((time.time() - start_time) * 1000)
          print(f"   [{elapsed:3d}ms] {part.text}")

  total_time = int((time.time() - start_time) * 1000)

  print(f"\n   Total time: ~{total_time}ms")
  print(f"   WAIT_ALL would take: ~100ms (slowest branch)")
  print(f"   WAIT_N={WAIT_N} took:  ~30ms (2nd fastest branch)")
  print(f"   Speedup vs WAIT_ALL: ~{100 // max(total_time, 1)}x")

  print("\nExample complete!\n")


if __name__ == "__main__":
  asyncio.run(main())
