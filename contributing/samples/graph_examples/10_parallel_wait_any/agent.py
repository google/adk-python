"""Example 10: Parallel Execution - WAIT_ANY (Race)

Demonstrates:
- Racing multiple data sources
- WAIT_ANY join strategy
- First-to-complete wins
- Automatic cancellation of slower nodes

Run modes:
- Default: python -m contributing.samples.graph_examples.10_parallel_wait_any.agent
- LLM: python -m contributing.samples.graph_examples.10_parallel_wait_any.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.10_parallel_wait_any.agent
"""

import asyncio

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

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class DataSourceAgent(BaseAgent):
  """Simulates fetching from different data sources."""

  def __init__(self, name: str, source_type: str, latency_ms: int, **kwargs):
    super().__init__(name=name, **kwargs)
    self._source_type = source_type
    self._latency_ms = latency_ms

  async def _run_async_impl(self, ctx):
    await asyncio.sleep(self._latency_ms / 1000.0)

    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        f"✅ Data from {self._source_type}"
                        f" ({self._latency_ms}ms)"
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
      tuple: (from_cache, from_database, from_api) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    from_cache = create_llm_agent(
        name="from_cache",
        instruction=(
            "Respond with 'Data from CACHE (50ms)' exactly. Respond quickly"
            " without delays."
        ),
    )
    from_database = create_llm_agent(
        name="from_database",
        instruction=(
            "Respond with 'Data from DATABASE (150ms)' exactly. Respond quickly"
            " without delays."
        ),
    )
    from_api = create_llm_agent(
        name="from_api",
        instruction=(
            "Respond with 'Data from API (300ms)' exactly. Respond quickly"
            " without delays."
        ),
    )

    return from_cache, from_database, from_api
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    from_cache = DataSourceAgent(
        name="from_cache", source_type="CACHE", latency_ms=50
    )
    from_database = DataSourceAgent(
        name="from_database", source_type="DATABASE", latency_ms=150
    )
    from_api = DataSourceAgent(
        name="from_api", source_type="API", latency_ms=300
    )

    return from_cache, from_database, from_api


async def main():
  print("\n" + "=" * 60)
  print("Example 10: Parallel Execution - WAIT_ANY (Race)")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  from_cache, from_database, from_api = create_agents()

  # Build graph
  graph = (
      GraphAgent(name="race_workflow")
      .add_node("from_cache", agent=from_cache)
      .add_node("from_database", agent=from_database)
      .add_node("from_api", agent=from_api)
      # Add parallel group with WAIT_ANY strategy (race!)
      .add_parallel_group(
          "data_race",
          ParallelNodeGroup(
              nodes=["from_cache", "from_database", "from_api"],
              join_strategy=JoinStrategy.WAIT_ANY,  # First to finish wins!
          ),
      )
      .set_start("from_cache")
      .set_end("from_cache")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="race_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("🏁 Starting data source race...")
  print("   Competitors:")
  print("   - Cache:    50ms")
  print("   - Database: 150ms")
  print("   - API:      300ms")
  print("   Strategy: WAIT_ANY (first to complete)\n")

  import time

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

  print(f"\n✅ Race complete in ~{total_time}ms!")
  print("   Winner: Cache (fastest source)")
  print("   Slower sources: Cancelled automatically")
  print("   Use case: Cache-DB-API fallback strategy\n")


if __name__ == "__main__":
  asyncio.run(main())
