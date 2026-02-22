"""Example 12: Parallel Execution with Checkpointing

Demonstrates:
- Enabling checkpointing alongside parallel branch execution
- Two parallel workers each writing independent results to session state
- Using CheckpointService to inspect checkpoint index after execution
- Checkpoint data stored under "_checkpoint_index" in session state

Run modes:
- Default: python -m contributing.samples.graph_examples.12_parallel_checkpointing.agent
- LLM: python -m contributing.samples.graph_examples.12_parallel_checkpointing.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.12_parallel_checkpointing.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import JoinStrategy
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.checkpoints import CheckpointService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class WorkerAgent(BaseAgent):
  """Worker that writes a result value to session state."""

  def __init__(self, name: str, result_key: str, result_value: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._result_key = result_key
    self._result_value = result_value

  async def _run_async_impl(self, ctx):
    ctx.session.state[self._result_key] = self._result_value
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        f"Worker '{self.name}':"
                        f" {self._result_key}={self._result_value!r}"
                    )
                )
            ]
        ),
    )


class CollectAgent(BaseAgent):
  """Collects and reports results from both workers."""

  async def _run_async_impl(self, ctx):
    result_a = ctx.session.state.get("result_a", "N/A")
    result_b = ctx.session.state.get("result_b", "N/A")
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        f"Collected: result_a={result_a!r},"
                        f" result_b={result_b!r}"
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
      tuple: (worker_a, worker_b, collect) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    worker_a = create_llm_agent(
        name="worker_a",
        instruction=(
            "Respond with \"Worker 'worker_a': result_a='processed_by_a'\""
            " exactly."
        ),
    )
    worker_b = create_llm_agent(
        name="worker_b",
        instruction=(
            "Respond with \"Worker 'worker_b': result_b='processed_by_b'\""
            " exactly."
        ),
    )
    collect = create_llm_agent(
        name="collect",
        instruction=(
            "Respond with \"Collected: result_a='processed_by_a',"
            " result_b='processed_by_b'\" exactly."
        ),
    )

    return worker_a, worker_b, collect
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    worker_a = WorkerAgent(
        name="worker_a", result_key="result_a", result_value="processed_by_a"
    )
    worker_b = WorkerAgent(
        name="worker_b", result_key="result_b", result_value="processed_by_b"
    )
    collect = CollectAgent(name="collect")

    return worker_a, worker_b, collect


async def main():
  print("\n" + "=" * 60)
  print("Example 12: Parallel Execution with Checkpointing")
  print("=" * 60 + "\n")

  # Setup services
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)

  # Create agents (deterministic or LLM based on USE_LLM flag)
  worker_a, worker_b, collect = create_agents()

  # Build graph with checkpointing enabled
  graph = (
      GraphAgent(name="parallel_checkpoint_workflow", checkpointing=True)
      .add_node("worker_a", agent=worker_a)
      .add_node("worker_b", agent=worker_b)
      .add_node("collect", agent=collect)
      # Parallel group: both workers run concurrently with WAIT_ALL
      .add_parallel_group(
          "workers",
          ParallelNodeGroup(
              nodes=["worker_a", "worker_b"],
              join_strategy=JoinStrategy.WAIT_ALL,
          ),
      )
      # Edges: worker_a (entry for parallel group) -> collect
      .add_edge("worker_a", "collect")
      .set_start("worker_a")
      .set_end("collect")
  )

  # Execute
  runner = Runner(
      app_name="parallel_checkpoint_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("Executing parallel workflow with checkpointing=True")
  print("   Graph: [worker_a || worker_b] -> collect\n")

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   [{event.author}] {part.text}")

  # Inspect checkpoint data from session state
  session = await session_service.get_session(
      app_name="parallel_checkpoint_demo",
      user_id="user1",
      session_id="session1",
  )

  checkpoint_index = session.state.get("_checkpoint_index", {})
  print(f"\n   Checkpoint index entries: {len(checkpoint_index)}")

  if checkpoint_index:
    for cp_id, cp_info in checkpoint_index.items():
      agent = cp_info.get("agent", "unknown")
      print(f"   Checkpoint '{cp_id[:24]}...': agent={agent}")
  else:
    print("   (No checkpoints recorded — checkpointing requires")
    print("    CheckpointCallback for automatic per-node checkpoints)")

  # Show worker results persisted in session state
  print(f"\n   Session state after execution:")
  print(f"   result_a = {session.state.get('result_a', 'N/A')!r}")
  print(f"   result_b = {session.state.get('result_b', 'N/A')!r}")

  # Use checkpoint_service to manually create a checkpoint post-execution
  cp_metadata = await checkpoint_service.create_checkpoint(
      session=session,
      description="Post parallel execution snapshot",
      agent_name="parallel_checkpoint_workflow",
  )
  print(f"\n   Manual checkpoint created: {cp_metadata.checkpoint_id}")
  print(f"   Checkpoint state keys: {list(cp_metadata.state_snapshot.keys())}")

  print("\nExample complete!\n")
  print("   Use CheckpointCallback for automatic per-node checkpointing")
  print("   Use checkpoint_service.restore_checkpoint() for resume")


if __name__ == "__main__":
  asyncio.run(main())
