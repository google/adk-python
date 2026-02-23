"""Example 14: Parallel Execution + Rewind

Demonstrates:
- Parallel node execution
- Invocation tracking in parallel workflows
- Rewinding to parallel node
- Re-execution of parallel group

Run modes:
- Default: python -m contributing.samples.graph_examples.14_parallel_rewind.agent
- LLM: python -m contributing.samples.graph_examples.14_parallel_rewind.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.14_parallel_rewind.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.agents.graph import rewind_to_node
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class TaskAgent(BaseAgent):
  """Agent that executes a task."""

  def __init__(self, name: str, task_name: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._task_name = task_name
    self._count = 0

  async def _run_async_impl(self, ctx):
    self._count += 1
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=(
                        f"✅ {self._task_name} completed (execution"
                        f" #{self._count})"
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
      tuple: (task1, task2, merge) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    task1 = create_llm_agent(
        name="task1",
        instruction=(
            "Respond with 'Data fetch completed (execution #X)' where X is the"
            " execution count. Track this in your context."
        ),
    )
    task2 = create_llm_agent(
        name="task2",
        instruction=(
            "Respond with 'Data transform completed (execution #X)' where X is"
            " the execution count. Track this in your context."
        ),
    )
    merge = create_llm_agent(
        name="merge",
        instruction=(
            "Respond with 'Merge results completed (execution #X)' where X is"
            " the execution count. Track this in your context."
        ),
    )

    return task1, task2, merge
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    task1 = TaskAgent(name="task1", task_name="Data fetch")
    task2 = TaskAgent(name="task2", task_name="Data transform")
    merge = TaskAgent(name="merge", task_name="Merge results")

    return task1, task2, merge


async def main():
  print("\n" + "=" * 60)
  print("Example 14: Parallel Execution + Rewind")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  task1, task2, merge = create_agents()

  # Build graph
  graph = (
      GraphAgent(name="parallel_rewind_workflow")
      .add_node("task1", agent=task1)
      .add_node("task2", agent=task2)
      .add_node("merge", agent=merge)
      # Add parallel group
      .add_parallel_group(
          "parallel_tasks", ParallelNodeGroup(nodes=["task1", "task2"])
      )
      .add_edge("task1", "merge")
      .add_edge("task2", "merge")
      .set_start("task1")
      .set_end("merge")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="parallel_rewind_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("🚀 First execution (parallel tasks)...\n")

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   {part.text}")

  # Check invocations
  session = await session_service.get_session(
      app_name="parallel_rewind_demo", user_id="user1", session_id="session1"
  )
  node_invocations = session.state.get("node_invocations", {})

  print(f"\n📊 Invocation Tracking:")
  for node_name, invocations in node_invocations.items():
    print(f"   {node_name}: {len(invocations)} invocation(s)")

  # Rewind to task1 (part of parallel group)
  print(f"\n⏪ Rewinding to 'task1' (parallel group node)...")
  await rewind_to_node(
      graph,
      session_service,
      app_name="parallel_rewind_demo",
      user_id="user1",
      session_id="session1",
      node_name="task1",
      invocation_index=-1,
  )

  print("   ✅ Rewind successful!")

  # Re-execute from rewind point
  print("\n🚀 Re-execution after rewind (parallel group re-runs)...\n")

  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   {part.text}")

  print("\n✅ Example complete!")
  print("\n   Key Points:")
  print("   - Rewind works with parallel nodes")
  print("   - Entire parallel group re-executes")
  print("   - Invocations tracked per node")
  print("   - Execution counts show: task1=#1, task2=#2, merge=#2\n")


if __name__ == "__main__":
  asyncio.run(main())
