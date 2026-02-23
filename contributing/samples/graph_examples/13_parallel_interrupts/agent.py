"""Example 13: Interrupt Inside a Parallel Group Branch

Demonstrates:
- Configuring an interrupt on one branch of a parallel group
- Using interrupt_service.is_active() to detect a pending interrupt
- Calling interrupt_service.resume() to continue after the interrupt fires
- branch_a runs normally; branch_b triggers the AFTER interrupt

Run modes:
- Default: python -m contributing.samples.graph_examples.13_parallel_interrupts.agent
- LLM: python -m contributing.samples.graph_examples.13_parallel_interrupts.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.13_parallel_interrupts.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph import JoinStrategy
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

SESSION_ID = "session1"


# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class BranchAgent(BaseAgent):
  """A simple agent representing one parallel branch."""

  def __init__(self, name: str, label: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._label = label

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Branch '{self._label}' executed")]
        ),
    )


class JoinAgent(BaseAgent):
  """Merges results after parallel branches complete."""

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text="Both branches joined — workflow resumed")]
        ),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents():
  """Create agents based on USE_LLM mode.

  Returns:
      tuple: (branch_a, branch_b, join) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    branch_a = create_llm_agent(
        name="branch_a",
        instruction="Respond with \"Branch 'A (normal)' executed\" exactly.",
    )
    branch_b = create_llm_agent(
        name="branch_b",
        instruction=(
            "Respond with \"Branch 'B (interrupted)' executed\" exactly."
        ),
    )
    join = create_llm_agent(
        name="join",
        instruction=(
            "Respond with 'Both branches joined — workflow resumed' exactly."
        ),
    )

    return branch_a, branch_b, join
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    branch_a = BranchAgent(name="branch_a", label="A (normal)")
    branch_b = BranchAgent(name="branch_b", label="B (interrupted)")
    join = JoinAgent(name="join")

    return branch_a, branch_b, join


async def main():
  print("\n" + "=" * 60)
  print("Example 13: Interrupt Inside a Parallel Group Branch")
  print("=" * 60 + "\n")

  interrupt_service = InterruptService()

  # Create agents (deterministic or LLM based on USE_LLM flag)
  branch_a, branch_b, join = create_agents()

  # Build graph with interrupt configured AFTER branch_b
  graph = (
      GraphAgent(
          name="parallel_interrupt_workflow",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER,
              nodes=["branch_b"],
          ),
      )
      .add_node("branch_a", agent=branch_a)
      .add_node("branch_b", agent=branch_b)
      .add_node("join", agent=join)
      # Parallel group: both branches run concurrently
      .add_parallel_group(
          "parallel_branches",
          ParallelNodeGroup(
              nodes=["branch_a", "branch_b"],
              join_strategy=JoinStrategy.WAIT_ALL,
          ),
      )
      # Edges: branch_a (parallel group entry) -> join
      .add_edge("branch_a", "join")
      .set_start("branch_a")
      .set_end("join")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="parallel_interrupt_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("Graph: [branch_a || branch_b] -> join")
  print("Interrupt configured AFTER branch_b\n")

  # Pre-register session so we can interact with interrupt service
  interrupt_service.register_session(SESSION_ID)

  # Collect events while running; concurrently check for interrupt and resume
  events_received = []

  async def resume_after_interrupt() -> None:
    """Poll until interrupt is active, then resume."""
    for _ in range(50):
      await asyncio.sleep(0.05)
      if interrupt_service.is_paused(SESSION_ID):
        print(
            "   Interrupt detected on 'branch_b' "
            "(interrupt_service.is_paused() == True)"
        )
        print("   Calling interrupt_service.resume() to continue...")
        await interrupt_service.resume(SESSION_ID)
        return
    print("   (interrupt not triggered within poll window)")

  new_message = types.Content(parts=[types.Part(text="Start")])

  resume_task = asyncio.create_task(resume_after_interrupt())

  async for event in runner.run_async(
      user_id="user1", session_id=SESSION_ID, new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          events_received.append((event.author, part.text))

  await resume_task

  print("\n   Events received during execution:")
  for author, text in events_received:
    print(f"   [{author}] {text}")

  is_still_active = interrupt_service.is_active(SESSION_ID)
  print(f"\n   interrupt_service.is_active(session_id): {is_still_active}")

  join_reached = any("join" in author for author, _ in events_received)
  print(f"   Join node reached: {join_reached}")

  print("\nExample complete!\n")
  print("   branch_b triggered an AFTER interrupt during parallel execution")
  print("   interrupt_service.resume() allowed the workflow to continue")
  print("   Use interrupt_service.cancel() instead to abort the workflow")


if __name__ == "__main__":
  asyncio.run(main())
