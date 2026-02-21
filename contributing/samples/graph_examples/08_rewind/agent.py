"""Example 8: Rewind Integration

Demonstrates:
- Invocation tracking per node
- Rewinding to specific node execution
- Re-execution after rewind
- State restoration

Run modes:
- Default: python -m contributing.samples.graph_examples.08_rewind.agent
- LLM: python -m contributing.samples.graph_examples.08_rewind.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.08_rewind.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
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


class CounterAgent(BaseAgent):
  """Agent that tracks execution count."""

  def __init__(self, name: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._count = 0

  async def _run_async_impl(self, ctx):
    self._count += 1
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=f"✅ {self.name} executed (count: {self._count})"
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
      tuple: (step1, step2, step3) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    step1 = create_llm_agent(
        name="step1",
        instruction=(
            "Respond with 'step1 executed (count: X)' where X is the execution"
            " count. Track this in your context."
        ),
    )
    step2 = create_llm_agent(
        name="step2",
        instruction=(
            "Respond with 'step2 executed (count: X)' where X is the execution"
            " count. Track this in your context."
        ),
    )
    step3 = create_llm_agent(
        name="step3",
        instruction=(
            "Respond with 'step3 executed (count: X)' where X is the execution"
            " count. Track this in your context."
        ),
    )

    return step1, step2, step3
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    step1 = CounterAgent(name="step1")
    step2 = CounterAgent(name="step2")
    step3 = CounterAgent(name="step3")

    return step1, step2, step3


async def main():
  print("\n" + "=" * 60)
  print("Example 8: Rewind Integration")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  step1, step2, step3 = create_agents()

  # Build graph
  graph = (
      GraphAgent(name="rewind_workflow")
      .add_node("step1", agent=step1)
      .add_node("step2", agent=step2)
      .add_node("step3", agent=step3)
      .add_edge("step1", "step2")
      .add_edge("step2", "step3")
      .set_start("step1")
      .set_end("step3")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="rewind_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("🚀 First execution...\n")

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
      app_name="rewind_demo", user_id="user1", session_id="session1"
  )
  node_invocations = session.state.get("node_invocations", {})

  print(f"\n📊 Invocation Tracking:")
  for node_name, invocations in node_invocations.items():
    print(f"   {node_name}: {len(invocations)} invocation(s)")

  # Rewind to step2
  print(f"\n⏪ Rewinding to 'step2'...")
  await rewind_to_node(
      graph,
      session_service,
      app_name="rewind_demo",
      user_id="user1",
      session_id="session1",
      node_name="step2",
      invocation_index=-1,  # Last invocation
  )

  print("   ✅ Rewind successful! State restored to before step2")

  # Re-execute from rewind point
  print("\n🚀 Re-execution after rewind...\n")

  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   {part.text}")

  print("\n✅ Example complete!")
  print("   Note: step1 count stays at 1, step2 & step3 executed again\n")


if __name__ == "__main__":
  asyncio.run(main())
