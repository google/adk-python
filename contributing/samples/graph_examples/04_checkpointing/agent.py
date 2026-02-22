"""Example 4: Checkpointing & Resume

Demonstrates:
- Automatic checkpointing at each node
- Listing checkpoints
- State persistence
- Checkpoint metadata

Run modes:
- Default: python -m contributing.samples.graph_examples.04_checkpointing.agent
- LLM: python -m contributing.samples.graph_examples.04_checkpointing.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.04_checkpointing.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
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


class StepAgent(BaseAgent):
  """Agent that represents a workflow step."""

  def __init__(self, name: str, step_num: int, **kwargs):
    super().__init__(name=name, **kwargs)
    self._step_num = step_num

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"✅ Completed step {self._step_num}")]
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
        instruction="Respond with 'Completed step 1' exactly.",
    )
    step2 = create_llm_agent(
        name="step2",
        instruction="Respond with 'Completed step 2' exactly.",
    )
    step3 = create_llm_agent(
        name="step3",
        instruction="Respond with 'Completed step 3' exactly.",
    )

    return step1, step2, step3
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    step1 = StepAgent(name="step1", step_num=1)
    step2 = StepAgent(name="step2", step_num=2)
    step3 = StepAgent(name="step3", step_num=3)

    return step1, step2, step3


async def main():
  print("\n" + "=" * 60)
  print("Example 4: Checkpointing & Resume")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  step1, step2, step3 = create_agents()

  # Setup checkpoint service
  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service)

  # Build graph with checkpointing enabled
  graph = (
      GraphAgent(name="checkpoint_workflow", checkpointing=True)
      .add_node("step1", agent=step1)
      .add_node("step2", agent=step2)
      .add_node("step3", agent=step3)
      .add_edge("step1", "step2")
      .add_edge("step2", "step3")
      .set_start("step1")
      .set_end("step3")
  )

  # Execute with checkpointing
  runner = Runner(
      app_name="checkpoint_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("🚀 Executing workflow with checkpointing enabled...\n")

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   {part.text}")

  # Get session and check checkpoint data
  session = await session_service.get_session(
      app_name="checkpoint_demo", user_id="user1", session_id="session1"
  )

  checkpoint_data = session.state.get("graph_checkpoint", {})
  print(f"\n📊 Checkpoint Information:")
  print(f"   Last checkpoint at: {checkpoint_data.get('node', 'N/A')}")
  print(f"   Iteration: {checkpoint_data.get('iteration', 'N/A')}")

  # Show execution path
  path = session.state.get("graph_path", [])
  print(f"   Execution path: {' → '.join(path)}")

  print("\n✅ Example complete!")
  print("   Note: Checkpoints created at each node for state persistence\n")


if __name__ == "__main__":
  asyncio.run(main())
