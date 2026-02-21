"""Example 1: Basic GraphAgent Workflow

Demonstrates:
- Creating a simple directed graph
- Adding nodes (agents)
- Adding edges (transitions)
- Setting start and end nodes
- Executing the workflow

Run modes:
- Default: python -m contributing.samples.graph_examples.01_basic.agent
- LLM: python -m contributing.samples.graph_examples.01_basic.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.01_basic.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class SimpleAgent(BaseAgent):
  """A simple agent that outputs a message."""

  def __init__(self, name: str, message: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._message = message

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._message)]),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents():
  """Create agents based on USE_LLM mode.

  Returns:
      tuple: (validate, process, complete) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    validate = create_llm_agent(
        name="validate",
        instruction=(
            "You are a validation agent. Respond with '✅ Validation passed' to"
            " confirm the workflow started successfully."
        ),
    )
    process = create_llm_agent(
        name="process",
        instruction=(
            "You are a processing agent. Respond with '⚙️ Processing data' to"
            " indicate you're processing the workflow."
        ),
    )
    complete = create_llm_agent(
        name="complete",
        instruction=(
            "You are a completion agent. Respond with '✅ Workflow complete' to"
            " signal successful workflow completion."
        ),
    )

    return validate, process, complete
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    validate = SimpleAgent(name="validate", message="✅ Validation passed")
    process = SimpleAgent(name="process", message="⚙️  Processing data")
    complete = SimpleAgent(name="complete", message="✅ Workflow complete")

    return validate, process, complete


async def main():
  print("\n" + "=" * 60)
  print("Example 1: Basic GraphAgent Workflow")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  validate, process, complete = create_agents()

  # Build graph using convenience API (fluent pattern)
  graph = (
      GraphAgent(name="basic_workflow")
      .add_node("validate", agent=validate)
      .add_node("process", agent=process)
      .add_node("complete", agent=complete)
      .add_edge("validate", "process")
      .add_edge("process", "complete")
      .set_start("validate")
      .set_end("complete")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="basic_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("🚀 Executing workflow: validate → process → complete\n")

  new_message = types.Content(parts=[types.Part(text="Start workflow")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   {part.text}")

  print("\n✅ Example complete!\n")


if __name__ == "__main__":
  asyncio.run(main())
