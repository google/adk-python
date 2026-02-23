"""Example 6: Interrupt with Condition-Based Action Selection

Demonstrates:
- Condition-based interrupt routing without an LLM
- Using InterruptService.pause() and send_message() to inject decisions
- Checking queued messages with list_queued_messages()
- Resuming or cancelling based on message content
- Note: InterruptReasoner requires an LLM — this example shows deterministic routing

Run modes:
- Default: python -m contributing.samples.graph_examples.06_interrupts_reasoning.agent
- LLM: python -m contributing.samples.graph_examples.06_interrupts_reasoning.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.06_interrupts_reasoning.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class DraftAgent(BaseAgent):
  """Generates a draft for review."""

  def __init__(self, name: str, content: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._content = content

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Draft created: {self._content}")]
        ),
    )


class ReviewAgent(BaseAgent):
  """Processes the approved draft."""

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text="Draft approved — publishing review")]
        ),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents(content: str):
  """Create agents based on USE_LLM mode.

  Args:
      content: Draft content to generate

  Returns:
      tuple: (draft, review) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    draft = create_llm_agent(
        name="draft",
        instruction=f"Respond with 'Draft created: {content}' exactly.",
    )
    review = create_llm_agent(
        name="review",
        instruction=(
            "Respond with 'Draft approved — publishing review' exactly."
        ),
    )

    return draft, review
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    draft = DraftAgent(name="draft", content=content)
    review = ReviewAgent(name="review")

    return draft, review


async def run_scenario(scenario_name: str, decision_message: str) -> None:
  """Run a single interrupt scenario with a given decision message."""
  print(f"\n   Scenario: {scenario_name}")
  print(f"   Decision message: '{decision_message}'")

  interrupt_service = InterruptService()

  # Create agents (deterministic or LLM based on USE_LLM flag)
  draft, review = create_agents("First draft of the document")

  graph = (
      GraphAgent(
          name="interrupt_routing_workflow",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER,
              nodes=["draft"],
          ),
      )
      .add_node("draft", agent=draft)
      .add_node("review", agent=review)
      .add_edge("draft", "review")
      .set_start("draft")
      .set_end("review")
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="interrupt_routing_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  session_id = f"session_{scenario_name.lower().replace(' ', '_')}"

  # Register session so we can interact with it before the graph does
  interrupt_service.register_session(session_id)

  # Queue the decision message (simulates external human input)
  await interrupt_service.pause(session_id)
  await interrupt_service.send_message(
      session_id, decision_message, action="route"
  )

  # Peek at the queued messages to determine action
  queued = interrupt_service.list_queued_messages(session_id)
  if queued:
    msg_text = queued[0].text
    if "APPROVE" in msg_text:
      print("   Condition met: APPROVE found — resuming workflow")
      await interrupt_service.resume(session_id)
    else:
      print("   Condition not met: APPROVE absent — cancelling workflow")
      await interrupt_service.cancel(session_id)

  events_received = []
  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id=session_id, new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          events_received.append(f"[{event.author}] {part.text}")

  for line in events_received:
    print(f"   {line}")

  interrupted = not any("review" in e.lower() for e in events_received)
  print(f"   Review reached: {not interrupted}")


async def main():
  print("\n" + "=" * 60)
  print("Example 6: Interrupt with Condition-Based Action Selection")
  print("=" * 60 + "\n")
  print("Note: InterruptReasoner requires an LLM. This example uses")
  print("      deterministic condition-based routing instead.\n")
  print("Graph: draft -> [INTERRUPT] -> review")

  # Scenario 1: approve message -> resume -> review node runs
  await run_scenario("Approve", "APPROVE: content looks good")

  # Scenario 2: reject message -> cancel -> review node skipped
  await run_scenario("Reject", "REJECT: needs revision")

  print("\nExample complete!\n")
  print("   In production, replace condition check with InterruptReasoner")
  print("   (requires LLM) for natural language action selection.")


if __name__ == "__main__":
  asyncio.run(main())
