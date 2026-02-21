"""Example 2: Conditional Routing

Demonstrates:
- Conditional edges based on state
- Multiple routing paths
- State-based decision making

Run modes:
- Default: python -m contributing.samples.graph_examples.02_conditional_routing.agent
- LLM: python -m contributing.samples.graph_examples.02_conditional_routing.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.02_conditional_routing.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class ValidatorAgent(BaseAgent):
  """Validates input and sets quality score."""

  def __init__(self, name: str, score: int, **kwargs):
    super().__init__(name=name, **kwargs)
    self._score = score

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(
                    text=f"✅ Validation complete (score: {self._score})"
                )
            ]
        ),
    )


class ProcessAgent(BaseAgent):
  """Process based on quality."""

  def __init__(self, name: str, quality: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._quality = quality

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"⚙️  {self._quality} quality processing")]
        ),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents(test_score: int):
  """Create agents based on USE_LLM mode.

  Args:
      test_score: The score to use for validation

  Returns:
      tuple: (validate, high_quality, medium_quality, low_quality) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    validate = create_llm_agent(
        name="validate",
        instruction=(
            "You are a validation agent. Respond with 'Validation complete"
            f" (score: {test_score})' exactly."
        ),
    )
    high_quality = create_llm_agent(
        name="high_quality",
        instruction=(
            "You are a high quality processor. Respond with 'HIGH quality"
            " processing' exactly."
        ),
    )
    medium_quality = create_llm_agent(
        name="medium_quality",
        instruction=(
            "You are a medium quality processor. Respond with 'MEDIUM quality"
            " processing' exactly."
        ),
    )
    low_quality = create_llm_agent(
        name="low_quality",
        instruction=(
            "You are a low quality processor. Respond with 'LOW quality"
            " processing' exactly."
        ),
    )

    return validate, high_quality, medium_quality, low_quality
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    validate = ValidatorAgent(name="validate", score=test_score)
    high_quality = ProcessAgent(name="high_quality", quality="HIGH")
    medium_quality = ProcessAgent(name="medium_quality", quality="MEDIUM")
    low_quality = ProcessAgent(name="low_quality", quality="LOW")

    return validate, high_quality, medium_quality, low_quality


async def main():
  print("\n" + "=" * 60)
  print("Example 2: Conditional Routing")
  print("=" * 60 + "\n")

  # Test with different scores
  for test_score in [95, 75, 45]:
    print(f"🎯 Testing with score: {test_score}")

    # Create agents (deterministic or LLM based on USE_LLM flag)
    validate, high_quality, medium_quality, low_quality = create_agents(
        test_score
    )

    # Build graph with conditional routing
    graph = (
        GraphAgent(name="conditional_workflow")
        .add_node(
            "validate",
            agent=validate,
            output_mapper=lambda output, state: GraphState(
                data={**state.data, "score": test_score},
            ),
        )
        .add_node("high_quality", agent=high_quality)
        .add_node("medium_quality", agent=medium_quality)
        .add_node("low_quality", agent=low_quality)
        # Conditional edges based on score
        .add_edge(
            "validate",
            "high_quality",
            condition=lambda s: s.data.get("score", 0) >= 80,
        )
        .add_edge(
            "validate",
            "medium_quality",
            condition=lambda s: 50 <= s.data.get("score", 0) < 80,
        )
        .add_edge(
            "validate",
            "low_quality",
            condition=lambda s: s.data.get("score", 0) < 50,
        )
        .set_start("validate")
        .set_end("high_quality")
        .set_end("medium_quality")
        .set_end("low_quality")
    )

    # Execute
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="routing_demo",
        agent=graph,
        session_service=session_service,
        auto_create_session=True,
    )

    new_message = types.Content(parts=[types.Part(text="Start")])
    async for event in runner.run_async(
        user_id="user1",
        session_id=f"session_{test_score}",
        new_message=new_message,
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            print(f"   {part.text}")

    print()

  print("✅ Example complete!\n")


if __name__ == "__main__":
  asyncio.run(main())
