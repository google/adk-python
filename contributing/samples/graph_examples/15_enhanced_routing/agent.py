"""Example 15: Enhanced Routing (Priority, Weight, Fallback)

Demonstrates:
- Priority-based routing (higher priority evaluated first)
- Weighted random selection (probabilistic routing)
- Fallback edges (priority=0 always matches)

Run modes:
- Default: python -m contributing.samples.graph_examples.15_enhanced_routing.agent
- LLM: python -m contributing.samples.graph_examples.15_enhanced_routing.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.15_enhanced_routing.agent
"""

import asyncio

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import EdgeCondition
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


class SimpleAgent(BaseAgent):
  """Agent that outputs a message."""

  def __init__(self, name: str, message: str, **kwargs):
    super().__init__(name=name, **kwargs)
    self._message = message

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=self._message)]),
    )


class ScoreAgent(BaseAgent):
  """Agent that sets a risk score."""

  def __init__(self, name: str, score: float, **kwargs):
    super().__init__(name=name, **kwargs)
    self._score = score

  async def _run_async_impl(self, ctx):
    ctx.session.state["risk_score"] = self._score
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Risk score: {self._score}")]
        ),
    )


# ===========================
# Agent Factory
# ===========================


def create_agents_priority(score: float):
  """Create agents for priority routing example.

  Returns:
      tuple: (analyze, critical, warning, normal) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    analyze = create_llm_agent(
        name="analyze",
        instruction=f"Respond with 'Risk score: {score}' exactly.",
    )
    critical = create_llm_agent(
        name="critical",
        instruction=(
            "Respond with 'CRITICAL: Immediate action required' exactly."
        ),
    )
    warning = create_llm_agent(
        name="warning",
        instruction="Respond with 'WARNING: Review recommended' exactly.",
    )
    normal = create_llm_agent(
        name="normal",
        instruction="Respond with 'NORMAL: No action needed' exactly.",
    )

    return analyze, critical, warning, normal
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    analyze = ScoreAgent(name="analyze", score=score)
    critical = SimpleAgent(
        name="critical", message="🚨 CRITICAL: Immediate action required"
    )
    warning = SimpleAgent(
        name="warning", message="⚠️  WARNING: Review recommended"
    )
    normal = SimpleAgent(name="normal", message="✅ NORMAL: No action needed")

    return analyze, critical, warning, normal


def create_agents_weighted():
  """Create agents for weighted routing example.

  Returns:
      tuple: (start, server_a, server_b, server_c) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    start = create_llm_agent(
        name="start",
        instruction="Respond with 'Starting load balancer...' exactly.",
    )
    server_a = create_llm_agent(
        name="server_a",
        instruction="Respond with '   → Routed to Server A' exactly.",
    )
    server_b = create_llm_agent(
        name="server_b",
        instruction="Respond with '   → Routed to Server B' exactly.",
    )
    server_c = create_llm_agent(
        name="server_c",
        instruction="Respond with '   → Routed to Server C' exactly.",
    )

    return start, server_a, server_b, server_c
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    start = SimpleAgent(name="start", message="Starting load balancer...")
    server_a = SimpleAgent(name="server_a", message="   → Routed to Server A")
    server_b = SimpleAgent(name="server_b", message="   → Routed to Server B")
    server_c = SimpleAgent(name="server_c", message="   → Routed to Server C")

    return start, server_a, server_b, server_c


def create_agents_fallback(score: float):
  """Create agents for fallback routing example.

  Returns:
      tuple: (validate, premium, standard, fallback) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    validate = create_llm_agent(
        name="validate",
        instruction=f"Respond with 'Risk score: {score}' exactly.",
    )
    premium = create_llm_agent(
        name="premium",
        instruction="Respond with 'Premium path (VIP users)' exactly.",
    )
    standard = create_llm_agent(
        name="standard",
        instruction="Respond with 'Standard path (regular users)' exactly.",
    )
    fallback = create_llm_agent(
        name="fallback",
        instruction="Respond with 'Fallback path (default handler)' exactly.",
    )

    return validate, premium, standard, fallback
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    validate = ScoreAgent(name="validate", score=score)
    premium = SimpleAgent(name="premium", message="🌟 Premium path (VIP users)")
    standard = SimpleAgent(
        name="standard", message="📦 Standard path (regular users)"
    )
    fallback = SimpleAgent(
        name="fallback", message="🔒 Fallback path (default handler)"
    )

    return validate, premium, standard, fallback


async def main():
  print("\n" + "=" * 60)
  print("Example 15: Enhanced Routing")
  print("=" * 60 + "\n")

  # ===== Example 1: Priority-based Routing =====
  print("📊 Example 1: Priority-based Routing\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  analyze, critical, warning, normal = create_agents_priority(0.85)

  graph1 = (
      GraphAgent(name="priority_routing")
      .add_node("analyze", agent=analyze)
      .add_node("critical", agent=critical)
      .add_node("warning", agent=warning)
      .add_node("normal", agent=normal)
  )

  # Set output mapper to persist risk_score in state
  def store_score(output, state):
    new_state = GraphState(data=state.data.copy())
    new_state.data["risk_score"] = 0.85  # Score from analyze agent
    return new_state

  graph1.nodes["analyze"].output_mapper = store_score

  # Priority-based routing: higher priority evaluated first
  graph1.nodes["analyze"].edges = [
      EdgeCondition(
          target_node="critical",
          condition=lambda s: s.data.get("risk_score", 0) > 0.9,
          priority=10,  # Highest priority
      ),
      EdgeCondition(
          target_node="warning",
          condition=lambda s: s.data.get("risk_score", 0) > 0.7,
          priority=5,  # Medium priority - THIS WILL MATCH
      ),
      EdgeCondition(
          target_node="normal",
          priority=0,  # Fallback (priority=0 always matches if no other matched)
      ),
  ]

  graph1.set_start("analyze")
  graph1.set_end("critical")
  graph1.set_end("warning")
  graph1.set_end("normal")

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="routing_demo",
      agent=graph1,
      session_service=session_service,
      auto_create_session=True,
  )

  async for event in runner.run_async(
      user_id="user1",
      session_id="session1",
      new_message=types.Content(parts=[types.Part(text="Analyze")]),
  ):
    if event.content and event.content.parts and event.content.parts[0].text:
      print(f"   {event.content.parts[0].text}")

  print("\n   💡 Score was 0.85 → matched 'warning' (priority=5)")
  print("   💡 'critical' didn't match (0.85 < 0.9)")
  print("   💡 'normal' fallback not needed (higher priority matched)\n")

  # ===== Example 2: Weighted Random Routing =====
  print("🎲 Example 2: Weighted Random Routing\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  start, server_a, server_b, server_c = create_agents_weighted()

  graph2 = (
      GraphAgent(name="weighted_routing")
      .add_node("start", agent=start)
      .add_node("server_a", agent=server_a)
      .add_node("server_b", agent=server_b)
      .add_node("server_c", agent=server_c)
  )

  # Weighted routing: all at same priority, different weights
  graph2.nodes["start"].edges = [
      EdgeCondition(
          target_node="server_a",
          condition=lambda s: True,  # All match
          priority=1,  # Same priority
          weight=0.5,  # 50% probability
      ),
      EdgeCondition(
          target_node="server_b",
          condition=lambda s: True,
          priority=1,  # Same priority
          weight=0.3,  # 30% probability
      ),
      EdgeCondition(
          target_node="server_c",
          condition=lambda s: True,
          priority=1,  # Same priority
          weight=0.2,  # 20% probability
      ),
  ]

  graph2.set_start("start")
  graph2.set_end("server_a")
  graph2.set_end("server_b")
  graph2.set_end("server_c")

  runner2 = Runner(
      app_name="weighted_demo",
      agent=graph2,
      session_service=session_service,
      auto_create_session=True,
  )

  # Run multiple times to show distribution
  counts = {"server_a": 0, "server_b": 0, "server_c": 0}
  trials = 20

  print(f"   Running {trials} trials with weights (A:50%, B:30%, C:20%):\n")

  for i in range(trials):
    async for event in runner2.run_async(
        user_id="user1",
        session_id=f"session_weighted_{i}",
        new_message=types.Content(parts=[types.Part(text="Route")]),
    ):
      if event.content and event.content.parts and event.author in counts:
        text = event.content.parts[0].text
        counts[event.author] += 1
        print(f"   Trial {i+1:2d}: {text}")

  print(f"\n   📊 Distribution after {trials} trials:")
  print(
      f"   Server A: {counts['server_a']:2d}/{trials}"
      f" ({counts['server_a']/trials*100:.0f}%)"
  )
  print(
      f"   Server B: {counts['server_b']:2d}/{trials}"
      f" ({counts['server_b']/trials*100:.0f}%)"
  )
  print(
      f"   Server C: {counts['server_c']:2d}/{trials}"
      f" ({counts['server_c']/trials*100:.0f}%)\n"
  )

  # ===== Example 3: Fallback Edge =====
  print("🛡️  Example 3: Fallback Edge (priority=0)\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  validate, premium, standard, fallback = create_agents_fallback(0.5)

  graph3 = (
      GraphAgent(name="fallback_routing")
      .add_node("validate", agent=validate)
      .add_node("premium", agent=premium)
      .add_node("standard", agent=standard)
      .add_node("fallback", agent=fallback)
  )

  def store_score_fallback(output, state):
    new_state = GraphState(data=state.data.copy())
    new_state.data["risk_score"] = 0.5
    # Don't set is_vip or is_standard - will fall through to fallback
    return new_state

  graph3.nodes["validate"].output_mapper = store_score_fallback

  graph3.nodes["validate"].edges = [
      EdgeCondition(
          target_node="premium",
          condition=lambda s: s.data.get("is_vip", False),
          priority=10,  # High priority - won't match
      ),
      EdgeCondition(
          target_node="standard",
          condition=lambda s: s.data.get("is_standard", False),
          priority=5,  # Medium priority - won't match
      ),
      EdgeCondition(
          target_node="fallback",
          priority=0,  # FALLBACK - always matches if reached
      ),
  ]

  graph3.set_start("validate")
  graph3.set_end("premium")
  graph3.set_end("standard")
  graph3.set_end("fallback")

  runner3 = Runner(
      app_name="fallback_demo",
      agent=graph3,
      session_service=session_service,
      auto_create_session=True,
  )

  async for event in runner3.run_async(
      user_id="user1",
      session_id="session_fallback",
      new_message=types.Content(parts=[types.Part(text="Validate")]),
  ):
    if event.content and event.content.parts and event.content.parts[0].text:
      print(f"   {event.content.parts[0].text}")

  print("\n   💡 No is_vip or is_standard flag set")
  print("   💡 All higher priority edges failed to match")
  print("   💡 Fallback (priority=0) caught it!\n")

  print("=" * 60)
  print("✅ Enhanced Routing Complete!")
  print("=" * 60 + "\n")


if __name__ == "__main__":
  asyncio.run(main())
