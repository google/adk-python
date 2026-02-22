"""Advanced GraphAgent example with all features.

This example demonstrates a research paper writing workflow with:
- Checkpointing (save/resume)
- LLM-based interrupt reasoning
- Custom observability callbacks
- Flexible interrupt timings
- Immediate cancellation
- All interrupt actions

Run:
    python -m contributing.samples.graph_agent_advanced.agent
"""

import asyncio
import json
import os
from typing import Optional

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph.callbacks import create_nested_observability_callback
from google.adk.agents.graph.callbacks import NodeCallbackContext
from google.adk.agents.graph.interrupt_reasoner import InterruptReasoner
from google.adk.agents.graph.interrupt_reasoner import InterruptReasonerConfig
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.agents.llm_agent import LlmAgent
from google.adk.checkpoints.checkpoint_service import CheckpointService
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from google import genai

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ==============================================================================
# Custom Observability Callback (Rich, Multi-Content Events)
# ==============================================================================


async def research_observability_callback(
    ctx: NodeCallbackContext,
) -> Optional[Event]:
  """Custom observability for research workflow.

  Emits rich events with:
  - Node execution info
  - Current state snapshot
  - Progress indicator
  - Execution metadata
  """
  # Calculate progress
  total_nodes = len(ctx.invocation_context.agent.nodes)
  current_iteration = ctx.iteration
  progress = (
      current_iteration / ctx.invocation_context.agent.max_iterations
  ) * 100

  # Build rich content
  parts = [
      # Header with emoji and node name
      types.Part(text=f"📝 **Executing**: {ctx.node.name}"),
      # Progress indicator
      types.Part(
          text=f"Progress: {progress:.1f}% (iteration {current_iteration})"
      ),
      # Current state (formatted as JSON)
      # Automatic Pydantic serialization
      types.Part(text=f"**State**:\n{ctx.state.data_to_json()}"),
      # Execution metadata
      types.Part(
          text=(
              "**Metadata**:"
              f" Agent={ctx.node.agent.name if ctx.node.agent else 'function'},"
              f" Total nodes={total_nodes}"
          )
      ),
  ]

  return Event(
      author="observability",
      content=types.Content(parts=parts),
      actions=EventActions(
          escalate=False,
          state_delta={
              "observability_node": ctx.node.name,
              "observability_iteration": ctx.iteration,
              "observability_progress": progress,
              "observability_timestamp": asyncio.get_event_loop().time(),
          },
      ),
  )


# ==============================================================================
# Research Workflow Nodes (Agents)
# ==============================================================================


def _create_research_agents():
  """Create fresh agent instances (avoids single-parent conflicts)."""
  return (
      LlmAgent(
          name="literature_review",
          model=_MODEL,
          instruction=(
              "You are a research literature reviewer. Search for and summarize"
              " key papers related to the given topic. Output a JSON list of"
              " papers with title, authors, and key findings."
          ),
      ),
      LlmAgent(
          name="hypothesis_generator",
          model=_MODEL,
          instruction=(
              "You are a research hypothesis generator. Based on the literature"
              " review, propose 3-5 testable hypotheses. Output a JSON list of"
              " hypotheses with rationale."
          ),
      ),
      LlmAgent(
          name="methodology_designer",
          model=_MODEL,
          instruction=(
              "You are a research methodology expert. Design experimental"
              " methods to test the hypotheses. Output a JSON dict with"
              " methodology sections: participants, materials, procedure."
          ),
      ),
      LlmAgent(
          name="paper_writer",
          model=_MODEL,
          instruction=(
              "You are an academic paper writer. Write a research paper with"
              " sections: Abstract, Introduction, Methods, Results, Discussion."
              " Use the literature review, hypotheses, methodology, and results"
              " from the state. Output well-structured academic prose."
          ),
      ),
      LlmAgent(
          name="peer_reviewer",
          model=_MODEL,
          instruction=(
              "You are a peer reviewer. Review the paper for: clarity,"
              " scientific rigor, statistical validity, and writing quality."
              " Provide a review with scores (1-10) and specific suggestions"
              " for improvement. Output JSON with scores and comments."
          ),
      ),
  )


# 4. Results Analyzer Agent (Simulated)
def analyze_results(state: GraphState, ctx) -> GraphState:
  """Simulate data analysis (in real scenario, this would run experiments).

  Args:
      state: Current graph state
      ctx: Invocation context (provides session, session_service, user_content, etc.)

  Returns:
      Updated GraphState with analysis results
  """
  # Simulate analysis results
  results = {
      "hypothesis_1": {
          "supported": True,
          "p_value": 0.023,
          "effect_size": 0.42,
      },
      "hypothesis_2": {
          "supported": False,
          "p_value": 0.156,
          "effect_size": 0.18,
      },
      "hypothesis_3": {
          "supported": True,
          "p_value": 0.001,
          "effect_size": 0.67,
      },
  }
  # Update state with results
  state.data["analysis_results"] = results
  return state


# ==============================================================================
# Conditional Routing Functions
# ==============================================================================


def needs_revision(state: GraphState) -> bool:
  """Check if paper needs revision based on peer review."""
  review = state.data.get("peer_review", {})
  # LLM agents store output as JSON string; parse if needed
  if isinstance(review, str):
    try:
      review = json.loads(review)
    except (json.JSONDecodeError, TypeError):
      return False
  avg_score = sum(review.get("scores", {}).values()) / max(
      len(review.get("scores", {})), 1
  )
  return avg_score < 7.0  # Needs revision if average score < 7/10


def revision_count_ok(state: GraphState) -> bool:
  """Check if we've exceeded max revisions."""
  return state.data.get("revision_count", 0) < 3


# ==============================================================================
# Build Research Workflow Graph
# ==============================================================================


def build_research_workflow(
    session_service: InMemorySessionService,
    checkpoint_service: CheckpointService,
    interrupt_service: InterruptService,
) -> GraphAgent:
  """Build advanced research workflow with all features enabled."""

  # Create LLM-based interrupt reasoner
  interrupt_reasoner = InterruptReasoner(
      config=InterruptReasonerConfig(
          model=_MODEL,
          available_actions=["continue", "rerun", "pause", "defer", "skip"],
          instruction=(
              "You are an interrupt reasoning agent for a research paper"
              " writing workflow. Analyze interrupt messages from researchers"
              " and decide the best action. Consider: Is the feedback about"
              " quality? Should we rerun with guidance? Should we pause for"
              " human review? Should we defer the feedback for later?"
          ),
      )
  )

  # Create GraphAgent with all features enabled
  graph = GraphAgent(
      name="research_workflow",
      description=(
          "Advanced research paper writing workflow with interrupt &"
          " observability"
      ),
      max_iterations=20,
      checkpointing=True,
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          mode=InterruptMode.AFTER,  # Check for interrupts after each node
          nodes=None,  # All nodes (can specify specific nodes like ["peer_reviewer"])
          reasoner=interrupt_reasoner,  # Use LLM to reason about interrupts
      ),
      # Custom observability callback
      before_node_callback=research_observability_callback,
      # Nested observability (shows hierarchy)
      after_node_callback=create_nested_observability_callback(),
  )

  # Create fresh agents (avoids single-parent conflicts across scenarios)
  (
      literature_agent,
      hypothesis_agent,
      methodology_agent,
      paper_writer_agent,
      peer_reviewer_agent,
  ) = _create_research_agents()

  # Add nodes
  graph.add_node("literature_review", agent=literature_agent)
  graph.add_node("generate_hypotheses", agent=hypothesis_agent)
  graph.add_node("design_methodology", agent=methodology_agent)
  graph.add_node("analyze_results", function=analyze_results)
  graph.add_node("write_paper", agent=paper_writer_agent)
  graph.add_node("peer_review", agent=peer_reviewer_agent)

  # Add edges (sequential workflow with revision loop)
  graph.add_edge("literature_review", "generate_hypotheses")
  graph.add_edge("generate_hypotheses", "design_methodology")
  graph.add_edge("design_methodology", "analyze_results")
  graph.add_edge("analyze_results", "write_paper")
  graph.add_edge("write_paper", "peer_review")

  # Conditional routing: if review is poor, revise
  graph.add_edge(
      "peer_review",
      "write_paper",  # Loop back to rewrite
      condition=lambda s: needs_revision(s) and revision_count_ok(s),
  )

  # If review is good or max revisions reached, finish at peer_review
  # No edge needed - peer_review naturally becomes an end node when no edge matches

  # Set start and end
  graph.set_start("literature_review")
  graph.set_end("peer_review")  # End at peer_review when revision not needed

  return graph


# ==============================================================================
# Example Usage Scenarios
# ==============================================================================


async def scenario_1_basic_execution():
  """Scenario 1: Basic execution with observability."""
  print("\n" + "=" * 80)
  print("SCENARIO 1: Basic Execution with Observability")
  print("=" * 80 + "\n")

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)
  interrupt_service = InterruptService()

  graph = build_research_workflow(
      session_service, checkpoint_service, interrupt_service
  )

  # Create session
  session = await session_service.create_session(
      app_name="research_workflow", user_id="researcher_1"
  )

  # Create runner
  runner = Runner(
      app_name="research_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,  # Session already created
  )

  # Run workflow
  print("Running research workflow...\n")
  async for event in runner.run_async(
      user_id="researcher_1",
      session_id=session.id,
      new_message=types.Content(
          parts=[
              types.Part(
                  text="Research topic: Impact of AI on software development"
              )
          ]
      ),
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"[{event.author}] {part.text[:200]}...")

  print("\n✅ Workflow completed!")
  print(f"Final state keys: {list(session.state.get('graph_data', {}).keys())}")


async def scenario_2_interrupt_with_reasoning():
  """Scenario 2: Send interrupt and let LLM reason about it."""
  print("\n" + "=" * 80)
  print("SCENARIO 2: Interrupt with LLM Reasoning")
  print("=" * 80 + "\n")

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)
  interrupt_service = InterruptService()

  graph = build_research_workflow(
      session_service, checkpoint_service, interrupt_service
  )

  session = await session_service.create_session(
      app_name="research_workflow", user_id="researcher_2"
  )

  # Send interrupt after 2 seconds (simulating human feedback)
  async def send_interrupt_after_delay():
    await asyncio.sleep(2)
    print(
        "\n🔔 Sending interrupt: 'The literature review missed key papers on"
        " neural architecture search'"
    )
    await interrupt_service.send_interrupt(
        session_id=session.id,
        text=(
            "The literature review missed key papers on neural architecture"
            " search. Please include them."
        ),
        action="defer",  # Suggest defer, but LLM will decide
        metadata={"feedback_type": "missing_references"},
    )

  # Run both concurrently
  interrupt_task = asyncio.create_task(send_interrupt_after_delay())

  # Create runner
  runner = Runner(
      app_name="research_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  async for event in runner.run_async(
      user_id="researcher_2",
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text="Research topic: Neural Architecture Search")]
      ),
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if "interrupt" in part.text.lower() or "defer" in part.text.lower():
          print(f"\n🔶 [{event.author}] {part.text}")

  await interrupt_task

  # Check interrupt decision
  decision = session.state.get("_last_interrupt_decision", {})
  print(
      f"\n📊 LLM Decision: {decision.get('action')} -"
      f" {decision.get('reasoning')}"
  )

  # Check deferred todos
  todos = session.state.get("_interrupt_todos", [])
  print(f"📝 Deferred todos: {len(todos)} items")
  if todos:
    print(f"   First todo: {todos[0].get('message', '')[:100]}...")


async def scenario_3_checkpointing_and_resume():
  """Scenario 3: Create checkpoints and resume from them."""
  print("\n" + "=" * 80)
  print("SCENARIO 3: Checkpointing and Resume")
  print("=" * 80 + "\n")

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)
  interrupt_service = InterruptService()

  graph = build_research_workflow(
      session_service, checkpoint_service, interrupt_service
  )

  session = await session_service.create_session(
      app_name="research_workflow", user_id="researcher_3"
  )

  # Run workflow and pause after "design_methodology"
  async def pause_after_methodology():
    await asyncio.sleep(3)
    print("\n⏸️  Pausing workflow after methodology design...")
    await interrupt_service.send_interrupt(
        session_id=session.id,
        text="Pause for team review",
        action="pause",
    )

  pause_task = asyncio.create_task(pause_after_methodology())

  # Create runner
  runner = Runner(
      app_name="research_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  print("Running workflow (will pause)...\n")
  async for event in runner.run_async(
      user_id="researcher_3",
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text="Research topic: Quantum Machine Learning")]
      ),
  ):
    pass

  await pause_task

  # List checkpoints
  checkpoints = await checkpoint_service.list_checkpoints(session)
  print(f"\n📦 Checkpoints created: {len(checkpoints)}")
  for cp in checkpoints:
    print(
        f"   - {cp.checkpoint_id}: {cp.metadata.get('graph_node', 'unknown')}"
    )

  # Resume from last checkpoint
  if checkpoints:
    last_checkpoint = checkpoints[-1]
    print(f"\n▶️  Resuming from checkpoint: {last_checkpoint.checkpoint_id}")

    restored_state = await checkpoint_service.restore_checkpoint(
        session, last_checkpoint.checkpoint_id
    )
    print(f"✅ Restored state keys: {list(restored_state.keys())}")


async def scenario_4_immediate_cancellation():
  """Scenario 4: Immediate cancellation (ESC-like) with state preservation."""
  print("\n" + "=" * 80)
  print("SCENARIO 4: Immediate Cancellation with State Preservation")
  print("=" * 80 + "\n")

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)
  interrupt_service = InterruptService()

  graph = build_research_workflow(
      session_service, checkpoint_service, interrupt_service
  )

  session = await session_service.create_session(
      app_name="research_workflow", user_id="researcher_4"
  )

  # Cancel immediately after 1.5 seconds
  async def cancel_immediately():
    await asyncio.sleep(1.5)
    print("\n🛑 Cancelling workflow immediately (ESC)...")
    await interrupt_service.cancel_session(session.id)

  cancel_task = asyncio.create_task(cancel_immediately())

  # Create runner
  runner = Runner(
      app_name="research_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  print("Running workflow (will cancel immediately)...\n")
  async for event in runner.run_async(
      user_id="researcher_4",
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text="Research topic: Large Language Models")]
      ),
  ):
    if "cancelled" in str(event.content).lower():
      print(f"\n⚠️  Cancellation event received: {event.content.parts[0].text}")

  await cancel_task

  # Check preserved state
  print(f"\n📊 Session state after cancel:")
  print(f"   - Cancelled: {session.state.get('graph_cancelled', False)}")
  print(
      "   - Cancelled at node:"
      f" {session.state.get('graph_cancelled_at_node', 'unknown')}"
  )
  print(f"   - Can resume: {session.state.get('graph_can_resume', False)}")
  print(f"   - Partial state saved: {bool(session.state.get('graph_data'))}")

  # Show partial domain data
  if session.state.get("graph_data"):
    partial_data = session.state["graph_data"]
    print(f"   - Partial data keys: {list(partial_data.keys())}")


async def scenario_5_all_interrupt_timings():
  """Scenario 5: Demonstrate all interrupt timings (BEFORE/AFTER/BOTH)."""
  print("\n" + "=" * 80)
  print("SCENARIO 5: All Interrupt Timings")
  print("=" * 80 + "\n")

  session_service = InMemorySessionService()
  checkpoint_service = CheckpointService(session_service=session_service)
  interrupt_service = InterruptService()

  # Create local agent instances for this scenario
  _, _, _, paper_writer_agent, peer_reviewer_agent = _create_research_agents()

  # Test BEFORE mode
  print("Testing InterruptMode.BEFORE (validate before execution)...\n")

  reasoner = InterruptReasoner(
      config=InterruptReasonerConfig(
          model=_MODEL,
          available_actions=["continue", "skip", "pause"],
      )
  )

  graph_before = GraphAgent(
      name="research_workflow_before",
      max_iterations=5,
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          mode=InterruptMode.BEFORE,  # Interrupt BEFORE node execution
          nodes=["peer_review"],  # Only before peer_review
          reasoner=reasoner,
      ),
  )

  # Add simple nodes
  graph_before.add_node("write", agent=paper_writer_agent)
  graph_before.add_node("peer_review", agent=peer_reviewer_agent)
  graph_before.add_edge("write", "peer_review")
  graph_before.add_edge("peer_review", "END")
  graph_before.set_start("write")
  graph_before.set_end("END")

  session_before = await session_service.create_session(
      app_name="before_test", user_id="researcher_5"
  )

  # Send interrupt before peer_review
  async def interrupt_before():
    await asyncio.sleep(1)
    print("🔔 Sending BEFORE interrupt: 'Skip peer review, paper is perfect'\n")
    await interrupt_service.send_interrupt(
        session_id=session_before.id,
        text="Skip peer review, this paper is already perfect",
        action="skip",
    )

  interrupt_task = asyncio.create_task(interrupt_before())

  # Create runner
  runner_before = Runner(
      app_name="before_test",
      agent=graph_before,
      session_service=session_service,
      auto_create_session=False,
  )

  async for event in runner_before.run_async(
      user_id="researcher_5",
      session_id=session_before.id,
      new_message=types.Content(
          parts=[types.Part(text="Write a paper on AI safety")]
      ),
  ):
    pass

  await interrupt_task

  print(
      "✅ BEFORE mode test complete. Peer review skipped:"
      f" {not bool(session_before.state.get('peer_review'))}\n"
  )


# ==============================================================================
# Main Entry Point
# ==============================================================================


async def main(run_all: bool = False):
  """Run example scenarios.

  Args:
      run_all: If True, runs all 5 scenarios (requires multiple LLM calls,
          may be slow). If False, runs only scenario 1 for quick validation.
          Set RUN_ALL_SCENARIOS=1 env var or pass --all flag to run all.
  """
  run_all = run_all or os.getenv("RUN_ALL_SCENARIOS", "").strip() in (
      "1",
      "true",
  )

  print("\n" + "=" * 60)
  print("GraphAgent Advanced Examples")
  print("=" * 60)

  await scenario_1_basic_execution()

  if run_all:
    await scenario_2_interrupt_with_reasoning()
    await scenario_3_checkpointing_and_resume()
    await scenario_4_immediate_cancellation()
    await scenario_5_all_interrupt_timings()
    print("\nAll 5 scenarios completed.")
  else:
    print(
        "\nScenario 1 completed. Set RUN_ALL_SCENARIOS=1 to run all 5"
        " scenarios."
    )


if __name__ == "__main__":
  import sys

  asyncio.run(main(run_all="--all" in sys.argv))
