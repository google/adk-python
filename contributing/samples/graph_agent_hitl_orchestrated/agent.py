"""Composable HITL Orchestrated Pipeline — Multi-stage document processing.

Demonstrates how to compose HITL review loops as reusable NestedGraphNode
building blocks within a larger orchestrated pipeline.

Scenario: Multi-stage document pipeline
1. Orchestrator receives a document processing request
2. Classifier determines which processing stages are needed
3. Each stage is a self-contained GraphAgent with its own HITL review loop
4. Stages execute in sequence; each can loop internally for human review
5. Final stage aggregates results

Key concepts:
- HITL review as a **reusable building block** (NestedGraphNode)
- Inner graphs have independent interrupt timing and review cycles
- Outer orchestrator doesn't know about inner HITL details — clean abstraction
- Models real enterprise workflows: compliance review, multi-department approval

Outer graph:
    [classify] --> [process] --> [aggregate]

Inner graph (per stage):
    [execute] --> [review_gate] --> approved? --> [done]
                        |
                        v rejected
                   [revise] --> [review_gate]  (loop)

Run:
    python -m contributing.samples.graph_agent_hitl_orchestrated.agent
"""

import asyncio
import os

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph import InterruptService
from google.adk.agents.graph import NestedGraphNode
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


# ---------------------------------------------------------------------------
# LLM availability
# ---------------------------------------------------------------------------


def _has_llm() -> bool:
  return bool(
      os.environ.get("GOOGLE_API_KEY")
      or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
  )


# ---------------------------------------------------------------------------
# Fallback agents (deterministic, no LLM)
# ---------------------------------------------------------------------------


class StageAgent(BaseAgent):
  """Deterministic agent that performs a named stage operation."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, stage_type: str):
    super().__init__(name=name)
    object.__setattr__(self, "_stage_type", stage_type)

  async def _run_async_impl(self, ctx):
    stage_type = object.__getattribute__(self, "_stage_type")
    graph_data = ctx.session.state.get("graph_data", {})
    doc = graph_data.get("document", "")

    if stage_type == "extract":
      result = f"Extracted key entities from: {doc[:100]}"
    elif stage_type == "summarize":
      result = f"Summary of document: {doc[:80]}..."
    elif stage_type == "translate":
      result = f"Translated content: [{doc[:60]}] (translated)"
    else:
      result = f"Processed ({stage_type}): {doc[:80]}"

    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=result)]),
    )


class ReviseStageAgent(BaseAgent):
  """Deterministic revise agent for stage content."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  async def _run_async_impl(self, ctx):
    graph_data = ctx.session.state.get("graph_data", {})
    previous = graph_data.get("stage_content", "")
    feedback = graph_data.get("stage_feedback", "")
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"{previous}\n[Revised: {feedback}]")]
        ),
    )


# ---------------------------------------------------------------------------
# Inner review graph builder (reusable per stage)
# ---------------------------------------------------------------------------


def build_review_stage_graph(
    stage_name: str,
    stage_type: str,
    interrupt_service: InterruptService | None = None,
) -> GraphAgent:
  """Build a self-contained review graph for a single processing stage.

  Args:
      stage_name: Name for this stage graph (e.g., "extract_review")
      stage_type: Type of processing ("extract", "summarize", "translate")
      interrupt_service: Optional interrupt service for HITL

  Returns:
      GraphAgent with execute -> review_gate -> done/revise flow
  """
  execute_agent = StageAgent(
      name=f"{stage_type}_executor", stage_type=stage_type
  )
  revise_agent = ReviseStageAgent(name=f"{stage_type}_reviser")

  interrupt_config = None
  if interrupt_service:
    interrupt_config = InterruptConfig(
        mode=InterruptMode.BEFORE,
        nodes=["stage_review"],
    )

  graph = GraphAgent(
      name=stage_name,
      description=f"Review stage for {stage_type}",
      max_iterations=6,
      interrupt_service=interrupt_service,
      interrupt_config=interrupt_config,
  )

  # Execute stage
  graph.add_node(
      "execute",
      agent=execute_agent,
      output_mapper=lambda output, s: (
          s.data.update({"stage_content": output}),
          s,
      )[1],
  )

  # Review gate — checks interrupt queue for approval
  async def review_gate_fn(state: GraphState, ctx) -> str:
    svc = getattr(ctx, "_interrupt_service", None)
    sid = ctx.session.id if ctx.session else None
    if svc and sid:
      msg = await svc.check_interrupt(sid)
      if msg:
        approved = msg.action == "approve"
        state.data["stage_approved"] = approved
        state.data["stage_feedback"] = msg.text
        return (
            f"[{stage_type} review]"
            f" {'APPROVED' if approved else 'REVISE: ' + msg.text}"
        )
    # Auto-approve if no interrupt
    state.data["stage_approved"] = True
    state.data["stage_feedback"] = ""
    return f"[{stage_type} review] Auto-approved"

  graph.add_node(GraphNode(name="stage_review", function=review_gate_fn))

  # Revise node
  graph.add_node(
      "revise",
      agent=revise_agent,
      input_mapper=lambda s: (
          f"Revise: {s.data.get('stage_content', '')}\n"
          f"Feedback: {s.data.get('stage_feedback', '')}"
      ),
      output_mapper=lambda output, s: (
          s.data.update({"stage_content": output}),
          s,
      )[1],
  )

  # Done node — outputs final stage content
  async def done_fn(state: GraphState, ctx) -> str:
    return state.data.get("stage_content", "")

  graph.add_node(GraphNode(name="done", function=done_fn))

  # Edges
  graph.set_start("execute")
  graph.add_edge("execute", "stage_review")
  graph.add_edge(
      "stage_review",
      "done",
      condition=lambda s: s.data.get("stage_approved") is True,
  )
  graph.add_edge(
      "stage_review",
      "revise",
      condition=lambda s: s.data.get("stage_approved") is False,
  )
  graph.add_edge("revise", "stage_review")
  graph.set_end("done")

  return graph


# ---------------------------------------------------------------------------
# Outer orchestrator graph
# ---------------------------------------------------------------------------


def build_orchestrator(
    interrupt_service: InterruptService | None = None,
) -> GraphAgent:
  """Build the outer orchestration graph.

  Structure:
      [classify] --> [process] --> [aggregate]

  The `process` node is a NestedGraphNode wrapping a review-stage graph.

  Args:
      interrupt_service: Optional interrupt service for inner HITL

  Returns:
      Configured outer GraphAgent
  """

  # Classify: determine required stages from document
  async def classify_fn(state: GraphState, ctx) -> str:
    doc = state.data.get("input", "")
    state.data["document"] = doc

    # Simple rule-based classifier
    stages = []
    if any(kw in doc.lower() for kw in ["data", "entity", "extract"]):
      stages.append("extract")
    if any(kw in doc.lower() for kw in ["long", "summary", "summarize"]):
      stages.append("summarize")
    if any(kw in doc.lower() for kw in ["translate", "language", "i18n"]):
      stages.append("translate")
    if not stages:
      stages = ["extract", "summarize"]  # Default

    state.data["stages"] = stages
    return f"Classified: stages={stages}"

  # Build inner review graph for the primary stage
  # (In a production version, you'd dynamically select based on stages[0])
  inner_graph = build_review_stage_graph(
      stage_name="review_stage",
      stage_type="extract",
      interrupt_service=interrupt_service,
  )

  # Aggregate: combine results
  async def aggregate_fn(state: GraphState, ctx) -> str:
    stages = state.data.get("stages", [])
    stage_content = state.data.get("stage_content", "")
    document = state.data.get("document", "")

    result = (
        "Pipeline complete.\n"
        f"Document: {document[:100]}\n"
        f"Stages run: {stages}\n"
        f"Output: {stage_content[:200]}"
    )
    state.data["pipeline_result"] = result
    return result

  # Build outer graph
  graph = GraphAgent(
      name="document_pipeline",
      description="Multi-stage document processing with HITL review",
      max_iterations=20,
  )

  graph.add_node(GraphNode(name="classify", function=classify_fn))
  graph.add_node(
      NestedGraphNode(
          name="process",
          graph_agent=inner_graph,
          inherit_session=True,
          input_mapper=lambda s: s.data.get("document", ""),
      )
  )
  graph.add_node(GraphNode(name="aggregate", function=aggregate_fn))

  graph.set_start("classify")
  graph.add_edge("classify", "process")
  graph.add_edge("process", "aggregate")
  graph.set_end("aggregate")

  return graph


# ---------------------------------------------------------------------------
# Main — simulates multi-stage human review
# ---------------------------------------------------------------------------


async def main() -> None:
  """Run orchestrated HITL pipeline with simulated human interaction."""
  print("=" * 60)
  print("Composable HITL Orchestrated Pipeline")
  print("=" * 60)

  session_service = InMemorySessionService()
  interrupt_service = InterruptService()
  session_id = "hitl-orchestrated-demo"

  graph = build_orchestrator(interrupt_service=interrupt_service)

  await session_service.create_session(
      app_name="document_pipeline",
      user_id="reviewer",
      session_id=session_id,
  )

  runner = Runner(
      app_name="document_pipeline",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  interrupt_service.register_session(session_id)

  # Simulate: reject extract stage once, then approve
  await interrupt_service.send_message(
      session_id,
      "Missing key entity 'revenue' — please include financial data",
      action="revise",
  )
  await interrupt_service.send_message(
      session_id,
      "Extraction looks complete now",
      action="approve",
  )

  document = (
      "Q4 2025 Financial Report: Revenue grew 23% YoY. "
      "Key entities include quarterly data, revenue metrics, "
      "and department-level breakdowns. Extract all financial entities."
  )

  print(f"\nDocument: {document[:80]}...")
  print("-" * 40)

  async for event in runner.run_async(
      user_id="reviewer",
      session_id=session_id,
      new_message=types.Content(parts=[types.Part(text=document)]),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    if not text or text.startswith("[GraphMetadata]"):
      continue
    author = event.author or "system"
    print(f"[{author}] {text[:300]}")

  # Show final state
  session = await session_service.get_session(
      app_name="document_pipeline",
      user_id="reviewer",
      session_id=session_id,
  )
  if session:
    graph_data = session.state.get("graph_data", {})
    print(
        f"\nPipeline result: {graph_data.get('pipeline_result', 'N/A')[:200]}"
    )
    print(f"Stages: {graph_data.get('stages', [])}")
    debug_output = graph_data.get("_debug_process_output", "")
    if debug_output:
      print(f"Debug (nested output): {debug_output[:100]}")

  interrupt_service.unregister_session(session_id)
  print("\nDone.")


if __name__ == "__main__":
  asyncio.run(main())
