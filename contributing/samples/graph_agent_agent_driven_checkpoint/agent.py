"""GraphAgent agent-driven checkpoint: LLM proposes checkpoints via state flag.

Demonstrates the checkpoint_request_key pattern where an LLM agent decides
at runtime whether a checkpoint is warranted, rather than checkpointing
every node unconditionally.

Pattern:
- LlmAgent output_schema includes checkpoint_requested: bool = False
- GraphCheckpointCallback(checkpoint_request_key="analyzer.checkpoint_requested")
  reads the flag after the "analyzer" node finishes
- Checkpoint created only when LLM sets the flag (e.g., for high-risk findings)
- Flag clears automatically: StateReducer.OVERWRITE replaces output each run

Flow:
  analyzer → processor → reporter → END
  (may set checkpoint_requested=True based on task risk)

Why this pattern vs checkpointing=True?
- checkpointing=True: checkpoint after EVERY node unconditionally
- checkpoint_request_key: LLM reasons about criticality, checkpoints selectively

Run (requires Vertex AI or GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_agent_driven_checkpoint.agent
"""

import asyncio
import json
import os

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import StateReducer
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.graph.checkpoint_callback import GraphCheckpointCallback
from google.adk.checkpoints import CheckpointService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------


class AnalysisOutput(BaseModel):
  """Structured analysis output that includes a checkpoint proposal."""

  finding: str  # What was found
  risk_level: str  # "low" | "medium" | "high"
  justification: str  # Why this risk level
  checkpoint_requested: bool = False  # LLM sets True for high-risk findings


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _create_agents():
  """Create fresh agent instances (avoids single-parent conflicts)."""
  _analyzer = LlmAgent(
      name="analyzer",
      model=_MODEL,
      instruction=(
          "You are a risk analyzer. Analyze the input task and return"
          " structured JSON. Set checkpoint_requested=true ONLY when risk_level"
          " is 'high' (irreversible or destructive operations). For"
          " low/medium risk, set checkpoint_requested=false to avoid"
          ' unnecessary overhead. Return {"finding": "...", "risk_level":'
          ' "low|medium|high", "justification": "...", "checkpoint_requested":'
          " true|false}."
      ),
      output_schema=AnalysisOutput,
  )
  _processor = LlmAgent(
      name="processor",
      model=_MODEL,
      instruction=(
          "You are an action executor. Based on the analyzer's finding,"
          " describe what action was taken. Be concise (1 sentence)."
      ),
      output_key="processor_result",
  )
  _reporter = LlmAgent(
      name="reporter",
      model=_MODEL,
      instruction=(
          "You are a reporter. Summarize the analysis and action taken in one"
          " sentence for an audit log."
      ),
      output_key="report",
  )
  return _analyzer, _processor, _reporter


def build_agent_checkpoint_graph(
    session_service: InMemorySessionService,
) -> GraphAgent:
  """Build graph with agent-proposed checkpointing."""
  checkpoint_service = CheckpointService(session_service=session_service)

  # Only create checkpoints when the LLM explicitly requests one.
  # checkpoint_after=False disables automatic checkpoints.
  # checkpoint_request_key reads analyzer.checkpoint_requested from state.
  checkpoint_callback = GraphCheckpointCallback(
      checkpoint_service,
      checkpoint_before=False,
      checkpoint_after=False,  # no automatic checkpoints
      checkpoint_request_key="analyzer.checkpoint_requested",
  )

  graph = GraphAgent(
      name="agent_checkpoint_workflow",
      description="Workflow where the LLM decides when checkpoints are needed",
      max_iterations=10,
      after_node_callback=checkpoint_callback.after_node,
  )

  analyzer, processor, reporter = _create_agents()
  graph.add_node("analyzer", agent=analyzer, reducer=StateReducer.OVERWRITE)
  graph.add_node("processor", agent=processor, reducer=StateReducer.OVERWRITE)
  graph.add_node("reporter", agent=reporter, reducer=StateReducer.OVERWRITE)

  graph.set_start("analyzer")
  graph.add_edge("analyzer", "processor")
  graph.add_edge("processor", "reporter")
  graph.set_end("reporter")

  return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_scenario(
    task: str,
    session_service: InMemorySessionService,
    graph: GraphAgent,
    scenario_id: str,
) -> None:
  """Run a single scenario and print results."""
  session = await session_service.create_session(
      app_name="agent_checkpoint_workflow",
      user_id="user1",
      session_id=scenario_id,
  )

  runner = Runner(
      app_name="agent_checkpoint_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  assessment: dict = {}
  report_text: str = ""

  async for event in runner.run_async(
      user_id="user1",
      session_id=scenario_id,
      new_message=types.Content(parts=[types.Part(text=task)]),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    if not text:
      continue

    if event.author == "analyzer":
      try:
        assessment = json.loads(text)
      except (json.JSONDecodeError, TypeError):
        assessment = {}
      risk = assessment.get("risk_level", "?").upper()
      requested = assessment.get("checkpoint_requested", False)
      flag = " [CHECKPOINT REQUESTED]" if requested else ""
      print(f"  Analyzer: risk={risk}{flag}")
      print(f"    {assessment.get('justification', '')[:100]}")
    elif event.author == "reporter":
      report_text += text
      print(f"  Report: {text[:120]}")

  # Re-fetch to get updated checkpoint count
  fresh = await session_service.get_session(
      app_name="agent_checkpoint_workflow",
      user_id="user1",
      session_id=scenario_id,
  )
  if fresh is None:
    print(
        f"WARNING: session_service.get_session returned None, using stale copy"
    )
    fresh = session
  checkpoints = fresh.state.get("_checkpoint_index", {})
  requested_cps = {k for k in checkpoints if k.endswith("-requested")}
  print(
      f"  Checkpoints: {len(checkpoints)} total,"
      f" {len(requested_cps)} agent-requested"
  )


async def main() -> None:
  print("=== Agent-Driven Checkpoint: LLM Decides When to Checkpoint ===\n")
  print(
      "checkpoint_request_key='analyzer.checkpoint_requested' — only the LLM"
      " can trigger a checkpoint\n"
  )

  scenarios = [
      ("low", "Read the README file to understand the project structure"),
      (
          "medium",
          "Update the application config to change the log level to DEBUG",
      ),
      (
          "high",
          "Delete all rows from the users table in the production database",
      ),
  ]

  for label, task in scenarios:
    session_service = InMemorySessionService()
    graph = build_agent_checkpoint_graph(session_service)
    print(f"[{label.upper()} RISK] {task}")
    await run_scenario(task, session_service, graph, f"scenario-{label}")
    print()

  print(
      "Summary: only high-risk tasks should have agent-requested checkpoints."
  )


if __name__ == "__main__":
  asyncio.run(main())
