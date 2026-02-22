"""GraphAgent Human-In-The-Loop (HITL) with automatic checkpointing.

Demonstrates:
- Agent-driven risk assessment (agent decides when human approval is needed)
- InterruptService for runtime human messages
- InterruptReasoner for LLM-based approval decisions
- GraphCheckpointCallback for selective node-level checkpointing
- State preservation across human interactions

Flow:
  analyze → (checkpoint) → execute → (checkpoint) → END
      ↓
  [interrupt_config fires if risk_level == "high"]
      ↓
  InterruptReasoner processes human feedback
      ↓
  "continue" → execute; "pause" → stop

Why GraphAgent (not SequentialAgent)?
- SequentialAgent: cannot inspect agent output to conditionally interrupt
- GraphAgent: interrupt_config + interrupt_service read state → route or pause

Why GraphCheckpointCallback over checkpointing=True?
- checkpointing=True: checkpoints EVERY node
- GraphCheckpointCallback(checkpoint_nodes=...): checkpoints ONLY critical nodes

Run (requires GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_hitl.agent
"""

import asyncio
import json
import os

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph import InterruptReasoner
from google.adk.agents.graph import InterruptReasonerConfig
from google.adk.agents.graph import InterruptService
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


class RiskAssessment(BaseModel):
  """Structured risk assessment from analyzer agent."""

  action: str  # Description of the action to take
  risk_level: str  # "low" | "medium" | "high"
  justification: str  # Why this risk level was assigned


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _create_agents():
  """Create fresh agent instances (avoids single-parent conflicts)."""
  _analyzer = LlmAgent(
      name="analyzer",
      model=_MODEL,
      instruction=(
          "You are a risk assessment agent. Analyze the requested action and"
          " determine its risk level."
          ' Return {"action": "description of action", "risk_level":'
          ' "low|medium|high", "justification": "reason for risk level"}.'
          " High risk: irreversible actions (delete, overwrite, deploy to"
          " production). Low risk: read-only or reversible actions."
      ),
      output_schema=RiskAssessment,
  )
  _executor = LlmAgent(
      name="executor",
      model=_MODEL,
      instruction=(
          "You are an action executor. Confirm the action has been executed"
          " successfully and describe what was done."
      ),
      output_key="execution_result",
  )
  return _analyzer, _executor


# ---------------------------------------------------------------------------
# Interrupt reasoner
# ---------------------------------------------------------------------------

# LLM-based reasoner: processes human feedback and decides next action
# "continue" → proceed with execution
# "pause" → stop execution (escalate=True)
approval_reasoner = InterruptReasoner(
    InterruptReasonerConfig(
        model=_MODEL,
        available_actions=["continue", "pause"],
        instruction=(
            "You process human approval decisions for a risk-aware workflow."
            " If the human approves the action, return 'continue'."
            " If the human rejects or requests more review, return 'pause'."
        ),
    )
)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_hitl_graph(
    session_service: InMemorySessionService,
) -> tuple[GraphAgent, InterruptService]:
  """Build HITL graph with checkpointing and interrupt support."""
  interrupt_service = InterruptService()

  checkpoint_service = CheckpointService(session_service=session_service)
  # Only checkpoint the two critical nodes (not every node)
  checkpoint_callback = GraphCheckpointCallback(
      checkpoint_service,
      checkpoint_before=False,  # Only after
      checkpoint_after=True,
      checkpoint_nodes={"analyze", "execute"},
  )

  graph = GraphAgent(
      name="hitl_workflow",
      description=(
          "Risk-aware workflow with human approval for high-risk actions"
      ),
      max_iterations=10,
      interrupt_service=interrupt_service,
      interrupt_config=InterruptConfig(
          # Interrupt BEFORE execute for high-risk actions
          mode=InterruptMode.BEFORE,
          nodes=["execute"],  # Only check before "execute" node
          reasoner=approval_reasoner,
      ),
      after_node_callback=checkpoint_callback.after_node,
  )

  analyzer, executor = _create_agents()
  graph.add_node("analyze", agent=analyzer)
  graph.add_node(
      "execute",
      agent=executor,
      input_mapper=lambda s: (
          f"Execute this action: {s.data.get('analyzer', {}).get('action', '')}"
      ),
  )

  graph.set_start("analyze")
  graph.add_edge("analyze", "execute")
  graph.set_end("execute")

  return graph, interrupt_service


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_with_approval(
    request: str,
    approval_message: str,
    session_id: str,
) -> None:
  """Run HITL workflow with simulated human approval.

  In production: the approval_message would come from a real human
  (e.g., via API endpoint, Slack bot, or review UI).

  Args:
      request: The action to analyze and potentially execute
      approval_message: Human's approval feedback (simulated here)
      session_id: Unique session identifier
  """
  session_service = InMemorySessionService()
  graph, interrupt_service = build_hitl_graph(session_service)

  session = await session_service.create_session(
      app_name="hitl_workflow", user_id="user1", session_id=session_id
  )

  runner = Runner(
      app_name="hitl_workflow",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  # Register session with interrupt service BEFORE running
  interrupt_service.register_session(session_id)

  # Pre-queue human approval message (simulates human reviewing and approving)
  # In production: send this message from a separate process/API after
  # the graph escalates and notifies a human reviewer.
  await interrupt_service.send_message(
      session_id,
      approval_message,
      action="continue",  # "continue" = approve, "pause" = reject
  )

  print(f"\nRequest: {request}")
  print(f"Human approval pre-queued: '{approval_message}'")
  print("-" * 50)

  last_assessment: dict = {}
  last_execution: str = ""

  async for event in runner.run_async(
      user_id="user1",
      session_id=session_id,
      new_message=types.Content(parts=[types.Part(text=request)]),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    if not text:
      continue

    author = event.author
    if author == "analyzer":
      # output_schema serialises structured output as JSON text
      try:
        last_assessment = json.loads(text)
      except (json.JSONDecodeError, TypeError):
        last_assessment = {}
      risk = last_assessment.get("risk_level", "unknown").upper()
      action = last_assessment.get("action", "")
      print(f"[Analyzer] Risk: {risk} | Action: {action}")
    elif author == "executor":
      last_execution = text
      print(f"[Executor] {text[:200]}")
    elif author == "interrupt_reasoner":
      print(f"[Interrupt Reasoner] Decision: {text[:100]}")
    elif author == "checkpoint_service":
      print(f"[Checkpoint] Saved state at current node")

  # Show final state (tracked from events to avoid stale session reference)
  print(f"\nRisk level: {last_assessment.get('risk_level', 'N/A')}")
  print(
      f"Result: {last_execution[:200] if last_execution else '(not executed)'}"
  )

  # Re-fetch session: InMemorySessionService returns deepcopies so the local
  # `session` reference is stale. The runner's internal copy holds checkpoint state.
  fresh_session = await session_service.get_session(
      app_name="hitl_workflow", user_id="user1", session_id=session_id
  )
  if fresh_session is None:
    print(
        f"WARNING: session_service.create_session returned None, using stale"
        f" copy"
    )
    fresh_session = session
  checkpoints = fresh_session.state.get("_checkpoint_index", {})
  print(f"Checkpoints created: {len(checkpoints)}")

  # Cleanup
  interrupt_service.unregister_session(session_id)


async def main() -> None:
  print(
      "=== HITL Workflow: Agent-Driven Risk Assessment + Human Approval ===\n"
  )

  # Low risk: auto-approved (agent decides low risk, interrupt fires but
  # approval is already in queue)
  await run_with_approval(
      request="Read the contents of the config.yaml file",
      approval_message="Approved - read-only operation is safe",
      session_id="hitl-low-risk",
  )

  print("\n" + "=" * 60 + "\n")

  # High risk: human must review (pre-queued approval simulates human review)
  await run_with_approval(
      request="Delete all records from the production database",
      approval_message="Approved with caution - backup confirmed",
      session_id="hitl-high-risk",
  )


if __name__ == "__main__":
  asyncio.run(main())
