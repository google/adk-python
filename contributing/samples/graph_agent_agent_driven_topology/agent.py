"""GraphAgent agent-driven topology: function node injects nodes inside the runner.

Demonstrates the function-node-with-closure pattern for runtime topology
injection, contrasting with graph_agent_dynamic_topology which modifies the
graph from the OUTSIDE (event loop).

Key difference vs graph_agent_dynamic_topology:
- dynamic_topology: outer event loop calls graph.add_node() between events
- THIS sample: a function node inside the runner calls graph.add_node(),
  so topology injection happens entirely within the runner's execution.

How it works:
1. LlmAgent (planner) proposes steps via output_schema → stored in state
2. Function node "topology_applier" closes over the graph reference
3. topology_applier reads state.data["planner"], calls graph.add_node() and
   graph.add_edge() to connect itself → step_1 → step_2 → ... → step_N
4. Graph evaluates the newly added edges on its next routing decision and
   executes the injected step nodes

Why this is safe (cooperative asyncio):
- graph.add_node/add_edge have no locks; they do plain dict writes
- The function node runs synchronously within the async event loop
- No other coroutine can interleave while the function node executes

Flow:
  planner ──▶ topology_applier (fn, closes over graph)
                  ──▶ step_validate ──▶ step_transform ──▶ ... ──▶ END
                      (added at runtime by topology_applier)

Run (requires Vertex AI or GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_agent_driven_topology.agent
"""

import asyncio
import json
import os
from typing import Callable

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.invocation_context import InvocationContext
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


class PipelineDesign(BaseModel):
  """Structured pipeline design from planner agent."""

  steps: list[str]  # Ordered step names discovered at runtime
  description: str  # Pipeline overview


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------

planner = LlmAgent(
    name="planner",
    model=_MODEL,
    instruction=(
        "You are a data pipeline designer. Given a data processing task,"
        " identify 2-4 distinct processing steps required. Each step name"
        " must be a single lowercase word (e.g. validate, transform, enrich,"
        " aggregate, export). Return"
        ' {"steps": ["step1", "step2", ...], "description": "overview"}.'
    ),
    output_schema=PipelineDesign,
    # output_key auto-defaults to "planner"
)


# ---------------------------------------------------------------------------
# Step agent factory
# ---------------------------------------------------------------------------


def make_step_agent(step_name: str) -> LlmAgent:
  """Create a specialized agent for a specific pipeline step."""
  return LlmAgent(
      name=f"{step_name}_agent",
      model=_MODEL,
      instruction=(
          f"You perform the '{step_name}' step of a data pipeline."
          f" Describe what the '{step_name}' step does and confirm it"
          " completed successfully. Be concise (1-2 sentences)."
      ),
      output_key=f"step_{step_name}_result",
  )


# ---------------------------------------------------------------------------
# Function node factory: closes over graph reference
# ---------------------------------------------------------------------------


def make_topology_applier(graph: GraphAgent) -> Callable:
  """Return an async function node that injects pipeline steps into the graph.

  The returned function closes over `graph` so it can call add_node/add_edge
  from inside the runner — no event-loop mediation required.

  Args:
      graph: The GraphAgent to extend at runtime.

  Returns:
      An async function compatible with GraphNode(function=...).
  """

  async def topology_applier(state: GraphState, ctx: InvocationContext) -> str:
    """Read planner output from state and inject nodes into the graph."""
    raw = state.data.get("planner", "{}")
    if isinstance(raw, str):
      try:
        design = json.loads(raw)
      except (json.JSONDecodeError, TypeError):
        design = {}
    else:
      design = raw if isinstance(raw, dict) else {}

    steps = design.get("steps", [])
    description = design.get("description", "")

    if not steps:
      return "No steps discovered; pipeline ends here."

    prev = "topology_applier"
    injected = []

    for step_name in steps:
      node_name = f"step_{step_name}"
      if node_name not in graph.nodes:
        graph.add_node(node_name, agent=make_step_agent(step_name))
        graph.add_edge(prev, node_name)
        injected.append(node_name)
        print(f"  [INJECT] {prev} → {node_name}")
      else:
        print(f"  [SKIP]   {node_name} already exists")
      prev = node_name

    # Mark last step as the terminal node
    graph.set_end(prev)
    print(f"  [END]    {prev}")

    return (
        f"Injected {len(injected)} nodes for pipeline: {description}."
        f" Steps: {steps}"
    )

  return topology_applier


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent_topology_graph(
    session_service: InMemorySessionService,
) -> GraphAgent:
  """Build graph with agent-driven topology injection via function node."""
  checkpoint_service = CheckpointService(session_service=session_service)
  checkpoint_callback = GraphCheckpointCallback(
      checkpoint_service,
      checkpoint_before=False,
      checkpoint_after=True,
      # checkpoint_nodes=None → checkpoint ALL nodes including injected ones
  )

  graph = GraphAgent(
      name="agent_topology_pipeline",
      description=(
          "Adaptive pipeline where a function node injects steps at runtime"
      ),
      max_iterations=20,
      after_node_callback=checkpoint_callback.after_node,
  )

  # Static nodes: planner + topology_applier only
  graph.add_node("planner", agent=planner)
  graph.add_node(
      "topology_applier",
      function=make_topology_applier(graph),  # closes over graph
  )

  graph.set_start("planner")
  graph.add_edge("planner", "topology_applier")
  # topology_applier adds its own edges + set_end at runtime

  return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  print(
      "=== Agent-Driven Topology: Function Node Injects Steps Inside Runner"
      " ===\n"
  )
  print(
      "Contrast with graph_agent_dynamic_topology where the event loop\n"
      "modifies the graph from OUTSIDE the runner.\n"
  )

  session_service = InMemorySessionService()
  graph = build_agent_topology_graph(session_service)
  session_id = "agent-topology-1"

  session = await session_service.create_session(
      app_name="agent_topology_pipeline",
      user_id="user1",
      session_id=session_id,
  )

  runner = Runner(
      app_name="agent_topology_pipeline",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  task = (
      "Process a customer dataset: validate schema, enrich with geo data,"
      " aggregate by region, and export to Parquet"
  )
  print(f"Task: {task}\n")

  step_count = 0

  async for event in runner.run_async(
      user_id="user1",
      session_id=session_id,
      new_message=types.Content(parts=[types.Part(text=task)]),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    author = event.author

    if author == "topology_applier":
      print(f"[topology_applier] {text[:200]}\n")
    elif author.endswith("_agent"):
      step_count += 1
      print(f"[Step {step_count}] {author}: {text[:150]}")

  # Re-fetch session for checkpoint count
  fresh = await session_service.get_session(
      app_name="agent_topology_pipeline",
      user_id="user1",
      session_id=session_id,
  )
  if fresh is None:
    print(
        f"WARNING: session_service.get_session returned None, using stale copy"
    )
    fresh = session
  checkpoints = fresh.state.get("_checkpoint_index", {})
  print(f"\nPipeline complete. Steps executed: {step_count}")
  print(f"Checkpoints created: {len(checkpoints)}")
  print(f"Graph nodes after execution: {list(graph.nodes.keys())}")


if __name__ == "__main__":
  asyncio.run(main())
