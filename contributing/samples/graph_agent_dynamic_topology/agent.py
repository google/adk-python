"""GraphAgent dynamic topology: add nodes and edges at runtime.

Demonstrates adding entirely new nodes and edges AFTER graph construction
based on runtime decisions, with checkpointing of each processing step.

Difference from DynamicNode (patterns.py):
- DynamicNode: selects WHICH agent runs at a FIXED node position
- DynamicTopology: adds NEW nodes/edges to the graph at runtime

Use case: Adaptive pipeline that discovers required processing steps at
runtime (e.g., an LLM decides ETL steps needed for a given dataset).

Flow:
  planner ──(discovers steps)──→ [step_1, step_2, ...step_N]
     |                                      |
  checkpoint                          checkpoint (each step)
                                            |
                                          END

Why GraphAgent (not SequentialAgent)?
- SequentialAgent: fixed sequence of nodes defined at construction time
- GraphAgent: nodes and edges can be added at runtime before re-running

Run (requires GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_dynamic_topology.agent
"""

import asyncio
import json
import os

from google.adk.agents.graph import GraphAgent
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

  steps: list[str]  # Ordered list of processing step names
  description: str  # Brief description of the overall pipeline


# ---------------------------------------------------------------------------
# Planner agent (designs the pipeline steps at runtime)
# ---------------------------------------------------------------------------

planner = LlmAgent(
    name="planner",
    model=_MODEL,
    instruction=(
        "You are a data pipeline designer. Given a data processing task,"
        " identify 2-4 distinct processing steps required (e.g., 'validate',"
        " 'transform', 'enrich', 'aggregate', 'export'). Return {\"steps\":"
        ' ["step1", "step2", ...], "description": "pipeline overview"}.'
    ),
    output_schema=PipelineDesign,
    # output_key auto-defaults to "planner" (agent name)
)


# ---------------------------------------------------------------------------
# Step agent factory
# ---------------------------------------------------------------------------


def make_step_agent(step_name: str) -> LlmAgent:
  """Create a specialized agent for a specific pipeline step."""
  safe_name = step_name.replace(" ", "_").replace("-", "_")
  return LlmAgent(
      name=f"{safe_name}_agent",
      model=_MODEL,
      instruction=(
          f"You are a data processing agent performing the '{step_name}' step."
          f" Describe what the '{step_name}' step does to the input data and"
          " confirm it was completed successfully. Be concise (1-2 sentences)."
      ),
      output_key=f"step_{step_name}_result",
  )


# ---------------------------------------------------------------------------
# Graph builder with dynamic topology
# ---------------------------------------------------------------------------


def build_base_graph(
    session_service: InMemorySessionService,
) -> GraphAgent:
  """Build the initial graph with only the planner node."""
  checkpoint_service = CheckpointService(session_service=session_service)
  # Checkpoint after planner (topology decision) and after each dynamic step
  # We pass all=True initially; will filter in callback based on step names
  checkpoint_callback = GraphCheckpointCallback(
      checkpoint_service,
      checkpoint_before=False,
      checkpoint_after=True,
      # checkpoint_nodes=None means: checkpoint ALL nodes (planner + all steps)
  )

  graph = GraphAgent(
      name="dynamic_pipeline",
      description="Adaptive data pipeline with runtime-discovered steps",
      max_iterations=20,
      after_node_callback=checkpoint_callback.after_node,
  )

  graph.add_node("planner", agent=planner)
  graph.set_start("planner")
  # No edges or end node yet - will be added dynamically

  return graph


async def extend_graph_with_steps(
    graph: GraphAgent,
    steps: list[str],
) -> None:
  """Add discovered pipeline steps as nodes and edges to the graph.

  This modifies the graph BEFORE the runner continues execution.

  Args:
      graph: The GraphAgent to extend
      steps: Ordered list of step names discovered by planner
  """
  if not steps:
    return

  # Remove existing end nodes (planner was a dead end without steps)
  # Set the first dynamic step as the next node after planner
  prev_node = "planner"

  for i, step_name in enumerate(steps):
    safe_step = step_name.replace(" ", "_").replace("-", "_")
    node_name = f"step_{safe_step}"

    # Skip if node already exists (idempotent)
    if node_name in graph.nodes:
      print(f"  [TOPOLOGY] Node '{node_name}' already exists, skipping")
      prev_node = node_name
      continue

    # Create and add agent for this step
    step_agent = make_step_agent(step_name)
    graph.add_node(node_name, agent=step_agent)
    print(f"  [TOPOLOGY] Added node '{node_name}'")

    # Connect to previous node
    graph.add_edge(prev_node, node_name)
    print(f"  [TOPOLOGY] Added edge '{prev_node}' → '{node_name}'")

    prev_node = node_name

  # Set the last step as the end node
  graph.set_end(prev_node)
  print(f"  [TOPOLOGY] Set '{prev_node}' as end node")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  print("=== Dynamic Topology: Runtime Pipeline Discovery ===\n")

  session_service = InMemorySessionService()
  graph = build_base_graph(session_service)
  session_id = "dynamic-pipeline-1"

  session = await session_service.create_session(
      app_name="dynamic_pipeline", user_id="user1", session_id=session_id
  )

  runner = Runner(
      app_name="dynamic_pipeline",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  task = (
      "Process a CSV dataset: validate schema, transform types, enrich with"
      " external API, and export to JSON"
  )
  print(f"Task: {task}\n")

  topology_extended = False
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

    if author == "planner" and not topology_extended:
      # Planner emits structured JSON as event text (output_schema serialises to string)
      try:
        design = json.loads(text)
      except (json.JSONDecodeError, TypeError):
        design = {}
      steps = design.get("steps", [])
      description = design.get("description", "")

      print(f"Planner designed pipeline: {description}")
      print(f"Steps discovered: {steps}\n")

      # Dynamically add discovered steps to the graph
      await extend_graph_with_steps(graph, steps)
      topology_extended = True
      print()

    elif author.startswith("step_") or author.endswith("_agent"):
      step_count += 1
      print(f"[Step {step_count}] {author}: {text[:150]}")

    elif author == "checkpoint_service":
      pass  # Silent checkpoint acknowledgment

  # Show final state
  final_state = session.state.get("graph_data", {})
  print(f"\nPipeline completed. Steps executed: {step_count}")

  # Re-fetch session: InMemorySessionService returns deepcopies so the local
  # `session` reference is stale.
  fresh_session = await session_service.get_session(
      app_name="dynamic_pipeline", user_id="user1", session_id=session_id
  )
  if fresh_session is None:
    print(
        f"WARNING: session_service.create_session returned None, using stale"
        f" copy"
    )
    fresh_session = session
  checkpoints = fresh_session.state.get("_checkpoint_index", {})
  print(f"Checkpoints created: {len(checkpoints)}")
  print("\nStep results:")
  for key, value in final_state.items():
    if key.startswith("step_") and key.endswith("_result"):
      step_name = key[5:-7]  # strip "step_" prefix and "_result" suffix
      print(f"  {step_name}: {str(value)[:100]}")


if __name__ == "__main__":
  asyncio.run(main())
