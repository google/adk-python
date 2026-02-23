"""GraphAgent ReAct Pattern example.

Demonstrates the Reasoning + Acting (ReAct) loop using GraphAgent:
  reason → act → observe
  observe loops back to reason if "CONTINUE"
  observe ends if "COMPLETE"

Why GraphAgent (not LoopAgent/SequentialAgent)?
- SequentialAgent: cannot loop; fixed linear path
- LoopAgent: loops unconditionally or escalates; cannot inspect observation
  content to decide direction (reason vs. exit)
- GraphAgent: conditional edges read state → route to any node or exit

Run (requires GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_react_pattern.agent
"""

import asyncio
import os

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import StateReducer
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------


class ObservationResult(BaseModel):
  """Structured observation output from observer agent."""

  status: str  # "continue" or "complete"
  reasoning: str  # Why continue or why complete


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

reasoner = LlmAgent(
    name="reasoner",
    model=_MODEL,
    instruction=(
        "You are a reasoning agent. Analyse the task and any previous "
        "observations, then decide what action to take next. "
        "Write your reasoning in 1-3 sentences."
    ),
    output_key="reasoning",
)

actor = LlmAgent(
    name="actor",
    model=_MODEL,
    instruction=(
        "You are an action agent. Based on the reasoning provided, "
        "answer the question or perform the requested analysis using your "
        "knowledge. Do NOT write code or tool calls — just provide the "
        "factual answer or calculation result directly."
    ),
    output_key="action_result",
)

observer = LlmAgent(
    name="observer",
    model=_MODEL,
    instruction=(
        "You are an observation agent. Evaluate whether the action result "
        "fully answers the original task. "
        'Return {"status": "complete", "reasoning": "..."} if task is done, '
        'or {"status": "continue", "reasoning": "what is missing..."} if not.'
    ),
    output_schema=ObservationResult,  # Structured output
    # output_key auto-defaults to "observer" (agent name)
)


# ---------------------------------------------------------------------------
# Routing predicates
# ---------------------------------------------------------------------------


def _should_continue(state: GraphState) -> bool:
  """Check if ReAct loop should continue using structured output."""
  obs = state.get_parsed("observer", ObservationResult)
  return obs.status.lower() == "continue" if obs else False



# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_react_graph() -> GraphAgent:
  graph = GraphAgent(
      name="react_agent",
      description="ReAct pattern: Reasoning + Acting loop",
      max_iterations=10,
  )

  graph.add_node(
      "reason",
      agent=reasoner,
      input_mapper=lambda s: (
          f"Task: {s.data.get('task', '')}\n"
          f"Previous observation: {s.data.get('observation', 'none')}"
      ),
      reducer=StateReducer.OVERWRITE,
  )
  graph.add_node(
      "act",
      agent=actor,
      input_mapper=lambda s: s.data.get("reasoning", ""),
      reducer=StateReducer.OVERWRITE,
  )
  graph.add_node(
      "observe",
      agent=observer,
      input_mapper=lambda s: (
          f"Task: {s.data.get('task', '')}\n"
          f"Action result: {s.data.get('action_result', '')}"
      ),
      reducer=StateReducer.OVERWRITE,
  )

  graph.set_start("reason")
  graph.add_edge("reason", "act")
  graph.add_edge("act", "observe")

  # Loop back if not yet complete
  graph.add_edge("observe", "reason", condition=_should_continue)

  # Exit when complete
  graph.set_end("observe")

  return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  session_service = InMemorySessionService()
  graph = build_react_graph()

  session = await session_service.create_session(
      app_name="react_agent", user_id="user1"
  )

  task = "What are the key features of the Google Agent Development Kit (ADK)?"
  print(f"Task: {task}\n")

  # Seed the task into initial state (BEFORE calling runner)
  session.state["task"] = task

  # Use Runner instead of manual invocation context
  runner = Runner(
      app_name="react_agent",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  iteration = 0
  async for event in runner.run_async(
      user_id="user1",
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text=task)]),
  ):
    if event.content and event.content.parts:
      author = event.author
      text = event.content.parts[0].text or ""
      if author == "observer":
        iteration += 1
        # Parse from event text (JSON string from output_schema)
        try:
          obs = ObservationResult.model_validate_json(text.strip())
          status = obs.status.upper()
        except ValidationError:
          status = "UNKNOWN (parse error)"
        print(f"[iteration {iteration}] Observer: {status}")
      elif author in ("reasoner", "actor"):
        print(f"  [{author}]: {text[:120]}...")

  # Re-fetch fresh session state (create_session returns a deepcopy)
  fresh_session = await session_service.get_session(
      app_name="react_agent", user_id="user1", session_id=session.id
  )
  if fresh_session is None:
    print(
        "WARNING: session_service.get_session returned None, using stale copy"
    )
    fresh_session = session
  final_data = fresh_session.state.get("graph_data", {})
  final_state = GraphState(data=final_data)

  print("\nFinal observation:")
  obs = final_state.get_parsed("observer", ObservationResult)
  print(f"Status: {obs.status if obs else 'none'}")
  print(f"Reasoning: {obs.reasoning if obs else 'none'}")


if __name__ == "__main__":
  asyncio.run(main())
