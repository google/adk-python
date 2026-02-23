"""GraphAgent multi-agent research workflow example.

Demonstrates a coordinator → parallel researchers → merger → critic loop:

  coordinator
      ↓
  [researcher_a || researcher_b]  (ParallelNodeGroup, WAIT_ALL)
      ↓
  merger
      ↓
  critic ──REVISE──→ merger
      │
    APPROVED
      ↓
     END

Why GraphAgent (not ParallelAgent/SequentialAgent)?
- SequentialAgent: cannot run researcher_a and researcher_b concurrently.
- ParallelAgent: parallelises but cannot add coordinator before or critic+loop after.
- GraphAgent: combines sequential coordination, true parallel research, AND a
  conditional quality-review loop in one declarative graph.

Run (requires GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_multi_agent.agent
"""

import asyncio
import os

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import JoinStrategy
from google.adk.agents.graph import ParallelNodeGroup
from google.adk.agents.graph import StateReducer
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------


class ReviewResult(BaseModel):
  """Structured review output from critic agent."""

  decision: str  # "approve" or "revise"
  feedback: str  # Review comments


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

coordinator = LlmAgent(
    name="coordinator",
    model=_MODEL,
    instruction=(
        "You are a research coordinator. Given a research topic, split it into"
        " exactly two independent subtopics for parallel investigation. Output"
        " each subtopic on its own line prefixed with 'SUBTOPIC A:' and"
        " 'SUBTOPIC B:'."
    ),
    output_key="subtopics",
)

researcher_a = LlmAgent(
    name="researcher_a",
    model=_MODEL,
    instruction=(
        "You are a researcher specialising in the first subtopic. "
        "Write a concise research summary (3-5 sentences) with key findings."
    ),
    output_key="research_a",
)

researcher_b = LlmAgent(
    name="researcher_b",
    model=_MODEL,
    instruction=(
        "You are a researcher specialising in the second subtopic. "
        "Write a concise research summary (3-5 sentences) with key findings."
    ),
    output_key="research_b",
)

merger = LlmAgent(
    name="merger",
    model=_MODEL,
    instruction=(
        "You are a synthesis expert. Merge the two research summaries into a "
        "single coherent report. Highlight complementary insights."
    ),
    output_key="merged_report",
)

critic = LlmAgent(
    name="critic",
    model=_MODEL,
    instruction=(
        "You are a peer reviewer. Evaluate the merged report for clarity, "
        "completeness, and accuracy. "
        'Return {"decision": "approve", "feedback": "..."} if good, '
        'or {"decision": "revise", "feedback": "explanation..."} if needs work.'
    ),
    output_schema=ReviewResult,  # Structured output
    # output_key auto-defaults to "critic" (agent name)
)


# ---------------------------------------------------------------------------
# Routing predicates
# ---------------------------------------------------------------------------


def _needs_revision(state: GraphState) -> bool:
  """Check if critic requested revision using structured output."""
  review = state.get_parsed("critic", ReviewResult)
  return review.decision.lower() == "revise" if review else False


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_multi_agent_graph() -> GraphAgent:
  graph = GraphAgent(
      name="research_graph",
      description=(
          "Multi-agent research with parallel execution and quality loop"
      ),
      max_iterations=20,
  )

  graph.add_node("coordinator", agent=coordinator)

  graph.add_node(
      "researcher_a",
      agent=researcher_a,
      # Both researchers see the same coordinator output
      input_mapper=lambda s: s.data.get("subtopics", ""),
      reducer=StateReducer.OVERWRITE,
  )
  graph.add_node(
      "researcher_b",
      agent=researcher_b,
      input_mapper=lambda s: s.data.get("subtopics", ""),
      reducer=StateReducer.OVERWRITE,
  )

  graph.add_node(
      "merger",
      agent=merger,
      input_mapper=lambda s: (
          f"Research A:\n{s.data.get('research_a', '')}\n\n"
          f"Research B:\n{s.data.get('research_b', '')}"
      ),
      reducer=StateReducer.OVERWRITE,
  )
  graph.add_node("critic", agent=critic)

  # Register parallel group so branches execute concurrently
  graph.add_parallel_group(
      "researchers",
      ParallelNodeGroup(
          nodes=["researcher_a", "researcher_b"],
          join_strategy=JoinStrategy.WAIT_ALL,
      ),
  )

  graph.set_start("coordinator")
  graph.add_edge("coordinator", "researcher_a")
  graph.add_edge("coordinator", "researcher_b")
  graph.add_edge("researcher_a", "merger")
  graph.add_edge("researcher_b", "merger")
  graph.add_edge("merger", "critic")

  # Quality loop: revise if not approved
  graph.add_edge("critic", "merger", condition=_needs_revision)

  graph.set_end("critic")

  return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  session_service = InMemorySessionService()
  graph = build_multi_agent_graph()

  session = await session_service.create_session(
      app_name="research_graph", user_id="user1"
  )

  topic = "The impact of large language models on software engineering"
  print(f"Research topic: {topic}\n")

  # Use Runner instead of manual invocation context
  runner = Runner(
      app_name="research_graph",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,  # Session already created above
  )

  revision_count = 0
  async for event in runner.run_async(
      user_id="user1",
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text=topic)]),
  ):
    if not event.content or not event.content.parts:
      continue
    author = event.author
    text = event.content.parts[0].text or ""
    if author == "coordinator":
      print("Coordinator assigned subtopics.")
    elif author in ("researcher_a", "researcher_b"):
      print(f"  [{author}] research complete ({len(text)} chars)")
    elif author == "merger":
      revision_count += 1
      print(f"Merger produced report (revision {revision_count}).")
    elif author == "critic":
      # Parse critic output from the event text (JSON string)
      try:
        review = ReviewResult.model_validate_json(text.strip())
        decision = review.decision.upper()
      except Exception:
        decision = "UNKNOWN (parse error)"
      print(f"Critic verdict: {decision}")

  # Re-fetch fresh session state (create_session returns a deepcopy)
  fresh_session = await session_service.get_session(
      app_name="research_graph", user_id="user1", session_id=session.id
  )
  if fresh_session is None:
    print(
        "WARNING: session_service.get_session returned None, using stale copy"
    )
    fresh_session = session
  final_data = fresh_session.state.get("graph_data", {})
  final_state = GraphState(data=final_data)

  print("\nFinal merged report:")
  print(final_state.get_str("merged_report", "(none)")[:500])
  print("\nFinal review:")
  review = final_state.get_parsed("critic", ReviewResult)
  print(f"Decision: {review.decision if review else 'none'}")
  print(f"Feedback: {review.feedback[:200] if review else 'none'}")


if __name__ == "__main__":
  asyncio.run(main())
