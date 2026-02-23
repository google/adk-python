"""GraphAgent HITL Review Workflow — Content generation with human approval loop.

Demonstrates a complete review workflow pattern where human approval is a
first-class part of the graph execution flow (not an ad-hoc interruption):

1. LLM agent drafts content from a topic
2. Human reviews and approves or requests revision
3. If revision needed: loop back with feedback
4. If approved: proceed to publish

Graph structure:
    [draft] --> [review_gate] --> approved? --> [publish]
                      |
                      v rejected
                  [revise] --> [review_gate]  (loop)

Key concepts:
- InterruptService for structured pause/resume at deterministic points
- Conditional routing (approved -> publish, rejected -> revise)
- Execution path tracking shows review iteration history
- Fallback mode: runs without LLM (string templates) when no API key is set

Run:
    # With LLM (set GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=1):
    python -m contributing.samples.graph_agent_hitl_review.agent

    # Without LLM (deterministic fallback, no API key needed):
    python -m contributing.samples.graph_agent_hitl_review.agent
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
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# LLM availability check
# ---------------------------------------------------------------------------


def _has_llm() -> bool:
  """Check if LLM backend is configured via environment variables."""
  return bool(
      os.environ.get("GOOGLE_API_KEY")
      or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
  )


# ---------------------------------------------------------------------------
# Fallback agents (no LLM required)
# ---------------------------------------------------------------------------


class DraftAgent(BaseAgent):
  """Deterministic draft agent — produces content from a template."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  async def _run_async_impl(self, ctx):
    topic = ctx.session.state.get("graph_data", {}).get(
        "input", "unknown topic"
    )
    content = (
        f"Draft: This is an article about '{topic}'.\n"
        "It covers the key concepts, benefits, and practical applications "
        f"of {topic} in modern software engineering."
    )
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=content)]),
    )


class ReviseAgent(BaseAgent):
  """Deterministic revise agent — appends feedback-based revision."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  async def _run_async_impl(self, ctx):
    graph_data = ctx.session.state.get("graph_data", {})
    previous = graph_data.get("content", "")
    feedback = graph_data.get("review_feedback", "no feedback")
    revision = (
        f"{previous}\n\n"
        f"[Revised based on feedback: {feedback}]\n"
        "Additional details and corrections have been incorporated."
    )
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=revision)]),
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def review_gate_fn(state: GraphState, ctx) -> str:
  """Pause for human review and record approval decision.

  This is the core HITL mechanism: the function pauses execution via
  the interrupt service, waits for a human message, and records the
  decision in state.data for conditional routing.
  """
  interrupt_service = getattr(ctx, "_interrupt_service", None)
  session_id = ctx.session.id if ctx.session else None

  if interrupt_service and session_id:
    # Check for pre-queued interrupt message
    message = await interrupt_service.check_interrupt(session_id)
    if message:
      approved = message.action == "approve"
      state.data["approved"] = approved
      state.data["review_feedback"] = message.text
      decision = (
          "APPROVED" if approved else f"REVISION REQUESTED: {message.text}"
      )
      return f"[Review] {decision}"

  # No interrupt service or no message — auto-approve (for testing)
  state.data["approved"] = True
  state.data["review_feedback"] = ""
  return "[Review] Auto-approved (no interrupt service)"


async def publish_fn(state: GraphState, ctx) -> str:
  """Finalize and publish the approved content."""
  content = state.data.get("content", "(no content)")
  state.data["published"] = True
  return f"[Published] {content[:200]}"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_review_graph(
    interrupt_service: InterruptService | None = None,
    use_llm: bool = False,
) -> GraphAgent:
  """Build the content review workflow graph.

  Args:
      interrupt_service: Optional interrupt service for HITL. If None,
          review_gate auto-approves.
      use_llm: If True and LLM is available, use LlmAgent for draft/revise.
          Otherwise use deterministic fallback agents.

  Returns:
      Configured GraphAgent with draft -> review -> publish flow.
  """
  # Select agents based on LLM availability
  if use_llm and _has_llm():
    draft_agent = LlmAgent(
        name="drafter",
        model=_MODEL,
        instruction=(
            "You are a content writer. Write a short article (2-3 paragraphs) "
            "about the topic provided in the user input. Be informative and "
            "engaging."
        ),
        output_key="content",
    )
    revise_agent = LlmAgent(
        name="reviser",
        model=_MODEL,
        instruction=(
            "You are a content editor. Revise the following draft based on "
            "the reviewer's feedback.\n\n"
            "Current draft: {content}\n"
            "Feedback: {review_feedback}\n\n"
            "Produce an improved version."
        ),
        output_key="content",
    )
  else:
    draft_agent = DraftAgent(name="drafter")
    revise_agent = ReviseAgent(name="reviser")

  interrupt_config = None
  if interrupt_service:
    interrupt_config = InterruptConfig(
        mode=InterruptMode.BEFORE,
        nodes=["review_gate"],
    )

  graph = GraphAgent(
      name="content_review",
      description="Content generation with human review loop",
      max_iterations=10,
      interrupt_service=interrupt_service,
      interrupt_config=interrupt_config,
  )

  # Nodes
  graph.add_node(
      "draft",
      agent=draft_agent,
      output_mapper=lambda output, s: (s.data.update({"content": output}), s)[
          1
      ],
  )
  graph.add_node(
      GraphNode(
          name="review_gate",
          function=review_gate_fn,
      )
  )
  graph.add_node(
      "revise",
      agent=revise_agent,
      input_mapper=lambda s: (
          "Revise this draft based on feedback.\n"
          f"Draft: {s.data.get('content', '')}\n"
          f"Feedback: {s.data.get('review_feedback', '')}"
      ),
      output_mapper=lambda output, s: (s.data.update({"content": output}), s)[
          1
      ],
  )
  graph.add_node(
      GraphNode(
          name="publish",
          function=publish_fn,
      )
  )

  # Edges
  graph.set_start("draft")
  graph.add_edge("draft", "review_gate")
  graph.add_edge(
      "review_gate",
      "publish",
      condition=lambda s: s.data.get("approved") is True,
  )
  graph.add_edge(
      "review_gate",
      "revise",
      condition=lambda s: s.data.get("approved") is False,
  )
  graph.add_edge("revise", "review_gate")
  graph.set_end("publish")

  return graph


# ---------------------------------------------------------------------------
# Main — simulates human reviewer
# ---------------------------------------------------------------------------


async def main() -> None:
  """Run content review workflow with simulated human interaction."""
  print("=" * 60)
  print("HITL Content Review Workflow")
  print("=" * 60)

  session_service = InMemorySessionService()
  interrupt_service = InterruptService()
  session_id = "hitl-review-demo"

  graph = build_review_graph(
      interrupt_service=interrupt_service,
      use_llm=_has_llm(),
  )

  await session_service.create_session(
      app_name="content_review",
      user_id="reviewer",
      session_id=session_id,
  )

  runner = Runner(
      app_name="content_review",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  interrupt_service.register_session(session_id)

  # Simulate human: first reject with feedback, then approve on revision
  # Pre-queue messages (in production these come from a UI/API)
  await interrupt_service.send_message(
      session_id,
      "Needs more concrete examples and a conclusion paragraph",
      action="revise",
  )
  await interrupt_service.send_message(
      session_id,
      "Looks good now",
      action="approve",
  )

  print("\nTopic: 'Graph-based AI Agent Workflows'")
  print("-" * 40)

  async for event in runner.run_async(
      user_id="reviewer",
      session_id=session_id,
      new_message=types.Content(
          parts=[types.Part(text="Graph-based AI Agent Workflows")]
      ),
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
      app_name="content_review",
      user_id="reviewer",
      session_id=session_id,
  )
  if session:
    graph_data = session.state.get("graph_data", {})
    print(f"\nPublished: {graph_data.get('published', False)}")
    print(f"Review iterations: {graph_data.get('review_feedback', 'N/A')}")

  interrupt_service.unregister_session(session_id)
  print("\nDone.")


if __name__ == "__main__":
  asyncio.run(main())
