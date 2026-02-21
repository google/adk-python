#!/usr/bin/env python3
"""NestedGraphNode Pattern: Hierarchical Workflow Composition

Motivation (Hierarchical Planning)
------------------------------------
Hierarchical planning research (Sutton et al. 1999, "Between MDPs and
semi-MDPs") and modern LLM orchestration papers like ORION / LLM Compiler
(Kim et al. 2023) show that breaking complex tasks into nested sub-plans
improves both reasoning quality and task-level reuse.

In agentic terms: a *coordinator* decomposes the top-level goal into
sub-problems, each solved by a *sub-workflow* (a full GraphAgent) that can
be developed, tested, and reused independently.

Pattern: NestedGraphNode
------------------------
NestedGraphNode runs an entire GraphAgent as a single step inside a parent
graph.  The parent graph sees only the sub-workflow's final output — internal
steps are transparent.

Compare to the function-node alternative
-----------------------------------------
Without NestedGraphNode you must manually plumb the nested graph:

    async def run_research_step(state, ctx):
        sub_ctx = ctx.model_copy(update={...})
        result = ""
        async for event in research_graph._run_async_impl(sub_ctx):
            if event.content and event.content.parts:
                result = event.content.parts[0].text or ""
        return result

NestedGraphNode gives you:
    ✅ Session-inheritance control (inherit_session=True/False)
    ✅ Automatic metadata: sub-graph iteration count + execution path
    ✅ No manual context plumbing

Architecture (two-level hierarchy)
------------------------------------
Outer graph:
    plan ──► [research_step (NestedGraphNode)] ──► synthesize

Inner (research_step) sub-graph:
    search ──► extract ──► summarise
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import NestedGraphNode
from google.adk.agents.graph.callbacks import create_nested_observability_callback
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Inner sub-graph: a three-step research pipeline
# (search → extract key claims → summarise for the parent)
# ---------------------------------------------------------------------------
_searcher = LlmAgent(
    name="searcher",
    model=_MODEL,
    instruction="""
You are a research agent. Given a topic, produce 3-5 bullet-point facts
that are accurate and concise.  Start each bullet with "•".
""",
)

_extractor = LlmAgent(
    name="extractor",
    model=_MODEL,
    instruction="""
You receive bullet-point facts. Extract the 2 most important claims and
rewrite them as complete sentences.  Label them CLAIM-1 and CLAIM-2.
""",
)

_summariser = LlmAgent(
    name="summariser",
    model=_MODEL,
    instruction="""
You receive two claims. Write a single-paragraph research summary (≤4
sentences) that a domain expert would find useful.
""",
)


def build_research_subgraph() -> GraphAgent:
  """Returns a reusable three-step research sub-workflow."""
  g = GraphAgent(
      name="research_pipeline",
      before_node_callback=create_nested_observability_callback(),
  )
  g.add_node("search", agent=_searcher)
  g.add_node("extract", agent=_extractor)
  g.add_node("summarise", agent=_summariser)
  g.add_edge("search", "extract")
  g.add_edge("extract", "summarise")
  g.set_start("search")
  g.set_end("summarise")
  return g


# ---------------------------------------------------------------------------
# Outer graph: plan → nested research → final synthesis
# ---------------------------------------------------------------------------
_planner = LlmAgent(
    name="planner",
    model=_MODEL,
    instruction="""
You receive a broad research question.  Restate it as a focused single-topic
query suitable for a research assistant (one sentence, no preamble).
""",
)

_synthesiser = LlmAgent(
    name="synthesiser",
    model=_MODEL,
    instruction="""
You receive a research summary.  Write a concise final answer (2-3 sentences)
to the original user question, citing the key findings.
""",
)


def build_graph() -> GraphAgent:
  outer = GraphAgent(
      name="hierarchical_research",
      description="Coordinator → research sub-workflow → synthesis",
      before_node_callback=create_nested_observability_callback(),
  )

  outer.add_node("plan", agent=_planner)

  # The entire inner research pipeline runs as a single node
  outer.add_node(
      NestedGraphNode(
          name="research",
          graph_agent=build_research_subgraph(),
          inherit_session=True,  # share parent session → state visible to outer
      )
  )

  outer.add_node("synthesise", agent=_synthesiser)

  outer.add_edge("plan", "research")
  outer.add_edge("research", "synthesise")
  outer.set_start("plan")
  outer.set_end("synthesise")
  return outer


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------
async def run(question: str) -> str:
  graph = build_graph()
  svc = InMemorySessionService()
  runner = Runner(
      app_name="nested_graph_example", agent=graph, session_service=svc
  )
  await svc.create_session(
      app_name="nested_graph_example", user_id="user", session_id="s1"
  )
  final = ""
  async for event in runner.run_async(
      user_id="user",
      session_id="s1",
      new_message=types.Content(role="user", parts=[types.Part(text=question)]),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    if not text:
      continue
    if event.author == "observability":
      # Show node execution trace (from create_nested_observability_callback)
      print(f"  → {text}")
    elif not text.startswith("[GraphMetadata]"):
      final = text
  return final


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main():
  q = "What are the key developments in quantum computing hardware?"
  print(f"Question: {q}\n")
  answer = await run(q)
  print(f"Answer:\n{answer}")


if __name__ == "__main__":
  asyncio.run(main())
