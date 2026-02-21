#!/usr/bin/env python3
"""DynamicNode Pattern: Runtime Agent Selection

Motivation (Mixture-of-Experts)
--------------------------------
Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer" showed that routing inputs to specialised experts
beats a single monolithic model while keeping per-token compute fixed.

The same principle applies to agentic workflows: a *router* classifies the
complexity/type of each task, then a *gating function* selects the cheapest
adequate specialist—a fast flash-model for simple tasks, a slower pro-model
for hard tasks.

Pattern: DynamicNode
--------------------
DynamicNode is the first-class API for this pattern.  The `agent_selector`
callable runs at runtime, reads the current GraphState, and returns the
appropriate BaseAgent.

Compare to the function-node alternative
-----------------------------------------
Without DynamicNode you need a function node that manually dispatches:

    async def dispatch(state, ctx):
        agent = complex_agent if "hard" in state.data else simple_agent
        node_ctx = ctx.model_copy(update={...})
        output = ""
        async for event in agent.run_async(node_ctx):
            if event.content and event.content.parts:
                output = event.content.parts[0].text or ""
        return output

DynamicNode gives you:
    ✅ Metadata auto-tracking: which agent was selected (observability)
    ✅ Built-in fallback_agent when selector returns None
    ✅ Selection logic decoupled from execution boilerplate

Architecture
------------
    classify ──► route (DynamicNode) ──► end
                  │
                  ├─ selector returns simple_agent (flash, cheap)
                  └─ selector returns detailed_agent (pro, thorough)
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.agents.graph import DynamicNode
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Step 1: Classifier — assigns complexity label from the user's request
# ---------------------------------------------------------------------------
classifier = LlmAgent(
    name="classifier",
    model=_MODEL,
    instruction="""
You are a task complexity classifier.

Read the user's request and reply with EXACTLY one word:
  SIMPLE   – if the task is a quick factual lookup or short question
  COMPLEX  – if the task requires multi-step reasoning, analysis, or code

Reply with only the word, nothing else.
""",
)

# ---------------------------------------------------------------------------
# Step 2: Specialists — cheap flash model vs thorough pro model
# ---------------------------------------------------------------------------
simple_agent = LlmAgent(
    name="simple_responder",
    model=_MODEL,
    instruction="""
You are a concise assistant. Answer the user's question briefly (1-3 sentences).
""",
)

detailed_agent = LlmAgent(
    name="detailed_responder",
    model=_MODEL,
    instruction="""
You are a thorough analyst. Work through the problem step by step, show your
reasoning, and provide a complete, well-structured answer.
""",
)


# ---------------------------------------------------------------------------
# Step 3: Agent selector — called at runtime with current GraphState
# ---------------------------------------------------------------------------
def select_responder(state: GraphState) -> LlmAgent:
  """Route to simple_agent for SIMPLE tasks, detailed_agent otherwise.

  The classifier stored its output in state.data["classify"] via the
  default output_mapper (OVERWRITE reducer, key = node name).
  """
  classification = state.data.get("classify", "").upper()
  if "SIMPLE" in classification:
    return simple_agent
  return detailed_agent


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> GraphAgent:
  graph = GraphAgent(
      name="dynamic_routing",
      description="Routes each query to the cheapest adequate specialist",
  )

  # Node 1: classify complexity
  graph.add_node("classify", agent=classifier)

  # Node 2: DynamicNode selects the right specialist at runtime
  graph.add_node(
      DynamicNode(
          name="respond",
          agent_selector=select_responder,
          fallback_agent=simple_agent,  # safety net if selector returns None
      )
  )

  graph.add_edge("classify", "respond")
  graph.set_start("classify")
  graph.set_end("respond")
  return graph


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------
_graph = build_graph()


async def run(question: str) -> str:
  graph = _graph
  svc = InMemorySessionService()
  runner = Runner(
      app_name="dynamic_node_example", agent=graph, session_service=svc
  )
  await svc.create_session(
      app_name="dynamic_node_example", user_id="user", session_id="s1"
  )
  final = ""
  async for event in runner.run_async(
      user_id="user",
      session_id="s1",
      new_message=types.Content(role="user", parts=[types.Part(text=question)]),
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text or ""
      if text and not text.startswith("[GraphMetadata]"):
        final = text
  return final


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main():
  questions = [
      "What is the capital of France?",  # SIMPLE → flash model
      (  # COMPLEX → pro model
          "Explain how transformer attention scales with sequence length "
          "and what architectural changes help address this."
      ),
  ]
  for q in questions:
    print(f"\nQ: {q}")
    answer = await run(q)
    print(f"A: {answer}")


if __name__ == "__main__":
  asyncio.run(main())
