#!/usr/bin/env python3
"""DynamicParallelGroup Pattern: Tree of Thoughts / Self-Consistency

Motivation (Tree of Thoughts & Self-Consistency)
--------------------------------------------------
Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning":
sampling multiple independent reasoning paths and taking the majority answer
outperforms single chain-of-thought on arithmetic and commonsense benchmarks.

Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with LLMs":
exploring multiple partial-solution branches in parallel, then scoring and
selecting the best one, significantly improves performance on complex tasks.

Both techniques require spawning N concurrent agents whose count is determined
at runtime — exactly what DynamicParallelGroup provides.

Pattern: DynamicParallelGroup
------------------------------
DynamicParallelGroup generates the agent list at runtime from a callable, runs
them concurrently (with optional max_parallelism throttle), then feeds all
results to an aggregator function.

Compare to the static-parallelism alternative (ParallelNodeGroup)
------------------------------------------------------------------
ParallelNodeGroup is compiled at graph-construction time:

    group = ParallelNodeGroup(
        name="parallel",
        nodes=[node1, node2, node3],   # fixed list — must know N upfront
        join_strategy=JoinStrategy.WAIT_ALL,
    )

DynamicParallelGroup gives you:
    ✅ Runtime-determined N (read from state.data — e.g. user param)
    ✅ State-driven generation (e.g. one agent per item in a list)
    ✅ Custom aggregation logic (majority vote, best-score selection, …)
    ✅ Concurrency cap via max_parallelism (back-pressure / rate limiting)

Architecture (Tree of Thoughts)
---------------------------------
    generate_thoughts (DynamicParallelGroup)
          │  N independent "thought" agents run concurrently
          ▼
       evaluate    ← LlmAgent scores all thoughts
          │
          ▼
        select     ← LlmAgent picks the winning thought
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.agents.graph import DynamicParallelGroup
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph.callbacks import create_nested_observability_callback
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Thought generator — each instance produces one independent solution path
# ---------------------------------------------------------------------------
def make_thought_agent(idx: int) -> LlmAgent:
  return LlmAgent(
      name=f"thought_{idx}",
      model=_MODEL,
      instruction=f"""
You are creative problem-solver #{idx + 1}.  Given a problem, propose one
original, concrete solution approach.  Be specific (≤3 sentences).
Do NOT repeat approaches from other agents; bring a fresh angle.
""",
  )


# ---------------------------------------------------------------------------
# Runtime generator: reads state.data["num_thoughts"] or defaults to 3
# ---------------------------------------------------------------------------
def generate_thought_agents(state: GraphState):
  n = int(state.data.get("num_thoughts", "3"))
  return [make_thought_agent(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Aggregator: concatenate all thoughts with separators for the evaluator
# ---------------------------------------------------------------------------
def aggregate_thoughts(results: list, state: GraphState) -> str:
  lines = []
  for i, r in enumerate(results, 1):
    lines.append(f"=== Thought {i} ===\n{r.strip()}")
  return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator and selector
# ---------------------------------------------------------------------------
evaluator = LlmAgent(
    name="evaluator",
    model=_MODEL,
    instruction="""
You receive several numbered solution approaches.  For EACH one, write a
single line: "Thought N: <score 1-10> – <one-sentence rationale>".
""",
)

selector = LlmAgent(
    name="selector",
    model=_MODEL,
    instruction="""
You receive evaluations of solution approaches.  Choose the best-scoring one
and explain why it is the strongest approach (2-3 sentences).
""",
)


# ---------------------------------------------------------------------------
# Pre-processing: extract [num_thoughts=N] header from user input
# ---------------------------------------------------------------------------
def parse_config(state: GraphState, ctx) -> str:
  """Extract num_thoughts from the input message and store in state.

  Input format: "[num_thoughts=N] <actual problem>"
  Falls back to default 3 if the header is absent.
  """
  import re

  raw = state.data.get("input", "")
  m = re.match(r"\[num_thoughts=(\d+)\]\s*(.*)", raw, re.DOTALL)
  if m:
    state.data["num_thoughts"] = m.group(1)
    return m.group(2).strip()
  return raw


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph() -> GraphAgent:
  graph = GraphAgent(
      name="tree_of_thoughts",
      description="Parallel thought generation → evaluation → selection",
      before_node_callback=create_nested_observability_callback(),
  )

  # Extract num_thoughts config and clean the input text
  graph.add_node("config", function=parse_config)

  graph.add_node(
      DynamicParallelGroup(
          name="generate",
          agent_generator=generate_thought_agents,
          aggregator=aggregate_thoughts,
          max_parallelism=5,  # cap concurrent LLM calls
      )
  )
  graph.add_node("evaluate", agent=evaluator)
  graph.add_node("select", agent=selector)

  graph.add_edge("config", "generate")
  graph.add_edge("generate", "evaluate")
  graph.add_edge("evaluate", "select")
  graph.set_start("config")
  graph.set_end("select")
  return graph


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------
async def run(problem: str, num_thoughts: int = 3) -> str:
  # Encode num_thoughts in the message so state.data["input"] carries it,
  # and also pre-seed state via a thin wrapper node if needed.
  # The simplest approach: embed num_thoughts in a header the agent ignores.
  full_input = f"[num_thoughts={num_thoughts}] {problem}"

  graph = build_graph()
  svc = InMemorySessionService()
  runner = Runner(
      app_name="parallel_group_example", agent=graph, session_service=svc
  )
  await svc.create_session(
      app_name="parallel_group_example", user_id="user", session_id="s1"
  )

  final = ""
  async for event in runner.run_async(
      user_id="user",
      session_id="s1",
      new_message=types.Content(
          role="user", parts=[types.Part(text=full_input)]
      ),
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
      final += text
  return final


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main():
  problem = (
      "How can a small startup with limited budget compete with large "
      "incumbents in the enterprise software market?"
  )
  print(f"Problem: {problem}\n")
  print("Generating 4 parallel thought paths...\n")
  result = await run(problem, num_thoughts=4)
  print(f"Selected best approach:\n{result}")


if __name__ == "__main__":
  asyncio.run(main())
