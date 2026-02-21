"""Example 7: Node Callbacks (before_node_callback / after_node_callback)

Demonstrates:
- Registering before_node_callback and after_node_callback on a GraphAgent
- Measuring per-node execution time with time.perf_counter()
- Callbacks store start times in ctx.metadata and compute elapsed on exit
- Timing results are printed in async def main() after the run completes

Run modes:
- Default: python -m contributing.samples.graph_examples.07_callbacks.agent
- LLM: python -m contributing.samples.graph_examples.07_callbacks.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.07_callbacks.agent
"""

import asyncio
import time

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph.callbacks import NodeCallbackContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

# Shared dict to accumulate timing results from callbacks
_timings: dict[str, float] = {}


# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class FetchAgent(BaseAgent):
  """Simulates a data fetch step."""

  async def _run_async_impl(self, ctx):
    await asyncio.sleep(0.02)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text="Data fetched from source")]
        ),
    )


class ProcessAgent(BaseAgent):
  """Simulates a data processing step."""

  async def _run_async_impl(self, ctx):
    await asyncio.sleep(0.05)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text="Data processed and transformed")]
        ),
    )


class SaveAgent(BaseAgent):
  """Simulates a data persistence step."""

  async def _run_async_impl(self, ctx):
    await asyncio.sleep(0.01)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text="Data saved to storage")]),
    )


async def before_cb(ctx: NodeCallbackContext) -> None:
  """Record start time in shared timings dict keyed by node name."""
  _timings[f"_start_{ctx.node.name}"] = time.perf_counter()
  return None


async def after_cb(ctx: NodeCallbackContext) -> None:
  """Compute elapsed time and store in shared timings dict."""
  start_key = f"_start_{ctx.node.name}"
  start = _timings.get(start_key)
  if start is not None:
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _timings[ctx.node.name] = elapsed_ms
  return None


# ===========================
# Agent Factory
# ===========================


def create_agents():
  """Create agents based on USE_LLM mode.

  Returns:
      tuple: (fetch, process, save) agents
  """
  if use_llm_mode():
    print("🤖 Using LLM-powered agents (gemini-2.5-flash)\n")

    fetch = create_llm_agent(
        name="fetch",
        instruction=(
            "Respond with 'Data fetched from source' exactly. Respond quickly"
            " without delays."
        ),
    )
    process = create_llm_agent(
        name="process",
        instruction=(
            "Respond with 'Data processed and transformed' exactly. Respond"
            " quickly without delays."
        ),
    )
    save = create_llm_agent(
        name="save",
        instruction=(
            "Respond with 'Data saved to storage' exactly. Respond quickly"
            " without delays."
        ),
    )

    return fetch, process, save
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    fetch = FetchAgent(name="fetch")
    process = ProcessAgent(name="process")
    save = SaveAgent(name="save")

    return fetch, process, save


async def main():
  print("\n" + "=" * 60)
  print("Example 7: Node Callbacks")
  print("=" * 60 + "\n")

  # Create agents (deterministic or LLM based on USE_LLM flag)
  fetch, process, save = create_agents()

  # Build graph with before/after callbacks
  graph = (
      GraphAgent(
          name="callback_workflow",
          before_node_callback=before_cb,
          after_node_callback=after_cb,
      )
      .add_node("fetch", agent=fetch)
      .add_node("process", agent=process)
      .add_node("save", agent=save)
      .add_edge("fetch", "process")
      .add_edge("process", "save")
      .set_start("fetch")
      .set_end("save")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="callback_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print("Executing workflow: fetch -> process -> save")
  print("Callbacks will record timing for each node\n")

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"   [{event.author}] {part.text}")

  # Print timing results collected by callbacks
  print("\n   Node execution times (measured by callbacks):")
  for node_name in ["fetch", "process", "save"]:
    elapsed_ms = _timings.get(node_name)
    if elapsed_ms is not None:
      print(f"   [{node_name}] {elapsed_ms:.1f}ms")
    else:
      print(f"   [{node_name}] timing not recorded")

  print("\nExample complete!\n")
  print("   before_node_callback: stores perf_counter start per node")
  print("   after_node_callback:  computes elapsed ms and stores result")


if __name__ == "__main__":
  asyncio.run(main())
