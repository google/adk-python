"""Example 3: Cyclic Graph Execution

Demonstrates:
- Cyclic graphs with conditional back-edges (A -> B -> C -> A loop)
- Two routing patterns depending on execution mode:
  - Default mode: state_delta writes (ADK-standard for deterministic routing)
  - LLM mode: LlmAgent with include_contents='none' + dynamic instructions
    for clean context per iteration (Ralph Loop pattern), with
    output_mappers writing structured state for edge conditions
- Edge conditions reading from GraphState.data
- max_iterations guard to prevent infinite loops

Key design choice (LLM mode):
  Cyclic nodes (step, check) use include_contents='none' to prevent
  session history accumulation across iterations. Without this, the LLM
  sees all previous loop outputs and gets biased toward repeating earlier
  responses (context rot). This is the Ralph Loop pattern applied within
  ADK: each iteration gets clean context, state lives in session.state
  (synced from GraphState.data), not in conversation history.

  Dynamic instructions (callables) read the current counter from
  session.state, providing fresh context per iteration without any
  conversation history leakage.

Run modes:
- Default: python -m contributing.samples.graph_examples.03_cyclic_execution.agent
- LLM: python -m contributing.samples.graph_examples.03_cyclic_execution.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.03_cyclic_execution.agent
"""

import asyncio
import json
import re

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph.graph_state import GraphState
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

MAX_CYCLES = 3


# ===========================
# Deterministic Agents (BaseAgent)
# ===========================


class StartAgent(BaseAgent):
  """Initializes the counter."""

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text="Workflow started")]),
        actions=EventActions(state_delta={"counter": 0}),
    )


class StepAgent(BaseAgent):
  """Increments the counter and persists it via state_delta."""

  async def _run_async_impl(self, ctx):
    counter = ctx.session.state.get("counter", 0) + 1
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[types.Part(text=f"Step executed (counter={counter})")]
        ),
        actions=EventActions(state_delta={"counter": counter}),
    )


class CheckAgent(BaseAgent):
  """Reads counter and writes routing signal via state_delta."""

  async def _run_async_impl(self, ctx):
    counter = ctx.session.state.get("counter", 0)
    status = "CONTINUE" if counter < MAX_CYCLES else "DONE"
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(text=f"Check: counter={counter}, status={status}")
            ]
        ),
        actions=EventActions(state_delta={"status": status}),
    )


class EndAgent(BaseAgent):
  """Signals workflow completion."""

  async def _run_async_impl(self, ctx):
    counter = ctx.session.state.get("counter", 0)
    yield Event(
        author=self.name,
        content=types.Content(
            parts=[
                types.Part(text=f"Workflow complete after {counter} cycle(s)")
            ]
        ),
    )


# ===========================
# LLM Dynamic Instructions
# ===========================
# Callables that read current state per iteration — clean context each time.
# Used with include_contents='none' (Ralph Loop pattern).


def step_instruction(ctx):
  """Dynamic instruction: reads counter from session.state each iteration."""
  counter = ctx.state.get("counter", 0)
  return (
      f"The current counter value is {counter}. "
      f"Increment it by 1 and respond with ONLY the new number, "
      f"nothing else. Just the number."
  )


def check_instruction(ctx):
  """Dynamic instruction: reads counter from session.state each iteration."""
  counter = ctx.state.get("counter", 0)
  return (
      f"The current counter value is {counter} and the threshold is"
      f" {MAX_CYCLES}. If {counter} < {MAX_CYCLES}, respond with exactly:"
      f" CONTINUE. If {counter} >= {MAX_CYCLES}, respond with exactly:"
      f" DONE. One word only."
  )


# ===========================
# LLM Output Mappers
# ===========================
# Parse LLM text output into structured state keys for edge conditions.


def start_output_mapper(output: str, state: GraphState) -> GraphState:
  """Initialize counter=0 in state (LLM agent can't write state_delta)."""
  new_state = GraphState(data=state.data.copy())
  new_state.data["counter"] = 0
  new_state.data["start"] = output.strip()
  return new_state


def step_output_mapper(output: str, state: GraphState) -> GraphState:
  """Parse LLM number output -> counter in state."""
  new_state = GraphState(data=state.data.copy())
  text = str(output).strip()
  match = re.search(r"\d+", text)
  if match:
    counter = int(match.group())
  else:
    counter = state.data.get("counter", 0) + 1
  new_state.data["counter"] = counter
  new_state.data["step"] = f"Step executed (counter={counter})"
  return new_state


def check_output_mapper(output: str, state: GraphState) -> GraphState:
  """Parse LLM CONTINUE/DONE -> status routing signal in state."""
  new_state = GraphState(data=state.data.copy())
  text = str(output).strip().upper()
  status = "DONE" if "DONE" in text else "CONTINUE"
  counter = state.data.get("counter", 0)
  new_state.data["status"] = status
  new_state.data["check"] = f"Check: counter={counter}, status={status}"
  return new_state


# ===========================
# Graph Construction
# ===========================


async def main():
  print("\n" + "=" * 60)
  print("Example 3: Cyclic Graph Execution")
  print("=" * 60 + "\n")

  llm_mode = use_llm_mode()

  # Build graph
  graph = GraphAgent(name="cyclic_workflow", max_iterations=10)

  if llm_mode:
    print("🤖 Using LLM agents (gemini-2.5-flash)\n")

    start = create_llm_agent(
        name="start",
        instruction="Respond with exactly: 'Workflow started'",
    )
    # Cyclic nodes: include_contents='none' prevents session history
    # accumulation. Dynamic instructions read current counter from
    # session.state each iteration (Ralph Loop pattern).
    step = create_llm_agent(
        name="step",
        instruction=step_instruction,
        include_contents="none",
    )
    check = create_llm_agent(
        name="check",
        instruction=check_instruction,
        include_contents="none",
    )
    end = create_llm_agent(
        name="end_node",
        instruction=(
            "A cyclic workflow just completed. Summarize: the workflow ran"
            f" for {MAX_CYCLES} cycles. Respond in one sentence."
        ),
    )

    graph.add_node("start", agent=start, output_mapper=start_output_mapper)
    graph.add_node("step", agent=step, output_mapper=step_output_mapper)
    graph.add_node("check", agent=check, output_mapper=check_output_mapper)
    graph.add_node("end_node", agent=end)
  else:
    print("🎭 Using deterministic agents (BaseAgent)\n")

    start = StartAgent(name="start")
    step = StepAgent(name="step")
    check = CheckAgent(name="check")
    end = EndAgent(name="end_node")

    graph.add_node("start", agent=start)
    graph.add_node("step", agent=step)
    graph.add_node("check", agent=check)
    graph.add_node("end_node", agent=end)

  # Edges are identical in both modes — they read from state.data
  (
      graph.add_edge("start", "step")
      .add_edge("step", "check")
      .add_edge(
          "check",
          "step",
          condition=lambda s: s.data.get("status") == "CONTINUE",
      )
      .add_edge(
          "check",
          "end_node",
          condition=lambda s: s.data.get("status") == "DONE",
      )
      .set_start("start")
      .set_end("end_node")
  )

  # Execute
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="cyclic_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=True,
  )

  print(
      f"Executing cyclic workflow (max_cycles={MAX_CYCLES}, max_iterations=10)"
  )
  print("Graph: start -> step -> check -> (loop back or exit)\n")

  new_message = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id="user1", session_id="session1", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text and "#metadata" not in event.author:
          print(f"   [{event.author}] {part.text}")

  session = await session_service.get_session(
      app_name="cyclic_demo", user_id="user1", session_id="session1"
  )
  final_counter = session.state.get("counter")
  if final_counter is None:
    graph_data_raw = session.state.get("graph_data")
    if graph_data_raw:
      try:
        data = (
            json.loads(graph_data_raw)
            if isinstance(graph_data_raw, str)
            else graph_data_raw
        )
        final_counter = data.get("counter", 0)
      except (json.JSONDecodeError, TypeError):
        final_counter = 0

  if final_counter is None:
    final_counter = 0
  print(f"\n   Final counter value: {final_counter}")
  print(f"   Completed {final_counter} cycle(s) before exiting loop")

  print("\nExample complete!\n")


if __name__ == "__main__":
  asyncio.run(main())
