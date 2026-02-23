#!/usr/bin/env python3
"""Dynamic Task Queue Pattern Example

Demonstrates how to implement AI Co-Scientist pattern using GraphAgent
with a function node that dynamically dispatches to different agents.

This example shows:
1. Dynamic task queue management
2. Runtime agent dispatch based on task type
3. Dynamic task generation from agent outputs
4. State-based loop control
"""

import asyncio
import re
from typing import Any
from typing import Dict
from typing import List

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


# Mock agents for demonstration (replace with real agents)
class MockGenerationAgent(BaseAgent):
  """Mock agent that generates hypotheses."""

  async def _run_async_impl(self, ctx: InvocationContext):
    input_text = ""
    if ctx.user_content and ctx.user_content.parts:
      input_text = ctx.user_content.parts[0].text or ""

    # Simulate hypothesis generation
    hypothesis = f"Hypothesis: {input_text} leads to interesting results"

    # Generate follow-up tasks
    output = f"""{hypothesis}

TODO: review this hypothesis
TODO: experiment with variation A"""

    yield Event(
        author=self.name,
        content=types.Content(
            role="assistant", parts=[types.Part(text=output)]
        ),
    )


class MockReviewAgent(BaseAgent):
  """Mock agent that reviews hypotheses."""

  async def _run_async_impl(self, ctx: InvocationContext):
    input_text = ""
    if ctx.user_content and ctx.user_content.parts:
      input_text = ctx.user_content.parts[0].text or ""

    # Simulate review
    review = f"Review: {input_text} - APPROVED with score 8/10"

    yield Event(
        author=self.name,
        content=types.Content(
            role="assistant", parts=[types.Part(text=review)]
        ),
    )


class MockExperimentAgent(BaseAgent):
  """Mock agent that runs experiments."""

  async def _run_async_impl(self, ctx: InvocationContext):
    input_text = ""
    if ctx.user_content and ctx.user_content.parts:
      input_text = ctx.user_content.parts[0].text or ""

    # Simulate experiment
    result = f"Experiment: {input_text} - SUCCESS (confidence: 0.92)"

    yield Event(
        author=self.name,
        content=types.Content(
            role="assistant", parts=[types.Part(text=result)]
        ),
    )


# Initialize worker agents
generation_agent = MockGenerationAgent(name="generation_agent")
review_agent = MockReviewAgent(name="review_agent")
experiment_agent = MockExperimentAgent(name="experiment_agent")


def parse_new_tasks_from_result(result: str) -> List[Dict[str, str]]:
  """Extract TODO tasks from agent output.

  Looks for lines like:
  - TODO: review X
  - TODO: experiment with Y

  Returns:
      List of task dicts: [{"type": "review", "data": "X"}, ...]
  """
  tasks = []
  todo_pattern = r"TODO:\s*(review|experiment)\s+(.+)"

  for match in re.finditer(todo_pattern, result, re.IGNORECASE):
    task_type = match.group(1).lower()
    task_data = match.group(2).strip()
    tasks.append({"type": task_type, "data": task_data})

  return tasks


async def dynamic_task_dispatcher(
    state: GraphState, ctx: InvocationContext
) -> Dict[str, Any]:
  """Dispatch to agents based on dynamic task queue.

  This function:
  1. Reads task queue from state
  2. Pops next task
  3. Dispatches to appropriate agent
  4. Updates queue with any new tasks generated
  5. Returns updated state
  """
  task_queue = state.data.get("task_queue", [])

  if not task_queue:
    print("✅ Task queue empty - all tasks complete!")
    return {"all_complete": True, "tasks_remaining": 0}

  # Pop next task
  next_task = task_queue.pop(0)
  task_type = next_task["type"]
  task_data = next_task["data"]

  print(f"\n🔄 Processing task: [{task_type}] {task_data}")

  # Dynamic agent dispatch based on task type
  if task_type == "generate":
    agent = generation_agent
  elif task_type == "review":
    agent = review_agent
  elif task_type == "experiment":
    agent = experiment_agent
  else:
    raise ValueError(f"Unknown task type: {task_type}")

  # Create context for agent with task data
  agent_ctx = ctx.model_copy(
      update={
          "user_content": types.Content(
              role="user", parts=[types.Part(text=task_data)]
          )
      }
  )

  # Execute agent and collect result
  result = ""
  async for event in agent.run_async(agent_ctx):
    if event.content and event.content.parts:
      result += event.content.parts[0].text or ""

  print(f"   Result: {result[:100]}...")

  # Parse result for new tasks (dynamic task generation!)
  new_tasks = parse_new_tasks_from_result(result)
  if new_tasks:
    print(f"   Generated {len(new_tasks)} new tasks: {new_tasks}")
    task_queue.extend(new_tasks)

  # Update state
  state.data["task_queue"] = task_queue
  state.data["last_result"] = result
  completed = state.data.setdefault("completed_tasks", [])
  completed.append({"type": task_type, "data": task_data, "result": result})

  print(f"   Tasks remaining: {len(task_queue)}")

  return {"tasks_remaining": len(task_queue), "completed_count": len(completed)}


def build_dynamic_task_queue_graph() -> GraphAgent:
  """Build GraphAgent with dynamic task queue pattern.

  The graph has a single node that loops, processing tasks from a queue
  that can grow dynamically based on agent outputs.
  """
  graph = GraphAgent(
      name="ai_co_scientist",
      max_iterations=20,  # Prevent infinite loops
      description="Dynamic task queue with agent dispatch",
  )

  # Single dispatcher node that processes queue
  graph.add_node("task_dispatcher", function=dynamic_task_dispatcher)

  # Loop back to dispatcher while tasks remain.
  # Check task_queue directly (updated via output_mapper return value).
  # The return dict {"tasks_remaining": N} is stored under state.data["task_dispatcher"]
  # by the output mapper, so state.data.get("tasks_remaining") would always be 0.
  graph.add_edge(
      "task_dispatcher",
      "task_dispatcher",
      condition=lambda state: len(state.data.get("task_queue", [])) > 0,
  )

  graph.set_start("task_dispatcher")
  graph.set_end("task_dispatcher")  # Terminal when no edge condition matches

  return graph


async def main():
  """Run dynamic task queue example."""
  print("=" * 70)
  print("Dynamic Task Queue Pattern - AI Co-Scientist Example")
  print("=" * 70)

  # Build graph
  graph = build_dynamic_task_queue_graph()

  # Create session service
  session_service = InMemorySessionService()

  # Initialize with starting tasks
  initial_state = GraphState(
      data={
          "task_queue": [
              {"type": "generate", "data": "quantum computing approach"},
              {"type": "generate", "data": "machine learning approach"},
          ],
          "completed_tasks": [],
      }
  )

  # Seed domain data via create_session so it survives the deepcopy:
  # InMemorySessionService always deepcopies on get/create, so setting
  # session.state after create_session() would only mutate the returned copy.
  session = await session_service.create_session(
      app_name="dynamic_queue_demo",
      user_id="demo_user",
      state=initial_state.data,
  )

  # Create runner
  runner = Runner(
      app_name="dynamic_queue_demo",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,  # Session already created
  )

  # Run graph - dispatcher will process queue dynamically
  print("\n📋 Initial task queue:")
  for task in initial_state.data["task_queue"]:
    print(f"   - [{task['type']}] {task['data']}")

  print("\n" + "=" * 70)
  print("Starting execution...")
  print("=" * 70)

  async for event in runner.run_async(
      user_id="demo_user",
      session_id=session.id,
      new_message=types.Content(
          role="user", parts=[types.Part(text="Start task queue processing")]
      ),
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text or ""
      if text and "final_output" in text.lower():
        print(f"\n📊 {text}")

  # Print final statistics (re-fetch — create_session returned a deepcopy)
  fresh_session = await session_service.get_session(
      app_name="dynamic_queue_demo", user_id="demo_user", session_id=session.id
  )
  if fresh_session is None:
    print("\n⚠️  Could not retrieve final session state to print statistics.")
    return
  final_session = fresh_session
  final_data = final_session.state.get("graph_data", {})
  final_state = GraphState(data=final_data) if final_data else GraphState()

  print("\n" + "=" * 70)
  print("Execution Complete!")
  print("=" * 70)
  print(
      "Total tasks completed:"
      f" {len(final_state.data.get('completed_tasks', []))}"
  )
  print(f"\nCompleted tasks:")
  for i, task in enumerate(final_state.data.get("completed_tasks", []), 1):
    print(f"{i}. [{task['type']}] {task['data']}")


if __name__ == "__main__":
  asyncio.run(main())
