"""GraphAgent TODO queue orchestrator with checkpointing.

Demonstrates queue-based orchestration where items are processed one at a
time with a checkpoint after each completion. If execution is interrupted
(e.g., process crash), the queue resumes from the last checkpoint.

Features:
- Process a queue of TODO items sequentially
- Checkpoint after each item completion (resume-safe)
- Dynamic routing based on TODO type (data/notification/cleanup)
- Loop control: continues until queue is empty
- Selective checkpointing (only after processors, not fetcher/classifier)

Flow:
  fetcher → classifier → [processor_data | processor_notification | processor_cleanup]
       ↑                              ↓
       └──────── (has_more=True) ─────┘
                      ↓
               (has_more=False) → END

Why GraphAgent (not LoopAgent)?
- LoopAgent: unconditional loop, cannot route to different processors
- GraphAgent: conditional routing + state-driven loop control

Run (requires GOOGLE_API_KEY env var):
    python -m contributing.samples.graph_agent_todo_queue.agent
"""

import asyncio
import json
import os

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import StateReducer
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


class TodoClassification(BaseModel):
  """Structured TODO classification from classifier agent."""

  todo_id: str  # ID of the current TODO item
  todo_type: str  # "data_processing" | "notification" | "cleanup"
  priority: int  # 1 (highest) to 5 (lowest)
  has_more: bool  # Are there more items remaining in the queue?


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

fetcher = LlmAgent(
    name="fetcher",
    model=_MODEL,
    instruction=(
        "You are a TODO queue manager. Given the current queue state, fetch"
        " the next unprocessed TODO item. Describe the item briefly."
        " If the queue is empty, say 'Queue is empty'."
    ),
    output_key="current_todo",
)

classifier = LlmAgent(
    name="classifier",
    model=_MODEL,
    instruction=(
        "You are a TODO classifier. Classify the current TODO item."
        ' Return {"todo_id": "item-N", "todo_type":'
        ' "data_processing|notification|cleanup", "priority": 1-5,'
        ' "has_more": true/false}.'
        " has_more=true if there are more unprocessed items after this one."
    ),
    output_schema=TodoClassification,
    # output_key auto-defaults to "classifier" (agent name)
)

processor_data = LlmAgent(
    name="processor_data",
    model=_MODEL,
    instruction=(
        "You process data_processing TODO items. Describe what data"
        " transformation was performed. Be concise (1 sentence)."
    ),
    output_key="last_processed",
)

processor_notification = LlmAgent(
    name="processor_notification",
    model=_MODEL,
    instruction=(
        "You process notification TODO items. Describe what notification"
        " was sent. Be concise (1 sentence)."
    ),
    output_key="last_processed",
)

processor_cleanup = LlmAgent(
    name="processor_cleanup",
    model=_MODEL,
    instruction=(
        "You process cleanup TODO items. Describe what was cleaned up."
        " Be concise (1 sentence)."
    ),
    output_key="last_processed",
)


# ---------------------------------------------------------------------------
# Routing predicates
# ---------------------------------------------------------------------------


def _get_classifier(state: GraphState) -> dict:
  """Parse classifier output (may be a JSON string or a dict)."""
  val = state.data.get("classifier", {})
  if isinstance(val, str):
    try:
      return json.loads(val)
    except (json.JSONDecodeError, TypeError):
      return {}
  return val if isinstance(val, dict) else {}


def _is_data_task(state: GraphState) -> bool:
  return _get_classifier(state).get("todo_type") == "data_processing"


def _is_notification_task(state: GraphState) -> bool:
  return _get_classifier(state).get("todo_type") == "notification"


def _is_cleanup_task(state: GraphState) -> bool:
  return _get_classifier(state).get("todo_type") == "cleanup"


def _has_more_items(state: GraphState) -> bool:
  return _get_classifier(state).get("has_more", False) is True


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_todo_queue_graph(
    session_service: InMemorySessionService,
) -> GraphAgent:
  """Build TODO queue orchestrator with selective checkpointing."""
  checkpoint_service = CheckpointService(session_service=session_service)

  # Only checkpoint after each processor completion (not fetcher/classifier)
  # This ensures we can resume from the last COMPLETED item, not mid-classification
  checkpoint_callback = GraphCheckpointCallback(
      checkpoint_service,
      checkpoint_before=False,
      checkpoint_after=True,
      checkpoint_nodes={
          "processor_data",
          "processor_notification",
          "processor_cleanup",
      },
  )

  graph = GraphAgent(
      name="todo_queue",
      description="Queue-based TODO processing with resume-safe checkpointing",
      max_iterations=50,  # Process up to 50 TODO items
      after_node_callback=checkpoint_callback.after_node,
  )

  # Build graph structure
  graph.add_node("fetcher", agent=fetcher, reducer=StateReducer.OVERWRITE)
  graph.add_node("classifier", agent=classifier, reducer=StateReducer.OVERWRITE)
  graph.add_node(
      "processor_data", agent=processor_data, reducer=StateReducer.OVERWRITE
  )
  graph.add_node(
      "processor_notification",
      agent=processor_notification,
      reducer=StateReducer.OVERWRITE,
  )
  graph.add_node(
      "processor_cleanup",
      agent=processor_cleanup,
      reducer=StateReducer.OVERWRITE,
  )

  graph.set_start("fetcher")
  graph.add_edge("fetcher", "classifier")

  # Route to appropriate processor based on TODO type
  graph.add_edge("classifier", "processor_data", condition=_is_data_task)
  graph.add_edge(
      "classifier", "processor_notification", condition=_is_notification_task
  )
  graph.add_edge("classifier", "processor_cleanup", condition=_is_cleanup_task)

  # Loop back to fetch next item if queue not empty
  graph.add_edge("processor_data", "fetcher", condition=_has_more_items)
  graph.add_edge("processor_notification", "fetcher", condition=_has_more_items)
  graph.add_edge("processor_cleanup", "fetcher", condition=_has_more_items)

  # End at any processor when queue is empty
  graph.set_end("processor_data")
  graph.set_end("processor_notification")
  graph.set_end("processor_cleanup")

  return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  print("=== TODO Queue Orchestrator with Checkpointing ===\n")

  session_service = InMemorySessionService()
  graph = build_todo_queue_graph(session_service)

  session = await session_service.create_session(
      app_name="todo_queue", user_id="user1"
  )

  # Define the TODO queue as initial state
  todo_queue = [
      {
          "id": "todo-1",
          "type": "data_processing",
          "task": "Transform user CSV export to JSON format",
      },
      {
          "id": "todo-2",
          "type": "notification",
          "task": "Send weekly summary email to team",
      },
      {
          "id": "todo-3",
          "type": "cleanup",
          "task": "Delete temporary files from /tmp/exports",
      },
      {
          "id": "todo-4",
          "type": "data_processing",
          "task": "Aggregate daily metrics into monthly report",
      },
      {
          "id": "todo-5",
          "type": "notification",
          "task": "Notify ops team of deployment completion",
      },
  ]

  queue_json = json.dumps(todo_queue, indent=2)
  print(f"Queue contains {len(todo_queue)} items:")
  for item in todo_queue:
    print(f"  [{item['id']}] ({item['type']}) {item['task']}")
  print()

  runner = Runner(
      app_name="todo_queue",
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  processed_count = 0

  async for event in runner.run_async(
      user_id="user1",
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text=f"Process this TODO queue:\n{queue_json}")]
      ),
  ):
    if not event.content or not event.content.parts:
      continue
    text = event.content.parts[0].text or ""
    author = event.author

    if author in (
        "processor_data",
        "processor_notification",
        "processor_cleanup",
    ):
      processed_count += 1
      proc_type = author.replace("processor_", "")
      print(f"[{proc_type.upper()}] {text[:120]}")

  print(f"\nQueue processing complete. Items processed: {processed_count}")

  # Re-fetch session: InMemorySessionService returns deepcopies, so local
  # `session` is stale. The runner's internal copy has the checkpoint updates.
  fresh_session = await session_service.get_session(
      app_name="todo_queue", user_id="user1", session_id=session.id
  )
  if fresh_session is None:
    print(
        f"WARNING: session_service.create_session returned None, using stale"
        f" copy"
    )
    fresh_session = session
  checkpoints = fresh_session.state.get("_checkpoint_index", {})
  print(f"Checkpoints saved: {len(checkpoints)}")
  print(
      "Note: If interrupted, resume by restoring latest checkpoint and"
      " re-running."
  )


if __name__ == "__main__":
  asyncio.run(main())
