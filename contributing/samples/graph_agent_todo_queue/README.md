# GraphAgent TODO Queue — Conditional Loop with Type-Based Routing

This example demonstrates queue-based orchestration where items are fetched one
at a time, classified, routed to a type-specific processor, and then looped back
to fetch the next item — with a checkpoint after each completion so the queue is
resume-safe after an interruption.

## When to Use This Pattern

- Processing a heterogeneous queue where each item type needs a different handler
- Long-running batch jobs that must survive process restarts mid-queue
- Any loop that requires branching inside each iteration

## How to Run

```bash
GOOGLE_API_KEY=your_key python -m contributing.samples.graph_agent_todo_queue.agent
```

## Graph Structure

```
fetcher ──▶ classifier ──(data)──────▶ processor_data ──┐
                       ──(notification)▶ processor_notification ──┤ (has_more=True) ──▶ fetcher
                       ──(cleanup)────▶ processor_cleanup ──────┘
                                                              (has_more=False) ──▶ END
```

## Key Code Walkthrough

- **`add_edge("classifier", "processor_data", condition=_is_data_task)`** — three conditional branches route each item to the correct processor based on `todo_type` in state
- **Loop edges** — each processor connects back to `fetcher` when `has_more=True`; GraphAgent handles cycles that LoopAgent cannot route conditionally
- **`graph.set_end()` on all three processors** — any processor can be the terminal node when the queue drains
- **`GraphCheckpointCallback(checkpoint_nodes={"processor_data","processor_notification","processor_cleanup"})`** — checkpoints only after a full item completes, not mid-classification
- **`StateReducer.OVERWRITE`** — each loop iteration overwrites `current_todo` and `last_processed` rather than accumulating all iterations

