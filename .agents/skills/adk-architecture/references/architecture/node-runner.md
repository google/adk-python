# NodeRunner

NodeRunner is the per-node executor. It drives `BaseNode.run()`,
creates the child Context, enriches events, and writes results
to ctx.

## Two communication channels

The runtime has two distinct channels for data flow:

- **Context** — parent ↔ child communication. Output, route, state,
  resume_inputs, and interrupt_ids flow through ctx. The orchestrator
  reads ctx after the child completes to decide what to do next.
- **Event** — persistence and streaming. Events are appended to the
  session and streamed to the caller. They carry message, state
  deltas, function calls, and interrupt markers.

A node writes to **ctx** to communicate with its parent. A node
yields **Events** to persist data and stream messages to the user.

## Execution flow

```
Orchestrator
  │
  ├─ NodeRunner(node=child, parent_ctx=ctx)
  │    │
  │    ├─ _create_child_context()     → child Context
  │    ├─ _execute_node()             → iterate node.run()
  │    │    ├─ _track_event_in_context()  → write to ctx
  │    │    └─ _enqueue_event()           → enrich + persist
  │    ├─ _flush_output_and_deltas()  → emit deferred output
  │    └─ return child ctx
  │
  └─ reads ctx.output, ctx.route, ctx.interrupt_ids
```

1. **Create child Context** — inherits `_invocation_context` (shared
   singleton), builds `node_path` from parent, assigns `run_id`.

2. **Iterate `node.run()`** — for each yielded Event:

   **Track in context** — `_track_event_in_context` writes output,
   route, and interrupt_ids from the event to ctx (source of truth).

   **Enrich** — `_enrich_event` stamps metadata before persistence:
   - `event.author` — node name (or `event_author` override)
   - `event.invocation_id` — from InvocationContext
   - `event.node_info.path` — full path (e.g., `wf/child_a`)
   - `event.node_info.run_id` — unique per execution
   - `event.node_info.output_for` — ancestor paths when
     `use_as_output=True`

   **Flush deltas** — for non-partial events, `_flush_deltas` moves
   pending state/artifact deltas from `ctx.actions` onto the event
   before enqueueing.

   **Enqueue** — `ic.enqueue_event` puts the event on the shared
   process queue for session persistence.

3. **Flush deferred output** — if `ctx.output` was set directly
   (not via yield), `_flush_output_and_deltas` emits the output
   Event after `_run_impl` returns. Bundles any remaining
   state/artifact deltas onto the same Event.

4. **Return child ctx** — the orchestrator reads `ctx.output`,
   `ctx.route`, and `ctx.interrupt_ids`.

## Output delegation (`use_as_output`)

When a child is scheduled with `use_as_output=True`, its output
Event also counts as the parent's output. NodeRunner:

- Sets `ctx._output_delegated = True` on the parent
- Skips emitting the parent's own output Event
- Stamps `event.node_info.output_for` with ancestor paths
