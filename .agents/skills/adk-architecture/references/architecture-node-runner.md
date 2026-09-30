# NodeRunner

`NodeRunner` is the per-node executor. It creates the child Context, drives
`BaseNode.run()`, opens the node's span, enriches and enqueues events, retries
on failure, and returns the child Context to the caller.

## Two communication channels

- **Context** — parent ↔ child. Output, route, state, resume inputs and
  interrupt IDs flow through `ctx`. The orchestrator reads `ctx` after the
  child finishes to decide what happens next.
- **Event** — persistence and streaming. Events are appended to the session and
  streamed to the caller. They carry message content, state deltas, function
  calls and interrupt markers.

A node writes to `ctx` to talk to its parent. It yields Events to persist data
and stream to the user.

## Execution flow

```text
Orchestrator
  │
  ├─ NodeRunner(node=child, parent_ctx=ctx)
  │    │
  │    ├─ _create_child_context()       → child Context (attempt_count)
  │    ├─ start_as_current_node_span()  → ctx._telemetry_context
  │    ├─ _execute_node()               → iterate node.run()
  │    │    ├─ _track_event_in_context()    → write results to ctx
  │    │    └─ _enqueue_event()             → enrich + persist
  │    ├─ _flush_output_and_deltas()    → emit deferred output/route/deltas
  │    └─ return child ctx
  │
  └─ reads ctx.output, ctx.route, ctx.interrupt_ids, ctx.error
```

1. **Create child Context.** Shares the InvocationContext, builds `node_path`
   from the parent, assigns `run_id`, records `attempt_count`. If the session
   already holds events for this node path, resolved responses are rehydrated
   into `ctx._resume_inputs` before the node runs.

2. **Open the span** via `node_tracing.start_as_current_node_span`, storing the
   resulting `TelemetryContext` on `ctx`. Opening it inside the `try` is
   deliberate — exceptions are recorded on the span.

3. **Iterate `node.run()`.** For each yielded Event:

   - **Track in context** — `_track_event_in_context` copies output, route and
     `long_running_tool_ids` onto `ctx`, which is the source of truth. Route
     and `transfer_to_agent` are only picked up from *native* events (no
     author, or authored by this node), so a composite parent does not
     re-bubble a decision its own sub-agent already handled. The event ID is
     also registered on the node's span.
   - **Enrich** — `_enrich_event` stamps `author` (`ctx.event_author` or the
     node name), `invocation_id`, `node_info.path`, `branch`, and
     `isolation_scope`. For an event carrying output it also sets
     `node_info.output_for` to this node path plus its output ancestors.
     `node_info.run_id` is **not** stamped separately — it is a property
     derived from the last segment of `node_info.path` (`wf@1/child@2`).
   - **Flush deltas** — for non-partial events, pending state and artifact
     deltas move from `ctx.actions` onto the event.
   - **Enqueue** — `ic._enqueue_event(event)` puts it on the shared queue for
     session persistence.

4. **Flush deferred output.** If `ctx.output` or `ctx.route` were set directly
   rather than yielded, `_flush_output_and_deltas` emits one final Event after
   `_run_impl` returns, bundling any remaining deltas onto it.

5. **Return the child ctx.** The orchestrator reads `output`, `route`,
   `interrupt_ids` and `error`.

## Timeouts, retries and errors

- `node.timeout` wraps the iteration in `asyncio.wait_for`; a timeout raises
  `NodeTimeoutError`.
- On any exception, NodeRunner enqueues an Event with `error_code` and
  `error_message`, then consults `node.retry_config`. A retry sleeps for the
  configured delay, increments `attempt_count` and rebuilds the child Context
  from scratch. Retry count is **not** persisted, so it does not survive a
  resume.
- When retries are exhausted, the error is recorded on `ctx.error` and
  `ctx.error_node_path` and the Context is returned normally — NodeRunner does
  not re-raise to the orchestrator.
- `NodeInterruptedError` from a dynamic child is swallowed here: the child's
  interrupt IDs are already on `ctx`, so the caller just reads
  `ctx.interrupt_ids`.

## Output delegation (`use_as_output`)

When a child is scheduled with `use_as_output=True`, its output Event also
counts as the parent's output. NodeRunner sets `ctx._output_delegated`, drops
the output field from the parent's own event (keeping the event if it still
carries deltas), and stamps `node_info.output_for` with the ancestor paths.
