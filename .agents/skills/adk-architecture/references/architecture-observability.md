# Observability

Node tracing lives in `telemetry/node_tracing.py`. Spans are **scope-based**:
`NodeRunner` opens one around each node execution with an async context
manager, so the span closes on the way out even if the node raises.

```python
async with node_tracing.start_as_current_node_span(parent_ctx, node) as tel_ctx:
  ctx._telemetry_context = tel_ctx
  await self._execute_node(ctx, node_input)
```

## TelemetryContext, not a raw span

Each `Context` exposes `ctx.telemetry_context`, a frozen dataclass:

| Member | Purpose |
|---|---|
| `otel_context` | OTel context holding the current span. Passed to children so their spans parent correctly. |
| `add_event(event)` | Records an event ID as belonging to this node's span. |

The parent's `otel_context` is what makes the span tree mirror the node tree —
`start_as_current_node_span` passes `context.telemetry_context.otel_context` as
the explicit parent rather than relying on whatever OTel considers current.
Inside the span the child builds a fresh `TelemetryContext` from
`context_api.get_current()`.

When attaching OTel context by hand (`context_api.attach()`), pair it with
`detach()` in a `finally` — an unbalanced attach leaks the context into
whatever coroutine runs next on the loop.

## Which span a node gets

`start_as_current_node_span` dispatches on node type:

| Node type | Span |
|---|---|
| `BaseAgent` subclass | none of its own — passes through, the agent emits its own `invoke_agent {name}` span |
| `Workflow` | `invoke_workflow {name}` |
| anything else | `invoke_node {name}` |

`invoke_agent` follows OpenTelemetry semantic conventions v1.36 for backward
compatibility; `invoke_workflow` follows v1.41; `invoke_node` is not in any
semconv release yet.

## Attributes

| Attribute | Set on | Value |
|---|---|---|
| `gen_ai.operation.name` | all | `"invoke_workflow"` / `"invoke_node"` |
| `gen_ai.conversation.id` | all | `ctx.session.id` |
| `gen_ai.workflow.name` | workflow | the workflow's `name`, when non-empty |
| `gen_ai.workflow.nested` | workflow | `True` only for a nested workflow; the entrypoint workflow omits the attribute entirely |
| `gcp.vertex.agent.associated_event_ids` | all | IDs collected via `tel_ctx.add_event()`, stamped on span close when non-empty and the span is recording |

Nesting is detected through an OTel context key set by the first workflow in
the invocation. Because the key rides on the propagated `otel_context`, an
agent-as-tool that spins up its own runner still reports `nested=true`.

## Metrics

Recorded from `telemetry/_metrics.py` under the `gcp.vertex.agent` meter:

| Function | Recorded when |
|---|---|
| `record_workflow_invocation_duration` | an `invoke_workflow` span closes, tagged with `nested` and any error |
| `record_agent_invocation_duration` | an agent invocation completes |
| `record_invoke_agent_inference_calls` | per agent, count of model calls |
| `record_invoke_agent_tool_calls` | per agent, count of tool calls |
| `record_tool_execution_duration` | a tool finishes |
| `record_client_operation_duration`, `record_client_token_usage` | model client calls |

## Python logging

Use the `google_adk` logger namespace so callers can filter ADK output:

```python
logger = logging.getLogger("google_adk." + __name__)

logger.debug("node %s started.", ctx.node_path)
```

Use `%`-style arguments, not f-strings — the formatting is then skipped
entirely when the level is disabled.

`NodeRunner` already logs node start, node end, execute-loop boundaries,
rehydrated resume inputs, retries (`warning`) and unhandled exceptions
(`logger.exception`). Do not re-log those from inside a node.
