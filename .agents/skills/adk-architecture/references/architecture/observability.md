# Observability

## Design: span on Context

Each Context carries a `_span` field. Since Context forms a 1:1
parent-child tree with node executions (see [Context](context.md)),
span hierarchy follows naturally — no separate span management
needed.

```
Root Context._span (invocation)     ← Runner sets this
└── ctx[workflow]._span             ← NodeRunner creates
    ├── ctx[child_a]._span          ← NodeRunner creates
    │   ├── (call_llm span)         ← auto-parented
    │   └── (execute_tool span)     ← auto-parented
    ├── ctx[child_b]._span          ← NodeRunner creates
    │   └── ctx[grandchild]._span   ← nested
    └── ctx[child_c]._span          ← ctx.run_node()
```

**Runner** creates `root_ctx` and the `invocation` span, storing
it as `root_ctx._span`. This becomes the parent for all node spans.

**NodeRunner** creates each node's span, explicitly parented to
`parent_ctx._span`, stores it on `child_ctx._span`, and closes it
before returning (see [NodeRunner](node-runner.md) for the
execution flow).

**Always use `ctx._span` explicitly** — never rely on OTel's
implicit "current span" context. In a concurrent asyncio.Task
runtime, implicit context can be unreliable across concurrent
nodes. All tracing operations (attributes, logs, child spans)
should go through `ctx._span`.

**Span lifecycle:**

1. `NodeRunner.run()` creates span via `tracer.start_span()`,
   parented to `parent_ctx._span`, stored on `ctx._span`
2. Node executes; all tracing goes through `ctx._span` explicitly
3. `NodeRunner.run()` calls `ctx._span.end()` before returning
4. `BatchSpanProcessor` buffers ended spans, exports periodically
5. `OTLPSpanExporter` sends batch to the OTLP endpoint

**Interrupted nodes:** Span ends immediately when NodeRunner
returns — not left open waiting for resume. Otherwise the span
would be invisible to the backend until resume (which could be
minutes, hours, or never). The resumed execution starts a fresh
span in a new `Runner.run_async()` call (same invocation_id,
different trace — possibly on a different server).

## NodeRunner integration

**Context changes** — add `_span` field:

```python
class Context(ReadonlyContext):
    _span: Span | None = None
```

**NodeRunner.run():**

**NodeRunner.run() lifecycle:**

1. Create child ctx
2. Create span, parented to `parent_ctx._span`
3. Store on `ctx._span`
4. Set node attributes (name, path, run_id, type)
5. Execute node
   - Node can add custom attributes to `ctx._span` during
     execution (e.g., SingleAgentReactNode adds
     `gen_ai.agent.name`, `gen_ai.request_model`)
   - On interrupt: mark span `node.interrupted = True`
   - On error: set span status `ERROR`, record exception
6. Set result attributes (has_output, interrupted, resumed)
7. **Close span** (`ctx._span.end()`) — always, even on interrupt
8. Return ctx

Key points:
- Use `tracer.start_span()` with explicit parent context from
  `parent_ctx._span` — never rely on implicit OTel context in
  concurrent async code
- Span always ends before `run()` returns, even on interrupt

## Span attributes and semantic conventions

Set at span creation (available for sampling decisions):

| Attribute | Source | Example |
|---|---|---|
| `node.name` | `self._node.name` | `"call_llm"` |
| `node.path` | `ctx.node_path` | `"wf/child_a"` |
| `node.run_id` | `self._run_id` | `"child_a_abc123"` |
| `node.type` | `type(self._node).__name__` | `"CallLlmNode"` |

Set after execution (result attributes):

| Attribute | Source | Example |
|---|---|---|
| `node.has_output` | `ctx.output is not None` | `true` |
| `node.interrupted` | `bool(ctx.interrupt_ids)` | `false` |
| `node.resumed` | `bool(resume_inputs)` | `false` |

GenAI semantic conventions for node spans:

- `gen_ai.operation.name` = `"invoke_agent"` for agent nodes
- `gen_ai.operation.name` = `"execute_tool"` for tool nodes
- `gen_ai.agent.name`, `gen_ai.tool.name` as appropriate
- Span kind: `INTERNAL` (in-process orchestration)

## Correlated logs

Use the OTel Logs API for point-in-time occurrences within a
node's span. Context provides `emit_log()` for better DX —
wraps `set_span_in_context(self._span)` internally so callers
don't manage OTel context:

```python
# On Context:
def emit_log(self, body: str, **attributes):
    span_ctx = set_span_in_context(self._span)
    otel_logger.emit(
        LogRecord(body=body, attributes=attributes),
        context=span_ctx,
    )

# Usage:
ctx.emit_log('node.event.yielded',
    has_output=event.output is not None,
    has_message=event.content is not None,
)
```

## Python logging

Use the `google_adk` logger namespace:

| Level | What to log |
|---|---|
| `DEBUG` | Node started, node completed, event enqueued |
| `INFO` | Node interrupted, node resumed, dynamic node scheduled |
| `WARNING` | Node timeout, retry triggered |
| `ERROR` | Node failed, unhandled exception |

```python
logger = logging.getLogger("google_adk." + __name__)

logger.debug(
    'Node %s started (run_id=%s, path=%s)',
    node.name, run_id, ctx.node_path,
)
```

Use `%`-style formatting (lazy evaluation) for logging, not
f-strings.

## Metrics (future)

| Metric | Type | Description |
|---|---|---|
| `node.execution.duration` | Histogram | Per node type |
| `node.execution.count` | Counter | Per node type and status |
| `node.interrupt.count` | Counter | HITL interrupts |
| `node.resume.count` | Counter | Resumed executions |
| `workflow.active_nodes` | UpDownCounter | Currently executing |
