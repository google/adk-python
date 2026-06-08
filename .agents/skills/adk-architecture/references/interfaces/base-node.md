# BaseNode

BaseNode is the primitive unit of execution in the workflow runtime.
Every computation — LLM calls, tool execution, orchestration — is
a node. It is a Pydantic `BaseModel` subclass.

## The node contract

Every node follows a two-method pattern:

- `run()` is `@final` — normalizes yields to Events. Never override.
- `_run_impl()` is the extension point — subclasses implement their
  logic here as an async generator.

```python
class MyNode(BaseNode):
    async def _run_impl(self, *, ctx, node_input):
        result = do_work(node_input)
        yield result  # becomes Event(output=result)
```

**Why this split:** `run()` guarantees consistent normalization
regardless of what the subclass does. The subclass only thinks
about its domain logic.

**Normalization rules** (`run()` applies these to each yield):

- `None` → skipped
- `Event` → pass through
- `RequestInput` → interrupt Event
- any other value → `Event(output=value)`

**Generator conventions:**

A node can yield three types of data:

- **Output** — the node's result value. Flows between nodes
  (parent reads `ctx.output` after child completes). At most one
  per execution (second raises `ValueError`).
- **Message** — user-visible content streamed to the end user
  (e.g., progress text, partial responses). Multiple allowed.
- **Route** — Workflow-specific concept. Triggers conditional
  edges in the graph. Set via `ctx.route` or `event.actions.route`.

Additional rules:

- Yielding nothing produces no output event
- `yield None` is silently skipped

A custom node interacts with the runtime through two arguments:

- **`ctx`** (Context) — communicate results to the parent node
- **`node_input`** — data passed by the parent/orchestrator

## Output and streaming

Three ways to produce output (pick one per execution):

```python
# 1. Yield a value (most common)
async def _run_impl(self, *, ctx, node_input):
    yield compute(node_input)

# 2. Set ctx.output directly
async def _run_impl(self, *, ctx, node_input):
    ctx.output = compute(node_input)
    return
    yield  # generator contract

# 3. Yield an Event with output
async def _run_impl(self, *, ctx, node_input):
    yield Event(output=compute(node_input))
```

A second output raises `ValueError` — at most one per execution.

**Streaming messages** — yield Events with `message` to send
user-visible text (`message` is an alias for `content` on Event):

```python
async def _run_impl(self, *, ctx, node_input):
    yield Event(message='working...')
    yield final_result  # this is the output
```

## State and routing

**Mutating state:**

```python
async def _run_impl(self, *, ctx, node_input):
    ctx.state['key'] = 'value'  # recorded as state_delta
    yield result
```

**Setting route for conditional edges:**

```python
async def _run_impl(self, *, ctx, node_input):
    ctx.route = 'approve' if score > 0.8 else 'reject'
    yield node_input
```

## Advanced: child nodes and HITL

**Running child nodes** via `ctx.run_node()`:

```python
async def _run_impl(self, *, ctx, node_input):
    child_result = await ctx.run_node(some_node, node_input)
    yield f'child said: {child_result}'
```

Requires `rerun_on_resume = True` on the calling node.

**Requesting interrupt (HITL):**

```python
async def _run_impl(self, *, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]}'
        return
    yield Event(long_running_tool_ids={'fc-1'})
```

## Configuration reference

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | required | Unique identifier |
| `description` | `str` | `''` | Human-readable description |
| `rerun_on_resume` | `bool` | `False` | Re-execute on resume (required for `ctx.run_node()`) |
| `wait_for_output` | `bool` | `False` | Stay WAITING until output is yielded (for join nodes) |
| `retry_config` | `RetryConfig \| None` | `None` | Retry on failure |
| `timeout` | `float \| None` | `None` | Max execution time in seconds |
| `input_schema` | `SchemaType \| None` | `None` | Validate/coerce input data |
| `output_schema` | `SchemaType \| None` | `None` | Validate/coerce output data |
