# Advanced Workflow Patterns

Nested workflows, retries, timeouts, custom node classes, and the graph
validation rules that reject a malformed graph at construction time.

```python
from google.adk import Context, Event, Workflow
from google.adk.workflow import BaseNode, Edge, FunctionNode, RetryConfig, START
```

## Nested workflows

A `Workflow` is both an agent and a node, so one can sit inside another. The
inner workflow takes the predecessor's output as its `START` input, and its
terminal output flows on to the next node outside.

```python
inner = Workflow(name='inner_pipeline', edges=[('START', step_a, step_b)])

outer = Workflow(
    name='outer_pipeline',
    edges=[('START', pre_process, inner, post_process)],
)
```

## Retries

Every `RetryConfig` field defaults to `None`, which means "use the built-in
fallback" rather than "disabled":

| Field | Fallback | Meaning |
|---|---|---|
| `max_attempts` | 5 | Total attempts including the first; 0 or 1 disables retrying |
| `initial_delay` | 1.0 | Seconds before the first retry |
| `max_delay` | 60.0 | Ceiling on the computed delay |
| `backoff_factor` | 2.0 | Multiplier applied per attempt |
| `jitter` | 1.0 | Randomness factor; 0.0 removes it |
| `exceptions` | all | Exception classes or class-name strings to retry on |

```python
api_node = FunctionNode(
    func=flaky_api_call,
    name='api_call',
    retry_config=RetryConfig(max_attempts=5, exceptions=[TimeoutError]),
)
```

Delay for attempt *n* is
`min(initial_delay * backoff_factor ** n, max_delay) * (1 + random(0, jitter))`.

Read the current try from the context — it is 1 on the first attempt:

```python
def my_node(ctx: Context, node_input: str) -> str:
  if ctx.attempt_count > 1:
    logger.warning('retry %d', ctx.attempt_count)
  return 'result'
```

## Timeouts

`timeout` is a per-node wall-clock limit in seconds. Exceeding it raises
`NodeTimeoutError` (importable from `google.adk.workflow`), which the retry
machinery treats like any other exception.

## Custom node classes

`BaseNode` is a Pydantic model. Declare fields as fields, and override
`_run_impl` — **not** `run`, which is `@final` and does the normalization of
yielded values into events.

```python
from typing import Any, AsyncGenerator

from typing_extensions import override


class BatchProcessorNode(BaseNode):
  """Processes a list of items in fixed-size batches."""

  batch_size: int = 10

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    items = node_input if isinstance(node_input, list) else [node_input]
    results = []
    for i in range(0, len(items), self.batch_size):
      results.extend(await process_batch(items[i:i + self.batch_size]))
    yield Event(output=results)


batcher = BatchProcessorNode(name='batch_processor', batch_size=25)
```

`_run_impl` may yield an `Event`, a `RequestInput`, a bare value (wrapped as
`Event(output=...)`), or `None` (skipped). There is no `get_name()` to override
— the node's name is the `name` field.

### `BaseNode` fields

| Field | Default | Purpose |
|---|---|---|
| `name` | required | Node identity within the graph; must be unique |
| `description` | `''` | Human-readable label |
| `rerun_on_resume` | `False` | Re-run after an interrupt instead of taking the answer as output |
| `wait_for_output` | `False` | Finishing without an output event leaves the node WAITING, not COMPLETED |
| `retry_config` | `None` | Retry policy |
| `timeout` | `None` | Seconds before `NodeTimeoutError` |
| `input_schema` | `None` | Validates and coerces `node_input` |
| `output_schema` | `None` | Validates and coerces `event.output` |
| `state_schema` | `None` | Validates `ctx.state` writes; `app:`, `user:`, `temp:` keys bypass it |

### `wait_for_output`

With `wait_for_output=True`, a node that completes without emitting an output
event moves to WAITING rather than COMPLETED, and no downstream node fires. An
upstream predecessor can trigger it again later — useful for a node that
accumulates across several triggers before producing one answer.

```python
class CollectorNode(BaseNode):
  wait_for_output: bool = True

  @override
  async def _run_impl(self, *, ctx, node_input):
    collected = ctx.state.get('collected', []) + [node_input]
    yield Event(state={'collected': collected})
    if len(collected) >= 3:
      yield Event(output=collected)  # now COMPLETED, downstream fires
```

`JoinNode` reaches a similar result by a different mechanism — it sets
`_requires_all_predecessors`, so the orchestrator holds it until every
predecessor has run and then hands it all their outputs at once.

## Wrapping a tool as a node

`_ToolNode` is private and keyword-only. Its input must be a dict of tool
arguments, or `None`.

```python
from google.adk.tools import FunctionTool
from google.adk.workflow._tool_node import _ToolNode


def search(query: str) -> str:
  """Search for information."""
  return f'Results for: {query}'


tool_node = _ToolNode(tool=FunctionTool(search), name='search_node')

agent = Workflow(
    name='with_tool',
    edges=[('START', prepare_query, tool_node, process_results)],
)
```

## Graph validation

`Workflow` validates the graph when it is constructed, in this order. Each check
raises `ValueError` naming the offending node or edge.

1. No duplicate node names.
2. A `START` node exists.
3. No edge leaving `START` carries a route.
4. Every node is reachable from `START`, and `START` has no incoming edges.
5. No two edges share both a source and a target — routes are not part of edge
   identity.
6. At most one `__DEFAULT__` route per node, and `__DEFAULT__` never appears
   inside a list of routes.
7. No unconditional cycle — a cycle needs at least one routed edge.
8. Where a source declares `output_schema` and its target declares
   `input_schema`, the two must be the same schema.
9. No edge into a `mode='chat'` `LlmAgent` from anything but `START`, because a
   chat agent reads conversation history rather than a node input.

Nodes with no outgoing edges are the graph's terminals; their outputs become the
workflow's own output.

## Ways to declare edges

```python
edges = [
    ('START', node_a),                 # simple
    (node_a, node_b, 'route'),         # routed
    (node_a, (node_b, node_c)),        # fan-out
    ((node_b, node_c), join_node),     # fan-in
    ('START', node_a, node_b, node_c), # chain of three edges
    (classifier, {'ok': handler_a, 'err': handler_b}),  # routing map
]
```

`Edge` objects are the explicit form. It is a Pydantic model, so its fields are
keyword-only:

```python
edges = [
    Edge(from_node=START, to_node=node_a),
    Edge(from_node=node_a, to_node=node_b, route='success'),
]
```

To build the graph yourself, pass `graph=` instead of `edges=`:

```python
from google.adk.workflow._graph import Graph

graph = Graph.from_edge_items([('START', node_a), (node_a, node_b)])
agent = Workflow(name='my_workflow', graph=graph)
```

## Where the code lives

| Component | File |
|---|---|
| `Workflow` | `src/google/adk/workflow/_workflow.py` |
| `Graph`, `Edge`, `DEFAULT_ROUTE` | `src/google/adk/workflow/_graph.py` |
| graph validation rules | `src/google/adk/workflow/utils/_graph_validation.py` |
| `BaseNode`, `START` | `src/google/adk/workflow/_base_node.py` |
| `FunctionNode` | `src/google/adk/workflow/_function_node.py` |
| `@node`, `Node` | `src/google/adk/workflow/_node.py` |
| `JoinNode` | `src/google/adk/workflow/_join_node.py` |
| `_ParallelWorker` | `src/google/adk/workflow/_parallel_worker.py` |
| `_ToolNode` | `src/google/adk/workflow/_tool_node.py` |
| `RetryConfig` | `src/google/adk/workflow/_retry_config.py` |
| running an `LlmAgent` as a node | `src/google/adk/workflow/_llm_agent_wrapper.py` |
| dynamic node scheduling | `src/google/adk/workflow/_dynamic_node_scheduler.py` |
| `Context` | `src/google/adk/agents/context.py` |
| `Event` | `src/google/adk/events/event.py` |
| `RequestInput` | `src/google/adk/events/request_input.py` |
