# Node and @node

The `@node` decorator and `node()` function wrap Python functions, agents, tools, and existing nodes into configured workflow steps without mutating the original object. The `Node` class is the base class for custom node types that need their own fields and parallel worker support.

## Introduction

A workflow graph accepts functions, agents, tools, and other workflows directly in its edges, wrapping each one with default settings as the graph is built. You reach for `node` or `Node` when those defaults are not enough:

- **Configuring a function at definition time**: `@node(...)` attaches settings such as `timeout`, `retry_config`, `rerun_on_resume`, `parameter_binding`, and `parallel_worker` directly to a Python function.
- **Overriding settings or reusing a step across a graph**: Calling `node(step, name=..., timeout=...)` on an existing function, `Agent`, `BaseTool`, or `BaseNode` returns a configured copy, leaving the original untouched so it can be reused elsewhere in the same graph or in other workflows.
- **Writing a reusable node class**: Subclassing `Node` and implementing `run_node_impl` gives your custom node class all [`BaseNode`](../base_node/index.md) settings plus [`parallel_worker`](../parallel_worker/index.md) support out of the box.

## Get started

The following workflow uses `@node` to attach a timeout and retry policy to a function at definition time, and uses `node()` in the edge list to run the same verification agent at two different stages of the graph under distinct names:

```python
from google.adk import Agent, Workflow
from google.adk.workflow import node, RetryConfig, START

reviewer = Agent(
    name="reviewer",
    instruction="Check the draft in the user message for factual errors.",
)


@node(timeout=15.0, retry_config=RetryConfig(max_attempts=3))
async def fetch_draft(node_input: str) -> str:
  """Fetches the initial draft for the requested topic."""
  return f"Draft about {node_input}"


def revise(node_input: str) -> str:
  return f"Revised: {node_input}"


workflow = Workflow(
    name="editorial_pipeline",
    edges=[
        (
            START,
            fetch_draft,
            node(reviewer, name="initial_review"),
            revise,
            node(reviewer, name="final_review", timeout=30.0),
        ),
    ],
)
```

Two details in that graph matter whenever you wire steps together.

First, both `node` and `Node` are imported from `google.adk.workflow`, not from the top-level `google.adk` package.

Second, wrapping `reviewer` with `node(reviewer, name=...)` creates a separate copy of the agent for each stage. Putting the bare `reviewer` object into the edge list twice would refer to the same node instance in both places, which adds a cycle back to the first review step instead of running a second review after `revise`.

## How it works

`node` and `Node` both produce a [`BaseNode`](../base_node/index.md) that the workflow runner schedules, validates, and normalizes like any other step in the graph.

### Decorator versus function call

`node` inspects its first positional argument to decide whether you are decorating a function or wrapping an existing object:

1. **Decorator form, `@node` or `@node(...)`**: When called with keyword arguments only, or placed bare above a `def`, `node` wraps the function in a [`FunctionNode`](../function_node/index.md). If you also set `parallel_worker=True`, the resulting function node is wrapped for per-item parallel execution.
2. **Wrapper form, `node(node_like, ...)`**: When passed a function, `BaseNode`, `BaseAgent`, or `BaseTool` as its first argument, `node` returns a `BaseNode` configured with the keyword arguments you supplied:
   - A **callable** becomes a new `FunctionNode` with those settings.
   - A **`BaseNode`**, such as an existing `FunctionNode`, `JoinNode`, `Workflow`, or `Node` subclass, is copied via `model_copy` with the new settings applied. If you pass no overrides at all and `parallel_worker` is `False`, the original instance is returned as-is.
   - An **`LlmAgent`** is cloned with the overrides applied. When `rerun_on_resume` is not specified, `node()` sets it to `True` on the clone so that interrupted agent turns resume properly, and configures standalone workflow agents to run in `'single_turn'` mode.
   - A **`BaseTool`** is adapted into a workflow node that receives `node_input` as its tool arguments dictionary, runs the tool, and commits any changes the tool made to `tool_context.state` into the workflow state. If the tool is a [`NodeTool`](../../tools/node_tool/index.md), `node()` unwraps the underlying node and applies your overrides directly to it.

### Subclassing `Node`

`Node` sits between [`BaseNode`](../base_node/index.md) and your custom node classes. `BaseNode` defines the nine core settings and input/output validation, while `Node` adds the `parallel_worker` and `max_parallel_workers` fields and dispatches execution to `run_node_impl`.

When `parallel_worker` is `False`, calling the node runs your `run_node_impl` async generator once with the incoming `node_input`. When `parallel_worker` is `True`, `Node` clones itself with `parallel_worker=False`, preserving your subclass type and all custom Pydantic fields on the clone, and runs `run_node_impl` concurrently for each element of the input list.

## Configuration options

`@node` and `node()` accept the options below. Inherited [`BaseNode`](../base_node/index.md) fields such as `input_schema`, `output_schema`, `state_schema`, `wait_for_output`, and `description` are not passed to `node()`; for functions they are inferred from type hints and docstrings, and for `Node` subclasses they are passed to the subclass constructor.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `node_like` | `NodeLike \| None` | `None` | Positional target to wrap or copy: a callable, `BaseNode`, `BaseAgent`, `BaseTool`, or `'START'`. Omit when using `@node(...)` as a decorator. |
| `name` | `str \| None` | `None` | Overrides the node's identifier in the graph. Defaults to the function's `__name__` or the wrapped object's existing `name`. |
| `rerun_on_resume` | `bool \| None` | `None` | Whether an interrupted node re-executes from the top on resume. Defaults to `False` for functions and `True` for `LlmAgent`. |
| `retry_config` | `RetryConfig \| None` | `None` | Retry policy for failed attempts. See [RetryConfig](../retry_config/index.md). |
| `timeout` | `float \| None` | `None` | Per-attempt time limit in seconds before raising `NodeTimeoutError`. |
| `parallel_worker` | `bool` | `False` | Runs the node concurrently across each item of an input list. See [parallel worker mode](../parallel_worker/index.md). |
| `max_parallel_workers` | `int \| None` | `None` | Maximum number of list items processed at once when `parallel_worker=True`. `None` places no cap on concurrency. |
| `auth_config` | `AuthConfig \| None` | `None` | Requests user credentials before the function runs. Applies to callables and requires `rerun_on_resume=True`. |
| `parameter_binding` | `'state' \| 'node_input'` | `'state'` | How a wrapped function resolves parameters other than `ctx` and `node_input`. |

### `name`

Every distinct node in a workflow must have a unique name that is a valid Python identifier. When decorating a function, `name` defaults to `func.__name__`. When wrapping an existing node or agent with `node(existing, name="step_two")`, setting `name` produces a distinct copy so the same logic can appear at multiple positions in the graph without colliding or forming a loop.

### `rerun_on_resume`

Controls how the node behaves when a workflow resumes after pausing for human input or tool confirmation. Functions default to `False`, meaning the resuming input becomes the node's output without running the function body again. Setting `rerun_on_resume=True` re-executes the function from the start on resume, which is required whenever the node calls `ctx.run_node()`, yields a `RequestInput` and inspects `ctx.resume_inputs`, or uses `auth_config`.

When `parallel_worker=True`, `rerun_on_resume` is forced to `True` automatically so the worker can re-collect completed item outputs from the event log on resume.

### `parallel_worker` and `max_parallel_workers`

Available both as arguments to `node()` and as fields on `Node` subclasses. Setting `parallel_worker=True` maps the node over an input `list` concurrently and collects the outputs in the original order. `max_parallel_workers` throttles how many items run at the same time; passing `max_parallel_workers` while `parallel_worker=False`, or passing a value less than `1`, raises a validation error at construction time.

### `parameter_binding`

Applies when `node` wraps a Python callable. With the default `'state'`, parameters other than `ctx` and `node_input` are looked up by name in `ctx.state`. With `'node_input'`, `node_input` must be a dictionary or Pydantic model whose keys are unpacked into the function's parameters, and ADK infers `input_schema` and `output_schema` from the full parameter list so the node can also be called as an agent tool. See [Function nodes](../function_node/index.md) and [Node as tool](../../tools/node_tool/index.md).

## Advanced applications

Beyond decorating functions, `node` and `Node` solve three structural problems in workflow graphs: reusing a single component at multiple points in a graph, running an agent tool directly as a deterministic graph step, and authoring configurable node classes.

### Reusing a node or agent across multiple steps

In a workflow graph, node identity determines graph topology. Referencing the same node object in two edges tells the graph builder that both edges point to the same vertex, which creates a loop or a join rather than two separate steps. Wrapping the object in `node(..., name=...)` creates a fresh copy with its own name:

```python
from google.adk import Workflow
from google.adk.workflow import node, START


@node
def sanitize(node_input: str) -> str:
  return node_input.strip()


def enrich(node_input: str) -> str:
  return f"[enriched] {node_input} "


workflow = Workflow(
    name="reuse_pipeline",
    edges=[
        (
            START,
            node(sanitize, name="sanitize_raw"),
            enrich,
            node(sanitize, name="sanitize_enriched"),
        ),
    ],
)
```

Because `sanitize` is already a `FunctionNode` created by `@node`, passing it to `node(sanitize, name=...)` copies the node with the new name while keeping its underlying function and configuration intact.

### Running a `BaseTool` as a workflow step

When you already have a `BaseTool` written for an `LlmAgent`, you can run it deterministically inside a workflow graph without an LLM call in between. Pass the tool to `node()` to give it a workflow-specific name, timeout, or retry policy, and have the preceding node emit a dictionary matching the tool's arguments:

```python
from typing import Any
from google.adk import Workflow
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node, RetryConfig, START


class LookupOrderTool(BaseTool):
  """Fetches order status from a backend store."""

  def __init__(self):
    super().__init__(name="lookup_order", description="Looks up an order.")

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    order_id = args["order_id"]
    tool_context.state["last_order_id"] = order_id
    return {"order_id": order_id, "status": "shipped"}


def build_tool_args(node_input: str) -> dict[str, str]:
  return {"order_id": node_input}


lookup_step = node(
    LookupOrderTool(),
    name="lookup_order_step",
    timeout=10.0,
    retry_config=RetryConfig(max_attempts=2),
)

workflow = Workflow(
    name="order_lookup_workflow",
    edges=[(START, build_tool_args, lookup_step)],
)
```

Any writes the tool makes to `tool_context.state` are recorded on the node's output event and persisted to the workflow session state, so downstream nodes can read `last_order_id` as a state parameter.

### Building a custom node class with `Node`

When a workflow step carries custom configuration fields and you want callers to instantiate it like a built-in node, subclass `Node` and implement `run_node_impl`:

```python
from collections.abc import AsyncGenerator
from typing import Any

from google.adk import Context, Workflow
from google.adk.workflow import Node, START


class PrefixNode(Node):
  """Prepends a configured label to each input string."""

  prefix: str = "item"

  async def run_node_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield f"[{self.prefix}] {node_input}"


def produce_items(node_input: str) -> list[str]:
  return ["alpha", "beta", "gamma"]


tagger = PrefixNode(
    name="tagger",
    prefix="batch",
    parallel_worker=True,
    max_parallel_workers=2,
)

workflow = Workflow(
    name="custom_node_workflow",
    edges=[(START, produce_items, tagger)],
)
```

Because `PrefixNode` inherits from `Node` rather than `BaseNode`, passing `parallel_worker=True` and `max_parallel_workers=2` to `PrefixNode(...)` works without any extra code in `run_node_impl`. The method is written for a single item, and `Node` handles fan-out, concurrency throttling, and result ordering automatically.

## Limitations

Keep the following constraints in mind when wrapping nodes with `node()` or subclassing `Node`:

- **`node(existing_node)` without overrides does not copy the node.** When you pass a `BaseNode` to `node()` without setting `name`, `rerun_on_resume`, `retry_config`, `timeout`, or `parallel_worker=True`, `node()` returns the exact same instance. To use the same node at two places in one graph, you must pass a distinct `name` so a new copy is created and the two steps do not collide.
- **`auth_config` and `parameter_binding` do not apply when wrapping an existing `BaseNode`, `BaseAgent`, or `BaseTool`.** Those two options are only consumed when `node()` wraps a plain Python callable into a `FunctionNode`. Passing them alongside an already-constructed `FunctionNode` or `Agent` has no effect.
- **`parallel_worker` and `max_parallel_workers` are frozen on `Node` instances.** Both fields are declared with `frozen=True`, so mutating `my_node.parallel_worker = True` after construction raises a Pydantic validation error. Pass them to the constructor or to `node()`, or create a new instance with `model_copy(update={"parallel_worker": True})`.
- **`run_node_impl` must be an async generator.** Even if your `Node` subclass emits only a single value, `run_node_impl` must be defined with `async def` and `yield` its output rather than using `return`. A `Node` subclass also receives raw `node_input` without automatic `types.Content`-to-`str` conversion unless you set `input_schema`.

## Related samples

The following runnable samples demonstrate `@node` and `Node` in complete workflows:

- [Node Retries](../../../../contributing/samples/workflows/retry/agent.py): Decorating a workflow function with `@node(retry_config=...)` to retry transient failures.
- [Parallel Worker](../../../../contributing/samples/workflows/parallel_worker/agent.py): Using `@node(parallel_worker=True)` to process a list of items concurrently.
- [Node as Tool](../../../../contributing/samples/workflows/node_as_tool/agent.py): Exposing a `@node(rerun_on_resume=True)` generator function as an interactive agent tool.
- [Auth API Key](../../../../contributing/samples/workflows/auth_api_key/agent.py): Requesting credentials on a function node with `@node(auth_config=..., rerun_on_resume=True)`.
