# Dynamic Node Scheduling

Dynamic node scheduling allows you to execute workflow nodes dynamically at runtime using `ctx.run_node()`. This enables imperative workflow construction using standard Python control flow instead of static graph edges.

## Introduction

While static graph definitions (`Workflow(edges=[...])`) are suitable for many structured tasks, some scenarios require more flexibility. For example, you might need to:
- Loop a set of nodes until a condition is met (e.g., generator-evaluator loops).
- Run a variable number of tasks in parallel based on runtime input (dynamic fan-out).
- Conditionally execute nodes based on complex logic that is difficult to express in static edges.

`ctx.run_node()` allows a parent node to execute a child node (which can be a function, an Agent, or another Workflow) and await its result.

## Get started

The following example demonstrates how to dynamically execute a child agent from a parent node.

```python
from google.adk import Agent, Context, Event, Workflow
from google.adk.workflow import node

# Define a child agent
generate_headline = Agent(
    name="generate_headline",
    instruction="Write a catchy headline about the topic in the user message.",
)


# Define the parent orchestrator node (MUST have rerun_on_resume=True)
@node(rerun_on_resume=True)
async def orchestrate(ctx: Context, node_input: str) -> str:
  # Dynamically execute the child agent and await its output
  headline = await ctx.run_node(generate_headline, node_input=node_input)

  yield Event(output=headline)

# Build the workflow
root_agent = Workflow(
    name="root_agent",
    edges=[("START", orchestrate)],
)
```


## How it works

When `await ctx.run_node(node_like, ...)` is called:

1.  **Orchestrator Registration**: The workflow's `DynamicNodeScheduler` registers the child node execution.
2.  **State Tracking**: The execution state and events of the child node are tracked under the parent node's path (e.g., `parent_node@1/child_node@1`).
3.  **Resumption Support**: If the child node interrupts (e.g., waiting for user input), the parent node is also paused. When the workflow resumes, the parent node is re-run from the beginning (`rerun_on_resume=True`), but previous successful `ctx.run_node()` calls are replayed from history (cached outputs are returned) to avoid re-executing completed steps.

### Input Mapping

The `node_input` passed to `ctx.run_node(node, node_input=value)` is delivered differently depending on the type of the child node:

-   **Python Functions / FunctionNodes**: The `value` is passed directly to the function parameter named `node_input`. Other parameters are bound from the session state (default mode).
-   **Agents (Single-Turn Mode)**: The `value` is converted to a user-role message (`types.Content`) and appended to the session events history. The agent receives it as the incoming user message.
-   **Agents (Task Mode)**: The `value` is set as `user_content` in the `InvocationContext`, serving as the fallback first user turn for the task agent if it wasn't triggered by a tool call.

## Requirements & Rules

### 1. `rerun_on_resume=True` is Mandatory for Parents

Any node that calls `ctx.run_node()` **must** be configured with `rerun_on_resume=True`.
If the parent node does not have this setting, calling `ctx.run_node()` will raise a `ValueError` at runtime.

### 2. Function Parameter Mapping (`node_input` vs. Dict Binding)

By default, functions wrapped as nodes look up their arguments in the session state (state binding). However, the `node_input` argument passed to `ctx.run_node(..., node_input=value)` is passed directly to the node.

How you receive this input depends on how you define your function:

#### Pass-through `node_input` (Default)
To receive the raw `value` directly, the function's parameter must be named exactly `node_input`.

```python
# Correct: receives the raw value passed to node_input
def my_worker(node_input: str):
  return f"Done: {node_input}"

# Incorrect: will fail because it tries to look up 'data' in session state
def my_worker(data: str):
  return f"Done: {data}"
```

#### Binding Dictionary Keys to Parameters (`parameter_binding='node_input'`)
If you pass a dictionary to `node_input` (e.g., `node_input={'foo': 'bar'}`) and want to bind its keys to individual function parameters (e.g., `def my_worker(foo: str)`), you must configure the node with `parameter_binding='node_input'`.

You can configure this using the `@node` decorator with `parameter_binding='node_input'`:

```python
from google.adk.workflow import node

# Decorate with parameter_binding='node_input'
@node(parameter_binding='node_input')
def my_worker(foo: str):
  return f"Done: {foo}"

# Call via ctx.run_node
result = await ctx.run_node(my_worker, node_input={'foo': 'bar'}) # foo gets 'bar'
```


### 3. Nested Dynamic Nodes

If a dynamically scheduled node *itself* calls `ctx.run_node()`, it becomes a parent and must also have `rerun_on_resume=True`.
You should decorate the nested function with `@node(rerun_on_resume=True)` to ensure it has this property when executed:

```python
from google.adk.workflow import node

@node(rerun_on_resume=True)
async def inner_parent(ctx: Context):
  # Calls another dynamic node internally
  result = await ctx.run_node(some_child)
  yield Event(output=result)

# In the outer parent:
await ctx.run_node(inner_parent)
```


### 4. Generator Returns

In nodes that use `yield` (generators), you cannot use `return value` to produce the final output of the node due to Python syntax constraints. You must yield `Event(output=value)` instead.

## Method Signature

```python
async def run_node(
    self,
    node: NodeLike,
    node_input: Any = None,
    *,
    use_as_output: bool = False,
    run_id: str | None = None,
    use_sub_branch: bool = False,
    override_branch: str | None = None,
) -> Any:
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `node` | `NodeLike` | *Required* | The node to execute (Function, Agent, or Workflow). |
| `node_input` | `Any` | `None` | Input data to pass to the dynamic node. |
| `use_as_output` | `bool` | `False` | If `True`, the child node's output is used as the calling parent node's output. The parent's own output event is suppressed. Can only be set once per parent execution. |
| `run_id` | `str \| None` | `None` | Optional custom run ID. If provided, **must contain non-numeric characters** (e.g., `"run_a"`) to prevent collision with auto-generated IDs. |
| `use_sub_branch` | `bool` | `False` | If `True`, executes the node in a sub-branch (appending `node_name@run_id` to the branch path). Essential for parallel runs to isolate events. |
| `override_branch` | `str \| None` | `None` | Explicitly overrides the branch name for the execution context. |

## Advanced Applications

### Dynamic Fan-Out (Parallel Execution)

You can perform dynamic fan-out by scheduling multiple tasks in parallel using `asyncio.gather`. When doing this, you **must** set `use_sub_branch=True` to isolate the events of each parallel execution.

```python
import asyncio
from google.adk import Context, Event, Agent
from google.adk.workflow import node

worker = Agent(name="worker", instruction="Process {node_input}")

@node(rerun_on_resume=True)
async def parallel_orchestrator(ctx: Context, node_input: list[str]):
  tasks = []
  for topic in node_input:
    tasks.append(
        ctx.run_node(
            worker,
            node_input=topic,
            use_sub_branch=True, # Critical for parallel isolation
        )
    )

  # Await all tasks concurrently
  results = await asyncio.gather(*tasks)
  yield Event(output=results)
```

## Best Practices

- **Avoid Unsupervised Tasks**: Always `await` `ctx.run_node()` directly (or via `asyncio.gather`). Do **not** wrap it in `asyncio.create_task()` without awaiting it, as errors will be swallowed, and tasks won't be cancelled if the workflow is interrupted.
- **Manage Side Effects and Resumption**: Because a parent node with `rerun_on_resume=True` is executed from the beginning on resumption, any code with side effects (e.g., database writes, API calls) in the parent node will run again.
    - *Best Practice*: Keep the parent orchestrator node's logic as light as possible, containing mostly control flow and `ctx.run_node` calls.
    - *Best Practice*: Move any logic with side effects into dedicated child nodes and execute them via `ctx.run_node`. Since completed child nodes are cached and replayed, their side effects will *not* be executed again on resumption.


## Limitations

- **Replay Overhead**: Because the parent node is re-run from the beginning on resume, long-running parent node logic (outside of `ctx.run_node` calls) will be re-executed. Keep the orchestrator node logic light and delegate heavy lifting to child nodes.

## Related samples

- [Dynamic Nodes Sample](../../../../contributing/samples/workflows/dynamic_nodes/)
- [Dynamic Fan-Out / Fan-In Sample](../../../../contributing/samples/workflows/dynamic_fan_out_fan_in/)
