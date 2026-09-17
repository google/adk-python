# Dynamic node scheduling

Some execution paths cannot be drawn ahead of time, because which nodes run, or how many, depends on what the workflow finds out while it is running. For those, a node can call `ctx.run_node()` to run another node on the spot and wait for its result. You build the shape imperatively, with ordinary Python control flow such as a loop or a condition, instead of static graph edges.

## Introduction

A graph written as `Workflow(edges=[...])` handles structured work well, and most of the time it is what you want. What it cannot express is a shape that only becomes known once the workflow is under way, which covers cases like these:

- Looping over a set of nodes until a condition holds, as a generator-evaluator loop does.
- Running a number of tasks in parallel where the number itself comes from runtime input, which is dynamic fan-out.
- Deciding whether to run a node at all, on logic too involved to write down as edges.

In each of those, `ctx.run_node()` lets a parent node execute a child, which may be a function, an agent or another workflow, and await whatever it produces.

## Get started

In this example a parent node runs a child agent and passes what the agent produced back out as its own output. The `rerun_on_resume=True` on the parent is not optional: every node that calls `ctx.run_node()` has to set it.

```python
from google.adk import Agent, Context, Event, Workflow
from google.adk.workflow import node, START

# Define a child agent
generate_headline = Agent(
    name="generate_headline",
    instruction="Write a catchy headline about the topic in the user message.",
)


# Define the parent orchestrator node (MUST have rerun_on_resume=True).
# No return annotation: this is a generator, and it produces its output by
# yielding an Event rather than returning a value.
@node(rerun_on_resume=True)
async def orchestrate(ctx: Context, node_input: str):
  # Dynamically execute the child agent and await its output
  headline = await ctx.run_node(generate_headline, node_input=node_input)

  yield Event(output=headline)

# Build the workflow
root_agent = Workflow(
    name="root_agent",
    edges=[(START, orchestrate)],
)
```


## How it works

Three things follow from an `await ctx.run_node(node_like, ...)`.

1.  **The child runs outside the graph.** It executes even though no edge joins it to the parent, and the parent waits for whatever it produces.
2.  **Its state is tracked under the parent.** The child's execution state and events live at a path beneath the parent node's own, such as `parent_node@1/child_node@1`.
3.  **Resuming works through replay.** If the child interrupts, waiting for user input for instance, the parent is paused along with it. When the workflow resumes, the parent is re-run from the top, which is what `rerun_on_resume=True` is for, but `ctx.run_node()` calls that already succeeded are replayed from history and return their cached outputs rather than executing again.

### Input mapping

Where the `node_input` in `ctx.run_node(node, node_input=value)` ends up depends on what kind of child you are running:

-   **Python functions and `FunctionNode`s** receive the value directly, in the parameter named `node_input`. Other parameters are bound from the session state, as they are in the default mode.
-   **Agents in single-turn mode** get the value converted into a user-role message, a `types.Content`, which is appended to the session event history. The agent sees it as the incoming user message.
-   **Agents in task mode** get the value as `user_content` on the `InvocationContext`, which serves as the fallback first user turn for a task agent that was not triggered by a tool call.

## Requirements and rules

Four rules govern a node that schedules children: two about surviving an interrupt, one about how a child receives its input, and one about how the parent produces its own output.

### 1. `rerun_on_resume=True` is mandatory for parents

Any node that calls `ctx.run_node()` **must** be configured with `rerun_on_resume=True`, and a parent without it raises a `ValueError` at runtime the moment it makes the call. The reason is the replay described above: a parent that cannot be re-run cannot pick its children back up after an interrupt. `rerun_on_resume` is one of the options every node carries, and [BaseNode](../base_node/index.md) describes what it means for a node that does *not* schedule dynamic children.

### 2. Function parameter mapping

Functions wrapped as nodes look their arguments up in the session state by default, which is state binding. The `node_input` argument you pass to `ctx.run_node(..., node_input=value)` is the exception: it goes straight to the node.

How that value reaches your code depends on how you defined the function.

#### Pass-through `node_input`

To receive the raw value directly, name the function's parameter exactly `node_input`. Any other name sends the framework looking in session state instead, and a name that is not there raises `ValueError: Missing value for parameter "<name>"`.

```python
def my_worker(node_input: str):
  return f"Done: {node_input}"
```

#### Bind dictionary keys to parameters

When you want to pass several values at once, send a dictionary as `node_input` and have its keys bound to individual parameters. That takes `parameter_binding='node_input'` on the node, which you set through the `@node` decorator:

```python
from google.adk.workflow import node

# Decorate with parameter_binding='node_input'
@node(parameter_binding='node_input')
def my_worker(foo: str):
  return f"Done: {foo}"

# Call via ctx.run_node
result = await ctx.run_node(my_worker, node_input={'foo': 'bar'}) # foo gets 'bar'
```


### 3. Nested dynamic nodes

The first rule applies at every level. A dynamically scheduled node that *itself* calls `ctx.run_node()` has become a parent, so it needs `rerun_on_resume=True` as well. Decorate the nested function with `@node(rerun_on_resume=True)` so that it carries the property when it runs:

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


### 4. Generator returns

The parent nodes in these examples are all generators, since they use `yield`, and in a generator `return value` does not produce the node's output. Write `yield Event(output=value)` instead.

What `return value` does depends on the flavor of generator, and only one of the two tells you about it. An `async def` generator, which is what all the parents here are, rejects the line with a `SyntaxError` before anything runs. A plain generator accepts it and then throws the value away, so the node emits nothing at all and no error is raised to say so.

## Method signature

`ctx.run_node()` takes the node to run, the input to hand it, and six keyword
arguments that control how the child's run is recorded and what a child left
waiting gives back.

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
    override_isolation_scope: str | None = None,
    raise_on_wait: bool = False,
) -> Any: ...
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `node` | `NodeLike` | *Required* | The node to execute (Function, Agent, or Workflow). |
| `node_input` | `Any` | `None` | Input data to pass to the dynamic node. |
| `use_as_output` | `bool` | `False` | If `True`, the child node's output is used as the calling parent node's output. The parent's own output event is suppressed. Can only be set once per parent execution. |
| `run_id` | `str \| None` | `None` | Optional custom run ID. If provided, **must contain non-numeric characters**, such as `"run_a"`, to prevent collision with auto-generated IDs. |
| `use_sub_branch` | `bool` | `False` | If `True`, executes the node in a sub-branch (appending `node_name@run_id` to the branch path). Essential for parallel runs to isolate events. |
| `override_branch` | `str \| None` | `None` | Explicitly overrides the branch name for the execution context. |
| `override_isolation_scope` | `str \| None` | `None` | Overrides the isolation scope the child inherits from the parent. |
| `raise_on_wait` | `bool` | `False` | Changes what a child left `WAITING` gives back. `False` returns `None`; `True` raises `NodeInterruptedError`. |

`raise_on_wait` is narrower than its name suggests, because two situations look
alike from the parent's side and the flag only covers one of them.

Take first the child that *interrupts*, meaning one that yields a `RequestInput`
to ask the user something. That child always raises `NodeInterruptedError` out of
`ctx.run_node()`, whatever `raise_on_wait` is set to, and your code after the
`await` does not run at all on that pass. It is the behavior you want and it
needs no configuration.

What `raise_on_wait` actually covers is the other case, a child that finished
without producing an output and was left in the `WAITING` state. That happens
when the child has
`wait_for_output=True`, or when the child is a nested `Workflow`. By default
`ctx.run_node()` returns `None` for it, which is indistinguishable from a child
that ran and legitimately produced nothing, and the parent goes on to complete
as though the work were done. Set `raise_on_wait=True` and the call raises
`NodeInterruptedError` instead, so the parent is recorded as `WAITING` too
rather than falsely `COMPLETED`.

## Advanced applications

The pattern that brings most people to `ctx.run_node()` is a fan-out whose width is only known once the run has started.

### Dynamic fan-out

To fan out dynamically, schedule the child runs together and gather them with `asyncio.gather`. Every one of those runs **must** set `use_sub_branch=True`, which keeps each execution's events in a branch of its own; without it their events land on top of each other.

If all you need is one node applied to every item of a list, [parallel worker mode](../parallel_worker/index.md) does this for you with a flag. Build it by hand, as below, when you want control the flag does not offer.

```python
import asyncio
from google.adk import Context, Event, Agent
from google.adk.workflow import node

# The topic arrives as the agent's incoming user message. A brace placeholder
# such as {node_input} would be looked up in session state and raise KeyError.
worker = Agent(name="worker", instruction="Process the topic in the user message.")

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

## Best practices

**Do not leave a child run unsupervised.** Always `await` `ctx.run_node()` directly, or through `asyncio.gather`. Wrapping it in `asyncio.create_task()` and never awaiting the task swallows any error it raises, and the task is not canceled if the workflow is interrupted.

**Plan for the parent running twice.** A parent with `rerun_on_resume=True` is executed from the beginning when the workflow resumes, so anything in it with a side effect, a database write or an API call for instance, happens a second time. Two habits keep that from hurting:

- Keep the parent orchestrator light. It should be mostly control flow and `ctx.run_node` calls, with as little else as you can manage.
- Push the side effects down into child nodes and run those through `ctx.run_node`. Completed children are cached and replayed rather than re-executed, so their side effects do *not* happen again.

## Limitations

- **Replaying a parent costs whatever the parent costs.** Since it is re-run from the beginning on resume, any long-running logic sitting outside the `ctx.run_node` calls is paid for twice. The cost is the same argument for keeping the orchestrator thin and handing the heavy work to child nodes.

## Related samples

- [Dynamic Nodes](../../../../contributing/samples/workflows/dynamic_nodes/agent.py): a parent node driving a child agent in a loop until a condition holds.
- [Dynamic Fan-Out / Fan-In](../../../../contributing/samples/workflows/dynamic_fan_out_fan_in/agent.py): a variable number of parallel child runs gathered with `asyncio.gather`.
- [Use As Output](../../../../contributing/samples/workflows/use_as_output/agent.py): handing a child node's output straight out of the parent.
