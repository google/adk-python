# Workflow

`Workflow` is the graph-based orchestration node for a task that takes several steps. You describe the work as a graph, giving a list of edges that say which step feeds which, and the workflow schedules each node as soon as everything pointing into it has finished. Branches that do not depend on each other run at the same time, without you arranging for that.

## Introduction

A node can be a Python function, an `LlmAgent`, a tool, or another `Workflow`, so a single graph is free to mix all four. It provides three things:

- **Conditional and parallel transitions.** Branching and forking are declared in the edge list, so they never appear as control flow in your own code.
- **Interrupt and resume.** A workflow can pause for human input and pick up later. It rebuilds what already ran from the session history, so completed nodes do not run a second time.
- **Dynamic scheduling.** A node can spawn other nodes at runtime with `ctx.run_node()` and await them, outside the static edges entirely.

A `Workflow` is itself a node. That means you hand it to a `Runner` the way you would an agent, and it also means you can nest one workflow inside another.

## Get started

The example below defines a sequential workflow with two steps written as Python functions, where the output of the first is passed into the second.

```python
from google.adk import Workflow
from google.adk.workflow import START

# Plain Python functions are accepted as nodes. ADK wraps each one in a
# FunctionNode for you; use the @node decorator only when you need to set an
# option on the wrapper.
def step_one(node_input: str) -> str:
    return f"{node_input} -> step_one"

def step_two(node_input: str) -> str:
    return f"{node_input} -> step_two"

# Define the workflow graph using edges
workflow = Workflow(
    name="simple_workflow",
    edges=[
        (START, step_one, step_two),
    ],
)
```

Run that through a `Runner` with the message `hi` and it emits two events, one
per node: first `hi -> step_one`, then `hi -> step_one -> step_two`. Each node's
output becomes the next node's `node_input`, and whatever the last node produces
is the workflow's own output.

The import line catches almost everybody once. `START` is exported from
`google.adk.workflow`, and never from `google.adk`. The top-level package
exports only `Agent`, `Context`, `Event`, `Runner` and `Workflow`, so writing
`from google.adk import START` instead gets you an `ImportError` before
anything runs.

A bare function is already a node, which is why `step_one` and `step_two` go
into the edge list undecorated. ADK wraps each one in a `FunctionNode` on your
behalf, and you add the `@node` decorator only when you want to set an option
on that wrapper. [Function nodes](../function_node/index.md) goes into what
those options are.

## How it works

There are two distinct moments to keep apart. The graph is built when you
construct the workflow, and it is executed when you run it. Four behaviors
cover everything that happens between them.

- **Validation at construction.** Construction comes first. `Workflow(...)`
  builds the graph from the `edges` you passed and checks it, which is where a
  duplicate node name or an unconditional cycle gets caught. Finding those at
  construction rather than halfway through a run is the reason the check happens
  there.
- **Execution order.** A node runs once every one of its predecessors has
  completed, and its output is held for the nodes downstream to read.
  Parallelism falls out of that rather than being a separate feature: two
  branches that do not depend on each other become ready at the same moment, so
  they are started together.
- **Resume.** A workflow that was interrupted does not start again from nothing.
  It reconstructs from the session history which nodes had already completed and
  does not execute those a second time, then replays the flow up to the point
  where the interrupt happened and carries on from there.
- **Dynamic scheduling.** Alongside all of the above, a node can call
  `ctx.run_node()` to start and await another node with no edge joining the two.
  [Dynamic Nodes](../dynamic_nodes/index.md) covers when you would want that.

## Workflow output

A workflow is a node, so when it finishes it can produce an output of its own.
That output comes from the graph's **terminal nodes**, the ones with no outgoing
edges, and the rule depends on how many of them ran.

1.  **Single terminal output.** In the common case there is one terminal node,
    it executed, and it completed with a non-`None` result. That result is the
    workflow's output.
2.  **Multiple terminal nodes.** A graph can have more than one terminal node,
    which any fan-out gives you, and then what matters is how many of them
    actually executed. If only one did, as happens with conditional branching
    where a single path is taken, its output becomes the workflow's output. If
    several executed and produced outputs, which is exactly what parallel
    branches do, the workflow fails with a `ValueError` when it completes. A
    node has one output, and nothing in the edge list says which of the
    competing results should become it, so the run stops rather than picking
    one for you.
3.  **Aggregating outputs.** If you have parallel branches and you want their
    combined results, you have to send them into a `JoinNode` before the graph
    ends. The join synchronizes the branches, aggregates their outputs into one
    value, and becomes the single terminal node of the workflow.
    [JoinNode](../join_node/index.md) shows how that looks in practice.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `edges` | `list[EdgeItem]` | `[]` | The connections between nodes, as chain tuples or `Edge` objects. |
| `max_concurrency` | `int \| None` | `None` | Upper bound on graph-scheduled nodes running at once. `None` means unlimited. |
| `graph` | `Graph \| None` | `None` | The compiled graph. Left unset, it is built from `edges` during construction. |
| `rerun_on_resume` | `bool` | `True` | Inherited from `BaseNode`, but flipped to `True` here. |

Because a workflow is itself a node, every option a node has works here too.
`timeout` and `retry_config` govern failure and slowness. `input_schema`,
`output_schema` and `state_schema` validate what passes through the workflow.
`wait_for_output` and `description` behave as they do on any other node.
[BaseNode](../base_node/index.md) describes each of them.

### `edges`

The graph must contain the `START` sentinel, and nothing may point into it.
`START` may fan out to several nodes at once. Three shapes cover most graphs:

*   **Sequential chain.** Each element in the tuple runs after the one before it.
    ```python
    edges=[(START, step_a, step_b)]
    ```
*   **Parallel fan-out.** A nested tuple splits execution into branches that run
    at the same time.
    ```python
    edges=[
        (START, step_a, (step_b, step_c)),
    ]
    ```
*   **Conditional routing.** A dictionary maps a route value to a destination.
    ```python
    edges=[
        (START, step_a, {"route_x": step_b, "route_y": step_c}),
    ]
    ```
    In that case `step_a` must yield an `Event` with `route` set to `"route_x"` or
    `"route_y"`. Edges leaving `START` may not carry a route, since nothing has
    run yet that could emit one.

Beyond those three, [Workflow graphs](../graph/index.md) has the rest of the
syntax, along with the nine rules your graph is checked against before it runs.

### `max_concurrency`

Caps how many graph-scheduled nodes run at the same time, which matters when a
wide fan-out would otherwise exhaust a rate limit or a connection pool. The
snippet below throttles four parallel steps to two at a time:

```python
workflow = Workflow(
    name="throttled_workflow",
    edges=[(START, (step_a, step_b, step_c, step_d), join, collect)],
    max_concurrency=2,
)
```

The limit counts only nodes triggered by graph edges. Nodes spawned with
`ctx.run_node()` are exempt, because their parent awaits them inline and
throttling them would deadlock the workflow.

### `rerun_on_resume`

`BaseNode` defaults this to `False`, and `Workflow` overrides it to `True`. A
nested workflow therefore runs its own graph again when the parent resumes,
rather than completing immediately with the resuming input as its output. The
override exists because the alternative would take the resuming input as the
nested workflow's own answer and skip every step inside it. Completed child
nodes are replayed from history, so their side effects do not happen twice.

## Advanced applications

Three shapes take a graph past a fixed run of steps: a workflow used as a node
inside another one, parallel branches merged back into a single value, and
nodes chosen while the run is already under way.

### Nested workflows

A `Workflow` is itself a [`BaseNode`](../base_node/index.md), so you can drop one
into another workflow's edge list as an ordinary node. That is how a graph that
has grown too big to read gets broken into named pieces, each of which still
makes sense on its own.

### Joins for parallel branches

[`JoinNode`](../join_node/index.md) is how you bring parallel paths back
together. It waits for all of its predecessors to complete, then hands their
outputs to the next node as one aggregated value.

### Dynamic node execution

When the execution path cannot be written down ahead of time, a node can call
[`ctx.run_node()`](../dynamic_nodes/index.md) to run another node at runtime and
wait for its result, without any edge connecting the two.

## Limitations

- **Unconditional cycles are rejected.** The graph validator refuses a cycle made entirely of unconditional edges, because nothing would ever break out of it. A loop has to be conditional, which is to say controlled by routing logic.
- **Validation happens at construction, inside Pydantic.** A structural problem surfaces the moment you call `Workflow(...)` rather than when you run it, and Pydantic wraps the message in a `ValidationError`. That is a subclass of `ValueError`, so an `except ValueError` still catches it, but the graph error text is nested inside the Pydantic report rather than being the exception's first line.

## Related samples

Each of these is a runnable workflow that exercises one of the features above:

- [Sequence Workflow](../../../../contributing/samples/workflows/sequence/agent.py): steps running one after another.
- [Conditional Routing](../../../../contributing/samples/workflows/route/agent.py): branching on what a node emitted.
- [Looping Workflow](../../../../contributing/samples/workflows/loop/agent.py): a conditional cycle that repeats until it is done.
- [Nested Workflows](../../../../contributing/samples/workflows/nested_workflow/agent.py): a workflow used as a node inside another one.
- [Parallel Execution (Fan-Out/Fan-In)](../../../../contributing/samples/workflows/fan_out_fan_in/agent.py): branches running at once and then being joined.
- [Dynamic Nodes](../../../../contributing/samples/workflows/dynamic_nodes/agent.py): nodes scheduled at runtime through the context.
- [Node Retries](../../../../contributing/samples/workflows/retry/agent.py): error handling and a retry policy on a node.
