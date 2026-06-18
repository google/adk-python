# Workflow

`Workflow` is a graph-based orchestration node in ADK 2.0 that executes a Directed Acyclic Graph (DAG) of nodes. It manages the execution flow, including parallel branch execution, dynamic node scheduling, and resuming execution from session events.

## Introduction

The `Workflow` class allows developers to define complex, multi-step agent interactions as a graph of nodes. It acts as the orchestration engine, coordinating the execution of individual nodes (which can be agents, tools, or custom functions) based on defined edges and routing logic.

Key classes that depend on `Workflow`:
- **`Runner`**: The execution engine that runs the workflow.
- **`App`**: The container that registers the workflow as an agent.

Developer problems solved by `Workflow`:
- **Complex Orchestration**: Managing multi-agent systems with conditional transitions and parallel execution.
- **State Preservation and Resumption**: Workflows can pause for human input or external events and resume later, reconstructing their state from session history.
- **Dynamic Execution**: Support for spawning nodes dynamically at runtime based on data.

## Get started

The following example demonstrates a simple sequential workflow with two steps defined using Python functions.

```python
from google.adk import START, Workflow

# Define simple Python functions to act as workflow nodes
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



## How it works

- **Compilation**: During initialization, the `Workflow` compiles the provided `edges` into an internal `Graph` representation and validates it (e.g., checking for duplicate node names or unconditional cycles).
- **Execution Loop**: The core orchestration happens in `_run_impl()`. It maintains a loop that:
    1. Schedules "ready" nodes (nodes whose predecessors have completed).
    2. Runs each scheduled node in a separate `NodeRunner` as an asyncio task.
    3. Waits for tasks to complete.
    4. Handles completion by caching outputs and buffering triggers for downstream nodes.
- **Rehydration (Resume)**: When a workflow is resumed (e.g., after a human-in-the-loop interrupt), it scans the session history for previous execution events. It reconstructs the state of completed nodes to avoid re-running them and deterministically replays the execution flow up to the interrupt point.
- **Dynamic Scheduling**: Workflows support dynamic node execution via `ctx.run_node()`. The workflow registers a `DynamicNodeScheduler` on the context, allowing nodes to spawn and await other nodes dynamically, bypassing the static graph edges.

## Workflow Output

A workflow is itself a node, and when it finishes, it can produce an output. The output of a workflow is determined by the outputs of its **terminal nodes** (nodes with no outgoing edges):

1.  **Single Terminal Output**: The workflow's output is the output of the terminal node that executed and completed with a non-`None` result.
2.  **Multiple Terminal Nodes**: If your graph has multiple terminal nodes (e.g., parallel branches):
    *   If only **one** of them executes and produces an output (e.g., in a conditional branching scenario where only one path is taken), that output becomes the workflow's output.
    *   If **multiple** terminal nodes execute and produce outputs (e.g., parallel branches executing concurrently), the workflow will fail with a `ValueError` upon completion.
3.  **Aggregating Outputs**: If you have parallel branches and want to return their combined results, you **must** use a `JoinNode` to synchronize the branches and aggregate their outputs into a single output before the workflow ends. The `JoinNode` then acts as the single terminal node of the workflow.

## Configuration options

The `Workflow` class introduces specific configuration options to define the graph structure and control execution concurrency:

### `edges`
*   **Type**: `list[EdgeItem]`
*   **Default**: `[]`
*   **Description**: Defines the structure of the workflow graph by specifying connections between nodes. The graph must have a single entry point starting from the `START` sentinel node.
*   **Usage Patterns**:
    *   **Sequential Chain**: Define a list of tuples representing sequential steps.
        ```python
        edges=[(START, step_a, step_b)]
        ```
    *   **Parallel Fan-Out**: Route from one node to multiple downstream nodes in parallel.
        ```python
        edges=[
            (START, step_a, (step_b, step_c)),
        ]
        ```
    *   **Conditional Routing**: Route to different nodes based on a routing key returned by the predecessor.
        ```python
        edges=[
            (START, step_a, {"route_x": step_b, "route_y": step_c}),
        ]
        ```
        In this case, `step_a` must yield an `Event` with the `route` field set to either `"route_x"` or `"route_y"`.

### `max_concurrency`
*   **Type**: `int | None`
*   **Default**: `None` (unlimited)
*   **Description**: Limits the number of static graph nodes that can execute in parallel. This is useful for managing resource usage or rate limits when running workflows with large parallel branches.
*   **Usage Patterns**:
    *   **Throttling parallel tasks**: If you have a fan-out of 100 tasks but want to run at most 5 at a time:
        ```python
        workflow = Workflow(
            name="throttled_workflow",
            edges=[(START, list_of_parallel_nodes)],
            max_concurrency=5,
        )
        ```
    *   *Note*: This limit only applies to nodes scheduled via graph edges. Dynamic nodes spawned via `ctx.run_node()` are excluded from this limit to prevent deadlocks (as they are awaited inline by their parent node).


## Advanced applications

### Nested Workflows
A `Workflow` is itself a `BaseNode`, meaning it can be used as a node inside another workflow. This allows for modular and hierarchical workflow design.

### Joining Parallel Branches
You can use `JoinNode` to synchronize parallel execution paths. A `JoinNode` waits for all its predecessors to complete before it executes and aggregates their outputs.

### Dynamic Node Execution
For scenarios where the execution path cannot be predefined, nodes can use `ctx.run_node(node_instance, input)` to execute nodes dynamically.

## Limitations

- **Task-Mode Agents**: `LlmAgent` instances configured with `mode='task'` cannot be used as static nodes in a workflow graph. This is because the workflow scheduler currently overwrites the originating input on re-entry, causing the agent to lose its task brief. Use them as sub-agents or dispatch them dynamically.
- **Unconditional Cycles**: The workflow graph validator rejects graphs with unconditional cycles to prevent infinite loops. Cycles must be conditional (i.e., controlled by routing logic).

## Related samples

The following samples demonstrate various workflow features:
- [Sequence Workflow](../../../../contributing/samples/workflows/sequence/agent.py) - Basic sequential execution.
- [Conditional Routing](../../../../contributing/samples/workflows/route/agent.py) - Branching based on node outputs.
- [Looping Workflow](../../../../contributing/samples/workflows/loop/agent.py) - Iterative execution.
- [Nested Workflows](../../../../contributing/samples/workflows/nested_workflow/agent.py) - Workflows containing other workflows.
- [Parallel Execution (Fan-Out/Fan-In)](../../../../contributing/samples/workflows/fan_out_fan_in/agent.py) - Running tasks in parallel and joining them.
- [Dynamic Nodes](../../../../contributing/samples/workflows/dynamic_nodes/agent.py) - Executing nodes dynamically via context.
- [Node Retries](../../../../contributing/samples/workflows/retry/agent.py) - Configuring error handling and retries.
