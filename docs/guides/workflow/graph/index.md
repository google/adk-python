# Workflow graphs

A workflow in ADK 2.0 is a directed graph. The nodes are the work, the edges say what follows what, and execution moves along those edges until there is nothing left to run. The chain tuple shown on the [Workflow](../workflow/index.md) page is one of two ways to write that graph down, and whichever you use, the finished graph is checked against nine rules before it runs.

## Introduction

The graph is the execution plan for a multi-step agent interaction. It specifies three things:

- What work there is to do, in the shape of **nodes**.
- What order that work happens in, in the shape of **edges**.
- How data moves between steps, and where branches fork apart and merge back together.

None of it is checked lazily. The graph is compiled and validated the moment you instantiate the `Workflow` class, so a graph that could never have worked is rejected before a single node runs.

## Get started

The graph below is complete: `START` fans out to two nodes that run in parallel,
a `JoinNode` waits for both, and a final node reads the aggregated result. The
examples further down are fragments of a workflow shaped like this one rather
than complete programs.

```python
from typing import Any

from google.adk import Workflow
from google.adk.workflow import JoinNode, START

def word_count(node_input: str) -> int:
    return len(node_input.split())

def shout(node_input: str) -> str:
    return node_input.upper()

join = JoinNode(name="join")

def report(node_input: dict[str, Any]) -> str:
    return f"{node_input['shout']} ({node_input['word_count']} words)"

workflow = Workflow(
    name="fan_out_workflow",
    edges=[
        (START, (word_count, shout), join, report),
    ],
)
```

The single chain tuple expands into five edges: `START` to each of the two
workers, each worker to `join`, and `join` to `report`. Both workers receive the
same input, the user's message, because they both follow `START`.

## Core concepts

Both halves of a graph have a type you can name in your own code. `NodeLike`
covers everything that can serve as a node, and `Edge` is a single transition
written out in full.

### Nodes (`NodeLike`)

A node is one unit of execution in the workflow. Several different kinds of object can serve as one, and the type that covers all of them is called `NodeLike`:

1.  **Python functions.** Sync functions, async functions and generators all qualify. Pass one straight into a chain tuple and ADK wraps it in a `FunctionNode` for you, so a bare function is already a node. The `@node` decorator is there to configure that wrapper, for when you want a `retry_config`, a `timeout` or `rerun_on_resume` on it, and you never need it to make a function usable in the first place. The exception is an explicit `Edge`, which takes a `BaseNode` rather than a function, so a function used there has to be wrapped even when there is nothing to configure. See [Explicit edge objects](#2-explicit-edge-objects).
2.  **Agents.** `LlmAgent` instances, usually in `single_turn` mode. Pass one straight into a chain tuple, the way you would a function.
3.  **Tools.** `BaseTool` instances, which go into a chain tuple in the same way.
4.  **Workflows.** A `Workflow` is itself a [`BaseNode`](../base_node/index.md), so one workflow can be nested as a child node inside another.
5.  **`START`.** The sentinel that marks the entry point. It has to be present in the graph and nothing may point into it, since it never executes and so an edge arriving at it could never fire, but any number of edges may leave it, and the fan-out/fan-in sample opens three parallel branches straight from `START`. Import it from `google.adk.workflow` and not from `google.adk`, because the top-level package does not export it and the wrong import fails with `ImportError` before any node runs. [Workflow](../workflow/index.md) shows which names the top-level package does export. The string `"START"` is an accepted alias for the `START` object and the shipped samples use it, but the examples here stay with the object, since that is the form `Edge(from_node=START, ...)` requires.

### Edges (`Edge`)

An edge is a transition from a source node (`from_node`) to a destination node (`to_node`).

#### Unconditional edges

Edges are unconditional unless you say otherwise, which means that as soon as the source node completes, execution moves on to the destination.

#### Conditional edges

An edge can also carry one or more **routes**, each of which is a string, an integer or a boolean. An edge with routes is followed only when the source node emits one that matches.

To emit a route, the source node yields an `Event(route="my_route")`, or returns or yields an object that maps to that route.

#### Default route

For the cases you have not enumerated, you can add a fallback edge using `DEFAULT_ROUTE`, which you get with `from google.adk.workflow import DEFAULT_ROUTE` or write as the literal `"__DEFAULT__"`. That edge is taken when the source node emits a route and no specific conditional edge matches it.

Without a fallback, a route that matches no edge ends that branch where it stands. Nothing downstream of the source node runs, and the workflow carries on with whatever other branches it has.

---

## Graph syntax

However you build it, the graph reaches the framework the same way: as a list of `edges` passed to the `Workflow` constructor. There are two ways to write that list, and you can mix them.

### 1. Chain tuples

A chain tuple describes a run of nodes in the order they should execute, and nesting inside it is what gives you parallel and conditional transitions. The chain tuple is the shorter of the two syntaxes, and most graphs are written with it.

*   **Sequential Chain**:
    ```python
    edges=[
        (START, step_a, step_b, step_c),
    ]
    ```
    Each node runs once the one before it has finished, which gives you
    `START -> step_a -> step_b -> step_c`.

*   **Parallel Fan-Out**: Use a tuple of nodes to split execution into parallel branches.
    ```python
    edges=[
        (START, step_a, (step_b, step_c)),
    ]
    ```
    `step_a` runs after `START`, and once it finishes both `step_b` and
    `step_c` start at the same time.

    Watch the end of that graph, though. As written it finishes with two
    terminal nodes, and if both of them produce an output the run fails with a
    `ValueError` saying multiple terminal nodes produced output. Send the
    branches into a `JoinNode` first, the way the fan-out example above does.
    [Workflow Output](../workflow/index.md#workflow-output) explains why the
    rule exists.

*   **Conditional Routing**: Use a dictionary, called a routing map, to define conditional branches.
    ```python
    from google.adk.workflow import DEFAULT_ROUTE

    edges=[
        (START, step_a, {
            "success": step_b,
            "failure": step_c,
            DEFAULT_ROUTE: fallback_step,
        }),
    ]
    ```
    If `step_a` yields `Event(route="success")` execution goes to `step_b`, and
    if it yields `"failure"` it goes to `step_c` instead. Any other route falls
    through to `fallback_step`.

    A routing map value may itself be a tuple, in which case matching that route
    fans out to every node in it:

    ```python
    edges=[
        (START, step_a, {"success": (step_b, step_c), "failure": fallback_step}),
    ]
    ```

    That is a fan-out like the one above, so the same terminal-output rule
    applies to `step_b` and `step_c`.

### 2. Explicit edge objects

The second syntax names every edge individually with an `Edge` object. It is more to type, and worth it once a graph is large enough that a nested tuple stops being readable, or whenever you would rather see each transition spelled out.

```python
from google.adk.workflow import Edge, START, node

@node()
def step_a(node_input: str):
    ...

@node()
def step_b(node_input: str):
    ...

@node()
def step_c(node_input: str):
    ...

edges=[
    Edge(from_node=START, to_node=step_a),
    Edge(from_node=step_a, to_node=step_b, route="success"),
    Edge(from_node=step_a, to_node=step_c, route="failure"),
]
```

`Edge` is stricter than a chain tuple about what counts as a node. Both
`from_node` and `to_node` are annotated `BaseNode`, and Pydantic enforces that,
so a bare function is rejected with "Input should be a valid dictionary or
instance of BaseNode". Wrap the function first, with `@node()` as above or with
`FunctionNode(func=...)`. For the same reason the string `"START"` works in a
chain tuple but not here; pass the `START` object.

---

## Graph validation

Initializing a `Workflow` builds the graph and checks it, so that a structural mistake surfaces while you are still writing the graph rather than in the middle of a run. There are nine rules, and all of them are enforced.

### 1. Unique node names

Every distinct node object in the graph needs its own name, so two different function nodes both called `process_data` fail validation. The name is how the graph, the event stream and a `JoinNode`'s aggregated output all refer to a node, and none of those could tell the two apart. Rename one of them. If what you actually wanted was for the graph to come back to the same step twice, reuse the exact same object instance instead of creating a second one, and the rule stops applying.

### 2. START is present and has no incoming edges

The graph has to contain the `START` node, and no edge may name `START` as its destination. Any number of edges may leave it. A graph with no `START` at all, or one with an edge pointing back into it, is rejected. Execution begins at `START` and `START` itself never runs, so an edge arriving there could never fire and a graph without one has nowhere to begin.

### 3. Edges leaving START carry no route

An edge out of `START` must be unconditional, because nothing has run yet that could have emitted a route. Writing `(START, {"a": node_a})` fails with "Edges from START must not have routes". When you want to branch immediately, put the router in a node of its own and branch out of that.

### 4. Reachability from START

Every node in the graph must be reachable from `START`. A node nothing points at never becomes ready, so it would sit in the graph without ever running, and validation fails rather than quietly leaving it out. Defining a node and then forgetting to connect it is the usual way to trip this rule.

### 5. No duplicate edges

The same two nodes may be joined only once. Listing `Edge(from_node=A, to_node=B)` twice in the same edge list fails, which is what catches the copy-and-paste slip. Repeating an edge is not how you make `B` run more than once. A second run comes from giving `B` another predecessor, or from a conditional loop back to it.

### 6. Default route constraints

A node may have at most one outgoing `DEFAULT_ROUTE` edge, since two fallbacks would leave the choice undefined. `DEFAULT_ROUTE` also cannot share a list with other routes, so `route=["success", DEFAULT_ROUTE]` is invalid.

### 7. No unconditional cycles

A cycle made entirely of unconditional edges, such as `A -> B -> A` with no routes anywhere in it, is rejected, because nothing in the graph would ever break out of it. Conditional loops are allowed and are how you write iteration: the same `A -> B -> A` is fine as long as the edge from `B` back to `A` depends on a route.

### 8. Static schema matching

If a node has an `output_schema` and its successor has an `input_schema`, the two must be equal, and the graph is rejected when they are not. The rule catches people out because both schemas are often inferred rather than written: a function node takes its `output_schema` from its return type hint and its `input_schema` from its `node_input` hint, so two functions annotated with different Pydantic models fail the check without either schema having been set by hand. See [BaseNode](../base_node/index.md).

### 9. Chat agent wiring

An `LlmAgent` configured with `mode='chat'` may only follow the `START` node. A chat-mode agent manages its own conversational history and has no way to consume an input handed to it by a preceding node, so it cannot sit in the middle of a chain. Use `mode='single_turn'` for a step that takes input from the node before it.

## Limitations

- **Validation is structural, not behavioral.** The rules above are checked against the shape of the graph at construction time. Nothing checks that a node actually emits the routes its outgoing edges name. A node that emits a route no edge matches ends its branch quietly, with a warning in the log and no error.
- **Errors arrive wrapped in a Pydantic `ValidationError`.** Validation runs while the `Workflow` is being constructed, so the message above is nested in a Pydantic report rather than being the exception's first line. `ValidationError` subclasses `ValueError`, so an `except ValueError` still catches it.

## Related samples

Each of these is a working graph you can read for the shape rather than the syntax:

- [Conditional Routing](../../../../contributing/samples/workflows/route/agent.py): a router node with a three-way routing map.
- [Fan-Out / Fan-In](../../../../contributing/samples/workflows/fan_out_fan_in/agent.py): three edges out of `START`, rejoined by a `JoinNode`.
- [Sequence Workflow](../../../../contributing/samples/workflows/sequence/agent.py): the shortest chain tuple there is.
- [Looping Workflow](../../../../contributing/samples/workflows/loop/agent.py): a conditional cycle, which is the only kind the validator allows.
- [Multiple Triggers](../../../../contributing/samples/workflows/multi_triggers/agent.py): a node reached from more than one predecessor.
