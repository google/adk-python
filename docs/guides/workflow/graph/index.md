# Workflow Graphs

In ADK 2.0, workflows are represented as directed graphs where execution flows from node to node along defined edges. This guide explains the core concepts of **nodes**, **edges**, and **graphs**, how to define them, and the validation rules enforced by the framework.

## Introduction

A workflow graph defines the execution plan for your multi-step agent interactions. It specifies:
- What tasks to run (Nodes).
- The order of execution (Edges).
- How data flows and how branches fork or merge.

The graph structure is compiled and validated when you instantiate the `Workflow` class.

## Core Concepts

### Nodes (`NodeLike`)

A node represents a single unit of execution in the workflow. In ADK, you can use several types of objects as nodes (collectively referred to as `NodeLike`):

1.  **Python Functions**: Sync or async functions (and generators) decorated with `@node`. They are automatically wrapped in a `FunctionNode`.
2.  **Agents**: `LlmAgent` instances (typically in `single_turn` mode). They are automatically wrapped in an internal `_LlmAgentWrapper`.
3.  **Tools**: `BaseTool` instances. They are wrapped in a `ToolNode`.
4.  **Workflows**: A `Workflow` is itself a `BaseNode` and can be nested as a child node in another workflow.
5.  **`START`**: A special sentinel node that marks the entry point of the workflow. Every graph must have exactly one edge starting from `START`.

### Edges (`Edge`)

An edge defines a transition from a source node (`from_node`) to a destination node (`to_node`).

#### Unconditional Edges
By default, edges are unconditional. When the source node completes, execution immediately transitions to the destination node.

#### Conditional Edges (Routing)
An edge can be associated with one or more **routes** (a string, integer, or boolean). The edge is only followed if the source node explicitly emits a matching route.

To emit a route, the source node must yield an `Event(route="my_route")` (or return/yield an object that maps to that route).

#### Default Route
You can define a fallback edge using `DEFAULT_ROUTE` (imported as `from google.adk.workflow import DEFAULT_ROUTE` or using `"__DEFAULT__"`). This edge is followed if the source node emits a route, but no specific conditional edge matches it.

---

## Defining the Graph (Syntax)

You define the graph structure by passing a list of `edges` to the `Workflow` constructor. ADK supports two syntax styles:

### 1. Chain Tuples (Recommended)

Chain tuples provide a concise way to define sequential, parallel, and conditional transitions using Python tuples.

*   **Sequential Chain**:
    ```python
    edges=[
        (START, step_a, step_b, step_c),
    ]
    ```
    This defines: `START -> step_a -> step_b -> step_c`.

*   **Parallel Fan-Out**: Use a tuple of nodes to split execution into parallel branches.
    ```python
    edges=[
        (START, step_a, (step_b, step_c)),
    ]
    ```
    This defines: `START -> step_a`, and then `step_a -> step_b` AND `step_a -> step_c` in parallel.

*   **Conditional Routing**: Use a dictionary (Routing Map) to define conditional branches.
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
    If `step_a` yields `Event(route="success")`, it goes to `step_b`. If it yields `"failure"`, it goes to `step_c`. Any other route goes to `fallback_step`.

### 2. Explicit Edge Objects

For complex graphs or when you prefer explicit declarations, you can use `Edge` objects:

```python
from google.adk.workflow import Edge, START

edges=[
    Edge(from_node=START, to_node=step_a),
    Edge(from_node=step_a, to_node=step_b, route="success"),
    Edge(from_node=step_a, to_node=step_c, route="failure"),
]
```

---

## Graph Validation

When a `Workflow` is initialized, it builds an internal `Graph` representation and runs `validate_graph()` to catch structural errors early. The following rules are strictly enforced:

### 1. Unique Node Names
All distinct node objects in the graph must have unique names.
*   *Error*: If you have two different function nodes named `process_data`, validation will fail.
*   *Solution*: Ensure unique names, or reuse the exact same object instance if you want to route back to the same node.

### 2. Single START Entry Point
The graph must contain the `START` node, and `START` must not have any incoming edges.
*   *Error*: A graph without `START` or with an edge pointing back to `START` will fail validation.

### 3. Connectivity (Reachability)
All nodes in the graph must be reachable from the `START` node.
*   *Error*: If you define a node but do not connect it to the rest of the graph, validation will fail.

### 4. No Duplicate Edges
You cannot define duplicate edges between the same two nodes.
*   *Error*: `Edge(from_node=A, to_node=B)` and `Edge(from_node=A, to_node=B)` in the same list will fail.

### 5. Default Route Constraints
- A node can have at most one outgoing `DEFAULT_ROUTE` edge.
- `DEFAULT_ROUTE` cannot be combined with other routes in a list (e.g., `route=["success", DEFAULT_ROUTE]` is invalid).

### 6. No Unconditional Cycles
The graph must not contain cycles consisting entirely of unconditional edges (edges with no route).
*   *Allowed*: Conditional loops are allowed (e.g., `A -> B -> A` where `B -> A` is conditional on a route).
*   *Forbidden*: Unconditional loops (`A -> B -> A` with no routes) are rejected to prevent infinite execution loops.

### 7. Static Schema Matching
If a node defines an `output_schema` and its successor defines an `input_schema`, they must match exactly.
*   *Error*: Schema mismatch on transition edges will fail validation.

### 8. Chat Agent Wiring
`LlmAgent` instances configured with `mode='chat'` are only allowed to follow the `START` node.
*   *Reason*: Chat-mode agents manage their own conversational history and cannot consume direct inputs from preceding nodes in a workflow chain. For sequential steps, use `mode='single_turn'`.
