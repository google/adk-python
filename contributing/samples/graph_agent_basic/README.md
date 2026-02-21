# GraphAgent Basic Example — Conditional Routing

This example demonstrates a data validation pipeline using **conditional routing** based on runtime
state. The validator checks input quality and branches to either a processor (success path) or an
error handler (failure path), showing how GraphAgent enables state-dependent decision logic that
sequential or parallel agent composition alone cannot achieve.

## When to Use This Pattern

- Any workflow requiring "if X then A, else B" branching on agent output
- Input validation before expensive downstream processing
- Quality-gate patterns where the next step depends on a score or classification

## How to Run

```bash
adk run contributing/samples/graph_agent_basic
```

## Graph Structure

```
validate ──(valid=True)──▶ process
         ──(valid=False)─▶ error
```

## Key Code Walkthrough

- **`GraphNode(name="validate", agent=validator_agent)`** — wraps an `LlmAgent` as a graph node
- **`add_edge("validate", "process", condition=lambda s: s.data["valid"] == True)`** — conditional
  edge that only fires when the validation flag is set
- **Two end nodes** (`process` and `error`) — GraphAgent can have multiple terminal nodes
- **State propagation** — each node's output is written to `state.data[node_name]` and read by
  downstream condition functions
- **No cycles** — this is a simple directed acyclic graph; for loops see `graph_agent_dynamic_queue`

