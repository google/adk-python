# GraphAgent Pattern — NestedGraphNode (Hierarchical Composition)

This example demonstrates **hierarchical workflow decomposition** using `NestedGraphNode`. A
coordinator planner produces a focused query, a three-step research sub-graph (search → extract →
summarise) handles it as a single reusable unit, and a synthesiser produces the final answer.
The sub-graph is entirely encapsulated and independently testable.

## When to Use This Pattern

- Large workflows that benefit from breaking into independently developed sub-pipelines
- Reusable sub-workflows (the same research graph could be called from multiple parent graphs)
- Team boundaries: different teams own the outer orchestration and inner sub-workflows
- Recursive depth: sub-graphs can themselves contain `NestedGraphNode`s

## How to Run

```bash
adk run contributing/samples/graph_agent_pattern_nested_graph
```

## Graph Structure

```
Outer: plan ──▶ research (NestedGraphNode) ──▶ synthesise

Inner (research sub-graph):
       search ──▶ extract ──▶ summarise
```

## Key Code Walkthrough

- **`NestedGraphNode(name="research", graph_agent=build_research_subgraph())`** — wraps an entire
  `GraphAgent` as a single node in the parent graph
- **`inherit_session=True`** — the sub-graph shares the parent session's state, so outputs
  written inside are visible to the parent's synthesiser
- **`build_research_subgraph()`** — factory function that constructs and returns the inner
  `GraphAgent`; call it multiple times for independent instances
- **State bridging** — the sub-graph's final state is merged back; use `output_mapper` on the
  `NestedGraphNode` to control which keys are exposed to the outer graph
- **Telemetry and checkpointing** — propagate automatically into the sub-graph when enabled on
  the parent

