# GraphAgent Pattern — DynamicNode (Mixture of Experts)

This example implements **runtime agent selection** using `DynamicNode`. A classifier labels the
incoming task as SIMPLE or COMPLEX, then `DynamicNode` routes to a cheap fast model or a thorough
capable model accordingly — a sparse mixture-of-experts dispatch optimizing cost vs. quality.

## When to Use This Pattern

- Cost optimisation: route easy tasks to cheaper models, hard tasks to capable models
- Capability dispatch: pick a specialist agent based on detected task domain
- Fallback chains: try a fast agent first, escalate to a powerful agent on failure

## How to Run

```bash
adk run contributing/samples/graph_agent_pattern_dynamic_node
```

## Graph Structure

```
classify ──▶ respond (DynamicNode)
                 ├── simple_agent   (when classify output contains "SIMPLE")
                 └── detailed_agent (otherwise)
```

## Key Code Walkthrough

- **`DynamicNode(name="respond", agent_selector=select_responder)`** — the selector callable
  receives `GraphState` and returns the `BaseAgent` to invoke
- **`select_responder(state)`** — reads `state.data["classify"]` and returns the matching agent
- **`fallback_agent`** parameter — used when the selector returns `None`
- **Transparent to the graph** — downstream edges see `respond`'s output regardless of which
  agent was chosen
- **No graph-level changes needed** — swap agents by changing `select_responder`, not the graph
  topology

