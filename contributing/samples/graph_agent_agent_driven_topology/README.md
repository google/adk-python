# graph_agent_agent_driven_topology

Demonstrates the **function-node-with-closure** pattern: a function node that
holds a reference to the `GraphAgent` calls `graph.add_node()` / `graph.add_edge()`
from *inside* the runner, injecting the topology entirely within the execution loop.

Contrasts with `graph_agent_dynamic_topology` where the outer event loop modifies
the graph between emitted events.

## Pattern

```python
def make_topology_applier(graph: GraphAgent):
    async def topology_applier(state: GraphState, ctx: InvocationContext) -> str:
        design = json.loads(state.data.get("planner", "{}"))
        prev = "topology_applier"
        for step_name in design.get("steps", []):
            node_name = f"step_{step_name}"
            if node_name not in graph.nodes:
                graph.add_node(node_name, agent=make_step_agent(step_name))
                graph.add_edge(prev, node_name)
            prev = node_name
        graph.set_end(prev)
        return f"Injected nodes: {design.get('steps')}"
    return topology_applier

graph.add_node("topology_applier", function=make_topology_applier(graph))
```

## Why this is safe (cooperative asyncio)

`add_node` / `add_edge` perform plain dict writes with no locks.  Python's
cooperative scheduler suspends the graph loop while the function node runs
synchronously, so no other coroutine can interleave.

## Comparison

| | `dynamic_topology` | `agent_driven_topology` |
|-|-------------------|------------------------|
| Where mutation happens | Outer event loop | Inside runner (function node) |
| Graph reference | Passed in from outside | Closure over `graph` |
| Mediation | Event loop pauses between events | None — runs in one async frame |

## Flow

```
planner ──▶ topology_applier (fn, closes over graph)
                ──▶ step_validate ──▶ step_transform ──▶ ... ──▶ END
                    (nodes added at runtime by topology_applier)
```

## Run

```bash
GOOGLE_API_KEY=<key> python -m contributing.samples.graph_agent_agent_driven_topology.agent
# or Vertex AI:
GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=<region> GOOGLE_GENAI_USE_VERTEXAI=true \
  python -m contributing.samples.graph_agent_agent_driven_topology.agent
```
