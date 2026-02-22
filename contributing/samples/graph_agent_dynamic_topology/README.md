# GraphAgent Dynamic Topology — Runtime Node and Edge Discovery

This example demonstrates a pipeline where an LLM planner first decides which
processing steps are needed, then those steps are added as new nodes and edges
to the graph before execution continues — the graph shape is unknown at build time.

## When to Use This Pattern

- Adaptive pipelines where required steps depend on the input or task type
- ETL workflows where an LLM determines the transformation chain at runtime
- Any scenario requiring a variable number of agents chosen dynamically

## How to Run

```bash
GOOGLE_API_KEY=your_key python -m contributing.samples.graph_agent_dynamic_topology.agent
```

## Graph Structure

```
planner ──(discovers steps)──▶ step_1 ──▶ step_2 ──▶ ... ──▶ step_N ──▶ END
   │                                 (nodes and edges added at runtime)
checkpoint                       checkpoint    checkpoint        checkpoint
```

## Key Code Walkthrough

- **`graph.add_node()` / `graph.add_edge()` post-construction** — called inside `extend_graph_with_steps()` after the planner fires; GraphAgent supports topology mutations mid-run
- **`output_schema=PipelineDesign`** — planner returns a structured `steps: list[str]` that drives topology extension
- **`make_step_agent(step_name)`** — factory creates a specialized `LlmAgent` per discovered step
- **`graph.set_end(prev_node)`** — end node is reassigned to the last dynamically added step
- **`GraphCheckpointCallback(checkpoint_nodes=None)`** — checkpoints every node (planner + all steps) since the full set is not known ahead of time

