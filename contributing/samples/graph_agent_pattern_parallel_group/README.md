# GraphAgent Pattern — DynamicParallelGroup (Tree of Thoughts)

This example implements the **Tree of Thoughts** pattern using `DynamicParallelGroup`. N independent
reasoning paths are generated concurrently, then an evaluator scores them and a selector picks the
best. The number of parallel branches is determined at runtime from the graph state, and concurrency
is capped by `max_parallelism` to prevent resource exhaustion.

## When to Use This Pattern

- Diverge-and-converge workflows: generate many candidates in parallel, then select the best
- Tree of Thoughts / beam search style reasoning
- Ensemble approaches: run N agents independently and aggregate their outputs
- Any case where the number of parallel branches is data-dependent (unknown at graph-build time)

## How to Run

```bash
adk run contributing/samples/graph_agent_pattern_parallel_group
```

## Graph Structure

```
config (function) ──▶ generate (DynamicParallelGroup) ──▶ evaluate ──▶ select
                           ├── thought_agent_0
                           ├── thought_agent_1
                           └── thought_agent_N  (N from state.data["num_thoughts"])
```

## Key Code Walkthrough

- **`DynamicParallelGroup(name="generate", agent_generator=generate_thought_agents)`** — the
  generator callable receives `GraphState` at runtime and returns a list of `BaseAgent` instances
- **`max_parallelism=5`** — caps concurrent agent executions via an `asyncio.Semaphore`; prevents
  overloading the model API with too many simultaneous requests
- **`aggregator=aggregate_thoughts`** — combines the N results into a single string (using
  `=== Thought N ===` separators) before passing to the evaluator
- **`config` function node** — parses `[num_thoughts=N]` from the user message and writes
  `state.data["num_thoughts"]`; shows how function nodes can pre-process inputs
- **`select_responder`** — final selector reads evaluator scores and returns the winning thought

