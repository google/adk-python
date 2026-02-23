# GraphAgent Multi-Agent Research Workflow

Demonstrates a **multi-agent coordination pattern** using GraphAgent:
parallel research branches, sequential coordination, and a quality-review loop.

## Graph Structure

```
coordinator
    ↓
[researcher_a ║ researcher_b]  ← ParallelNodeGroup (WAIT_ALL)
    ↓
merger
    ↓
critic ──REVISE──→ merger
    │
  APPROVED
    ↓
   END
```

## What Each Agent Does

| Agent | Role | Output key |
|-------|------|-----------|
| coordinator | Splits topic into two subtopics | `subtopics` |
| researcher_a | Investigates subtopic A concurrently | `research_a` |
| researcher_b | Investigates subtopic B concurrently | `research_b` |
| merger | Synthesises findings into one report | `merged_report` |
| critic | Peer-reviews; routes to merger (REVISE) or ends (APPROVED) | `review` |

## When to Use

- Tasks that decompose into independent parallel workstreams
- Workflows needing a quality-review loop after synthesis
- Any pattern mixing sequential coordination, parallelism, and conditional loops

## Comparison with Other Workflow Agents

| Capability | SequentialAgent | ParallelAgent | **GraphAgent** |
|------------|----------------|---------------|----------------|
| Run researcher_a and researcher_b concurrently | ✗ | ✅ | ✅ |
| Pre-coordination step before parallel work | ✗ | ✗ | ✅ |
| Post-merge step after parallel work | ✗ | ✗ | ✅ |
| Conditional quality loop (critic → merger) | ✗ | ✗ | ✅ |
| Inspect state to decide routing | ✗ | ✗ | ✅ |

**ParallelAgent** can fan out but cannot add a coordinator before or a
critic-loop after — it has no concept of entry/exit coordination nodes.
**SequentialAgent** executes in a fixed order and cannot parallelise the
two researchers.

## Key Code

```python
# Register parallel group: researcher_a and researcher_b run concurrently
graph.add_parallel_group(
    "researchers",
    ParallelNodeGroup(
        nodes=["researcher_a", "researcher_b"],
        join_strategy=JoinStrategy.WAIT_ALL,
    ),
)

# Edges fan-out from coordinator to both researchers
graph.add_edge("coordinator", "researcher_a")
graph.add_edge("coordinator", "researcher_b")

# Both researchers converge at merger
graph.add_edge("researcher_a", "merger")
graph.add_edge("researcher_b", "merger")

# Conditional quality loop
graph.add_edge("critic", "merger", condition=lambda s: s.data.get("review","").startswith("REVISE"))
graph.set_end("critic")
```

## State Isolation

During parallel execution each branch (`researcher_a`, `researcher_b`) receives
an **isolated copy** of the shared state. Both write to independent output keys
(`research_a`, `research_b`), so there are no race conditions. After both
complete, states are merged automatically before `merger` runs.

## How to Run

```bash
cd /path/to/adk-python
source venv/bin/activate
export GOOGLE_API_KEY=<your-key>
python -m contributing.samples.graph_agent_multi_agent.agent
```

## Related Examples

- `contributing/samples/graph_examples/09_parallel_wait_all` — parallel basics
- `contributing/samples/graph_examples/14_parallel_rewind` — parallel + rewind
- `contributing/samples/graph_agent_advanced` — full research workflow with interrupts
