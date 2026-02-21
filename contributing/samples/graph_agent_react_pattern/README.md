# GraphAgent ReAct Pattern

Demonstrates the **ReAct (Reasoning + Acting)** loop using GraphAgent.

## Pattern

```
reason → act → observe
            ↑         |  CONTINUE
            └─────────┘
                       |  COMPLETE
                       ↓
                      END
```

Each iteration:
1. **reason** — analyse task + previous observation, decide next action
2. **act** — execute the chosen action, produce a result
3. **observe** — evaluate result; output `COMPLETE:` or `CONTINUE:`

GraphAgent routes `observe → reason` (continue) or exits (complete) based on
the `observation` state key — a conditional edge only GraphAgent can express.

## When to Use

- Multi-step problem solving where the number of iterations is unknown
- Tool-augmented agents (search → reason → act → observe loop)
- Any workflow where routing depends on the *content* of an intermediate output

## Comparison with Other Workflow Agents

| Capability | SequentialAgent | LoopAgent | **GraphAgent** |
|------------|----------------|-----------|----------------|
| Execute nodes in order | ✅ | ✅ | ✅ |
| Loop (repeat execution) | ✗ | ✅ | ✅ |
| Route based on state content | ✗ | ✗ (escalate only) | ✅ |
| Conditional exit mid-loop | ✗ | via escalate | ✅ |
| Route to *different* next node | ✗ | ✗ | ✅ |

**LoopAgent** can repeat, but its only conditional exit is via `escalate` — it
cannot inspect the `observation` field and route back to `reason` vs. exit.
**SequentialAgent** cannot loop at all.

## Key Code

```python
# Conditional back-edge: loop if not complete
graph.add_edge("observe", "reason", condition=_should_continue)

# No forward edge needed: set_end() exits when observe has no matching edge
graph.set_end("observe")
```

## How to Run

```bash
cd /path/to/adk-python
source venv/bin/activate
export GOOGLE_API_KEY=<your-key>
python -m contributing.samples.graph_agent_react_pattern.agent
```

## Related Examples

- `contributing/samples/graph_examples/02_conditional_routing` — basic conditional edges
- `contributing/samples/graph_examples/03_cyclic_execution` — cyclic loop without LLM
- `contributing/samples/graph_agent_advanced` — full research workflow
