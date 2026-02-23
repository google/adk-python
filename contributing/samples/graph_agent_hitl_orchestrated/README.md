# Composable HITL Orchestrated Pipeline

Demonstrates how to compose HITL review loops as reusable `NestedGraphNode` building blocks in a larger orchestrated pipeline.

## Graph Structure

**Outer graph** (`document_pipeline`):
```
[classify] --> [process] --> [aggregate]
```

**Inner graph** (`review_stage`, wrapped as `NestedGraphNode`):
```
[execute] --> [review_gate] --> approved? --> [done]
                    |
                    v rejected
               [revise] --> [review_gate]  (loop)
```

## Key Concepts

- **Reusable HITL block**: The inner review graph is built once and wrapped in `NestedGraphNode`
- **Clean abstraction**: Outer orchestrator doesn't know about inner HITL details
- **Independent review cycles**: Each inner graph has its own interrupt timing
- **Observability**: `_debug_process_output` tracks nested graph output (via Part B observability)
- **Rule-based classification**: `classify` node determines stages without LLM

## Running

```bash
# Without LLM (deterministic fallback):
python -m contributing.samples.graph_agent_hitl_orchestrated.agent

# With LLM:
export GOOGLE_API_KEY="your-key"
python -m contributing.samples.graph_agent_hitl_orchestrated.agent
```

## How It Works

1. `classify` reads input document, determines processing stages (extract/summarize/translate)
2. `process` (NestedGraphNode) runs the inner review graph with HITL loop
3. Inner `execute` performs the stage task, `review_gate` pauses for human approval
4. If rejected: inner `revise` incorporates feedback, loops back
5. If approved: inner `done` returns output to outer graph
6. `aggregate` combines results into final output

## Differences from `graph_agent_hitl_review`

- `graph_agent_hitl_review`: Standalone HITL review loop
- `graph_agent_hitl_orchestrated`: Wraps the review pattern as a NestedGraphNode in a larger pipeline, showing composability
