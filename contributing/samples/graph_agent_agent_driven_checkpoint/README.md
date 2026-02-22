# graph_agent_agent_driven_checkpoint

Demonstrates the `checkpoint_request_key` pattern: an LLM agent proposes
checkpoints via a boolean flag in its structured output schema, and
`GraphCheckpointCallback` creates the checkpoint only when the flag is set.

## Pattern

```python
class AnalysisOutput(BaseModel):
    finding: str
    risk_level: str
    checkpoint_requested: bool = False  # LLM sets True for high-risk

checkpoint_callback = GraphCheckpointCallback(
    checkpoint_service,
    checkpoint_after=False,                         # no automatic checkpoints
    checkpoint_request_key="analyzer.checkpoint_requested",  # agent-proposed
)
```

`checkpoint_request_key` accepts a dotted path `"<state_key>.<bool_field>"`.
After the named node completes, `after_node` reads the field from state and
creates a checkpoint only when it is truthy.  The flag resets automatically
because `StateReducer.OVERWRITE` replaces the entire output on the next run.

## Why this vs `checkpoint_after=True`?

| | `checkpoint_after=True` | `checkpoint_request_key` |
|-|------------------------|--------------------------|
| Trigger | Every node | LLM reasoning only |
| Overhead | Always | Only for high-risk tasks |
| Control | Infrastructure | Agent |

## Flow

```
analyzer → processor → reporter → END
   ↓ (high-risk only)
[checkpoint created]
```

## Run

```bash
GOOGLE_API_KEY=<key> python -m contributing.samples.graph_agent_agent_driven_checkpoint.agent
# or Vertex AI:
GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=<region> GOOGLE_GENAI_USE_VERTEXAI=true \
  python -m contributing.samples.graph_agent_agent_driven_checkpoint.agent
```

Expected: only the `HIGH RISK` scenario produces agent-requested checkpoints.
