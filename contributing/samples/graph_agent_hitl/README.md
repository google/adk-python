# GraphAgent Human-In-The-Loop (HITL) — Risk-Gated Approval

This example demonstrates a risk-aware workflow where an agent assesses action
risk at runtime and conditionally pauses for human approval before proceeding,
using `InterruptService`, `InterruptReasoner`, and `GraphCheckpointCallback`.

## When to Use This Pattern

- Any workflow requiring human sign-off before irreversible actions
- Automated pipelines that must escalate high-risk decisions to a reviewer
- Audit trails where state must be preserved across human interaction windows

## How to Run

```bash
GOOGLE_API_KEY=your_key python -m contributing.samples.graph_agent_hitl.agent
```

## Graph Structure

```
analyze ──(always)──▶ execute ──▶ END
   │                    ▲
   │           [interrupt_config fires if risk == "high"]
   │                    │
   └──── InterruptReasoner processes human feedback ───────┘
              "continue" → proceed  |  "pause" → stop
```

## Key Code Walkthrough

- **`InterruptConfig(mode=BEFORE, nodes=["execute"])`** — pauses before `execute` when risk is high
- **`InterruptReasoner`** — LLM that interprets human feedback and returns `continue` or `pause`
- **`InterruptService.send_message()`** — delivers human approval into the running graph
- **`GraphCheckpointCallback(checkpoint_nodes={"analyze","execute"})`** — saves state only at critical nodes, not every node
- **`output_schema=RiskAssessment`** — structured output lets condition logic read `risk_level` from state

