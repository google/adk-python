# HITL Content Review Workflow

Demonstrates a complete human-in-the-loop review workflow using GraphAgent.

## Graph Structure

```
[draft] --> [review_gate] --> approved? --> [publish]
                  |
                  v rejected
             [revise] --> [review_gate]  (loop)
```

## Key Concepts

- **InterruptService** pauses execution at the `review_gate` node for human input
- **Conditional routing** routes to `publish` (approved) or `revise` (rejected) based on `state.data["approved"]`
- **Fallback mode**: runs without LLM when no API key is configured (deterministic string templates)
- **Review loop**: `revise -> review_gate` loop with `max_iterations=10` safety limit

## Running

```bash
# Without LLM (deterministic fallback):
python -m contributing.samples.graph_agent_hitl_review.agent

# With LLM:
export GOOGLE_API_KEY="your-key"
python -m contributing.samples.graph_agent_hitl_review.agent

# With Vertex AI:
export GOOGLE_GENAI_USE_VERTEXAI=1
python -m contributing.samples.graph_agent_hitl_review.agent
```

## How It Works

1. `draft` node generates initial content (LLM or template)
2. `review_gate` node pauses via InterruptService, waits for human message
3. Human sends approval (`action="approve"`) or revision request (`action="revise"`)
4. If rejected: `revise` node incorporates feedback, loops back to `review_gate`
5. If approved: `publish` node finalizes content

In the demo, human messages are pre-queued to simulate the review interaction.

## Differences from `graph_agent_hitl`

- `graph_agent_hitl`: Demonstrates interrupt *mechanics* (risk assessment, InterruptReasoner)
- `graph_agent_hitl_review`: Demonstrates a *workflow pattern* where human approval is a required graph step with conditional routing
