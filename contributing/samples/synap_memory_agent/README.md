# Synap Memory Agent Sample

This sample demonstrates how to add persistent, cross-session memory to a Google ADK agent using [Synap](https://maximem.ai) — a managed long-term memory layer for AI agents.

## What it does

The agent is wired up with two `FunctionTool` instances from `synap-google-adk`:
- `search_memory` — semantic search over the user's stored memories
- `store_memory` — persist explicit facts the user mentions

On each turn the agent can recall what it knows about the user and save new facts for future sessions.

## Setup

```bash
pip install synap-google-adk maximem-synap
export SYNAP_API_KEY=<your-key>  # free key at https://synap.maximem.ai
```

## Run

```bash
adk run contributing/samples/synap_memory_agent
```

Try teaching it something on the first turn (e.g. *"I'm allergic to peanuts"*), then ask about it on a later turn — Synap will retrieve the relevant memory automatically.

## Resources

- [Synap documentation](https://docs.maximem.ai)
- [PyPI: `synap-google-adk`](https://pypi.org/project/synap-google-adk/)
- [Open source integration package](https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations/synap-google-adk)
