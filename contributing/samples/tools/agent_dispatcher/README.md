# Agent Dispatcher Sample

Demonstrates [`AgentDispatcherToolset`](../../../../src/google/adk/tools/agent_dispatcher/_agent_dispatcher_toolset.py) for [issue #4759](https://github.com/google/adk-python/issues/4759): dispatch a runtime specialist agent, then send follow-ups on the same persistent child session.

## What it shows

- `dispatch_agent` — create a named agent with instruction + user message (optional allowlisted tools)
- `message_agent` — continue on the same `dispatch_id` / session
- `get_agent_result` — read the latest status/result

## Run

From the repository root (with ADK installed and `GOOGLE_API_KEY` or Vertex credentials configured):

```bash
adk run contributing/samples/tools/agent_dispatcher
```

Try prompts like:

- "Research ADK and then ask the specialist for one more sentence."
- "Dispatch a worker to look up Gemini, then get the result."

## Notes

- Tools for dispatched agents must be listed in the toolset `tool_allowlist` (MVP security boundary).
- Sync-first: `dispatch_agent` awaits the child run. Background fan-out is a follow-up.
