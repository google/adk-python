# Agent Dispatcher Sample

Demonstrates [`AgentDispatcherToolset`](../../../../src/google/adk/tools/agent_dispatcher/_agent_dispatcher_toolset.py) for [issue #4759](https://github.com/google/adk-python/issues/4759).

## Capabilities

- **Background dispatch** (default): orchestrator continues while specialists run
- **Parallel multi-dispatch**: multiple `dispatch_agent` calls in one turn
- **`await_agent` / `get_agent_result`**: poll or wait for completion
- **`message_agent`**: follow-ups on the same persistent child session
- **Allowlisted tools + skills** via toolset constructor
- **Shared session service**: child sessions use the parent session service by default

## Run

```bash
adk run contributing/samples/tools/agent_dispatcher
```

Try:

- "Dispatch two researchers in parallel on ADK and Gemini, then await both."
- "Research ADK, then ask the same specialist a follow-up."
