# Event flow and where to look

## One user message becomes events

```text
Runner.run_async()
  Runner._exec_with_plugin()          # plugin hooks + persisting events
    agent.run_async()                 # BaseAgent: before/after agent callbacks
      LlmAgent._run_async_impl()      # yields events
        BaseLlmFlow.run_async()
          SingleFlow | AutoFlow       # AutoFlow adds agent transfer
            call_llm                  # request build + model call
            handle_function_calls_async()   # tool dispatch
```

`LlmAgent._llm_flow` picks `SingleFlow` only when
`disallow_transfer_to_parent` and `disallow_transfer_to_peers` are both set and
the agent has no sub-agents; otherwise it is `AutoFlow`. If an agent refuses to
transfer, check those two fields before suspecting the prompt.

Workflow-graph execution takes a different path: `LlmAgent._run_impl` runs the
agent as a node via `src/google/adk/workflow/`.

## Callback order

Both the plugin manager and the agent get a turn, plugins first:

Point | Plugin manager | Agent
--- | --- | ---
Before model | `run_before_model_callback` | `canonical_before_model_callbacks`
After model | `run_after_model_callback` | `canonical_after_model_callbacks`
Model error | `run_on_model_error_callback` | `canonical_on_model_error_callbacks`
Before tool | `run_before_tool_callback` | `canonical_before_tool_callbacks`
After tool | `run_after_tool_callback` | `canonical_after_tool_callbacks`
Tool error | `run_on_tool_error_callback` | `canonical_on_tool_error_callbacks`

The manager also exposes run-level hooks with no agent counterpart:
`run_on_user_message_callback`, `run_before_run_callback`,
`run_after_run_callback`, `run_on_event_callback`, and the agent/run error
hooks. Source: `src/google/adk/plugins/plugin_manager.py`.

A plugin callback that returns a value short-circuits the step, so an agent
that "ignores" its own callback is often a plugin that already answered.

## Event fields worth reading

`Event` serializes with a camelCase alias generator, so JSON from the HTTP API
or `adk run --jsonl` uses `invocationId`, `functionCall`, `nodeInfo`,
`longRunningToolIds`, while Python attribute access stays snake_case.

Field | Why it matters
--- | ---
`author` | `user` or the agent name — the fastest way to see which agent actually spoke.
`branch` | `agent_1.agent_2` path. Drives which history the agent can see.
`nodeInfo.path` | Node path inside a workflow, e.g. `wf/A@1/B@1`.
`content.parts` | `text`, `functionCall`, `functionResponse` — a turn with no `text` part is not a bug, it is a tool round trip.
`output` | Generic node output value. Absent on ordinary chat events.
`longRunningToolIds` | Present means the run is parked on a human-in-the-loop tool.
`actions.transferToAgent` | The agent handed control to a named agent.
`actions.escalate` | The agent gave up to its parent, typically ending a loop.
`actions.endOfAgent` | The agent finished.
`actions.stateDelta` / `artifactDelta` | State and artifact writes made by this event.

`isolationScope` also appears on task-agent events; it is internal, so read it
for orientation but do not build on it. Source:
`src/google/adk/events/event.py`, `src/google/adk/events/event_actions.py`.

## Source map

Area | File
--- | ---
Runner and event persistence | `src/google/adk/runners.py`
Flow driver, LLM call, callbacks | `src/google/adk/flows/llm_flows/base_llm_flow.py`
Request assembly (model, tools, schema) | `src/google/adk/flows/llm_flows/basic.py`
Which history reaches the model | `src/google/adk/flows/llm_flows/contents.py`
Tool dispatch and tool errors | `src/google/adk/flows/llm_flows/functions.py`
Agent transfer | `src/google/adk/flows/llm_flows/agent_transfer.py`
Agent config and validation | `src/google/adk/agents/llm_agent.py`
Invocation state and call limits | `src/google/adk/agents/invocation_context.py`
Task agents | `src/google/adk/agents/llm/task/`
Graph orchestration | `src/google/adk/workflow/`
Event model | `src/google/adk/events/event.py`
Session services | `src/google/adk/sessions/`
Plugin hook ordering | `src/google/adk/plugins/plugin_manager.py`
HTTP API (production-safe routes) | `src/google/adk/cli/api_server.py`
Dev-only routes, including traces | `src/google/adk/cli/dev_server.py`
Agent discovery | `src/google/adk/cli/utils/agent_loader.py`
Log setup | `src/google/adk/cli/utils/logs.py`
Tracing and span attributes | `src/google/adk/telemetry/tracing.py`
Event printer used by the CLI | `src/google/adk/utils/_debug_output.py`

`src/google/adk/cli/adk_web_server.py` is a deprecated shim; `AdkWebServer` now
just subclasses `DevServer`. Read `api_server.py` / `dev_server.py` instead.
