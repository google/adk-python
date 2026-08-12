---
name: adk-debug
description: Use when debugging ADK agents, inspecting sessions, testing agent behavior, troubleshooting tool calls, event flow issues, or diagnosing LLM/model problems.
---

# Debugging ADK Agents

Two debugging modes: `adk web` (browser UI + API) and `adk run` (CLI).

> [!NOTE]
> **Preference**: For most development and debugging tasks, `adk run` (CLI) is preferred as it is faster and more convenient. **Within `adk run`, query mode is preferred over interactive mode** because it requires less human intervention. However, `adk web` is still required for UI-specific issues, session management visualization, or debugging the API server itself.


---

## Mode 1: adk web (Browser UI + REST API)

Best for: visual inspection, session management, multi-turn testing.

### Dev server workflow

Before starting a server, ask the user:
1. **Is there already a running `adk web` server?** If yes, use it
   (check with `curl -s http://localhost:8000/health`).
2. **If not**, start one. Use `run_in_background` so it doesn't
   block. **Remember to shut it down when debugging is done.**

```bash
# Check if server is already running
curl -s http://localhost:8000/health

# Start server (if not running)
adk web path/to/agents_dir                    # default: http://localhost:8000
adk web -v path/to/agents_dir                 # verbose (DEBUG level)
adk web --reload_agents path/to/agents_dir    # auto-reload on file changes

# Shut down when done (if you started it)
# Kill the background process or Ctrl+C
```

> [!TIP]
> **Coding Agent Friendly Setup**: To allow a coding agent to read the server logs, recommend the user to start the server and redirect output to a file in a location the agent can read (e.g., the conversation's artifact directory or a shared workspace folder):
> ```bash
> adk web -v path/to/agents_dir 2>&1 | tee path/to/agent_readable_log.log
> ```
> This ensures both the user and the agent can inspect the full debug logs.

Web UI: `http://localhost:8000/dev-ui/`

### Session inspection via curl

```bash
# List sessions
curl -s http://localhost:8000/apps/{app_name}/users/{user_id}/sessions | python3 -m json.tool

# Get full session with events
curl -s http://localhost:8000/apps/{app_name}/users/{user_id}/sessions/{session_id} | python3 -m json.tool
```

Do NOT delete sessions after debugging — the user may want to
inspect them in the web UI.

### Summarize events

Fetch the session JSON and write a Python script to summarize
it. Do NOT use hardcoded inline scripts — the JSON schema may
change. Instead, fetch the raw JSON first:

```bash
curl -s http://localhost:8000/apps/{app_name}/users/{user_id}/sessions/{session_id} | python3 -m json.tool
```

Then write a script based on the actual structure you see.
Key fields to look for in each event: `author`, `branch`,
`content.parts` (text, functionCall, functionResponse),
`output`, `actions` (transferToAgent, requestTask, finishTask),
`nodeInfo.path`.

### Send test messages via curl

```bash
SESSION=$(curl -s -X POST http://localhost:8000/apps/{app_name}/users/test/sessions \
  -H "Content-Type: application/json" -d '{}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -N -X POST http://localhost:8000/run_sse \
  -H "Content-Type: application/json" \
  -d "{\"app_name\":\"{app_name}\",\"user_id\":\"test\",\"session_id\":\"$SESSION\",
       \"new_message\":{\"role\":\"user\",\"parts\":[{\"text\":\"your message here\"}]},
       \"streaming\":false}"
```

### Debug endpoints (traces)

```bash
# Trace for a specific event
curl -s http://localhost:8000/debug/trace/{event_id} | python3 -m json.tool

# All traces for a session
curl -s http://localhost:8000/debug/trace/session/{session_id} | python3 -m json.tool

# Health check
curl -s http://localhost:8000/health
```

### Extract LLM content history

Fetch trace data and inspect the `call_llm` spans. The LLM
request/response are in span attributes:

```bash
curl -s http://localhost:8000/debug/trace/session/{session_id} | python3 -m json.tool
```

Look for spans with `name: "call_llm"` and inspect their
`attributes.gcp.vertex.agent.llm_request` (JSON string of the
full request including `contents`, `config`, `model`).

### Key span attributes

| Attribute | Description |
|-----------|-------------|
| `gcp.vertex.agent.llm_request` | Full LLM request JSON (contents, config, model) |
| `gcp.vertex.agent.llm_response` | Full LLM response JSON |
| `gcp.vertex.agent.event_id` | Event ID — correlate with session events |
| `gen_ai.request.model` | Model name |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.response.finish_reasons` | Stop reason |

---

## Mode 2: adk run (CLI)

Best for: quick testing, scripting, CI/CD, headless debugging.

### Run interactively

```bash
adk run path/to/my_agent                      # interactive prompts
adk run -v path/to/my_agent                   # verbose logging
```

### Run with query (automated)

```bash
adk run path/to/my_agent "query"              # run with query
adk run --jsonl path/to/my_agent "query"      # output structured JSONL (noise reduced)
```

### When to use automated query mode

- **Fast & Lightweight**: Run tests quickly without starting the `adk web` dev server.
- **Easy Automation**: Perfect for CI/CD pipelines and regression scripts.
- **Highly Composable**: You can pipe the `--jsonl` output to standard tools like `jq`, `grep`, or `diff`.
- **Parallel Execution**: Each run is an isolated process. You can run multiple tests concurrently without port conflicts.
- **State Isolation**: Use `--in_memory` for fast, side-effect-free testing (no database updates).
- **Multi-Turn Support**: Remember to set a session ID if you need to maintain conversation state across turns.

> [!TIP]
> Always read the sample's `README.md` first to understand expected inputs and behaviors!

### Unit Tests vs. Sample Agents (When to use which)

Choosing the right testing strategy is crucial for efficiency and coverage:

- **Use Unit Tests when**:
  - Testing **isolated logic**, specific methods, or edge cases of a single component.
  - Verifying **data schemas**, Pydantic validations, or utility functions.
  - *Location*: `tests/unittests/`.

- **Use Sample Agents (Integration Testing) when**:
  - Developing features with **multi-level integration** (Runner + Agent + Workflow) or changes with wide impact.
  - Testing complex scenarios like **Human-in-the-Loop (HITL)** or long-running tools.
  - You need to verify the **real behavior** of the agent in a simulated environment.
  - *Location*: Create a sample under `contributing/agent_samples/` (refer to `adk-sample-creator`).

> [!IMPORTANT]
> **AI Assistant Reminder**: If you create a temporary sample agent for testing, you **MUST delete it** after verification is complete, unless the user explicitly asks to keep it.

### Exit Codes & Details

- **Exit Code 0**: Success.
- **Exit Code 1**: Error (e.g., API key missing, agent load failure).
- **Exit Code 2**: Paused (Workflow is waiting for human input/HITL).

For more options and flags, run:
```bash
adk run --help
```

### Event printing utility

```python
from google.adk.utils._debug_output import print_event

print_event(event, verbose=False)  # text responses only
print_event(event, verbose=True)   # tool calls, code execution, inline data
```

Location: `src/google/adk/utils/_debug_output.py`

### Programmatic debugging

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService

agent = Agent(name="test", model="gemini-2.5-flash", instruction="...")
runner = Runner(app_name="test", agent=agent, session_service=InMemorySessionService())

session = runner.session_service.create_session_sync(app_name="test", user_id="u")
for event in runner.run(user_id="u", session_id=session.id, new_message="hello"):
    print(f"{event.author}: {event.content}")
    if event.actions.transfer_to_agent:
        print(f"  -> transfer to {event.actions.transfer_to_agent}")
    if event.output:
        print(f"  -> output: {event.output}")
```

---

## Logging

Shared across both modes.

Set log level with `--log_level` (DEBUG, INFO, WARNING, ERROR, CRITICAL) or `-v` for DEBUG.
Logs write to `/tmp/agents_log/`. Tail latest: `tail -F /tmp/agents_log/agent.latest.log`
Logger name: `google_adk`. Setup: `src/google/adk/cli/utils/logs.py`

| Env Variable | Effect |
|---|---|
| `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | Include prompt/response in traces (default: `true`) |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Enable prompt/response in OTEL spans |
| `GOOGLE_CLOUD_PROJECT` | Required for `--trace_to_cloud` |

---

## Common Issues

### 1. Agent outputs raw JSON instead of calling tools

**Symptom:** Agent with `output_schema` dumps JSON text instead of calling tools.
**Cause:** `output_schema` sets `response_schema` on the LLM config, activating controlled generation (JSON-only mode).
**Check:** Look for `response_mime_type: "application/json"` in the LLM request.
**Location:** `src/google/adk/flows/llm_flows/basic.py`

### 2. Events missing from session / not visible to plugins

**Symptom:** Events from sub-agents don't appear in plugin callbacks or runner event stream.
**Cause:** Direct `append_event` calls inside components bypass the runner's event loop.
**Check:** Only the runner (`runners.py`) should call `append_event`. Components should yield events.

### 3. `NameError: name 'X' is not defined` at runtime

**Symptom:** `{"error": "name 'SomeClass' is not defined"}`
**Cause:** Class imported under `TYPE_CHECKING` but used at runtime (e.g., `isinstance()`).
**Fix:** Move import outside `TYPE_CHECKING` or use a local import.

### 4. Sub-agent doesn't have context from parent conversation

**Symptom:** Sub-agent only sees its own input, not the parent's history.
**Cause:** Branch isolation — sub-agents on a branch only see events on that branch.
**Fix:** Write the sub-agent's `description` to prompt the parent to include context in delegation input.

### 5. Agent validation errors at startup

**Symptom:** `ValueError` on agent construction.
**Common causes:**
- `"All tools must be set via LlmAgent.tools."` — Don't pass tools via `generate_content_config`
- `"System instruction must be set via LlmAgent.instruction."` — Don't set via `generate_content_config`
- `"Response schema must be set via LlmAgent.output_schema."` — Don't set via `generate_content_config`
**Location:** `src/google/adk/agents/llm_agent.py` — `validate_generate_content_config`

### 6. LLM calls exceeding limit

**Symptom:** `LlmCallsLimitExceededError: Max number of llm calls limit of N exceeded`
**Cause:** `run_config.max_llm_calls` limit reached.
**Fix:** Increase `max_llm_calls` in `RunConfig`, or investigate why the agent is looping.
**Location:** `src/google/adk/agents/invocation_context.py`

### 7. Tool errors silently swallowed

**Symptom:** Tool call fails but agent continues without expected result.
**Cause:** Errors are caught and returned as function response text. Set `on_tool_error_callback` to customize.
**Check:** Look for error text in function response events.

### 8. Agent not loading / not discovered

**Symptom:** `adk web` doesn't list the agent, or returns 404.
**Cause:** Agent directory must follow convention:
```
my_agent/
  __init__.py   # MUST contain: from . import agent
  agent.py      # MUST define: root_agent = Agent(...) OR app = App(...)
```

### 9. Sync tool blocking the event loop

**Symptom:** Agent hangs or becomes very slow.
**Cause:** Sync tools run in a thread pool (max 4 workers). All workers busy → new tool calls block.
**Fix:** Make tools async if they do I/O.

---

## LLM Finish Reasons

- `STOP` — normal completion
- `MAX_TOKENS` — output truncated (increase `max_output_tokens`)
- `SAFETY` — blocked by safety filters
- `RECITATION` — blocked for recitation

---

## Event Flow Architecture

```
User message
  -> Runner.run_async()
    -> Runner._exec_with_plugin()        # persists events, runs plugins
      -> agent.run_async()               # yields events
        -> LlmAgent._run_async_impl()
          -> BaseLlmFlow.run_async()       # Execution flow
            -> _AutoFlow or _SingleFlow   # Flow implementations
              -> call_llm               # LLM request + response
              -> execute_tools          # tool dispatch (functions.py)
```

---

## Callback Chain

**Before model call:** PluginManager `run_before_model_callback()` → agent `canonical_before_model_callbacks`
**After model call:** PluginManager `run_after_model_callback()` → agent `canonical_after_model_callbacks`
**Before/after tool call:** PluginManager `run_before_tool_callback()` / `run_after_tool_callback()` → agent callbacks

---

## Key Files for Debugging

| Area | File |
|---|---|
| Runner event loop | `src/google/adk/runners.py` |
| LLM request building | `src/google/adk/flows/llm_flows/basic.py` |
| Tool dispatch | `src/google/adk/flows/llm_flows/functions.py` |
| Multi-agent orchestration | `src/google/adk/workflow/` |
| Content/context building | `src/google/adk/flows/llm_flows/contents.py` |
| Task support | `src/google/adk/agents/llm/task/` |
| Agent config + validation | `src/google/adk/agents/llm_agent.py` |
| Event model | `src/google/adk/events/event.py` |
| Session services | `src/google/adk/sessions/` |
| Invocation context | `src/google/adk/agents/invocation_context.py` |
| Web server + debug endpoints | `src/google/adk/cli/adk_web_server.py` |
| Debug output printer | `src/google/adk/utils/_debug_output.py` |

---

## Debugging Checklist

1. **Start with logs** — `-v` flag, check `/tmp/agents_log/agent.latest.log`
2. **Inspect the session** — curl endpoints (`adk web`) or print events (`adk run`)
3. **Check event actions** — `transfer_to_agent`, `request_task`, `finish_task`, `escalate`
4. **Check event.output** — single_turn and task agents set output here
5. **Check traces** — `/debug/trace/session/{id}` for model/token usage
6. **Verify agent structure** — `__init__.py` imports, `root_agent` or `app` defined
7. **Check tool responses** — look for error text in function response events
8. **Check LLM finish reason** — `STOP`, `MAX_TOKENS`, `SAFETY`
9. **Test in isolation** — create a minimal agent with just the problem tool/config
