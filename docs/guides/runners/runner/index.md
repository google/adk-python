# Runner and InMemoryRunner

`Runner` is the top-level execution engine ADK uses to manage session lifecycles, resolve persistent state, dispatch agent invocations, and stream events back to callers. `InMemoryRunner` is the default in-memory implementation suitable for local development, CLI applications, and unit testing.

## Introduction

Executing an LLM agent or workflow involves coordinating session storage, artifact management, plugin callbacks, and multi-turn message state. Directly instantiating flows or managing raw session objects mixes execution infrastructure with agent business logic.

`Runner` acts as the execution boundary between external callers and internal
ADK agent trees. A runner binds a root agent, workflow node, or `App` container
to session, memory, artifact, and credential services. It provides standard
execution and lifecycle methods (`run_async`, `run`, `run_live`, `run_debug`,
`rewind_async`, and `close`) that handle session lookup and creation, user event
append, context assembly, plugin hook dispatch, structured event streaming
(`Event`), session rewind, and resource teardown.

ADK provides `InMemoryRunner` for development with built-in in-memory session and artifact services, while production deployments use `Runner` with persistent services (e.g. database-backed session stores).

## Get started

Wrap a root agent in an `App`, attach it to an `InMemoryRunner`, create a session, and invoke `run_async` with a structured `types.Content` message:

```python
root_agent = LlmAgent(
    name="greeter",
    instruction="Greet users politely and answer their questions.",
)

app = App(
    name="greeter_app",
    root_agent=root_agent,
)

runner = InMemoryRunner(app=app)

# In an async function:
# 1. Create a session explicitly using the session service
session = await runner.session_service.create_session(
    app_name=app.name,
    user_id="user_123",
    session_id="session_456",
)

# 2. Run agent turn with the created session
async for event in runner.run_async(
    user_id="user_123",
    session_id=session.id,
    new_message=types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello, ADK!")],
    ),
):
  if event.content and event.content.parts:
    for part in event.content.parts:
      if part.text:
        print(event.author, part.text)
```

The runner retrieves session `session_456` under application `greeter_app`, appends the user message, executes `root_agent`, and yields generated `Event` objects containing model responses and tool outputs.

## How it works

The execution lifecycle coordinates between `Runner`, `BaseSessionService`, `PluginManager`, `InvocationContext`, and the root `BaseAgent`:

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Runner as Runner / InMemoryRunner
    participant Session as BaseSessionService
    participant Plugins as PluginManager
    participant Agent as Root Agent / Workflow

    Caller->>Runner: run_async(user_id, session_id, new_message, run_config, abort_signal)
    Runner->>Session: get_session(app_name, user_id, session_id)
    alt Session Not Found & auto_create_session=True
        Runner->>Session: create_session(app_name, user_id, session_id)
    end
    Runner->>Runner: Construct InvocationContext & apply RunConfig
    Runner->>Session: append_event(user message)
    Runner->>Plugins: run_before_run_callback()
    Runner->>Agent: run_async(invocation_context)
    loop Stream Events
        Agent-->>Runner: Yield Event
        Runner->>Plugins: run_on_event_callback()
        Runner->>Session: append_event(event) unless event.partial
        Runner-->>Caller: Yield Event
    end
    Runner->>Plugins: run_after_run_callback()
    Runner->>Runner: Event compaction, if configured
```

1. **Session Resolution & Normalization:** `Runner` normalizes its root target to an `App`. On `run_async`, it retrieves the active `Session` from `session_service` using `app.name`, `user_id`, and `session_id`. If the session is missing and `auto_create_session=True`, a new session is created automatically; otherwise `SessionNotFoundError` is raised.
2. **Context & Event Ingestion:** `Runner` constructs an `InvocationContext`
   linking session state (`app:`, `user:`, `temp:`), artifact service, memory
   service, and plugin manager, then appends the caller's `new_message` to the
   session as a user `Event`.
3. **Execution & Event Streaming:** The runner executes the root agent generator. Generated `Event` instances pass through `PluginManager.run_on_event_callback()` before being yielded to the caller. If `abort_signal` is set during execution, the runner cancels active tasks, emits an `INVOCATION_ABORTED` event, and seals any pending tool calls.
4. **Session Persistence & Compaction:** Each non-partial event is appended to
   `session_service` before it is yielded to the caller; partial streaming
   chunks (`event.partial=True`) are yielded without being persisted. If the
   invocation's `events_compaction_config` is set, event compaction runs after
   the last event is yielded before the generator finishes. It starts as the
   one on `App`, and a plugin's `before_run_callback` can replace it or set it
   to `None`.

## Configuration options

`Runner` and `RunConfig` introduce the following options:

### Runner Options

Constructor arguments passed when initializing `Runner(...)` or `InMemoryRunner(...)`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `app` | `App \| None` | `None` | Recommended entry point: the `App` container binding root agent, plugins, and app-wide configs. Mutually exclusive with `agent` and `node`. |
| `agent` | `BaseAgent \| None` | `None` | Root agent parameter, wrapped into an `App` internally. Mutually exclusive with `app` and `node`. |
| `node` | `BaseNode \| None` | `None` | Root workflow or node parameter, wrapped into an `App` internally. Mutually exclusive with `app` and `agent`. |
| `app_name` | `str \| None` | `None` | Application name; optional override for `app.name` when `app` is passed. On `Runner`, required when `agent` is passed and defaults to `node.name` when `node` is passed. On `InMemoryRunner`, defaults to `"InMemoryRunner"` whenever neither `app` nor `app_name` is passed, including with `agent` or `node`. |
| `session_service` | `BaseSessionService` | *(required for `Runner`)* | Session storage backend for retrieving and persisting conversation sessions. Accepted by `Runner` only; `InMemoryRunner` wires `InMemorySessionService` automatically. |
| `memory_service` | `BaseMemoryService \| None` | `None` | Long-term memory backend for cross-session retrieval. Accepted by `Runner` only; `InMemoryRunner` wires `InMemoryMemoryService` automatically. |
| `artifact_service` | `BaseArtifactService \| None` | `None` | Service for storing binary payloads and files outside session events. Accepted by `Runner` only; `InMemoryRunner` wires `InMemoryArtifactService` automatically. |
| `credential_service` | `BaseCredentialService \| None` | `None` | Service for resolving and persisting tool authentication credentials. Accepted by `Runner` only. |
| `auto_create_session` | `bool` | `False` | Automatically create a new session if `session_id` is not found during `run_async` or `run_live`. Accepted by `Runner` only, not by `InMemoryRunner`. |
| `plugin_close_timeout` | `float` | `5.0` | Timeout in seconds when closing plugins during `await runner.close()`. |
| `plugins` | `list[BasePlugin] \| None` | `None` | Deprecated on `Runner`: pass plugins on `App(plugins=[...])` instead. |

### `run_async` Options

Arguments passed per-invocation to `runner.run_async(...)`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `user_id` | `str` | *required* | User identifier of the session. |
| `session_id` | `str` | *required* | Session identifier. |
| `invocation_id` | `str \| None` | `None` | Invocation identifier to resume an interrupted invocation. |
| `new_message` | `types.Content \| None` | `None` | Message to append to the session. Omit when resuming. |
| `state_delta` | `dict[str, Any] \| None` | `None` | State changes to apply to the session before execution. |
| `run_config` | `RunConfig \| None` | `None` | Per-invocation execution configuration. |
| `yield_user_message` | `bool` | `False` | Yield the user-message event before agent or node events. |
| `abort_signal` | `asyncio.Event \| None` | `None` | Cooperative cancellation signal; see [Runner Execution Cancellation](abort.md). |

### RunConfig Options

Passed per-invocation to `runner.run_async(..., run_config=RunConfig(...))`. See [RunConfig](../../agents/run_config/index.md) for the full configuration reference, including `StreamingMode` and `ToolThreadPoolConfig`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `custom_metadata` | `dict[str, Any] \| None` | `None` | Caller metadata copied onto `InvocationContext`, merged into emitted events, and persisted to the session. |
| `get_session_config` | `GetSessionConfig \| None` | `None` | Fine-grained session retrieval and event window loading configuration. |
| `model_input_context` | `list[types.Content] \| None` | `None` | Transient unpersisted context added to model input for the current invocation. |
| `max_llm_calls` | `int` | `500` | Maximum limit on LLM calls per run execution. |

## Advanced applications

### Custom persistent runner

In production, pass database-backed session and memory services to `Runner` to serve persistent sessions:

```python
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

app = App(name="customer_support", root_agent=root_agent)
session_service = DatabaseSessionService(db_url="postgresql://...")

runner = Runner(
    app=app,
    session_service=session_service,
)
```

### Automatic session creation

By default, calling `runner.run_async` with a non-existent `session_id` raises
`SessionNotFoundError`. To automatically create a new session when one is
missing without an explicit `create_session` call, set
`auto_create_session=True` when constructing `Runner`:

```python
runner = Runner(
    app=app,
    session_service=session_service,
    auto_create_session=True,
)
```

`InMemoryRunner` does not accept `auto_create_session`, and passing it raises
`TypeError`. With `InMemoryRunner`, create the session explicitly via
`runner.session_service.create_session(...)`, or use
`await runner.run_debug(...)`, which creates or reuses its own debug session.

### Rewinding a session and closing resources

To undo a turn and restore session state and artifacts to the point before a
specific `invocation_id`, call `await runner.rewind_async(...)` with `user_id`,
`session_id`, and `rewind_before_invocation_id`.

Both `Runner` and `InMemoryRunner` implement the async context manager protocol
(`async with runner:`) and `await runner.close()`, which closes active toolsets,
closes registered plugins within `plugin_close_timeout`, and flushes
`session_service`.

## Limitations

*   **App versus Bare Agent:** `Runner(agent=...)` wraps the agent in an unvalidated `App` without `context_cache_config`, `events_compaction_config`, or `resumability_config`. Always pass an `App` via `app=` for production applications.
*   **InMemoryRunner Volatility:** `InMemoryRunner` uses `InMemorySessionService` by default. Session state is stored in memory and lost when the process terminates.

## Related guides & samples

*   [App Container](../../apps/app/index.md) — Guide on `App` configuration, plugins, and cross-cutting features.
*   [Runner Execution Cancellation](abort.md) — Guide on halting agent and workflow execution cleanly using abort signals.
*   [Runner Live Streaming](live.md) — Guide on real-time bidirectional streaming with `run_live` and `LiveRequestQueue`.
*   [Runner Rewind](rewind.md) covers `rewind_async`, which rewinds a session to the point before a given invocation and reverts the state and artifact changes made after it.
*   [Session and BaseSessionService](../../sessions/session/index.md) — Guide on session storage backends and state scoping.
*   [Agent-to-Agent Sample](../../../../contributing/samples/a2a/a2a_basic/agent.py) — Multi-agent application executed via `Runner`.
