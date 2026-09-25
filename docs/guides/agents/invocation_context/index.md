# InvocationContext

`InvocationContext` represents the execution environment and service container for a single invocation turn of an agent application.

## Introduction

An ADK application structures conversational execution into discrete invocation turns. When a user sends a message to `Runner.run_async`, the runner creates an `InvocationContext` instance to manage the lifecycle, state, and backing services for that specific execution turn.

While `Context` serves as the developer-facing facade for tools, callbacks, and workflow nodes (exposing delta-aware state and event accumulators), `InvocationContext` acts as the underlying runtime foundation. It holds references to persistent storage providers (`session_service`, `artifact_service`, `memory_service`, `credential_service`), tracks the active `Session`, retains the initial user input (`user_content`), records the turn identifier (`invocation_id`), manages branch hierarchy (`branch`), and governs turn-wide control flags (`end_invocation`).

Custom agents subclassing `BaseAgent` receive `InvocationContext` directly in their `_run_async_impl` generator, while higher-level components access these underlying services transparently through `Context`.

## Get started

Custom agents inspect turn metadata, evaluate user input content, and manage branching execution directly through `InvocationContext`.

The example below implements a diagnostic agent that subclasses `BaseAgent`, inspects the current `InvocationContext` attributes, and returns turn execution metadata.

```python
import asyncio
from collections.abc import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents import InvocationContext
from google.adk.apps import App
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


class DiagnosticAgent(BaseAgent):
  """Agent that inspects InvocationContext runtime attributes."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          user_text = part.text

    branch_name = ctx.branch or "root"
    response_text = (
        f"Processed message: {user_text} | Turn ID: {ctx.invocation_id} | "
        f"Branch: {branch_name}"
    )

    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        branch=ctx.branch,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)],
        ),
    )


async def main() -> None:
  agent = DiagnosticAgent(name="diagnostic")
  app = App(name="diagnostic_app", root_agent=agent)
  session_service = InMemorySessionService()
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="diagnostic_app",
      user_id="user_123",
  )

  message = types.Content(
      role="user",
      parts=[types.Part.from_text(text="Ping")],
  )

  async for event in runner.run_async(
      user_id="user_123",
      session_id=session.id,
      new_message=message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print("Agent output:", part.text)


if __name__ == "__main__":
  asyncio.run(main())
```

When executed, `Runner.run_async` generates an invocation identifier (prefixed with `e-`), wraps the input message in `user_content`, and passes the initialized `InvocationContext` to `DiagnosticAgent._run_async_impl`.

## How it works

The execution lifecycle of an ADK application is organized hierarchically into invocations, agent calls, and steps:

```
┌─────────────────────────────── invocation ────────────────────────────────┐
│ ┌──────────── llm_agent_call_1 ────────────┐ ┌─────── agent_call_2 ─────┐ │
│ │ ┌──── step_1 ────────┐ ┌───── step_2 ──┐ │ │                          │ │
│ │ │ [call_llm] [tools] │ │ [call_llm]    │ │ │ [execute_sub_agent]      │ │
│ │ └────────────────────┘ └───────────────┘ │ │                          │ │
│ └──────────────────────────────────────────┘ └──────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

1. **Invocation**: Initiated by `Runner.run_async` upon receiving a user message. It instantiates an `InvocationContext` with a fresh `invocation_id` and runs until the root agent or transferred agents finish generating responses.
2. **Agent call**: A single agent's execution span, managed by `BaseAgent.run_async`. When an agent delegates or transfers to another agent, the parent context clones via `model_copy(update={'agent': self})`, tracking the active agent while preserving shared session and service references.
3. **Step**: A single LLM query and corresponding tool executions within an agent call.

### Service dependency container

`InvocationContext` acts as the dependency injection hub for all persistence and integration backends configured on `Runner`:

* `session_service`: Reads and writes conversation events, session state, and session metadata.
* `artifact_service`: Persists versioned binary objects, documents, and media out of the event stream.
* `memory_service`: Stores summarized or completed sessions and executes similarity searches across past conversations.
* `credential_service`: Resolves authenticated credentials and tokens for secure tool executions.
* `plugin_manager`: Orchestrates lifecycle hooks and callbacks across all agents and tools.

When tools, callbacks, or workflow nodes execute, the runtime constructs a `Context` instance wrapping the `InvocationContext`. Operations such as `ctx.save_artifact` and `ctx.search_memory` route directly to the respective `artifact_service` and `memory_service` instances, while `ctx.save_credential` and `ctx.load_credential` interact with `credential_service`. In contrast, `ctx.get_credential` resolves credentials directly from the `credential_by_key` dictionary cached on `InvocationContext` and never touches `credential_service`.

### Branch isolation and event routing

In multi-agent architectures and workflow graphs, sub-agents may execute concurrently or perform isolated sub-tasks. `InvocationContext` tracks the current execution path via `branch`:

* Root agents run on `branch=None`. In `_get_events(current_branch=True)`, a `branch=None` setting matches all user events regardless of branch. By contrast, an empty-string branch (`branch=""`) represents an explicit, unbranched scope in workflow code that matches no branched events.
* Sub-agents run on hierarchical, dot-separated branches (such as `coordinator.researcher`).
* The internal `_get_events(current_branch=True)` method filters session history so that sub-agents do not inadvertently read conversational noise or tool confirmations from sibling branches.

### Event synchronization

`InvocationContext` maintains an internal asynchronous event queue. When components emit non-partial events via `_enqueue_event`, execution pauses until the runner main loop commits the event to session storage. This ensures state and history consistency before downstream nodes or steps proceed. Partial events (such as real-time audio or token streaming) flow through without blocking.

## Configuration options

The `InvocationContext` model exposes properties and methods across several functional areas:

### Identity and invocation lifecycle

Fields governing the active turn identity, session association, and agent-call short-circuiting.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `invocation_id` | Field | `str` | Unique identifier generated for the current invocation turn (read-only). |
| `user_content` | Field | `Content \| None` | The user content that initiated this invocation (read-only). |
| `session` | Field | `Session` | Reference to the active session object (read-only). |
| `branch` | Field | `str \| None` | Dot-separated hierarchical branch identifier (e.g. `agent_1.agent_2`). |
| `node_path` | Field | `str \| None` | Path of the executing agent in the workflow call stack. |
| `isolation_scope` | Field | `str \| None` | Internal scope tag for filtering events visible to the agent. |
| `end_invocation` | Field | `bool` | Flag read by `BaseAgent.run_async` and `BaseLlmFlow` to short-circuit the current agent call and its LLM step loop. |
| `app_name` | Property | `str` | Name of the active application derived from the session. |
| `user_id` | Property | `str` | User identifier associated with the active session. |

### Services and providers

Backing services and configuration objects injected by the runner.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `session_service` | Field | `BaseSessionService` | Service handling session creation, event persistence, and state updates. |
| `artifact_service` | Field | `BaseArtifactService \| None` | Service for persisting versioned binary artifacts. |
| `memory_service` | Field | `BaseMemoryService \| None` | Long-term memory service for cross-session storage and retrieval. |
| `credential_service` | Field | `BaseCredentialService \| None` | Service resolving authentication credentials. |
| `context_cache_config` | Field | `ContextCacheConfig \| None` | Configuration for LLM prompt context caching. |
| `plugin_manager` | Field | `PluginManager` | Manager coordinating plugin execution and callbacks. |

### Agent execution and state management

Fields and methods for tracking execution status and state across agents in the hierarchy.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `agent` | Field | `BaseAgent \| BaseNode \| None` | Reference to the currently executing agent or workflow node. |
| `agent_states` | Field | `dict[str, dict[str, Any]]` | Internal state mapping for agents active within the invocation. |
| `end_of_agents` | Field | `dict[str, bool]` | Status flags indicating whether individual agents have completed execution. |
| `set_agent_state` | Method | `(agent_name, *, agent_state, end_of_agent) -> None` | Sets or clears execution state for a specific agent in the invocation. |
| `reset_sub_agent_states` | Method | `(agent_name) -> None` | Recursively resets states of all sub-agents under the specified agent. |
| `populate_invocation_agent_states` | Method | `() -> None` | Restores agent states from session history for resumable workflows. |

### Realtime and streaming infrastructure

Configuration fields supporting live bidirectional audio, streaming tools, and token limits.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `live_request_queue` | Field | `LiveRequestQueue \| None` | Queue for receiving real-time input chunks in live sessions. |
| `active_streaming_tools` | Field | `dict[str, ActiveStreamingTool] \| None` | Active background streaming tools for live agents. |
| `active_non_blocking_tool_tasks` | Field | `dict[str, Task[Any]] \| None` | Running background tool tasks executing concurrently with live audio. |
| `transcription_cache` | Field | `list[TranscriptionEntry] \| None` | Cached transcriptions and audio buffers for live streaming. |
| `run_config` | Field | `RunConfig \| None` | Runtime limits and metadata configuration (such as `max_llm_calls`). |
| `resumability_config` | Field | `ResumabilityConfig \| None` | Configuration controlling session pause and resume capabilities. |
| `events_compaction_config` | Field | `EventsCompactionConfig \| None` | Policy for compacting historical session events. |

## Advanced applications

### Sub-agent branch isolation

In hierarchical multi-agent teams, delegating work to child agents can pollute the primary conversation history if all events share a single branch. Setting hierarchical branches isolates execution sub-trees.

The example below demonstrates how an agent clones `InvocationContext` with a sub-branch to run a child agent in an isolated scope.

```python
from collections.abc import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents import InvocationContext
from google.adk.events import Event


class ParentAgent(BaseAgent):
  """Parent agent that invokes a child agent on a scoped sub-branch."""

  child_agent: BaseAgent

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Construct a child branch path
    child_branch = f"{ctx.branch or self.name}.{self.child_agent.name}"

    # Clone the invocation context with the isolated branch identifier
    child_ctx = ctx.model_copy(
        update={
            "branch": child_branch,
            "agent": self.child_agent,
        }
    )

    # Execute the child agent within its isolated branch context
    async for event in self.child_agent.run_async(child_ctx):
      yield event
```

Events emitted by `child_agent` carry `branch="ParentAgent.ChildAgent"`. Peer sub-agents filtering events by their own branch cannot see these internal exchanges, preventing cross-branch context contamination.

### Short-circuiting an agent call

A callback, request processor, or custom agent implementation can short-circuit the current agent's execution lifecycle by setting `end_invocation = True` on its `InvocationContext`:

* When set during `before_agent_callback`, `BaseAgent.run_async` skips both `_run_async_impl` and `after_agent_callback` for that agent.
* When set inside `_run_async_impl` or by a `BaseLlmFlow` request processor or tool authentication handler, `BaseLlmFlow` halts remaining request processors and exits the agent's LLM step loop, and `BaseAgent.run_async` skips `after_agent_callback`.

The example below implements a custom guardrail agent that validates credentials and sets `ctx.end_invocation = True` to skip its own `after_agent_callback` when authentication fails.

```python
from collections.abc import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents import InvocationContext
from google.adk.events import Event
from google.genai import types


class SecurityGuardAgent(BaseAgent):
  """Short-circuits its agent call lifecycle if credentials are missing."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    auth_token = ctx.session.state.get("auth_token")

    if not auth_token:
      # Short-circuit this agent's lifecycle (skips after_agent_callback)
      ctx.end_invocation = True

      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          branch=ctx.branch,
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text="Access denied: Missing authentication token."
                  )
              ],
          ),
      )
      return

    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        branch=ctx.branch,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Authentication verified.")],
        ),
    )
```

Only `BaseAgent.run_async` (via `_run_with_lifecycle`) and `BaseLlmFlow` read `ctx.end_invocation`; `Runner.run_async` never reads this flag. Furthermore, because `BaseAgent._create_invocation_context` creates a per-agent shallow copy via `parent_context.model_copy(update={'agent': self})`, setting `ctx.end_invocation = True` mutates only the current agent's context copy and does not propagate back to the parent agent or stop subsequent sibling agents in a `SequentialAgent`.

### Resumable agent state restoration

When building human-in-the-loop workflows or pausing for long-running operations, `InvocationContext` supports restoring execution states across turns.

When `Runner` resumes an interrupted session:
1. `populate_invocation_agent_states()` iterates over events in `session.events` for the current invocation.
2. It rebuilds `agent_states` and `end_of_agents` dictionaries to match the state of each agent prior to the pause.
3. The runner resumes execution from the paused node without re-executing completed agents.

## Limitations

`InvocationContext` is configured with `extra="forbid"`. Passing unknown keyword arguments to its constructor raises a Pydantic `ValidationError`, while attempting to assign undeclared attributes to an existing instance raises a plain `ValueError`. Store custom application data in `session.state` or `run_config.custom_metadata` instead.

An `InvocationContext` instance is bound strictly to a single invocation turn. It is instantiated when `Runner.run_async` begins and is not preserved or serialized directly to persistent storage. All state that must persist across turns must be written to `session.state` or recorded as an `Event`.

The `invocation_id`, `user_content`, and `session` attributes are initialized at the start of the turn and are treated as immutable throughout the invocation.

Internal methods prefixed with an underscore (such as `_enqueue_event`, `_get_events`, and `_find_matching_function_call`) and internal attributes like `isolation_scope` are runtime implementation details that may change without notice.

## Related samples

- [Three-Layer Transfer](../../../../contributing/samples/multi_agent/three_layer_transfer/agent.py) - Demonstrates multi-agent transfers and context propagation across parent and child agents.
- [Abort Execution](../../../../contributing/samples/core/abort/agent.py) - Demonstrates aborting execution and handling cancellations during agent runs.
- [Parallel Worker](../../../../contributing/samples/workflows/parallel_worker/agent.py) - Demonstrates parallel sub-agent execution with branch management.
- [Debug Logging Plugin](../../../../contributing/samples/plugins/plugin_debug_logging/agent.py) - Demonstrates inspecting invocation lifecycle events using plugins.
