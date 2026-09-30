# App

The top-level container for an ADK application. An `App` binds a root agent to
the settings that belong to the application as a whole — its name, its plugins,
and the configs for context caching, event compaction, and resumability.

## Introduction

An agent describes one participant in a conversation: a model, an instruction, a
set of tools, maybe some sub-agents. Several things a real deployment needs are
not properties of any single agent. The application has one name that sessions
are keyed by. Plugins observe every agent, model call, and tool call in the tree.
Context caching, event compaction, and resumability apply to the whole agent tree
at once, not to one node of it.

`App` is where those live. It is a Pydantic model holding a `root_agent` plus
that application-wide configuration, so the settings travel with the agent
definition instead of being spread across whichever call site builds the
`Runner`. There is no separate root-node field: a workflow's root `BaseNode`
goes in `root_agent` too.

`Runner` normalizes its input to an `App` internally, so a bare agent still
works, but only an `App` can carry the cross-cutting configs. Passing `app=` is
the current path; see [App versus a bare agent](#app-versus-a-bare-agent).

## Get started

Define the agent and wrap it in an `App`.

The example below builds a weather agent with a single tool and puts it in an
`App` named `weather_app` alongside a `LoggingPlugin`. The `App` is what makes
the plugin possible here: `Runner(plugins=...)` is deprecated, and it is also
the only place the caching, compaction, and resumability configs can be set.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import LoggingPlugin


def get_weather(city: str) -> str:
  """Returns a one-line weather report for the given city."""
  return f"It is sunny in {city}."


root_agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction="Answer weather questions using the get_weather tool.",
    tools=[get_weather],
)

app = App(
    name="weather_app",
    root_agent=root_agent,
    plugins=[LoggingPlugin()],
)
```

## Running your app

`adk run`, `adk web`, and `adk api_server` build the `Runner` for you. They look
for a module-level variable named `app` in the agent module first, and fall back
to `root_agent` only when no `App` is found, so exporting the `App` above is all
these commands need to pick up the plugins and the cross-cutting configs.

When an agent is loaded this way, the `Runner` logs a warning if the app name
does not match the directory the agent was loaded from, and names the directory
it expected. Renaming the directory or the app to agree silences it.

To drive the app yourself instead, create a session service, hand the `App` to a
`Runner`, and run one user turn, printing each event as it arrives.

```python
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def main() -> None:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=app.name, user_id="user"
  )
  runner = Runner(app=app, session_service=session_service)
  async for event in runner.run_async(
      user_id="user",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part(text="What is the weather in Zurich?")],
      ),
  ):
    if event.content and event.content.parts:
      print(event.author, event.content.parts[0].text)


if __name__ == "__main__":
  asyncio.run(main())
```

Note that the session is created under `app.name`. The session service keys
every session by app name, so the name on the `App` and the name used to look up
sessions have to agree.

## App versus a bare agent

`Runner` accepts either an `App` or a plain agent, and turns the plain agent into
an `App` before doing anything else. The two paths are not equivalent.

```python
# Current: the App carries the application-wide configuration.
runner = Runner(app=app, session_service=session_service)

# Legacy: the agent is wrapped in an App for you.
runner = Runner(
    app_name="weather_app",
    agent=root_agent,
    session_service=session_service,
)
```

The legacy form is what ADK 1.x accepted, and it is still supported. It differs
in three ways:

*   The wrapping skips `App`'s validation, so an app name that `App` would
    reject is accepted here.
*   `context_cache_config`, `events_compaction_config`, and
    `resumability_config` are left unset. There is no `Runner` argument for
    them; an `App` is the only way to set them.
*   `Runner(plugins=[...])` is deprecated and raises a `DeprecationWarning`.
    Passing both `app` and `plugins` raises `ValueError` — put the plugins on
    the `App`.

When `app` and `app_name` are both given, `app_name` wins for session lookups
while `app.name` is unchanged. Passing both is rarely what you want.

## Fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | *required* | The application name. Sessions are keyed by it. |
| `root_agent` | `BaseAgent` or `BaseNode` | *required* | The entry point for execution. `BaseNode` is the workflow node base class in `google.adk.workflow`, so a `Workflow` can be the root too. |
| `plugins` | `list[BasePlugin]` | `[]` | Application-wide plugins. Their callbacks fire for every agent, model call, and tool call. |
| `context_cache_config` | `ContextCacheConfig \| None` | `None` | Enables context caching for every LLM agent in the app. Absent means caching is off. |
| `events_compaction_config` | `EventsCompactionConfig \| None` | `None` | Summarizes older session events so the context stops growing without bound. |
| `resumability_config` | `ResumabilityConfig \| None` | `None` | Lets an invocation pause on a long-running function call and resume later. |

`App` forbids unknown keywords, so a misspelled field name raises a
`ValidationError` rather than being silently ignored.

The name must start with a letter and may then contain letters, digits,
underscores, and hyphens. `"user"` is rejected because it is reserved for
end-user input. `validate_app_name` is exported from `google.adk.apps.app` if
you want to check a name before constructing the app.

The three config types are imported from different places:

```python
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
```

## Services attach to the Runner, not the App

An `App` holds declarative configuration only. The session, artifact, memory,
and credential services are constructor arguments of `Runner`, because they are
deployment wiring rather than part of the application's definition. The same
`App` can therefore be run against in-memory services in a test and persistent
ones in production, unchanged.

`session_service` is the one required service. For local development,
`InMemoryRunner` supplies in-memory session, artifact, and memory services and
accepts the same `App`:

```python
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(app=app)
```

## Configuring the cross-cutting features

Each config is inert until you set it on the `App`.

```python
app = App(
    name="weather_app",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(
        cache_intervals=10, ttl_seconds=1800, min_tokens=2048
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5, overlap_size=1
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

`EventsCompactionConfig` needs at least one trigger, and its two triggers are
each a pair that must be set together: `compaction_interval` with
`overlap_size` for a sliding window, or `token_threshold` with
`event_retention_size` for a token budget. Leaving `summarizer` unset makes ADK
build an `LlmEventSummarizer` from the root agent's model.

## Limitations

*   **Experimental configs**: `EventsCompactionConfig`, `ResumabilityConfig`,
    and `ContextCacheConfig` all emit an experimental warning on construction
    and may change without notice.
*   **`EventsCompactionConfig` is not re-exported**: `google.adk.apps` exports
    only `App` and `ResumabilityConfig`. Import `EventsCompactionConfig` from
    `google.adk.apps.app`.
*   **Resumption is best-effort**: a tool that may be resumed has to be
    idempotent, because resumption guarantees at-least-once execution, and any
    in-memory state is lost across the pause.

## Related samples

*   [Application configuration](../../../../contributing/samples/core/app)
