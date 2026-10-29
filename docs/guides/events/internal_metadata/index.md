# Reserved custom_metadata keys

ADK reserves `Event.custom_metadata` keys that start with `__adk_internal_` for its own components, and callers cannot set them. ADK also marks events that are restored into a new session, so that its components can tell them apart.

## Introduction

`Event.custom_metadata` is a free-form dictionary, and several paths let a caller put keys into it: `RunConfig.custom_metadata`, the events passed to the create-session API, a saved session file loaded with `adk run --resume`, and the metadata a remote A2A agent sends. That freedom is useful for tagging events with request IDs or tracing data, but it means a component cannot trust a key it finds on an event, because a caller could have written it.

Reserving a prefix gives ADK components a namespace that none of those paths can write. A component, such as a plugin that signs tool calls, can store its data under a reserved key, knowing that a value it reads back was written by ADK code or by someone with direct access to the session store.

Restored events get a marker for a related reason. An event restored from outside the session was not produced by the agent in that session, and a component may need to treat it differently, for example by refusing to run a tool call it contains.

## Get started

A run's `custom_metadata` is copied onto the events of that run, except for reserved keys. In this example, the events carry `request_id`, and the reserved key is dropped:

```python
from google.adk.agents.run_config import RunConfig

run_config = RunConfig(
    custom_metadata={
        "request_id": "req-1",
        "__adk_internal_example": True,
    }
)

async for event in runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=message,
    run_config=run_config,
):
  ...  # event.custom_metadata has "request_id" but not the reserved key.
```

## How it works

Reserved keys are removed at each point where a caller hands ADK an event or its metadata:

- **`RunConfig.custom_metadata`**: The runner drops reserved keys before merging the run's metadata into the user message and into every event the run produces. They are also left out of the `custom_metadata` that callbacks and tools read from their context. Other keys are merged as before.
- **Create-session API**: Events passed in the request body lose their reserved keys and are marked as restored before they are stored.
- **`adk run --resume`**: Events loaded from the saved session file are handled the same way as events passed to the create-session API.
- **Remote A2A agents**: Reserved keys in the `custom_metadata` that a remote agent sends are dropped when its message is converted to an ADK event.

On restore, the reserved keys are removed first and the marker is set afterwards, so a caller cannot supply a marker of its own. The marker is itself a reserved key. Its name is not part of the public API and can change, so application code should not read it. The same removal also drops any reserved data that an event carried in the session it was copied from, such as a stamp that is only valid in that session.

Keys that ADK code sets on its own events are kept. Agents, plugins, and callbacks run inside ADK, so the reserved namespace is available to them. When a plugin's `on_event_callback` returns a replacement event with its own `custom_metadata`, the reserved keys of the original event are carried over.

Reserved keys stay on stored events. So that API output does not change, they are removed from these outputs:

- **API server**: events returned by the session endpoints (create, get, list and update) and streamed by `/run`, `/run_sse` and `/run_live`.
- **CLI**: session files written by `adk run --save_session`, and events printed with `--jsonl`.
- **A2A**: metadata sent to remote clients and agents.

An event that had no `custom_metadata` before it was restored is returned without one. Code that reads events from the session service directly, such as evaluation results (`EvalCaseResult.session_details`), sees them as stored. Removing keys from these outputs is not logged.

## Limitations

- **Session store writes are not filtered.** The reserved prefix stops callers that go through ADK's entry points. Anyone who can write to the session store directly can write any key, so a component that relies on a reserved key for security still has to verify it, for example with an HMAC.
- **Other restore paths are not marked.** Only the create-session API and `adk run --resume` mark restored events. Code that copies events into a session by calling `append_event` itself should remove `__adk_internal_*` keys and mark the events as restored, because data carried over from another session can be invalid in the new one. `ToolCallIntegrityPlugin.prepare_restored_event` does both.
- **The marker is set on every restored event.** Events without function calls are marked too. Only code that reads events from the session service or the session store directly sees the marker.
- **Dropped keys are logged at debug level.** ADK does not raise an error when it drops a reserved key, so a caller that sets one by mistake sees it only in debug logs.
