# Runner

`Runner` is the public entry point for executing agents and workflows. It owns
the invocation lifecycle: building the `InvocationContext`, draining the event
queue onto the session, and wiring in the artifact, session, memory and
credential services plus the plugin manager.

`InMemoryRunner` is the batteries-included subclass that supplies in-memory
services; it accepts either an `agent=` or a `node=`.

## Entrance methods

### `run_async`

The main asynchronous entry point. Use this in production.

- Yields events as they are produced; does not block concurrent calls for other
  queries.
- Runs event compaction after the invocation when the app has
  `events_compaction_config` set.

| Argument | Meaning |
|---|---|
| `user_id` | User ID of the session. |
| `session_id` | Session ID. |
| `invocation_id` | Set to resume an interrupted invocation. |
| `new_message` | Message to append to the session. Optional — omit it when resuming. |
| `state_delta` | State changes to apply to the session. |
| `run_config` | Run config for the agent. |
| `yield_user_message` | Yield the user-message event before agent/node events. |

### `run`

Synchronous convenience wrapper for local testing: runs the async path on a
background thread and re-yields its events. Takes `user_id`, `session_id`,
`new_message`, `state_delta` and `run_config` — no `invocation_id`, so it
cannot resume.

### `run_live`

Audio/video streaming entry point, driven by a `LiveRequestQueue` rather than a
single `new_message`.

### `run_debug`

Convenience harness for local iteration: takes one message or a list of
messages, creates the session if needed, and prints the exchange (`quiet` and
`verbose` control how much).
