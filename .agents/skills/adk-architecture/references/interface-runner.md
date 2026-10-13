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
| `abort_signal` | Optional `asyncio.Event` for cooperative cancellation. Setting it halts the active invocation cleanly. |

### `run`

Synchronous convenience wrapper for local testing: runs the async path on a
background thread and re-yields its events. Takes `user_id`, `session_id`,
`new_message`, `state_delta` and `run_config` — no `invocation_id`, so it
cannot resume.

### `run_live`

Audio/video streaming entry point, driven by a `LiveRequestQueue`
(`from google.adk.live import LiveRequestQueue`) rather than a single
`new_message`.

### `run_debug`

Convenience harness for local iteration: takes one message or a list of
messages, creates the session if needed, and prints the exchange (`quiet` and
`verbose` control how much).

## Execution cancellation (`abort_signal`)

When `abort_signal` is passed to `run_async` and set during execution:

- `InvocationContext.is_aborted` flips to `True` and active agent or workflow
  tasks are cancelled cleanly without surfacing an unhandled `CancelledError` to
  the caller.
- `Runner._synthesize_abort_events_if_needed` appends and yields synthetic
  abort `Event`s with `error_code='INVOCATION_ABORTED'`. When function calls are
  dangling, it emits one abort event per `(author, branch, isolation_scope)`
  tuple sealing those calls with synthetic `FunctionResponse(response={'error':
  'Invocation was aborted by client.'})` parts so each response pairs with its
  call in the issuing agent's view. If nothing is dangling, it yields a single
  root-authored abort event instead. This preserves session log invariants so
  that subsequent conversational turns or resumes do not fail on dangling tool
  calls or re-run the cancelled tool.
- `PluginManager.run_after_run_callback` still executes before `run_async`
  exits.
- In the FastAPI server (`/run_sse`), client HTTP disconnects automatically set
  `abort_signal` on the active runner invocation.
