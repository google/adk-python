# BufferedSessionService

`BufferedSessionService` wraps another session service and writes appended
events to it in the background, so that an agent turn does not wait for the
store on every event. Reads, flushes, and deletes wait for the writes they
depend on, and a write that keeps failing is reported instead of lost.

## Introduction

A turn appends many events: the user message, model responses, function calls
and responses, and state changes. With a remote store, each `append_event` is a
round trip, and the turn pays for all of them before the user sees a reply.

`BufferedSessionService` applies each event to the session object and returns
at once. A background writer then sends the events to the wrapped store in the
order they were appended. It works with any `BaseSessionService`, and the store
ends up holding the same session that an unbuffered service would have written.

## Get started

Wrap the store and pass the result to the `Runner`. The runner does not flush
the service yet, so flush the session after each run, before you tell the
client that the turn is done. Close the service on shutdown, so that the writes
still in flight reach the store.

```python
from google.adk.runners import Runner
from google.adk.sessions import BufferedSessionService
from google.adk.sessions import DatabaseSessionService

store = DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db")
session_service = BufferedSessionService(store)
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

async for event in runner.run_async(
    user_id=user_id, session_id=session_id, new_message=message
):
  ...
await session_service.flush_session(
    app_name="my_app", user_id=user_id, session_id=session_id
)

# On shutdown:
await session_service.aclose()
```

`aclose` waits for every buffered write and stops the writers. It leaves the
wrapped store open, because the store may have other users. If a buffered
write failed, `aclose` raises `SessionPersistenceError`, and so does
`Runner.close`, which flushes the session service.

## How it works

### Writes

Each session has its own writer, which sends the session's events in order.
Sessions do not wait for each other, with one exception: `app:` and `user:`
state is shared, so changes to it reach the store in the order they were
appended across sessions.

Each write carries the version of the session object it was appended to, as a
direct write would. A store that checks versions, such as `DatabaseSessionService`,
rejects the write of a session object that is behind with `StaleSessionError`,
failing only that stale session object's writes while leaving a newer session
object's writes untouched. If a store instead merges in what other writers
stored and retries, the service copies those events and that state to the
session object, as the store would without buffering.

### Reads

`get_session`, `list_sessions`, `get_user_state`, and `create_session` wait for
the buffered writes whose effects they would return, then read the store. A
read never raises a write failure.

### Flushing

`flush_session` waits until one session's buffered writes are in the store,
`flush` waits for every session, and both then flush the wrapped store:

```python
await session_service.flush_session(
    app_name=session.app_name, user_id=session.user_id, session_id=session.id
)
await session_service.flush()
```

`delete_session` flushes the session before it deletes it.

### Failures

A failed write is not retried unless you turn retries on. When a write fails,
the service drops that write and the session's later buffered writes, because
storing a later event without the earlier one would leave a gap in the history.
The session's next `flush_session`, `append_event`, or `delete_session` then
raises `SessionPersistenceError`, whose `__cause__` is the error of the last
attempt. A flush of every session reports each failure once.

The session object still holds the dropped events, so an append to it raises
`SessionPersistenceError` too. Read the session again with `get_session` and
continue from that copy. `StaleSessionError` and `SessionNotFoundError` are not
retried, because another attempt fails the same way.

Code that handles `StaleSessionError` directly on unbuffered stores (such as
skipping stale post-run compaction or returning HTTP 409) should also check
`__cause__` on `SessionPersistenceError`:

```python
try:
  await session_service.append_event(session, event)
  await session_service.flush_session(
      app_name=session.app_name, user_id=session.user_id, session_id=session.id
  )
except StaleSessionError:
  ...
except SessionPersistenceError as err:
  if isinstance(err.__cause__, StaleSessionError):
    ...
  else:
    raise
```

## Configuration options

`BufferedSessionServiceConfig` holds the options of the service:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `retry_config` | `RetryConfig` | `RetryConfig(max_attempts=1)` | How a failed write is retried. Off by default. |
| `max_write_delay` | `timedelta` | `timedelta(0)` | How long the first queued write of a session waits for others to join it. |
| `max_batch_events` | `int` | `50` | The most events sent in one `append_events` call. |

```python
from google.adk.sessions import BufferedSessionServiceConfig
from google.adk.workflow import RetryConfig

session_service = BufferedSessionService(
    store,
    config=BufferedSessionServiceConfig(retry_config=RetryConfig(max_attempts=3)),
)
```

`retry_config` is the workflow `RetryConfig`. With retries on, it waits about
1, 2, 4, and 8 seconds between attempts by default, with jitter. Retries are
off by default because a retry can resend an event that the store already
committed. Turn them on only for a store that keeps such an event once.
`DatabaseSessionService` rejects the resent event as stale instead of storing
it twice, so the write is reported as failed although the store has it.

When the wrapped store overrides `BaseSessionService.append_events(session,
events)`, the writer sends a session's queued events in one call, up to
`max_batch_events` per call. An override must store all of the events or none of
them, in order, with the same effect on `session` and the store as calling
`append_event` for each event. The base method appends events one at a time, and
ADK's built-in session services keep it, so for them each event is sent in its
own `append_event` call.

`max_write_delay` holds back the first queued write of a session so that nearby
events join it in one `append_events` call. A flush or read that waits on the
session sends queued writes right away without waiting out the delay.

## Advanced applications

### Use it with adk web and adk api_server

`adk web` and `adk api_server` build the session service from
`--session_service_uri`. To buffer it, register a URI scheme in a `services.py`
beside your agents, as the
[ServiceRegistry guide](../../cli/service_registry/index.md) describes. This
factory wraps the service that the rest of a `buffered:` URI names:

```python
# my_agents/services.py
from typing import Any

from google.adk.cli.service_registry import get_service_registry
from google.adk.sessions import BufferedSessionService


def buffered_session_factory(uri: str, **kwargs: Any) -> BufferedSessionService:
  """Builds buffered:<URI> as a BufferedSessionService over <URI>."""
  inner_uri = uri.removeprefix("buffered:")
  store = get_service_registry().create_session_service(inner_uri, **kwargs)
  if store is None:
    # Needs the optional `db` extra, so it is imported only when used.
    from google.adk.sessions import DatabaseSessionService

    kwargs.pop("agents_dir", None)
    store = DatabaseSessionService(db_url=inner_uri, **kwargs)
  return BufferedSessionService(store)


get_service_registry().register_session_service(
    "buffered", buffered_session_factory
)
```

Then pass the URI on the command line:

```bash
adk api_server my_agents --session_service_uri=buffered:sqlite:///sessions.db
```

### Event loops

The writers run on the event loop of the first append. While writes are in
flight, a call from another loop raises `RuntimeError`. Once they complete, the
service moves to the loop of the next call, so code that runs each request in
its own `asyncio.run` can share one service if it flushes before the loop ends.
An application that runs several event loops at once needs one service per loop.
The service is not thread-safe.

## Limitations

*   **Experimental.** The API may change.
*   `Runner.run` closes its event loop when the run ends, which cancels the
    writes still in flight. Use `run_async` and flush after each run.
*   A crash loses the writes that had not reached the store.
*   Two processes that write to one session conflict as they would without
    buffering. A store that checks versions rejects one of the writes, and the
    service reports it as a `SessionPersistenceError`.
*   An `app:` or `user:` change waits for earlier changes to that scope from
    other sessions, including their retries.
*   Up to 1024 sessions keep a failure to report. Past that, the oldest is
    forgotten and a warning is logged.
*   A failed write is logged by the type of its error only, because the message
    can hold session content.
