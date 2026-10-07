# Session and BaseSessionService

`Session` is the conversation record — its id, its owner, its state, and its
ordered event history. `BaseSessionService` is the storage interface that
creates, reads, lists, and deletes those records and appends events to them.

## Introduction

An agent run is stateless on its own: the model sees only what you give it. A
`Session` is what carries a conversation across turns, holding the event history
that becomes the model's context and a `state` dict that agents and tools read
and write.

`Session` is a plain Pydantic model and never talks to storage itself.
Everything that persists a session goes through a `BaseSessionService`, which
declares four abstract methods — `create_session`, `get_session`,
`list_sessions`, `delete_session` — plus a concrete `append_event` that every
backend inherits. That split is why the same agent code runs unchanged against
an in-process dict during development and a shared database in production: you
swap the service, not the agent. `Runner` takes a `session_service` as a
required argument and drives `get_session` and `append_event` for you, so most
applications call the service directly only to create, list, and delete
sessions.

## Get started

`InMemorySessionService` needs no configuration. This example creates a
session, appends two events, and reads the result back.

```python
import asyncio

from google.adk.events import Event
from google.adk.sessions import InMemorySessionService

APP_NAME = "hello_world"
USER_ID = "user-123"


async def main() -> None:
  session_service = InMemorySessionService()

  # 1. Create. Omit session_id to have one generated for you.
  session = await session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
      state={"locale": "en-US"},
  )

  # 2. Append events. Each one lands in session.events, and any state the
  # event carries is merged into session.state.
  await session_service.append_event(
      session, Event(author="user", message="What is the weather?")
  )
  await session_service.append_event(
      session,
      Event(
          author="weather_agent",
          message="It is sunny.",
          state={"last_city": "Zurich"},
      ),
  )

  # 3. Read it back. get_session returns None when nothing is stored.
  loaded = await session_service.get_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=session.id
  )
  assert loaded is not None
  print(len(loaded.events), loaded.state)


if __name__ == "__main__":
  asyncio.run(main())
```

This prints `2 {'locale': 'en-US', 'last_city': 'Zurich'}`.

Every method is keyword-only except `append_event`, which takes the session and
the event positionally. A session is identified by the triple
`(app_name, user_id, session_id)`, not by `session_id` alone, so all three are
required on every read.

## How it works

### The lifecycle

`create_session` generates a UUID when you do not pass `session_id`, and raises
`AlreadyExistsError` (from `google.adk.errors.already_exists_error`) when you
pass one that is already taken. `get_session` returns `None` for a missing
session rather than raising. `list_sessions` returns a `ListSessionsResponse`
ordered by `last_update_time`, oldest first, with the event history omitted.

`append_event` is where the two copies of a session meet. The base
implementation applies the event's `actions.state_delta` to the in-memory
`Session` you hold and appends to `session.events`; each backend overrides it to
write the event to storage as well. Partial events (`event.partial` is true) are
returned untouched and never stored, which is how streaming chunks stay out of
the history. Appending to a session that storage does not know about raises
`SessionNotFoundError` on every backend, so a deleted or never-created session
fails loudly instead of dropping the event.

### State scoping

Keys in `state` are scoped by prefix, and the prefixes are constants on `State`:

| Prefix | Constant | Scope |
| --- | --- | --- |
| none | | This session only. |
| `app:` | `State.APP_PREFIX` | Every session of the app. |
| `user:` | `State.USER_PREFIX` | Every session of this user within the app. |
| `temp:` | `State.TEMP_PREFIX` | The current invocation only; never persisted. |

Write prefixed keys like any other key, in `create_session(state=...)` or in an
event's state delta. The service routes them to the right storage scope and
merges them back into `session.state` on read, prefix included. `temp:` keys are
the exception: they are applied to the in-memory session so later agents in the
same invocation can read them, then stripped from the event before it is
written.

`get_user_state(app_name=..., user_id=...)` reads user-scoped state without a
session id, returning raw keys with the `user:` prefix removed — useful for
bootstrapping context before `create_session`. It is not abstract, and the
default implementation raises `NotImplementedError`, so a custom backend that
does not override it will fail this call.

### Trimming what you load

Pass a `GetSessionConfig` to bound the history you read back. It lives in
`google.adk.sessions.base_session_service`, not in the package root:

```python
from google.adk.sessions.base_session_service import GetSessionConfig

# The 20 most recent events. Use num_recent_events=0 for metadata and state
# only, or after_timestamp=<unix seconds> to cut the history by time instead.
recent = await session_service.get_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=session_id,
    config=GetSessionConfig(num_recent_events=20),
)
```

The service applies these filters, so on a database backend they reduce what is
read, not just what you see.

## Choosing a session service

| Service | Import | Use it when |
| --- | --- | --- |
| `InMemorySessionService` | `google.adk.sessions` | Developing and testing. State lives in process dicts and the class documents itself as unsuitable for multi-threaded production. |
| `DatabaseSessionService` | `google.adk.sessions` | You need durability, or several processes sharing one conversation. Backed by a SQLAlchemy async engine; requires the `db` extra. |
| `VertexAiSessionService` | `google.adk.sessions` | You are deploying on Vertex AI Agent Engine and want its managed session store. Requires the `gcp` extra. |
| `SqliteSessionService` | `google.adk.sessions.sqlite_session_service` | You want a local SQLite file and no server. This is what the ADK CLI uses; note it is not re-exported from the package root. |

`DatabaseSessionService` takes either a URL or an engine you already own, and
exactly one of the two:

```python
from google.adk.sessions import DatabaseSessionService

async with DatabaseSessionService("sqlite+aiosqlite:///./sessions.db") as svc:
  await svc.prepare_tables()  # optional; otherwise done on first use
  session = await svc.create_session(app_name=APP_NAME, user_id=USER_ID)
```

Use an async driver in the URL — `sqlite+aiosqlite`, `postgresql+asyncpg`, and
so on. Passing `db_engine=<AsyncEngine>` instead reuses your application's
engine, and the service will not dispose of one it did not create. As an async
context manager it closes the engine it owns on exit; call `close()` yourself
otherwise.

`VertexAiSessionService` differs in one respect worth knowing before you switch
to it: `app_name` is not a free-form string there. It must be the reasoning
engine id or the full `projects/.../locations/.../reasoningEngines/N` resource
name, unless you pass `agent_engine_id` to the constructor.

## Advanced applications

### Wiring a service into a Runner

*   **Problem solved**: one place decides where every conversation is stored.
*   **Implementation**: pass the service to `Runner(session_service=...)` and
    create the session before the first run. `Runner` defaults
    `auto_create_session` to `False`, so an unknown `session_id` raises
    `SessionNotFoundError` instead of silently starting a new conversation.

### Writing your own backend

*   **Problem solved**: your sessions belong in a store ADK does not ship.
*   **Implementation**: subclass `BaseSessionService` and implement the four
    abstract methods. Override `append_event` to persist the event and call
    `await super().append_event(session, event)` so the in-memory session stays
    in step. Override `get_user_state` if your store can answer it, and `flush`
    if you buffer writes — the base `flush` is a no-op that `Runner` calls when
    it closes.

### Detecting a stale session

*   **Problem solved**: two workers hold the same `Session` object and both
    append, so one would silently overwrite the other's history.
*   **Implementation**: nothing to write. `DatabaseSessionService` tracks a
    storage revision per session and raises `ValueError` from `append_event`
    when the in-memory copy has fallen behind. Recover by calling `get_session`
    again and replaying the append against the fresh session.

## Limitations

*   **`InMemorySessionService` is not for production**: nothing survives a
    restart, nothing is shared between workers, and it does not lock.
*   **`list_sessions` returns partial sessions**: the event history is dropped,
    and how much of `state` is populated depends on the backend. Load what you
    need with `get_session`.
