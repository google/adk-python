# RedisSessionService

`RedisSessionService` keeps ADK sessions in Redis, so several processes share
one conversation without a relational database between them. Unlike every other
session backend, its sessions expire. The service puts a deadline on every key
it writes, seven days by default, so a conversation nobody touches for a week
is gone.

To try it you need a Redis server the process can reach and the `redis` extra
installed. There is no account to create and no cloud service to configure, so
a Redis container on your laptop is enough.

## Introduction

Sessions have to live somewhere every worker can reach.
`DatabaseSessionService` answers that with SQL and `VertexAiSessionService`
with a managed service. If you already run Redis for cache and queues, you can
keep sessions there instead of adding a database purely for chat history, so
you operate nothing new and a session read is a single key lookup.

It implements `BaseSessionService`, so the calls are the ones the
[Session guide](../../../sessions/session/index.md) describes. Two things set
it apart, and both matter more than the API surface. Sessions carry a
time-to-live, which makes it a good fit for conversations that are supposed to
age out and a bad fit for a permanent record. And it has no
optimistic-concurrency check, so two workers appending to the same session both
succeed and the second one's write is the one that survives.

Both `RedisSessionService` and its configuration class are exported from
`google.adk.integrations.redis`.

## Get started

Three steps get the service running.

1.  Start a Redis server the process can reach. A container on your own machine
    is enough.

2.  Install the extra, which pulls in `redis` 4.2 or newer for `redis.asyncio`:

    ```bash
    pip install "google-adk[redis]"
    ```

3.  Point the service at your server and pass it to a `Runner`, which handles
    the session for every turn. The application creates the session and then
    runs the agent against it:

    ```python
    from google.adk.agents import Agent
    from google.adk.apps import App
    from google.adk.integrations.redis import RedisSessionService
    from google.adk.integrations.redis import RedisSessionServiceConfig
    from google.adk.runners import Runner
    from google.genai import types

    APP_NAME = "hello_world"
    USER_ID = "user-123"

    root_agent = Agent(
        name="weather_agent",
        description="Answers questions about the weather.",
        instruction="Answer the user's question in one sentence.",
    )


    async def main() -> None:
      session_service = RedisSessionService(
          config=RedisSessionServiceConfig(uri="redis://localhost:6379/0")
      )
      runner = Runner(
          app=App(name=APP_NAME, root_agent=root_agent),
          session_service=session_service,
      )

      session = await session_service.create_session(
          app_name=APP_NAME,
          user_id=USER_ID,
          state={"locale": "en-US", "user:theme": "dark"},
      )

      async for event in runner.run_async(
          user_id=USER_ID,
          session_id=session.id,
          new_message=types.Content(
              role="user",
              parts=[types.Part.from_text(text="What is the weather in Zurich?")],
          ),
      ):
        if event.content and event.content.parts:
          print(event.author, event.content.parts[0].text)

      loaded = await session_service.get_session(
          app_name=APP_NAME, user_id=USER_ID, session_id=session.id
      )
      print(len(loaded.events), loaded.state)
    ```

The runner writes the user message and everything the agent produces to Redis
as the turn runs, so the final read returns those events and the merged state,
`{'locale': 'en-US', 'user:theme': 'dark'}`. Creating the session first is
deliberate: `Runner` leaves `auto_create_session` at `False`, so an unknown
`session_id` raises rather than starting a new conversation.

If you pass no `config` at all, the service connects to `localhost:6379`,
database `0`, with the seven-day expiry. Those defaults match a container on
your own machine, where the server is local and a lost conversation costs
nothing. Set the connection details and `ttl_seconds` yourself for a
deployment, because neither the address nor the retention period is likely to
be the same there.

## How it works

Three things follow from storing a session as a single Redis value: the shape
of the keys, the deadline on them, and how the service reaches the server.

### What is stored, and where

The service writes three kinds of Redis key, all plain JSON strings:

| Key | Contents |
| --- | --- |
| `{key_prefix}{app_name}:{user_id}:{session_id}` | The whole `Session` as one JSON document: its id, its session-scoped state, and the complete event list. |
| `{key_prefix}user_state:{app_name}:{user_id}` | The `user:`-scoped state for that user, with the prefix stripped. |
| `{key_prefix}app_state:{app_name}` | The `app:`-scoped state for that app, with the prefix stripped. |

Only session-scoped keys go in the session document. `app:` and `user:` keys
are split out to their own keys on write, then merged back in with their
prefixes restored on every read. That is how a second session for the same user
starts out already knowing that user's preferences. `temp:` keys are applied to
the in-memory session and never written anywhere.

Storing the whole conversation in one string has one consequence that governs
everything else: **every event appended re-serializes and rewrites the entire
event history.** A hundred-turn conversation writes the whole hundred turns on
turn one hundred and one. That is fine for the short conversations Redis suits
and it is the wrong shape for a transcript that grows without bound.

### Expiry

Redis keeps its data in memory, so a session store that expires nothing grows
for as long as the application runs. Every write therefore sets a TTL of
`ttl_seconds`, which defaults to `604800`, or seven days: long enough to
outlast an ordinary conversation, short enough that abandoned ones do not
accumulate. The clock restarts on each write, so an active conversation stays
alive indefinitely and an abandoned one disappears a week after its last turn.
`get_session` then returns `None`, indistinguishable from a session that never
existed.

The app-state and user-state keys carry the same TTL, but they are only
rewritten when a state delta actually touches them. If your app writes
`app:config` once at startup and never again, it loses that value seven days
later while its sessions are still in use. If you keep long-lived shared state
here, either write it periodically or turn expiry off.

Setting `ttl_seconds` to `0` or a negative number disables expiry entirely; the
service then passes no expiry to Redis and the keys persist until something
deletes them.

### Connecting

The client is created on the first call, not in the constructor. So if the
`redis` package is missing, you find out the first time you touch a session, as
an `ImportError` with an install hint, rather than at import or construction
time.

There are three ways to say where Redis is, and they are checked in this order:

1.  A `redis.asyncio.Redis` you pass as `redis_client=`. The config's connection
    fields are then ignored entirely, though `ttl_seconds` and `key_prefix`
    still apply.
2.  `config.uri`, passed to `redis.asyncio.from_url`. **A URI ignores the
    `host`, `port`, `password`, `ssl` and `db` fields.** Everything has to be
    in the URI, which means TLS is `rediss://` and not `redis://` plus
    `ssl=True`.
3.  Otherwise the individual fields, used to construct `redis.asyncio.Redis`
    directly.

Whichever path is taken, the service sets `decode_responses=True` on a client
it creates. A client you pass in yourself should have it too, since the service
reads Redis values as strings.

## Configuration options

`RedisSessionServiceConfig` is a Pydantic model, and every field has a default.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uri` | `Optional[str]` | `None` | Full connection URI. When set, the five fields below are not used. |
| `host` | `Optional[str]` | `"localhost"` | Server hostname. |
| `port` | `Optional[int]` | `6379` | Server port. |
| `password` | `Optional[str]` | `None` | Password for authentication. |
| `ssl` | `bool` | `False` | Whether to connect with TLS. |
| `db` | `int` | `0` | Redis database index. |
| `ttl_seconds` | `int` | `604800` | Seconds before a key expires, refreshed on every write. Zero or less disables expiry. |
| `key_prefix` | `str` | `"adk:session:"` | Prefix on every key the service creates. |

The constructor itself takes `config` and `redis_client`, both positional or
keyword and both optional.

`key_prefix` is the field to change when more than one application shares a
Redis instance, because it is the only thing separating their keys. Give each
app its own prefix and `list_sessions` stops seeing the other's sessions. Be
careful about changing it once data exists, because the old keys then become
unreachable: the service never scans outside its current prefix.

`ttl_seconds` deserves a deliberate decision rather than the default. Seven days
is right for support chats and wrong for anything a user expects to find next
month.

## Advanced applications

Two arrangements come up once an application outgrows the default client:
handing the service a client it already builds, and limiting how much history
a run loads.

### Reuse an existing client

*   **Problem solved**: your application already builds a Redis client, with a
    connection pool, TLS material and retry policy it wants to keep.
*   **Implementation**: pass it as `redis_client=`. The service uses it as-is
    and never replaces it.

    ```python
    import redis.asyncio as redis_asyncio

    client = redis_asyncio.Redis.from_url(
        "rediss://cache.internal:6379/0", decode_responses=True
    )
    session_service = RedisSessionService(
        config=RedisSessionServiceConfig(ttl_seconds=86400),
        redis_client=client,
    )
    ```

    You own that client's lifetime. The service has no `close()` and never
    disconnects, so closing the client is your application's job. That is also
    the only way it ever gets closed, including for a client the service
    created for itself.

### Bound what a read pulls back

*   **Problem solved**: a long conversation means a large JSON document, and
    you only need the tail of it.
*   **Implementation**: put a `GetSessionConfig` on the run config, exactly as
    with any other backend, and the runner applies it when it loads the
    session. Be aware that this trims after the fact. The whole document is
    fetched from Redis and parsed either way, so the saving is in what your
    agent sees, not in what crosses the network.

    ```python
    from google.adk.agents.run_config import RunConfig
    from google.adk.sessions.base_session_service import GetSessionConfig

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(
            get_session_config=GetSessionConfig(num_recent_events=20)
        ),
    ):
      print(event.author)
    ```

## Limitations

*   **No stale-write detection.** `DatabaseSessionService` and
    `FirestoreSessionService` raise `StaleSessionError` when two holders of the
    same session both append. This service does not check: the second write
    replaces the key, and the first worker's events are gone with no error
    anywhere. Do not run two workers on one session.
*   **The whole history is one Redis value.** Each append rewrites all of it,
    and a Redis string tops out at 512 MB. Long-running conversations belong in
    a backend that stores events as rows or documents.
*   **`list_sessions` scans, and returns everything.** It walks the keyspace
    with `SCAN` and parses every matching session, including its full event
    history, unlike other backends which drop events from the listing. On a
    large keyspace this is slow and the result is large. It also returns newest
    first, where the interface documents oldest first.
*   **`delete_session` deletes only the session.** The user-state and app-state
    keys survive, so a new session for the same user still sees that user's
    stored preferences. Delete those keys yourself if you need a real erasure.
*   **A colon in an `app_name` or `user_id` breaks key scoping.** Ids are
    concatenated into the key with `:` separators and nothing escapes them, so
    a `user_id` of `"b:c"` produces keys that
    `list_sessions(app_name=..., user_id="b")` also matches. Keep colons out of
    both.
*   **A key that fails to parse is skipped with a warning.** `list_sessions`
    logs and moves on rather than raising, so a corrupt or foreign key under
    your prefix silently reduces the result instead of failing loudly.

## Related guides

*   [Session and BaseSessionService](../../../sessions/session/index.md), the
    interface this service implements and how the backends compare.
*   [State](../../../sessions/state/index.md), for what `app:`, `user:` and
    `temp:` mean before they are split across Redis keys.
