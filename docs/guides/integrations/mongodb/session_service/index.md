# MongoDbSessionService

`MongoDbSessionService` is a `BaseSessionService` that stores sessions, their event history, and `app:`, `user:`, and session-scoped state in a MongoDB database. It lets several runner processes share sessions through one database, on Atlas or on any MongoDB deployment PyMongo can reach.

## Introduction

`InMemorySessionService` loses every session when the process exits, and it cannot share a session between two replicas of the same service. A durable backend fixes both, and `MongoDbSessionService` is the one to use when the application already keeps its data in MongoDB.

The `Runner` calls the service on every turn: it loads the session with `get_session`, and it records each non-partial event with `append_event`. The service merges the three state scopes into the single `session.state` view the agent sees. Keys prefixed with `app:` are shared by every session of the app, keys prefixed with `user:` are shared by every session of one user, and the remaining keys belong to the one session. Keys prefixed with `temp:` live on the in-memory session only and are never written.

The service is experimental. It sits behind `FeatureName.MONGODB_SESSION_SERVICE`, which is on by default, so the first construction logs a one-time experimental warning.

## Get started

Install the extra, which brings in `pymongo`:

```shell
pip install 'google-adk[mongodb]'
```

Construct the service with a connection string and a database name, and hand it to the `Runner`:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.runners import Runner

agent = LlmAgent(
    name="assistant",
    instruction="You are a helpful assistant.",
)

session_service = MongoDbSessionService(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="my_app",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)
```

The service creates the client from the connection string and owns it, so call `await session_service.close()` during shutdown to release its connections.

## How it works

The service keeps four collections in the configured database. Each session, event, and shared state bucket is its own document, keyed by an `_id` subdocument made of its identifiers:

| Collection | Document key | Contents |
| :--- | :--- | :--- |
| `sessions` | `{app_name, user_id, session_id}` | Session-scoped state, timestamps, and a revision counter. |
| `events` | `{app_name, user_id, session_id, event_id}` | One serialized event. |
| `app_states` | `<app_name>` | State under the `app:` prefix. |
| `user_states` | `{app_name, user_id}` | State under the `user:` prefix. |

State and events are stored JSON-encoded rather than as BSON subdocuments, because MongoDB restricts field names containing `.` or starting with `$`, and a state key may contain either. Values that are not JSON-serializable are coerced when the service writes them, and the session returned by `create_session` already carries the coerced values, so it matches what a later `get_session` reads back.

`append_event` persists only the event's `state_delta`. A direct assignment to `session.state` outside an event is not saved, which matches the other database-backed services.

### Atomic writes and concurrency

One `append_event` touches up to four documents: the session, the event, and the app and user state buckets. On a replica set, a sharded cluster, or Atlas, the service writes them in one multi-document transaction, so they commit or abort together, and a concurrent writer to shared app or user state loses to a retry instead of silently overwriting the other.

A standalone `mongod` cannot run transactions. The first write there fails its transaction attempt, the service logs a warning, and every later write runs sequentially without one. Writes are then not atomic across documents. A single-node replica set is enough to give a development deployment transactions.

Every session document carries a revision that `append_event` checks and increments. A session loaded before another writer appended to it is stale, and appending to it raises `StaleSessionError` instead of overwriting the newer history. The caller reloads the session with `get_session` and retries. Within one process the service also serializes appends to the same session, but that only orders local callers; across replicas the revision check is the guard.

An event id identifies one event within its session. Appending a second event that reuses an id already stored raises `AlreadyExistsError` and leaves the stored event and session state unchanged.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `database_name` | `str` | required | Database holding the four collections. |
| `connection_string` | `str \| None` | `None` | MongoDB URI the service builds and owns a client from. |
| `mongo_client` | `MongoClient \| None` | `None` | An existing client the caller owns. |
| `sessions_collection` | `str` | `"sessions"` | Collection for session documents. |
| `events_collection` | `str` | `"events"` | Collection for event documents. |
| `app_state_collection` | `str` | `"app_states"` | Collection for app-scoped state. |
| `user_state_collection` | `str` | `"user_states"` | Collection for user-scoped state. |
| `ensure_indexes` | `bool` | `True` | Whether to create the recommended secondary indexes on first use. |

Exactly one of `connection_string` and `mongo_client` must be set, and passing both or neither raises `ValueError`. A client the service builds from `connection_string` is closed by `close()`; a client passed as `mongo_client` stays open, because the caller may share it with other components.

PyMongo sets no time limit on an operation by default, and every session call runs on the turn's critical path. Add `timeoutMS` to the connection string, or pass `timeoutMS=` when building your own `MongoClient`, to bound each operation:

```python
session_service = MongoDbSessionService(
    connection_string=(
        "mongodb+srv://user:password@cluster.mongodb.net/?timeoutMS=10000"
    ),
    database_name="my_app",
)
```

The four collection names let several applications or environments share one database without colliding.

`ensure_indexes` covers the two reads that are not `_id` lookups: `get_session` reads events by app, user, and session ordered by time, and `list_sessions` filters sessions by app and user. Without indexes both become collection scans as the data grows, so the service creates them on first use: `app_user_session_ts` on the events collection and `app_user` on the sessions collection. Creating an index that already exists is a no-op. Indexes only speed up reads, so an index-creation error never fails a session call: the service logs a warning and reads run unindexed. A database user without the `createIndex` privilege, or an existing index that conflicts (the same name with other keys, or the same keys under another name), gets one warning and no retry; after any other error the next call tries again. Pass `ensure_indexes=False` to run under a user that deliberately lacks the privilege without the warning.

## Advanced applications

### Sharing one client across components

When an application also uses `MongoDbToolset`, or already configures its own pool and TLS settings, pass one `MongoClient` to both, so the process keeps a single connection pool:

```python
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.integrations.mongodb import MongoDbToolset
from pymongo import MongoClient

client = MongoClient("mongodb+srv://user:password@cluster.mongodb.net/")

toolset = MongoDbToolset(database_name="catalog", mongo_client=client)
session_service = MongoDbSessionService(
    database_name="my_app",
    mongo_client=client,
)
```

Neither component closes `client`, so close it during application shutdown.

### Loading part of a long history

A session with a long history costs a large read on every turn. Pass a `GetSessionConfig` to load only the recent events, or only those at or after a timestamp:

```python
from google.adk.sessions.base_session_service import GetSessionConfig

session = await session_service.get_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
    config=GetSessionConfig(num_recent_events=20),
)
```

## Limitations

- On a deployment without transactions, a failure part way through `append_event` can leave some of its documents written and others not.
- Concurrent appends to one session from several processes are resolved by `StaleSessionError`, so callers that write from more than one replica need to reload and retry.
- The service does not expire sessions, so remove old ones with `delete_session`.
- The service takes a synchronous `MongoClient` so one client can be shared with `MongoDbToolset`, and runs each PyMongo call in a worker thread. Cancelling a turn does not stop a call already running there, so set `timeoutMS` to bound it.
