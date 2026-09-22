# FirestoreSessionService

`FirestoreSessionService` stores each session as a Firestore document and each
event as a document beneath it, giving you a durable multi-process session
store with no database server to run. Writes go through Firestore transactions
and carry a revision number, so a worker holding a stale session is stopped
rather than allowed to overwrite a newer one.

To try it you need a Google Cloud project with a Firestore database in Native
mode, credentials that can read and write it, and the
`google-cloud-firestore` client library. Firestore bills per document read,
write and delete, so the cost of running an agent on it tracks the number of
turns in your conversations.

## Introduction

Between `DatabaseSessionService`, which needs a SQL database you operate, and
`VertexAiSessionService`, which ties sessions to an Agent Engine deployment,
there is a gap: a managed store you can point at from anywhere.
`FirestoreSessionService` fills it. It suits an agent already deployed on Cloud
Run or Cloud Functions, where adding a Postgres instance is the largest part of
the architecture.

It implements `BaseSessionService`, so the calls are the ones the
[Session guide](../../../sessions/session/index.md) describes. What is specific
to it is the document layout, which you have to understand before you can grant
IAM on it or query it by hand, and the fact that events are separate documents
rather than one blob. That second point has the larger practical effect:
appending a turn to a thousand-turn conversation writes one small document, not
the whole history.

### Import it

The service is imported from its own module rather than from the package:

```python
from google.adk.integrations.firestore.firestore_session_service import (
    FirestoreSessionService,
)
```

If `google-cloud-firestore` is not installed, that import raises `ImportError`
with an install hint, so you find out before you construct anything.

## Get started

Three steps get the service running.

1.  Create a Firestore database in Native mode in a Google Cloud project, and
    make credentials that can read and write it available to the process.

2.  Install the client library, either directly or through the `extensions`
    extra:

    ```bash
    pip install google-cloud-firestore
    ```

3.  Construct the service and pass it to a `Runner`, which handles the session
    for every turn. If you construct it with no arguments, it builds a default
    `AsyncClient`, which picks up the project and credentials the Google Cloud
    client libraries normally resolve from the environment. The application
    creates the session and then runs the agent against it:

    ```python
    from google.adk.agents import Agent
    from google.adk.apps import App
    from google.adk.integrations.firestore.firestore_session_service import (
        FirestoreSessionService,
    )
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
      session_service = FirestoreSessionService()
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

The runner writes the user message and everything the agent produces to
Firestore as the turn runs, so the final read returns those events and the
merged state, `{'locale': 'en-US', 'user:theme': 'dark'}`. Creating the session
first is deliberate: `Runner` leaves `auto_create_session` at `False`, so an
unknown `session_id` raises rather than starting a new conversation.

To choose the project, credentials or database explicitly, build the client
yourself and pass it:

```python
from google.cloud import firestore

session_service = FirestoreSessionService(
    client=firestore.AsyncClient(project="my-project")
)
```

## How it works

Four behaviors are worth understanding before you deploy on this service: where
the documents sit, how two writers to one session are resolved, what a read
fetches, and what deletion leaves behind.

### The document layout

Sessions live in a nested hierarchy under one root collection, and each session
owns an `events` subcollection:

```
adk-session/{app_name}/users/{user_id}/sessions/{session_id}
adk-session/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}
```

Session-scoped state is not a document of its own. It is a field on the session
document, described below. Shared state does get two collections of its own:

```
app_states/{app_name}
user_states/{app_name}/users/{user_id}
```

**Those two are at the top level of the database, not under the root
collection.** If you change `root_collection`, the sessions move and the shared
state stays where it was, so two services with different root collections still
read and write each other's `app:` and `user:` state. If you were planning to
separate two environments in one database that way, it separates the
conversations and leaves the shared state common to both.

A session document carries an identity, a pair of timestamps, its state, and a
revision counter. `id`, `appName` and `userId` are the identity, `createTime`
and `updateTime` the timestamps, and `revision` the number the concurrency
check reads. The session-scoped state goes into `state` as a JSON string, so it
is opaque to a Firestore query, while the `app:` and `user:` documents are
written natively, field by field, which keeps their datetimes and numbers
readable. On read all three are merged into one dict with the `app:` and
`user:` prefixes restored, which is why the state you get back is wider than
the session document. `temp:` keys are applied to the in-memory session and
written nowhere.

Each event is its own document, holding the serialized event under
`event_data`, a server `timestamp`, and `appName` and `userId` for querying.

### Optimistic concurrency

The session document carries an integer `revision`, and a `Session` loaded from
this service remembers the value it was loaded at. Appending an event runs
inside a transaction that re-reads the revision and raises `StaleSessionError`
when it has moved, and you can import that exception from `google.adk.errors`.
Nothing is written in that case: no event document, no state update, no
revision bump.

Because the runner writes through this service, that exception surfaces from
the run. Catch it around the turn:

```python
from google.adk.errors import StaleSessionError

try:
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session_id,
      new_message=message,
  ):
    print(event.author)
except StaleSessionError:
  print("Another worker advanced this session.")
```

Recover by running the turn again. The runner reloads the session from
Firestore at the start of each run, so the retry sees the other worker's
events.

Within one process, appends to the same session are serialized, so concurrent
tasks in one worker queue up instead of losing the race against each other.

### Reads and the event query

`get_session` fetches the session document, its events, and both shared-state
documents. A `GetSessionConfig` shapes the event query on the server rather
than in Python:

*   `num_recent_events=N` becomes `limit_to_last(N)` on a query ordered by
    timestamp.
*   `num_recent_events=0` skips the event query altogether, which makes it the
    cheap way to ask whether a session exists and read its state.
*   `after_timestamp` becomes a `>=` filter, converted to an aware UTC datetime
    first so the comparison does not shift by the host's UTC offset.

`list_sessions` returns sessions with empty event lists, ordered oldest first.
If you pass a `user_id`, it queries that user's `sessions` collection. If you
leave it out, it runs a collection-group query across every user's `sessions`
collection, filtered on `appName`, which is a different kind of query with its
own indexing requirements in Firestore, so expect to create an index the first
time you call it that way.

### Deletion

`delete_session` runs in a fixed order:

1.  Mark the session document `status: "DELETING"`, in a transaction.
2.  Delete the event documents in batches of 500.
3.  Delete the session document.

If an `append_event` arrives while the marker is set, it raises `ValueError`
rather than writing into a session that is going away.

The marking step is best-effort: if it fails, the deletion proceeds anyway, so
a concurrent append during that window is possible.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `client` | `Optional[firestore.AsyncClient]` | `None` | An async Firestore client. A default one is built when omitted. |
| `root_collection` | `Optional[str]` | `None`, resolving to `"adk-session"` | Top-level collection holding the session hierarchy. |

`root_collection` also reads the `ADK_FIRESTORE_ROOT_COLLECTION` environment
variable, which is how you point a deployment at a different collection without
a code change. The constructor argument wins over the variable, and the
variable wins over the built-in default.

The other four collection names are not constructor parameters. Two of them
name the subcollections under a session, `sessions` and `events`, and the other
two name the top-level shared-state collections, `app_states` and
`user_states`. Each has a constant naming it: `DEFAULT_SESSIONS_COLLECTION` and
`DEFAULT_EVENTS_COLLECTION` for the first pair,
`DEFAULT_APP_STATE_COLLECTION` and `DEFAULT_USER_STATE_COLLECTION` for the
second. All four are importable from the same module when you need to name a
collection in an IAM rule or a query of your own.

## Limitations

*   **`get_user_state` is not implemented.** Calling it raises
    `NotImplementedError`. Reading a user's state requires loading one of their
    sessions.
*   **App and user state ignore `root_collection`.** They are top-level
    collections, so two services with different root collections share them.
    Use separate Firestore databases to isolate environments, not separate root
    collections.
*   **Session state is an opaque JSON string.** The `state` field on a session
    document cannot be filtered or indexed by Firestore. Only `app:` and
    `user:` state is stored as real fields.
*   **The service has no `close()`.** A client it created for itself is never
    closed. Pass your own client if your application needs to control that.
*   **Deletion is not atomic.** Events are removed in 500-document batches
    before the session document goes, so a failure part way through leaves
    orphaned event documents under a session that may itself be gone.
*   **A session you constructed by hand skips the revision check.** The check
    only runs when the session carries a revision marker, which it does after
    `create_session` or `get_session`. Build a `Session` yourself and appends
    proceed unchecked.

## Related guides

*   [Session and BaseSessionService](../../../sessions/session/index.md), the
    interface this service implements and how the backends compare.
*   [State](../../../sessions/state/index.md), for what `app:`, `user:` and
    `temp:` mean before they are split across documents.
