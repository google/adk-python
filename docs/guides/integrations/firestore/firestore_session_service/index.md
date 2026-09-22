# FirestoreSessionService

The `FirestoreSessionService` provides persistent session storage using Google
Cloud Firestore as the backend. It manages multi-user and multi-application state
along with event history to support long-running agent interactions.

## Introduction

The service allows developers to store and retrieve session data across multiple
application instances. It handles the complexities of state management by
merging application-wide, user-specific, and session-private data into a single
view. This service is essential for building agents that need to remember past
interactions or maintain consistent state for a user across different devices
and sessions.

The service depends on the `google-cloud-firestore` library and uses the
Firestore `AsyncClient` for non-blocking database operations. It implements
optimistic concurrency control to prevent data loss when multiple processes
attempt to update the same session simultaneously.

## Get started

Initialize the service to manage persistent sessions in a Firestore database.
Create a session to begin tracking a user interaction history.

```python
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
from google.adk.events.event import Event

# Initialize the session service with default settings
service = FirestoreSessionService()

# Create a session for a specific user and application
session = await service.create_session(
    app_name="customer_support",
    user_id="user_123",
    state={"app:version": "2.0", "user:name": "Alice"}
)

# Append an interaction event to the session
event = Event(invocation_id="inv_001", author="user")
await service.append_event(session, event)
```

## How it works

The service manages session data through a structured document hierarchy and a
prefixed state management system.

### Storage hierarchy

The service organizes data using a nested collection structure. The primary
hierarchy for sessions is as follows:

```
adk-session (root)
↳ <app name>
  ↳ users
    ↳ <user ID>
      ↳ sessions
        ↳ <session ID>
          ↳ events
            ↳ <event ID>
```

Shared state configurations are stored in separate top-level collections to
allow access across different sessions:

```
app_states
↳ <app name>

user_states
↳ <app name>
  ↳ users
    ↳ <user ID>
```

### State management

The service distinguishes between three scopes of state based on key prefixes:

1.  **Application state**: Keys starting with `app:` are stored in the
    `app_states` collection and are shared by all users of that application.
2.  **User state**: Keys starting with `user:` are stored in the `user_states`
    collection and are shared by all sessions for a specific user within an
    application.
3.  **Session state**: Unprefixed keys are stored within the specific session
    document.
4.  **Temporary state**: Keys starting with `temp:` are applied to the local
    session object in memory but are never persisted to Firestore.

When a session is retrieved, the service merges these scopes into a single
dictionary. Application and user states are written natively to Firestore,
preserving rich types such as `datetime` objects. Session-specific state is
serialized to a JSON string, and the service coerces values to JSON-safe formats
to ensure storage compatibility.

### Concurrency and locking

To ensure data consistency, the service implements two layers of protection:

-   **Optimistic concurrency**: Every session document includes a `revision`
    field. When appending an event, the service checks that the revision in
    Firestore matches the revision held by the local `Session` object. If the
    document has been updated by another process, the service raises a
    `StaleSessionError`.
-   **Local serialization**: The service uses internal `asyncio.Lock` instances
    to serialize `append_event` calls for the same session within a single
    Python process. This prevents race conditions where multiple tasks in the
    same instance attempt to update the same session object simultaneously.

## Configuration options

The service allows configuration of the Firestore client and the root storage
location.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `client` | `Optional[firestore.AsyncClient]` | `None` | An optional Firestore client for database operations. |
| `root_collection` | `Optional[str]` | `None` | The name of the top-level collection for session data. |

A developer can choose to provide an existing `AsyncClient` if the application
manages its own database connections. If the `client` is `None`, the service
creates a new instance automatically.

The `root_collection` defaults to `adk-session`. If the `root_collection`
parameter is `None`, the service also checks the `ADK_FIRESTORE_ROOT_COLLECTION`
environment variable before falling back to the default.

## Advanced applications

The service provides several methods for managing the lifecycle of sessions
beyond simple creation and retrieval.

### Session history control

The `get_session` method accepts a `GetSessionConfig` object to optimize
database reads. A developer can set `num_recent_events` to limit the number of
events retrieved from history. Setting this value to `0` allows the service to
verify a session exists without downloading its entire event transcript. The
`after_timestamp` option filters the history to only include events occurring
after a specific time.

### Lifecycle management

The `list_sessions` method retrieves all sessions for a given application. If a
`user_id` is provided, the results are filtered to that specific user. This
method requires a Firestore collection group index for the `sessions`
collection to function across multiple users.

The `delete_session` method handles the removal of a session and all its
associated events. The service first marks the session status as `DELETING` to
block further appends, then deletes events in batches of 500 to remain within
Firestore transaction limits before finally removing the session document.

## Limitations

The service requires the `google-cloud-firestore` package to be installed in the
environment.

The internal locking mechanism only synchronizes updates within a single Python
process. In a distributed environment with multiple server instances,
applications must handle `StaleSessionError` to manage concurrent updates to
the same session document.

Listing sessions across all users for an application requires the manual
creation of a collection group index in the Google Cloud Console or via the
`gcloud` CLI. Without this index, `list_sessions` calls that do not specify a
`user_id` will fail.
