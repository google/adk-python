# RedisSessionService

RedisSessionService provides a persistent storage backend for application
sessions, user state, and events using Redis.

## Introduction

RedisSessionService allows an application to persist session data across
restarts and share state between multiple instances of a service. It manages
the lifecycle of Session objects, including their event history and
multi-scoped state.

The service handles three distinct levels of state: application-scoped state,
user-scoped state, and session-scoped state. By using Redis as the backend, the
unit ensures that state updates made in one session are visible to other
sessions belonging to the same user or application. It is primarily used by the
Runner and other session-aware components to maintain continuity in
conversations and workflows.

## Get started

To use Redis as a session store, provide a configuration object to the
RedisSessionService and use it to create or retrieve sessions.

```python
from google.adk.integrations.redis._redis_session_service import RedisSessionService
from google.adk.integrations.redis._config import RedisSessionServiceConfig

# Configure the service to connect to a local Redis instance
config = RedisSessionServiceConfig(
    uri="redis://localhost:6379",
    ttl_seconds=3600,
    key_prefix="myapp:"
)
service = RedisSessionService(config=config)

# Create a new session with initial state
session = await service.create_session(
    app_name="assistant_app",
    user_id="user_882",
    state={"user:name": "Alice", "step": "initial"}
)
```

## How it works

RedisSessionService stores data using a key-value structure where keys are
derived from the application name, user ID, and session ID. When a session is
requested, the service performs concurrent lookups for the session data, the
user-wide state, and the application-wide state.

The service manages state scoping through key prefixes. Keys starting with
`app:` are stored in a shared application record, keys starting with `user:` are
stored in a shared user record, and all other keys remain local to the specific
session. When a session is loaded, the service merges these scopes into a
single state dictionary for the caller.

When an event is appended via `append_event`, the service inspects the
`state_delta`. If the delta contains updates to `app:` or `user:` keys, the
service synchronizes those specific Redis records. This mechanism allows a
change in one session to propagate to all other sessions for that user
immediately.

## Configuration options

The service is configured through the `RedisSessionServiceConfig` class.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uri` | `Optional[str]` | `None` | A Redis connection string. |
| `host` | `Optional[str]` | `None` | The hostname of the Redis server. |
| `port` | `Optional[int]` | `None` | The port of the Redis server. |
| `password` | `Optional[str]` | `None` | The password for authentication. |
| `ssl` | `bool` | `False` | Whether to use a secure SSL connection. |
| `db` | `int` | `0` | The Redis database index to use. |
| `ttl_seconds` | `int` | `0` | Expiration time for all session keys. |
| `key_prefix` | `str` | `""` | A prefix added to every Redis key. |

The `uri` option takes precedence over individual `host` and `port` settings.
If `uri` is not provided, `host` defaults to `localhost` and `port` defaults to
`6379` during client initialization.

The `ttl_seconds` value determines how long session and state data persist in
Redis. If set to `0`, keys do not expire. When a non-zero TTL is set, every
update to a session or its associated state refreshes the expiration timer.

The `key_prefix` is useful for namespacing data when multiple applications
share the same Redis database.

## Advanced applications

You can control how much history is retrieved when loading a session by
providing a `GetSessionConfig`. This is useful for reducing network payload
sizes when a session has a very long event history.

```python
from google.adk.sessions.base_session_service import GetSessionConfig

# Retrieve only the 5 most recent events from the session
config = GetSessionConfig(num_recent_events=5)
session = await service.get_session(
    app_name="assistant_app",
    user_id="user_882",
    session_id="sess_123",
    config=config
)
```

If your application already manages a Redis connection pool, you can pass a
pre-configured `redis.asyncio.Redis` client directly to the service
constructor.

```python
import redis.asyncio as redis

shared_client = redis.Redis(host="custom-host", decode_responses=True)
service = RedisSessionService(redis_client=shared_client)
```

## Limitations

RedisSessionService requires the `redis` Python package to be installed. You
can install it using `pip install google-adk[redis]`.

State keys prefixed with `temp:` are treated as transient. While they are
available in the session object immediately after creation or an update, they
are not persisted to Redis and will disappear when the session is reloaded
from the store.

The `list_sessions` operation uses the Redis `SCAN` command. While this is
safer than `KEYS` for production environments, performance may degrade if the
database contains millions of keys without a specific `key_prefix` to narrow
the search space.
