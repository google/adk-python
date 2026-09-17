# Redis Integration for ADK

This integration provides Redis-backed persistent session storage for the Google Agent Development Kit (ADK).

## Features

- **Session Persistence:** Store and retrieve agent sessions, conversation events, and state in Redis using `redis.asyncio`.
- **App & User State Scoping:** Automatic merging and synchronization of `app:`, `user:`, and `session:` scoped state across turns and sessions.
- **TTL Support:** Automatically expire stale sessions after a configurable duration (`ttl_seconds`).
- **Flexible Connection Options:** Connect via connection URI (`redis://` / `rediss://`), individual connection parameters (`host`, `port`, `password`, `ssl`, `db`), or a pre-configured `redis.asyncio.Redis` client.
- **Event Filtering:** Retrieve sessions with filtered event histories based on timestamps or recent event counts.

## Installation / Dependencies

### Open-Source ADK

Install the `redis` package alongside ADK:

```bash
pip install google-adk redis
```

or if ADK is already installed:

```bash
pip install redis
```

## Quick Start

```python
from google.adk.agents import Agent
from google.adk.integrations.redis import RedisSessionService
from google.adk.integrations.redis import RedisSessionServiceConfig
from google.adk.runners import Runner

# 1. Configure the Redis session service
config = RedisSessionServiceConfig(
    uri="redis://localhost:6379/0",
    ttl_seconds=86400 * 7,  # 7 days
)
session_service = RedisSessionService(config=config)

# 2. Define your agent
agent = Agent(
    name="assistant",
    instructions="You are a helpful AI assistant.",
)

# 3. Create the Runner with app_name and session_service
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service,
)
```

## Configuration

`RedisSessionServiceConfig` supports the following options:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uri` | `Optional[str]` | `None` | Redis connection URI (e.g. `redis://[:password@]host:port/db` or `rediss://...` for SSL). If set, takes precedence over individual connection fields. |
| `host` | `Optional[str]` | `"localhost"` | Redis server hostname. |
| `port` | `Optional[int]` | `6379` | Redis server port. |
| `password` | `Optional[str]` | `None` | Password for Redis authentication. |
| `ssl` | `bool` | `False` | Whether to use SSL/TLS for Redis connections. |
| `db` | `int` | `0` | Redis database index. |
| `ttl_seconds` | `int` | `604800` (7 days) | TTL for session keys in seconds. Set to `0` or negative to disable expiration. |
| `key_prefix` | `str` | `"adk:session:"` | Prefix for all Redis keys created by the session service. |

### Using Connection Parameters

```python
from google.adk.integrations.redis import RedisSessionService
from google.adk.integrations.redis import RedisSessionServiceConfig

config = RedisSessionServiceConfig(
    host="redis.example.com",
    port=6379,
    password="my_secret_password",
    ssl=True,
    db=0,
    ttl_seconds=86400 * 3,  # 3 days
    key_prefix="myapp:sessions:",
)
session_service = RedisSessionService(config=config)
```

### Using a Pre-configured Redis Client

You can also pass an existing `redis.asyncio.Redis` instance directly:

```python
from google.adk.integrations.redis import RedisSessionService
import redis.asyncio as redis_asyncio

client = redis_asyncio.Redis.from_url(
    "redis://localhost:6379/0", decode_responses=True
)
session_service = RedisSessionService(redis_client=client)
```

## Key Schema & State Scoping

`RedisSessionService` stores session data and state using the following Redis key structure:

| Key Pattern | Data | Description |
| :--- | :--- | :--- |
| `{key_prefix}{app_name}:{user_id}:{session_id}` | JSON (`Session`) | Contains session ID, state, events list, and last update timestamp. Expired based on `ttl_seconds`. |
| `{key_prefix}user_state:{app_name}:{user_id}` | JSON (`dict`) | Persisted user-scoped state (keys prefixed with `user:`). Shared across sessions for the same user. Expired based on `ttl_seconds`. |
| `{key_prefix}app_state:{app_name}` | JSON (`dict`) | Persisted app-scoped state (keys prefixed with `app:`). Shared across all sessions and users for the app. Expired based on `ttl_seconds`. |

When events append state deltas:
- `user:<key>` updates are synchronized to the user state key.
- `app:<key>` updates are synchronized to the app state key.
- Upon session creation or event updates, state from all scopes is merged into the session state.

## Direct Service Usage

`RedisSessionService` implements `BaseSessionService` and can be used directly for session management:

```python
from google.adk.integrations.redis import RedisSessionService
from google.adk.sessions.base_session_service import GetSessionConfig

session_service = RedisSessionService()

# Create a new session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    state={"user:theme": "dark", "topic": "weather"},
)

# Retrieve a session
retrieved = await session_service.get_session(
    app_name="my_app",
    user_id="user_123",
    session_id=session.id,
    config=GetSessionConfig(num_recent_events=10),
)

# List all sessions for a user
response = await session_service.list_sessions(
    app_name="my_app",
    user_id="user_123",
)

# Delete a session
await session_service.delete_session(
    app_name="my_app",
    user_id="user_123",
    session_id=session.id,
)
```
