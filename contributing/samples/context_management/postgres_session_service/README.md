# Using PostgreSQL with DatabaseSessionService

This sample demonstrates how to configure `DatabaseSessionService` to use PostgreSQL for persisting sessions, events, and state.

## Overview

ADK's `DatabaseSessionService` supports multiple database backends through SQLAlchemy. This guide shows how to:

- Set up PostgreSQL as the session storage backend
- Configure async connections with `asyncpg`
- Understand the auto-generated schema
- Run the sample agent with persistent sessions

## Prerequisites

- **PostgreSQL Database**: A running PostgreSQL instance (local or cloud)
- **asyncpg**: Async PostgreSQL driver for Python

## Installation

Install the required Python packages:

```bash
pip install google-adk asyncpg greenlet
```

## Database Schema

`DatabaseSessionService` automatically creates the following tables on first use:

### sessions

| Column      | Type         | Description                 |
| ----------- | ------------ | --------------------------- |
| app_name    | VARCHAR(128) | Application identifier (PK) |
| user_id     | VARCHAR(128) | User identifier (PK)        |
| id          | VARCHAR(128) | Session UUID (PK)           |
| state       | JSON         | Session state as JSON       |
| create_time | TIMESTAMP    | Creation timestamp          |
| update_time | TIMESTAMP    | Last update timestamp       |

### events

| Column        | Type         | Description                 |
| ------------- | ------------ | --------------------------- |
| id            | VARCHAR(128) | Event UUID (PK)             |
| app_name      | VARCHAR(128) | Application identifier (PK) |
| user_id       | VARCHAR(128) | User identifier (PK)        |
| session_id    | VARCHAR(128) | Session reference (PK, FK)  |
| invocation_id | VARCHAR(256) | Invocation identifier       |
| timestamp     | TIMESTAMP    | Event timestamp             |
| event_data    | JSON         | Event content as JSON       |

### app_states

| Column      | Type         | Description                 |
| ----------- | ------------ | --------------------------- |
| app_name    | VARCHAR(128) | Application identifier (PK) |
| state       | JSON         | Application-level state     |
| update_time | TIMESTAMP    | Last update timestamp       |

### user_states

| Column      | Type         | Description                 |
| ----------- | ------------ | --------------------------- |
| app_name    | VARCHAR(128) | Application identifier (PK) |
| user_id     | VARCHAR(128) | User identifier (PK)        |
| state       | JSON         | User-level state            |
| update_time | TIMESTAMP    | Last update timestamp       |

### adk_internal_metadata

| Column | Type         | Description    |
| ------ | ------------ | -------------- |
| key    | VARCHAR(128) | Metadata key   |
| value  | VARCHAR(256) | Metadata value |

## Configuration

### Connection URL Format

```text
postgresql+asyncpg://username:password@host:port/database
```

### Basic Usage

```python
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.runners import Runner

# Initialize with PostgreSQL URL
session_service = DatabaseSessionService(
    "postgresql+asyncpg://postgres:postgres@localhost:5432/adk_sessions"
)

# Use with Runner
runner = Runner(
    app_name="my_app",
    agent=my_agent,
    session_service=session_service,
)
```

### Advanced Configuration

Pass additional SQLAlchemy engine options:

```python
session_service = DatabaseSessionService(
    "postgresql+asyncpg://postgres:postgres@localhost:5432/adk_sessions",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
)
```

## Running the Sample

### 1. Start PostgreSQL

Using Docker:

```bash
docker compose up -d
```

Or use an existing PostgreSQL instance.

### 2. Configure Connection

Create a `.env` file:

```bash
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/adk_sessions
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_ENTERPRISE=true
```

Or run export command.

```bash
export POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/adk_sessions
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_ENTERPRISE=true
```

### 3. Run the Agent

```bash
python main.py
```

Or use the ADK:

```bash
adk run .
```

## Session Persistence

Sessions and events are persisted across application restarts:

```python
# First run - creates a new session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user1",
    session_id="persistent-session-123",
)

# Later run - retrieves the existing session
session = await session_service.get_session(
    app_name="my_app",
    user_id="user1",
    session_id="persistent-session-123",
)
```

## State Management

PostgreSQL's JSON type stores state data as written:

- **Session state**: Stored in `sessions.state`
- **User state**: Stored in `user_states.state`
- **App state**: Stored in `app_states.state`

Tables created by older ADK versions keep their `jsonb` columns, which still
reject NUL (`\u0000`) in strings. To switch an existing database to `json`:

```sql
ALTER TABLE sessions ALTER COLUMN state TYPE json USING state::json;
ALTER TABLE events ALTER COLUMN event_data TYPE json USING event_data::json;
ALTER TABLE app_states ALTER COLUMN state TYPE json USING state::json;
ALTER TABLE user_states ALTER COLUMN state TYPE json USING state::json;
```

Each statement rewrites the table under an `ACCESS EXCLUSIVE` lock, blocking
reads and writes until it finishes, so run it during a maintenance window.
Values already stored keep their `jsonb`-normalized form.

## Production Considerations

1. **Connection Pooling**: Use `pool_size` and `max_overflow` for high-traffic applications
1. **SSL/TLS**: Always use encrypted connections in production
1. **Backups**: Implement regular backup strategies for session data
1. **Indexing**: The default schema includes primary key indexes; add additional indexes based on query patterns
1. **Monitoring**: Monitor connection pool usage and query performance
