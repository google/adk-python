# MongoDB Integration for ADK

This integration connects the Google Agent Development Kit (ADK) to MongoDB.
It provides search tools agents can call, plus MongoDB-backed session and
memory services. It works with MongoDB Atlas and self-managed MongoDB 8.0+
deployments.

> **Experimental:** All classes in this integration are experimental and their
> APIs may change in future releases.

## Features

- **Vector Search Tool (`mongodb_vector_search`):** Runs Atlas Vector Search
  (`$vectorSearch`) queries against a collection, with optional pre-filtering,
  configurable limits, and result projection.
- **Hybrid Search Tool (`mongodb_hybrid_search`):** Combines full-text search
  and vector search with reciprocal rank fusion (`$rankFusion`), with tunable
  vector/text weights.
- **Session Persistence (`MongoDbSessionService`):** Stores sessions, events,
  and `app:` / `user:` / session-scoped state in MongoDB, with optimistic
  concurrency control for concurrent writers.
- **Long-term Memory (`MongoDbMemoryService`):** Ingests session events into a
  memory collection and recalls them by keyword matching.
- **Flexible Connection Options:** Pass a connection string and let the
  integration create/own the PyMongo client, or bring your own pre-configured
  `pymongo.MongoClient`.

## Installation / Dependencies

Install ADK with the `mongodb` extra, which pulls in `pymongo`:

```bash
pip install google-adk[mongodb]
```

or if ADK is already installed:

```bash
pip install "pymongo>=4.9,<5"
```

## Requirements

| Component | Requirement |
| :--- | :--- |
| `mongodb_vector_search` | A **vector search index** on the collection (MongoDB Atlas or MongoDB 8.0+). |
| `mongodb_hybrid_search` | A **vector search index** and a **full-text search index** on the collection, and a deployment that supports `$rankFusion` (MongoDB Atlas or MongoDB 8.0+). |
| `MongoDbSessionService` | Any MongoDB deployment reachable by PyMongo (no search indexes needed). |
| `MongoDbMemoryService` | Any MongoDB deployment reachable by PyMongo (no search indexes needed). |

## Quick Start: Search Tools

`MongoDbToolset` exposes `mongodb_vector_search` and `mongodb_hybrid_search`
to the agent. The client, database name, and settings are bound on the
toolset and hidden from the model.

```python
from google.adk.agents import Agent
from google.adk.integrations.mongodb import MongoDbToolset

toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="products_db",
)

agent = Agent(
    model="gemini-2.5-flash",
    name="product_search_agent",
    instruction=(
        "Search the products collection with your MongoDB tools. "
        "Embed the user's query yourself before calling the search tools."
    ),
    tools=[toolset],
)
```

The embedding vector for `query_embedding` must be produced by your own
embedding model — typically exposed to the agent as an additional tool — and
must match the dimensions and content of the vectors stored in the
collection.

To bring your own client instead of a connection string:

```python
from pymongo import MongoClient
from google.adk.integrations.mongodb import MongoDbToolset

client = MongoClient("mongodb+srv://user:pass@cluster.mongodb.net/")
toolset = MongoDbToolset(database_name="products_db", mongo_client=client)
```

## Quick Start: Session Service

```python
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.runners import Runner

session_service = MongoDbSessionService(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="my_app",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)
```

### Document Layout

`MongoDbSessionService` stores data in four collections (names configurable):

| Collection | Document key | Contents |
| :--- | :--- | :--- |
| `sessions` | `<app_name>/<user_id>/<session_id>` | Session-scoped state (JSON-encoded), timestamps, and an optimistic-concurrency `revision`. |
| `events` | `<app_name>/<user_id>/<session_id>/<event_id>` | Full serialized event under `event_data`. |
| `app_states` | `<app_name>` | App-scoped state (`app:` prefixed keys). |
| `user_states` | `<app_name>/<user_id>` | User-scoped state (`user:` prefixed keys). |

State buckets are stored JSON-encoded so state keys containing characters
MongoDB forbids in document fields (e.g. `.`, `$`) round-trip safely. Event
appends use per-session locking plus a revision check, and raise
`StaleSessionError` if the session was modified in storage since it was
loaded.

## Quick Start: Memory Service

```python
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.runners import Runner

memory_service = MongoDbMemoryService(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="my_app",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    memory_service=memory_service,
)
```

Events ingested via `add_session_to_memory` are stored as memory documents
(one per event) keyed by `<app_name>/<user_id>/<session_id>/<event_id>`, each
carrying the lowercase keywords extracted from its text content (a compact
English stop-word list is ignored; pass `stop_words` to customize).
`search_memory` returns entries whose keyword array intersects the query's
keywords. Ingestion is idempotent — re-adding a session overwrites its
memories rather than duplicating them.

## Configuration: `MongoDbToolSettings`

`MongoDbToolSettings` customizes the defaults used by the search tools:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `default_vector_index_name` | `str` | `"vector_index"` | Name of the vector search index to query. |
| `default_search_index_name` | `str` | `"default"` | Name of the full-text search index used by hybrid search. |
| `default_embedding_field` | `str` | `"embedding"` | Document field that stores embedding vectors. |
| `default_limit` | `int` | `4` | Default number of documents returned by a search operation. |
| `max_results` | `int` | `50` | Maximum number of documents a search operation may return. |
| `default_num_candidates` | `int` | `100` | Default number of nearest neighbors considered by vector search. |

```python
from google.adk.integrations.mongodb import MongoDbToolset
from google.adk.integrations.mongodb import MongoDbToolSettings

toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="products_db",
    settings=MongoDbToolSettings(default_limit=8, max_results=20),
)
```

Per-call `limit`, `num_candidates`, `index_name`, `embedding_field`, and
`output_fields` arguments passed by the model override these defaults
(`limit` is still capped by `max_results`).
