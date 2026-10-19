# MongoDB Integration for ADK

This integration connects the Google Agent Development Kit (ADK) to MongoDB.
It provides search tools agents can call, plus MongoDB-backed session and
memory services. It works with MongoDB Atlas and self-managed MongoDB 8.2+
deployments.

## Features

- **Vector Search Tool (`mongodb_vector_search`):** Runs Atlas Vector Search
  (`$vectorSearch`) queries against a collection, with optional pre-filtering,
  configurable limits, and result projection.
- **Hybrid Search Tool (`mongodb_hybrid_search`):** Combines full-text search
  and vector search with reciprocal rank fusion (`$rankFusion`), with tunable
  vector/text weights.
- **Session Persistence (`MongoDbSessionService`):** Stores sessions, events,
  and `app:` / `user:` / session-scoped state in MongoDB. Multi-document
  writes run in a transaction (replica set, sharded cluster, or Atlas
  required) so concurrent writers cannot lose each other's state updates;
  on deployments without transaction support (standalone mongod) the
  service falls back to sequential writes with optimistic concurrency
  control on the session document.
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
| `mongodb_vector_search` | A **vector search index** on the collection, and a deployment that supports `$vectorSearch` (MongoDB Atlas 6.0.11+/7.0.2+, or self-managed MongoDB 8.2+). |
| `mongodb_hybrid_search` | A **vector search index** and a **full-text search index** on the collection, and a deployment that supports `$rankFusion` (MongoDB Atlas 8.0+). |
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
        "Search the products collection with your MongoDB tools."
    ),
    tools=[toolset],
)
```

Both tools take the user's question as plain text (`query`) and embed it on
the way through — see [Embedding modes](#embedding-modes) below — so the
model never produces or handles a vector itself.

To bring your own client instead of a connection string:

```python
from pymongo import MongoClient
from google.adk.integrations.mongodb import MongoDbToolset

client = MongoClient("mongodb+srv://user:pass@cluster.mongodb.net/")
toolset = MongoDbToolset(database_name="products_db", mongo_client=client)
```

## Embedding modes

Every search embeds its query text before running `$vectorSearch`. Two modes
are supported, selected with `MongoDbToolSettings.use_mongodb_auto_embedding`:

| Mode | Flag | How it works | Requirements |
| :--- | :--- | :--- | :--- |
| **Google embedding models** (default) | `use_mongodb_auto_embedding=False` | The toolset embeds the query through the genai client using `vertex_ai_embedding_model_name` (default `text-embedding-005`) and sends the vector as `queryVector`. | Google credentials for the embedding API; stored document embeddings produced by the same model. |
| **MongoDB Automated Embedding** (Preview) | `use_mongodb_auto_embedding=True` | The raw query text goes to `$vectorSearch` as `query.text`, and MongoDB generates the embedding with the Voyage AI model configured on the index. The genai client is never called. | MongoDB Atlas or self-managed MongoDB 8.3+ with `mongot` 1.70.1+ and Voyage AI API keys configured on `mongot`; the searched field indexed as the `autoEmbed` type. Optionally set `mongodb_auto_embedding_model` (e.g. `"voyage-4"`) to override the index's model per query. |

```python
from google.adk.integrations.mongodb import MongoDbToolset
from google.adk.integrations.mongodb import MongoDbToolSettings

# Default: embed queries with Google's text-embedding-005.
toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="products_db",
)

# Atlas Automated Embedding (Preview): Atlas embeds queries for you.
auto_toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="products_db",
    settings=MongoDbToolSettings(
        use_mongodb_auto_embedding=True,
        default_embedding_field="description",  # the autoEmbed text field
        mongodb_auto_embedding_model="voyage-4",  # optional
    ),
)
```

Automated Embedding is a
[Preview feature](https://www.mongodb.com/docs/preview-features/) available on
MongoDB Atlas and on self-managed deployments running MongoDB 8.3+ with
`mongot` 1.70.1+ and Voyage AI API keys
[configured on `mongot`](https://www.mongodb.com/docs/search/self-managed/current/configuration/automated-embedding/).

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
MongoDB forbids in document fields (e.g. `.`, `$`) round-trip safely.

### Concurrency model and deployment targets

The service targets two deployment shapes:

- **Replica set, sharded cluster, or Atlas:** every multi-document write
  (state buckets, the session revision bump, and the event insert) runs in a
  multi-document transaction, so it commits or aborts atomically.
- **Standalone mongod (or any deployment without transactions):** the first
  write attempts a transaction, the failure verdict is cached, and later
  writes run sequentially with an optimistic-concurrency revision check on
  the session document.

Within one process, a per-session `asyncio.Lock` serializes `append_event`
calls for the same session. That lock is intentionally **in-process only**:
when several runner replicas write to the same session, the revision check
on the session document is the cross-process guard — the losing writer gets
a `StaleSessionError`, reloads the session, and retries. Do not rely on the
in-process lock for correctness across replicas.

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

Retrieval is intentionally keyword-only: it needs no embedding model, no
Google credentials, and no vector index, so it works on any MongoDB
deployment (not just Atlas) at zero per-query cost. Vector recall is a
possible follow-up that can build on this integration's own search tools.

## Secondary indexes (`ensure_indexes`)

The services never create indexes on their own: every write and most reads
are keyed by `_id`, which MongoDB indexes implicitly. Two read patterns are
not `_id` lookups, though, and become collection scans as data grows:

- `get_session` reads `events` by `(app_name, user_id, session_id)` sorted
  by `timestamp`.
- `search_memory` reads `memories` by `(app_name, user_id)` with a `$in`
  match on the `keywords` array.

Pass `ensure_indexes=True` to have the services create the recommended
indexes on first use:

```python
session_service = MongoDbSessionService(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="my_app",
    ensure_indexes=True,
)
memory_service = MongoDbMemoryService(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="my_app",
    ensure_indexes=True,
)
```

This creates:

| Collection | Index | Keys |
| :--- | :--- | :--- |
| `events` | `app_user_session_ts` | `(app_name, user_id, session_id, timestamp DESC)` |
| `sessions` | `app_user` | `(app_name, user_id)` |
| `memories` | `app_user_keywords` | `(app_name, user_id, keywords)` (multikey) |

Index creation is idempotent — re-running with the same spec is a no-op — so
the flag is safe to leave on. It requires the MongoDB user to hold the
`createIndex` privilege on the database, and it runs outside the write
transactions (`createIndexes` is not allowed inside one). At small scale the
flag changes nothing measurably; enable it before the `events` and
`memories` collections grow large.

## Required permissions (least-privilege setup)

Do not hand the integration an admin user. The components need very little,
and two separate database users keep the privileges separated:

| Database user | Used by | Minimum privileges |
| :--- | :--- | :--- |
| **Search user** (read-only) | `MongoDbToolset` | `find` on each collection the agent may search. |
| **State user** (read-write) | `MongoDbSessionService`, `MongoDbMemoryService` | `find`, `insert`, `update`, `remove` on `sessions` and `events` (session deletion); `find`, `insert`, `update` on `app_states`, `user_states` and `memories` (upserts only, never deletes). Add `createIndex` on those collections only if you enable `ensure_indexes`. |

```javascript
// mongosh, as an admin user: one read-only role for the search tools, one
// read-write role scoped to the state collections for the service.
use admin;
db.createRole({
  role: "adk_search_reader",
  privileges: [
    { resource: { db: "products_db", collection: "products" }, actions: ["find"] },
  ],
  roles: [],
});
db.createRole({
  role: "adk_state_writer",
  privileges: ["sessions", "events"].map((c) => ({
    resource: { db: "my_app", collection: c },
    actions: ["find", "insert", "update", "remove"],
  })).concat(["app_states", "user_states", "memories"].map((c) => ({
    resource: { db: "my_app", collection: c },
    actions: ["find", "insert", "update"],
  }))),
  roles: [],
});
```

Creating and dropping **search indexes** is an administrative action, not
something the tools do at run time — run setup with a separate user holding
the `createSearchIndexes`, `listSearchIndexes`, and `dropSearchIndexes`
actions (on Atlas, the built-in `Project Data Access Admin` role covers
this), and keep those privileges off the runtime users above.

### The model chooses the collection

A `MongoDbToolset` binds one **database**; the model then picks
`collection_name` on every call. Every collection in that database that
carries a search index is reachable by the agent, so a database-wide `read`
role exposes all of them to prompt-driven queries. Scope accordingly:

- Prefer per-collection privileges (as above) over a database-wide `read`
  role, so even a misdirected query fails with an authorization error instead
  of returning data.
- Put only collections the agent may read into the bound database, or give
  the toolset its own database.
- For a hard allowlist enforced in code, add a `before_tool_callback`, which
  sees the call arguments the model chose:

```python
def allow_products_only(tool, args, tool_context):
  if args.get("collection_name") != "products":
    return {
        "status": "ERROR",
        "error_details": "Only the products collection may be searched.",
    }
  return None  # None lets the call proceed.

agent = Agent(
    ...,
    tools=[toolset],
    before_tool_callback=allow_products_only,
)
```

## Configuration: `MongoDbToolSettings`

`MongoDbToolSettings` customizes the defaults used by the search tools:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `default_vector_index_name` | `str` | `"vector_index"` | Name of the vector search index to query. |
| `default_search_index_name` | `str` | `"default"` | Name of the full-text search index used by hybrid search. |
| `default_embedding_field` | `str` | `"embedding"` | Document field searched by `$vectorSearch`: the field storing embedding vectors, or the `autoEmbed` text field when `use_mongodb_auto_embedding=True`. |
| `default_limit` | `int` | `4` | Default number of documents returned by a search operation. |
| `max_results` | `int` | `50` | Maximum number of documents a search operation may return. |
| `default_num_candidates` | `int` | `100` | Default number of nearest neighbors considered by vector search. |
| `timeout_ms` | `int` | `60000` | Time limit in milliseconds for a single search operation. |
| `vertex_ai_embedding_model_name` | `str` | `"text-embedding-005"` | Google embedding model used for query vectors (default mode). |
| `vertex_ai_embedding_output_dimensionality` | `int \| None` | `None` | Length of the query embedding, for models that support shortening it. |
| `use_mongodb_auto_embedding` | `bool` | `False` | Send query text to Atlas and let it embed (Automated Embedding, Preview) instead of calling a Google model. |
| `mongodb_auto_embedding_model` | `str \| None` | `None` | Voyage AI model for auto-embedded queries; defaults to the index's model. |

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
