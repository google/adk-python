# PostgreSQL / pgvector Memory Integration for ADK

This integration provides a self-hosted, semantic memory service for the Google
Agent Development Kit (ADK), backed by PostgreSQL with the
[pgvector](https://github.com/pgvector/pgvector) extension.

It complements the built-in memory services:

- `InMemoryMemoryService` is keyword-only and loses data on restart (prototyping
  only).
- `VertexAiMemoryBankService` and `VertexAiRagMemoryService` provide semantic
  memory but require Google Cloud.

`PgVectorMemoryService` gives you semantic (vector) memory that persists across
restarts and runs entirely on a database you control, which suits on-premise,
air-gapped, and data-residency-constrained deployments.

## Features

- **Semantic search:** Ranks memories by cosine similarity over embeddings using
  a pgvector HNSW index, instead of keyword overlap.
- **Self-hosted:** Vectors live in your own PostgreSQL database. No managed Cloud
  service is required for storage.
- **Idempotent ingestion:** Re-ingesting a session updates existing rows in
  place rather than creating duplicates.
- **Scoped memory:** Memories are isolated per `(app_name, user_id)`.
- **Pluggable embeddings:** Embeds through the `google-genai` client by default,
  or any provider via an injected `embedder` callable.

## Installation / Dependencies

Install the optional `pgvector` extra alongside ADK:

```bash
pip install "google-adk[pgvector]"
```

You also need a PostgreSQL database with the `vector` extension available. The
service creates the extension, table, and indexes on first use.

## Quick Start

```python
from google.adk.integrations.pgvector import PgVectorMemoryService
from google.adk.integrations.pgvector import PgVectorMemoryServiceConfig
from google.adk.runners import Runner

# 1. Configure the memory service.
memory_service = PgVectorMemoryService(
    PgVectorMemoryServiceConfig(
        dsn="postgresql://user:password@localhost:5432/adk",
        embedding_model="gemini-embedding-001",
        embedding_dimension=768,
    )
)

# 2. Wire it into your Runner.
runner = Runner(
    app_name="my_app",
    agent=agent,
    memory_service=memory_service,
)

# Ingesting and searching:
await memory_service.add_session_to_memory(session)
response = await memory_service.search_memory(
    app_name="my_app", user_id="user1", query="what did we decide about billing?"
)
```

## Configuration

`PgVectorMemoryServiceConfig` supports the following options:

| Option | Default | Description |
| --- | --- | --- |
| `dsn` | `None` | PostgreSQL connection string. Required unless a `connection_pool` is injected. |
| `table_name` | `adk_memory_entries` | Table that stores memory entries. |
| `embedding_model` | `gemini-embedding-001` | `google-genai` embedding model. |
| `embedding_dimension` | `768` | Stored vector dimension; must match the model output and stays fixed for the table. |
| `top_k` | `10` | Maximum memories returned per search. |
| `hnsw_m` | `16` | pgvector HNSW `m` parameter. |
| `hnsw_ef_construction` | `64` | pgvector HNSW `ef_construction` parameter. |
| `distance_threshold` | `None` | Optional cosine-distance ceiling; farther memories are dropped. |

## Advanced usage

Inject a pre-configured pool or a non-Google embedding provider:

```python
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool("postgresql://user:password@localhost:5432/adk")

async def my_embedder(texts):
  # Return one vector per text from any provider.
  ...

memory_service = PgVectorMemoryService(
    PgVectorMemoryServiceConfig(embedding_dimension=1024),
    connection_pool=pool,
    embedder=my_embedder,
)
```
