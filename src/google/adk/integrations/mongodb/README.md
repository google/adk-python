# MongoDB Integration for ADK

This integration connects the Google Agent Development Kit (ADK) to MongoDB.
It provides search tools agents can call against collections on MongoDB Atlas
or self-managed MongoDB 8.2+ deployments.

## Features

- **Vector Search Tool (`mongodb_vector_search`):** Runs Atlas Vector Search
  (`$vectorSearch`) queries against a collection, with optional pre-filtering,
  configurable limits, and result projection.
- **Hybrid Search Tool (`mongodb_hybrid_search`):** Combines full-text search
  and vector search with reciprocal rank fusion (`$rankFusion`), with tunable
  vector/text weights.
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

## Required permissions (least-privilege setup)

Do not hand the integration an admin user. The search tools only read, so the
database user needs nothing more than `find` on each collection the agent may
search:

```javascript
// mongosh, as an admin user.
use admin;
db.createRole({
  role: "adk_search_reader",
  privileges: [
    { resource: { db: "products_db", collection: "products" }, actions: ["find"] },
  ],
  roles: [],
});
```

Creating and dropping **search indexes** is an administrative action, not
something the tools do at run time — run setup with a separate user holding
the `createSearchIndexes`, `listSearchIndexes`, and `dropSearchIndexes`
actions (on Atlas, the built-in `Project Data Access Admin` role covers
this), and keep those privileges off the runtime user above.

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

## Deploying to Agent Engine

Agent Engine packages the app object with cloudpickle, and a live
`pymongo.MongoClient` (sockets, locks, background threads) cannot cross that
boundary. `MongoDbToolset` is therefore picklable: the client is dropped at
pickle time and rebuilt on the runtime from the connection string. A toolset
built from a caller-owned `mongo_client` or `genai_client` cannot be pickled;
pass a connection string and leave `genai_client` unset so the query-embedding
client is built from the ambient environment (`GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION`, set automatically on Agent Engine; add
`GOOGLE_GENAI_USE_VERTEXAI=TRUE` to the deployment env vars).

### Keep the connection string out of the deployment package

A toolset constructed with `connection_string="mongodb+srv://user:PASSWORD@…"`
carries the URI — password included — as a plaintext attribute, and pickling
writes it into the uploaded deployment package. To keep the credential out
of the package, name an environment variable instead:

```python
toolset = MongoDbToolset(
    database_name="products_db",
    connection_string_env_var="MONGODB_URI",  # only the name is stored
)
```

Only the variable's name is stored and pickled; the value is resolved where
the client is built — locally at construction, or on the Agent Engine
runtime after unpickling. Map the variable to a Secret Manager secret at
deploy time so the plaintext value never touches the deployment spec either
(the Agent Engine service account needs the Secret Manager Secret Accessor
role on the secret):

```python
from google.cloud.aiplatform_v1.types import SecretRef
from vertexai import agent_engines

remote = agent_engines.create(
    app,
    requirements=[...],
    env_vars={
        "MONGODB_URI": SecretRef(
            secret="mongodb-uri",
            version="latest",
        ),
    },
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
