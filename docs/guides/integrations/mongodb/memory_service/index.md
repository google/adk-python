# MongoDbMemoryService

`MongoDbMemoryService` stores completed session events as keyword-indexed documents in a MongoDB collection and retrieves matching turns for later sessions. It gives an agent long-term recall backed by any MongoDB deployment without requiring an embedding model or a vector search index.

## Introduction

When a session ends, the events from that conversation remain in the session service, where a new session does not automatically see them. `MongoDbMemoryService` implements `BaseMemoryService` to bridge sessions across time: `add_session_to_memory` and `add_events_to_memory` ingest text turns into a MongoDB collection, and `search_memory` returns the turns whose extracted keywords overlap with a query.

`Runner` wires the memory service into the agent runtime, so the `load_memory` and `preload_memory` tools and the memory helpers on `Context` all read and write through the same MongoDB collection. Because retrieval relies on keyword matching over an array field rather than vector embeddings, the service runs against any MongoDB deployment reachable by PyMongo, including standalone servers, replica sets, and MongoDB Atlas, with no per-query embedding cost.

## Get started

Install the `mongodb` extra, which pulls in `pymongo`:

```shell
pip install 'google-adk[mongodb]'
```

The following example attaches `MongoDbMemoryService` and the `load_memory` tool to an agent through `Runner`, ingests a finished session, and starts a new session that can recall what was said earlier:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory

agent = LlmAgent(
    name="assistant_agent",
    instruction=(
        "Answer the user's questions. Call load_memory when the answer might"
        " come from an earlier conversation."
    ),
    tools=[load_memory],
)

session_service = InMemorySessionService()
memory_service = MongoDbMemoryService(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="my_app",
)

runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service,
    memory_service=memory_service,
)
```

After a turn completes, call `await callback_context.add_events_to_memory(events=callback_context.session.events[-1:])` (or pass new events directly to `await memory_service.add_events_to_memory(...)`) to ingest only the latest delta, or pass a finished `Session` to `await memory_service.add_session_to_memory(session)`.

## How it works

### Ingestion

`add_events_to_memory` (and `add_session_to_memory`, which delegates to it) inspects each event, concatenates its non-thought text parts, normalizes the text to Unicode NFC, and extracts lowercase word tokens (`\w+`) that are not in `stop_words`. A token that mixes Latin and non-Latin scripts also yields each of its single-script runs, so `使用MongoDB存储` is indexed under `mongodb` as well, matching `InMemoryMemoryService` and `SqliteMemoryService`. Events with no text parts, or whose text contains only stop words, are skipped.

Each indexed event is stored as one document in `memories_collection` with a compound `_id` of `{"app_name": <app_name>, "user_id": <user_id>, "session_id": <session_id>, "event_id": <event_id>}` (using `"__unknown_session_id__"` when `session_id` is omitted). Events from a single call are written in one unordered `bulk_write` of `ReplaceOne(..., upsert=True)` operations keyed by that `_id`, so re-adding events or sessions overwrites earlier documents in place rather than creating duplicates.

### Retrieval

`search_memory` extracts keywords from the `query` string using the same tokenization and stop-word filter. When the query yields no keywords, the service returns an empty `SearchMemoryResponse` without querying MongoDB.

Otherwise, the service runs an aggregation pipeline on `memories_collection` that matches documents for `app_name` and `user_id` whose `keywords` array contains at least one of the query keywords, scores each document by the number of query keywords it matches (`$size` of `$setIntersection`), sorts by score descending and `timestamp` descending as a tie-breaker, and caps the result set at `max_results` on the server. Each returned document is validated back into a `MemoryEntry` with its original `Content`, `author`, `id`, and ISO-formatted `timestamp`. If multiple stored events carry the same author, text, and timestamp, only the first is kept. Documents whose stored content cannot be parsed log a warning and are skipped so a malformed record does not fail the search.

## Configuration options

`MongoDbMemoryService` accepts its connection, collection name, stop-word set, result limit, and index behavior as keyword arguments on construction:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `database_name` | `str` | Required | Name of the MongoDB database that holds the memories collection. |
| `connection_string` | `str \| None` | `None` | MongoDB URI used to create a `pymongo.MongoClient` owned by the service. |
| `mongo_client` | `MongoClient \| None` | `None` | Pre-configured `pymongo.MongoClient` supplied and owned by the caller. |
| `memories_collection` | `str` | `"memories"` | Name of the collection where memory documents are stored. |
| `stop_words` | `set[str] \| None` | `None` | Custom set of words ignored during keyword extraction, replacing the default English stop-word list. |
| `max_results` | `int` | `50` | Maximum number of memories returned per `search_memory` call. |
| `ensure_indexes` | `bool` | `True` | Creates the `app_user_keywords` compound multikey index on first use; `False` skips the attempt. |

`database_name` selects the database within the MongoDB deployment. All applications and users sharing a `MongoDbMemoryService` instance write to the same database and are isolated from one another by the `app_name` and `user_id` fields on each document.

Exactly one of `connection_string` and `mongo_client` must be provided; passing both or neither raises `ValueError`. When `connection_string` is used, the service creates a `MongoClient` on construction (which PyMongo connects lazily on first use) and closes it when `await memory_service.close()` is called. When `mongo_client` is used, the caller retains ownership of the client and `close()` leaves it open.

`memories_collection` defaults to `"memories"`. Override it when several environments or services share one database and need separate collection names.

`stop_words` defaults to a compact set of common English function words such as `"the"`, `"and"`, and `"is"`. Without stop-word filtering, nearly every query would share a common word with nearly every stored turn. Pass a custom set when your conversations are in another language or when domain-specific filler words should be ignored; words in the custom set are normalized to lowercase.

`max_results` caps the number of memories returned by `search_memory` after ranking by keyword overlap, and must be a positive integer.

`ensure_indexes` controls whether the service creates the recommended `(app_name, user_id, keywords)` compound multikey index, named `app_user_keywords`, on the first write or keyword search. It defaults to `True` so reads do not degrade into collection scans as the `memories` collection grows. If the database user lacks the `createIndex` privilege, the service logs one warning and falls back to unindexed reads; any other index-creation error propagates and is retried on the next operation. Pass `ensure_indexes=False` to skip the attempt on a deliberately least-privileged user.

## Advanced applications

### Sharing a MongoClient across services and tools

When an application uses both `MongoDbToolset` and `MongoDbMemoryService`, or manages its own connection pool and TLS settings, pass a shared `MongoClient` via `mongo_client` so the process maintains a single connection pool:

```python
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.integrations.mongodb import MongoDbToolset
from pymongo import MongoClient

client = MongoClient("mongodb+srv://user:password@cluster.mongodb.net/")

toolset = MongoDbToolset(database_name="catalog", mongo_client=client)
memory_service = MongoDbMemoryService(
    database_name="my_app",
    mongo_client=client,
)
```

Because `client` was passed in, calling `await memory_service.close()` or `await toolset.close()` will not close `client`; close `client` during application shutdown.

### Customizing stop words

Pass `stop_words` to replace the built-in English list with a custom set of words to ignore—for example, stop words for another language or domain-specific filler terms that appear in almost every turn:

```python
from google.adk.integrations.mongodb import MongoDbMemoryService

memory_service = MongoDbMemoryService(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="support_app",
    stop_words={"a", "an", "the", "ticket", "account", "support"},
)
```

## Limitations

- The service is experimental and its API may change between releases. The `MONGODB_MEMORY_SERVICE` feature is enabled by default, so constructing the service emits a warning when enabled and raises `RuntimeError` when disabled.
- Retrieval is keyword-based (Unicode-aware `\w+` tokens after NFC normalization and lowercasing) and does not perform semantic or embedding similarity search. A query matches a stored event only when they share at least one non-stop-word token; scripts written without spaces between words (such as Chinese or Japanese) are split only at Latin/non-Latin boundaries, not segmented into individual words.
- `add_session_to_memory`, `add_events_to_memory`, and `search_memory` are implemented; `custom_metadata` on `add_events_to_memory` is ignored, and calling `add_memory` raises `NotImplementedError`.
- Non-text parts, such as images or function calls, are stored but not indexed for keyword search. Model thought parts (`thought=True`) are dropped before storage, so `search_memory` never returns them.
