# MongoDbToolset

`MongoDbToolset` gives an agent two read-only search tools over one MongoDB database: `mongodb_vector_search`, which ranks documents by embedding similarity, and `mongodb_hybrid_search`, which fuses that ranking with a full-text one. Both tools take the user's question as text and embed it on the way through, so the model never produces or handles a vector.

## Introduction

Retrieval over a document collection is the usual reason to reach for MongoDB from an agent. The collection already holds the documents and, next to each one, an embedding of it; what the agent needs is a way to turn a question into the handful of documents that answer it.

`MongoDbToolset` is a `BaseToolset` that supplies that. You bind it to a database and a connection, and it returns two `FunctionTool` instances the model can call. Each call embeds the query text with a Vertex AI embedding model, runs one MongoDB aggregation, and returns the matching documents with a `search_score` field attached.

The two tools differ in how they rank:

- **`mongodb_vector_search`** uses the `$vectorSearch` stage alone. It finds documents whose stored embedding is closest to the query embedding, so it retrieves on meaning and tolerates wording the documents never use.
- **`mongodb_hybrid_search`** runs a `$vectorSearch` and a `$search` full-text query as two arms of a `$rankFusion` stage, and combines the two rankings with reciprocal rank fusion. It recovers the exact matches that similarity alone ranks poorly, such as a product code or a surname.

The configuration lives in `MongoDbToolSettings`, a Pydantic model covering the embedding model, the index and field names, and the limits every search is held to.

## Get started

Install the extra, which brings in `pymongo`:

```shell
pip install 'google-adk[mongodb]'
```

The collection needs an [Atlas Vector Search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) over the field holding the embeddings, and hybrid search additionally needs an Atlas Search index for the text field. Both tools read those indexes by name and never create them.

Bind the toolset to a database and hand it to an agent:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.mongodb import MongoDbToolset

toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="catalog",
)

agent = LlmAgent(
    name="catalog_agent",
    description="Answers questions about the product catalog.",
    instruction=(
        "Answer questions about products by searching the `products`"
        " collection. Cite the product name in your answer."
    ),
    tools=[toolset],
)
```

The instruction names the collection, because the model chooses it on each call. See [What the model chooses](#what-the-model-chooses) for what that means for the data the toolset can reach.

Credentials for the embedding model come from Application Default Credentials, and credentials for MongoDB come from the connection string.

## How it works

Both tools follow the same three steps: validate the arguments the model supplied, embed the query text, then run one aggregation and return its documents.

Neither tool raises. A failure at any step is caught and returned as `{"status": "ERROR", "error_details": "..."}`, so the model sees the failure as a tool result and can act on it rather than ending the invocation. A successful call returns `{"status": "SUCCESS", "rows": [...]}`. Values that JSON cannot carry, an `ObjectId` among them, are converted to their string form so the result survives the round trip to the model.

### The pipelines

`mongodb_vector_search` builds a three-stage pipeline: `$vectorSearch`, then an `$addFields` that copies the relevance score into `search_score`, then a `$project` that shapes the result. The score gets its own stage because a single `$project` cannot both exclude the embedding field and compute a new one.

`mongodb_hybrid_search` puts a `$vectorSearch` and a `$search` pipeline into `$rankFusion` as named arms, applies the combination weights, and then reuses the same two result stages. Each arm returns five times the final limit, bounded by `num_candidates`, rather than the final limit itself. Fusion can only reorder the documents the arms hand it, so an arm that stopped at the final limit would hide every document the other arm would have promoted.

### What the toolset binds, and what the model supplies

The toolset binds the MongoDB client, the database name, the settings, and the embedding client. Those four are hidden from the function declaration the model sees, so they cannot be steered by a prompt.

Everything else is a tool argument the model fills in: the collection, the index names, the embedding field, the fields to return, the result limit, the candidate count, the fusion weights, and a pre-filter.

### What the model chooses

The model chooses the collection on each call, so a toolset bound to `database_name` reaches every collection in that database that carries a search index. Point `database_name` at data the agent is allowed to read, and keep anything else in another database.

`tool_filter` does not narrow this, because it selects whole tools and never sees call arguments. A policy about which collections or fields are in bounds belongs in an agent's `before_tool_callback`, which does see them:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.mongodb import MongoDbToolset


def restrict_to_products(tool, args, tool_context):
  if tool.name.startswith("mongodb_") and args.get("collection_name") != (
      "products"
  ):
    return {
        "status": "ERROR",
        "error_details": "Only the products collection may be searched.",
    }
  return None


agent = LlmAgent(
    name="catalog_agent",
    description="Answers questions about the product catalog.",
    instruction="Answer questions about products.",
    tools=[
        MongoDbToolset(
            connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
            database_name="catalog",
        )
    ],
    before_tool_callback=restrict_to_products,
)
```

Returning a dict from the callback skips the tool and hands that dict back to the model as the result.

### Filters

The optional `filter` argument narrows the documents a search considers, and the model writes it. Only the operators that `$vectorSearch.filter` accepts are allowed: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$and`, `$or`, `$not` and `$nor`. A filter using anything else fails with an error result before any pipeline is built.

The allowlist exists because hybrid search sends the same filter to two places. It reaches `$vectorSearch.filter` on one arm and `$match` on the other, and `$match` accepts all of MQL. Without the allowlist the same filter would mean one thing to the vector arm and another to the text arm, and `$where` and `$expr` would let the model run server-side expressions that the vector arm cannot express at all.

Note that `$vectorSearch` only filters on fields declared as filter fields in the vector search index. A filter on any other field returns no rows rather than an error.

## Configuration options

`MongoDbToolset` takes its connection and its policy separately. `connection_string` or `mongo_client` says where MongoDB is, `database_name` says which database is in bounds, `tool_filter` selects which of the two tools to expose, `genai_client` supplies the client used for embedding, and `settings` carries everything below. Exactly one of `connection_string` and `mongo_client` is required, and passing both raises `ValueError`.

`MongoDbToolSettings` holds the rest:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vertex_ai_embedding_model_name` | `str` | `"text-embedding-005"` | The Vertex AI model used to embed the query text. |
| `vertex_ai_embedding_output_dimensionality` | `int \| None` | `None` | Length of the query embedding, for models that support shortening it. |
| `default_vector_index_name` | `str` | `"vector_index"` | Vector search index used when the model does not name one. |
| `default_search_index_name` | `str` | `"default"` | Full-text search index used by hybrid search when the model does not name one. |
| `default_embedding_field` | `str` | `"embedding"` | Document field holding the stored vectors. |
| `default_limit` | `int` | `4` | Documents returned when the model does not ask for a number. |
| `max_results` | `int` | `50` | Ceiling on the documents any one search returns. |
| `default_num_candidates` | `int` | `100` | Nearest neighbors the vector search considers. |
| `timeout_ms` | `int` | `60000` | Time limit for a single search operation. |

The embedding model and its dimensionality have to match the vectors already in the collection. Similarity between vectors from two different models is a number without meaning, and nothing in MongoDB rejects the comparison as long as the lengths happen to agree, so a mismatch shows up as results that are merely unhelpful. Set `vertex_ai_embedding_output_dimensionality` when the stored vectors were produced with a truncated embedding, and leave it unset to take the model's native length.

The three `default_*` names cover the common case where a database uses one naming convention throughout. The model can override each of them per call, which is what lets one toolset serve collections that index different fields.

`default_limit` and `max_results` bound how much the model can ask for. The model proposes a `limit`; the tool takes the smaller of that and `max_results`, and falls back to `default_limit` when the model asks for a number MongoDB would reject. `default_num_candidates` is the ANN over-request: the vector search examines that many neighbors and returns the best `limit` of them, so raising it improves recall and costs latency.

`timeout_ms` bounds a search two ways. It becomes `timeoutMS` on a client the toolset creates, and `maxTimeMS` on every aggregation, which is what bounds a client you passed in yourself. A bound matters here because the search runs on a worker thread that cannot be cancelled, and the model steers how much work the server is asked for.

## Advanced applications

### Reusing a client you already have

Pass `mongo_client` instead of `connection_string` when your application already holds a `pymongo.MongoClient`, or when you need connection options the toolset does not expose, such as a custom TLS configuration or a server API version:

```python
from google.adk.integrations.mongodb import MongoDbToolset
from pymongo import MongoClient

client = MongoClient("mongodb+srv://user:password@cluster.mongodb.net/", tls=True)

toolset = MongoDbToolset(database_name="catalog", mongo_client=client)
```

The toolset does not take ownership of a client you pass in, so `close()` leaves it open and closing it stays your job. `timeout_ms` still applies, as `maxTimeMS` on each aggregation.

### Exposing only one of the two tools

`tool_filter` takes the unprefixed tool names. Use it to drop hybrid search on a deployment whose MongoDB version does not support `$rankFusion`, or to keep the model's tool list short:

```python
toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="catalog",
    tool_filter=["vector_search"],
)
```

### Supplying the embedding client

Pass `genai_client` to control the credentials, project, or location used for embedding, rather than taking them from the ambient environment:

```python
from google.adk.integrations.mongodb import MongoDbToolset
from google.genai import Client

toolset = MongoDbToolset(
    connection_string="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="catalog",
    genai_client=Client(vertexai=True, project="my-project", location="us-central1"),
)
```

### Tuning the fusion

`mongodb_hybrid_search` accepts `vector_weight` and `text_weight`, both defaulting to `1.0`. Raise the text weight for a corpus where the exact term matters more than the paraphrase, such as part numbers, and raise the vector weight where the question and the document rarely share vocabulary. A negative weight is treated as zero, which removes that arm from the ranking.

## Limitations

- The toolset is experimental. Its API may change between releases, and using it emits a warning unless the feature is explicitly enabled.
- `$vectorSearch` requires MongoDB Atlas 6.0.11 or 7.0.2 and later, or self-managed MongoDB 8.2 and later. `$rankFusion`, and therefore hybrid search, requires MongoDB Atlas 8.0 and later.
- Both tools read. There is no tool for inserting, updating, or deleting documents, and none for creating the indexes the searches depend on.
- One toolset covers one database. Searching a second database means a second toolset.
- Results are not paginated. A search returns its first `limit` documents, and asking for the next page means asking for a larger limit.
- Every search embeds its query, which adds a Vertex AI round trip to each call and fails the call when embedding fails.
