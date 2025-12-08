---
name: bq_remote_model
description: BigQuery Remote Models - Create remote models connecting to Vertex AI endpoints for text generation (Gemini, Claude, Llama), embeddings, and custom deployed models. Use AI.GENERATE_TEXT and AI.GENERATE_EMBEDDING functions.
keywords:
  - remote model
  - create remote model
  - vertex ai
  - generate text
  - ai.generate_text
  - generate embedding
  - ai.generate_embedding
  - gemini
  - claude
  - llama
  - text generation
  - embeddings
  - llm
  - foundation model
  - hugging face
  - model garden
  - endpoint
  - list connections
  - connection_id
---

# BQ Remote Model Skill (Remote Models with Vertex AI)

Create and use remote models that connect BigQuery to Vertex AI endpoints for text generation, embeddings, and custom ML models.

**IMPORTANT**: Remote models require a BigQuery connection to Vertex AI. This is different from BQML (which trains models in BigQuery) and BQ AI Operator (which uses managed AI functions like AI.CLASSIFY).

## Prerequisites

1. **A BigQuery connection to Vertex AI is required** for all remote models.

2. **Grant the connection's service account the Vertex AI User role**

## Connection Workflow (ALWAYS Follow This)

**CRITICAL**: Remote models require a `connection_id` to a BigQuery connection to Vertex AI.

### ⚠️ IMPORTANT: Location Matching Rule

**The connection location MUST match your dataset location!**

| Dataset Location | Connection Location | Example |
|------------------|---------------------|---------|
| `US` (multi-region) | `us` | `us.my_vertex_connection` |
| `EU` (multi-region) | `eu` | `eu.my_vertex_connection` |
| `us-central1` (regional) | `us-central1` | `us-central1.my_vertex_connection` |

**Common Error**: Using `us-central1.my_connection` with a dataset in `US` multi-region will fail with "Dataset not found in location us-central1".

**How to check dataset location**:
```sql
SELECT option_value FROM `project.dataset.INFORMATION_SCHEMA.SCHEMATA_OPTIONS` WHERE option_name = 'location'
```

### Step 1: Determine Your Dataset Location

Before listing connections, identify where your target dataset is located:
- Most BigQuery public datasets are in `US` multi-region
- Your own datasets might be in `US`, `EU`, or a specific region like `us-central1`

### Step 2: List Connections in the SAME Location

Use the `list_connections` tool with the **same location as your dataset**:

```
# For datasets in US multi-region:
list_connections(project_id="your-project", location="us")

# For datasets in us-central1:
list_connections(project_id="your-project", location="us-central1")
```

This returns all available connections with their `connection_id` and `service_account`.

### Step 3: Use an Existing Connection If Available

If `list_connections` returns connections, **use one of them**. Pick a connection that:
- Has `connection_type: "CLOUD_RESOURCE"` (required for Vertex AI)
- Is in the **SAME location as your dataset**

Use the `connection_id` from the result, formatted as `location.connection_id`:
- Example: If connection_id is `my_ai_connection` in location `us`, use `us.my_ai_connection`

### Step 4: Only Create a New Connection If None Exist

**Only if `list_connections` returns empty or no suitable connections**, create a new one in the **same location as your dataset**:

```
# For US multi-region datasets:
create_connection(project_id="your-project", location="us", connection_id="my_vertex_connection")

# For us-central1 datasets:
create_connection(project_id="your-project", location="us-central1", connection_id="my_vertex_connection")
```

This automatically:
1. Creates the connection
2. Grants the Vertex AI User role to the service account (required for remote models)

### Connection ID Formats

When using connections in SQL:
- `us.my_connection` (location.connection_name) - **Preferred**
- `project_id.us.my_connection` (fully qualified)

---

## CREATE REMOTE MODEL Syntax

### For Google/Partner Models (Gemini, Claude)

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
REMOTE WITH CONNECTION `project.region.connection_id`
OPTIONS (ENDPOINT = 'endpoint_name');
```

### ⚠️ DEFAULT MODEL: Always Use Gemini 2.5 Pro

**ALWAYS use `gemini-2.5-pro` as the default model** unless the user specifically requests a different model.

```sql
-- RECOMMENDED: Use gemini-2.5-pro by default
CREATE OR REPLACE MODEL `project.dataset.gemini_model`
REMOTE WITH CONNECTION `us.my_connection`
OPTIONS (ENDPOINT = 'gemini-2.5-pro');
```

**Common ENDPOINT values:**
| Model | Endpoint | When to Use |
|-------|----------|-------------|
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | **DEFAULT** - Best quality, use for all tasks unless specified otherwise |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Only if user requests faster/cheaper processing |
| Claude 3.5 Sonnet | `claude-3-5-sonnet@20240620` | Only if user specifically requests Claude |
| Text Embedding | `text-embedding-004` | For embeddings/vector search |
| Gemini Embedding | `gemini-embedding-001` | For embeddings (larger dimension) |

**Legacy models (avoid):** `gemini-2.0-flash`, `gemini-1.5-pro` - Use 2.5 versions instead.

### For Open Models (Hugging Face / Model Garden)

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
REMOTE WITH CONNECTION `project.region.connection_id`
OPTIONS (
  HUGGING_FACE_MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf',
  HUGGING_FACE_TOKEN = 'your_token',  -- Optional, for gated models
  MACHINE_TYPE = 'n1-standard-4',
  MIN_REPLICA_COUNT = 1,
  MAX_REPLICA_COUNT = 3,
  ENDPOINT_IDLE_TTL = INTERVAL 1 HOUR
);
```

---

## AI.GENERATE_TEXT - Text Generation

Generate text using LLMs like Gemini, Claude, or Llama.

### Basic Syntax

```sql
SELECT *
FROM AI.GENERATE_TEXT(
  MODEL `project.dataset.model_name`,
  (SELECT 'Your prompt here' AS prompt),
  STRUCT(
    1024 AS max_output_tokens,
    0.7 AS temperature,
    0.95 AS top_p
  )
);
```

### ⚠️ CRITICAL: Task-Specific Parameter Settings

**The `max_output_tokens` parameter is crucial** - set it appropriately for the task type:

| Task Type | max_output_tokens | temperature | Example Use Case |
|-----------|-------------------|-------------|------------------|
| **Summarization** | `512-2048` | `0.2-0.4` | Summarize articles, extract key points |
| **Long-form generation** | `2048-8192` | `0.5-0.7` | Write essays, detailed explanations |
| **Classification/Labeling** | `50-100` | `0.0-0.2` | Classify text, extract labels |
| **Short answers** | `100-256` | `0.2-0.3` | Q&A, simple extractions |
| **Creative writing** | `1024-4096` | `0.7-0.9` | Stories, creative content |

**Guidelines:**
- **Summarization tasks**: Use `max_output_tokens` of `512-1024` for single-document summaries, `1024-2048` for multi-document summaries
- **Classification tasks**: Use small `max_output_tokens` (50-100) since output is typically a single word or short phrase
- **Low temperature (0.0-0.3)**: For factual, deterministic outputs (classification, extraction)
- **High temperature (0.5-0.8)**: For creative, varied outputs (writing, brainstorming)

### Parameters for Gemini Models

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_output_tokens` | INT64 | 1-8192 | 128 | **IMPORTANT**: Set based on task (see table above) |
| `temperature` | FLOAT64 | 0.0-1.0 | 0.0 | Randomness (0=deterministic, 1=creative) |
| `top_p` | FLOAT64 | 0.0-1.0 | 0.95 | Nucleus sampling threshold |
| `stop_sequences` | ARRAY<STRING> | - | [] | Stop generation at these sequences |
| `ground_with_google_search` | BOOL | - | FALSE | Enable Google Search grounding |
| `request_type` | STRING | DEDICATED/SHARED | UNSPECIFIED | Resource allocation |

### Parameters for Claude Models

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| `max_output_tokens` | INT64 | 1-4096 | 128 |
| `top_k` | INT64 | 1-40 | - |
| `top_p` | FLOAT64 | 0.0-1.0 | - |

### Example: Text Summarization (Large max_output_tokens)

```sql
-- Step 1: Create the remote model (ALWAYS use gemini-2.5-pro by default)
CREATE OR REPLACE MODEL `project.bq_demo.gemini_model`
REMOTE WITH CONNECTION `us.my_vertex_connection`
OPTIONS (ENDPOINT = 'gemini-2.5-pro');  -- DEFAULT: Always use 2.5-pro unless specified

-- Step 2: Summarize text (use 512-1024 tokens for summaries)
SELECT
    title,
    ml_generate_text_result AS summary
FROM AI.GENERATE_TEXT(
    MODEL `project.bq_demo.gemini_model`,
    (SELECT
        title,
        CONCAT('Summarize this article in 2-3 paragraphs:\n\n', body) AS prompt
     FROM `bigquery-public-data.bbc_news.fulltext`
     LIMIT 5),
    STRUCT(
        1024 AS max_output_tokens,  -- LARGE for summarization
        0.3 AS temperature          -- Low for factual output
    )
);
```

### Example: Text Classification (Small max_output_tokens)

```sql
-- Classification task: Use small max_output_tokens since output is just a label
SELECT
    review_id,
    review_text,
    ml_generate_text_result AS sentiment
FROM AI.GENERATE_TEXT(
    MODEL `project.bq_demo.gemini_model`,
    (SELECT
        review_id,
        review_text,
        CONCAT('Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL. Only output the label, nothing else.\n\nReview: ', review_text) AS prompt
     FROM `project.dataset.reviews`
     LIMIT 10),
    STRUCT(
        50 AS max_output_tokens,   -- SMALL for classification (just a single word)
        0.0 AS temperature         -- Zero for deterministic output
    )
);
```

### Example: Batch Summarization from Table

```sql
SELECT
    review_id,
    review_text,
    ml_generate_text_result AS summary
FROM AI.GENERATE_TEXT(
    MODEL `project.bq_demo.gemini_model`,
    (SELECT
        review_id,
        review_text,
        CONCAT('Summarize this review in one sentence: ', review_text) AS prompt
     FROM `project.dataset.reviews`
     LIMIT 10),
    STRUCT(
        256 AS max_output_tokens,  -- Medium for short summaries
        0.2 AS temperature
    )
);
```

---

## AI.GENERATE_EMBEDDING - Text Embeddings

Generate vector embeddings for text, useful for semantic search and similarity.

### Syntax

```sql
SELECT *
FROM AI.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT content FROM table),
  STRUCT(
    'RETRIEVAL_DOCUMENT' AS task_type,
    768 AS output_dimensionality
  )
);
```

### Task Types

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `RETRIEVAL_QUERY` | Optimize for queries | Search queries |
| `RETRIEVAL_DOCUMENT` | Optimize for documents | Document indexing |
| `SEMANTIC_SIMILARITY` | Compute similarity | Finding similar texts |
| `CLASSIFICATION` | Text classification | Categorization |
| `CLUSTERING` | Group similar texts | Topic modeling |
| `QUESTION_ANSWERING` | Q&A tasks | FAQ systems |
| `FACT_VERIFICATION` | Verify facts | Fact checking |
| `CODE_RETRIEVAL_QUERY` | Code search | Code similarity |

### Dimensionality

| Model | Dimension Range | Default |
|-------|-----------------|---------|
| `gemini-embedding-001` | 1-3072 | 3072 |
| `text-embedding-004` | 1-768 | 768 |

### Example: Create Embeddings for Semantic Search

```sql
-- Step 1: Create embedding model
CREATE OR REPLACE MODEL `project.bq_demo.embedding_model`
REMOTE WITH CONNECTION `us.my_vertex_connection`
OPTIONS (ENDPOINT = 'text-embedding-004');

-- Step 2: Generate embeddings for documents
SELECT
    doc_id,
    title,
    ml_generate_embedding_result AS embedding
FROM AI.GENERATE_EMBEDDING(
    MODEL `project.bq_demo.embedding_model`,
    (SELECT doc_id, title, content FROM `project.dataset.documents` LIMIT 100),
    STRUCT('RETRIEVAL_DOCUMENT' AS task_type, 768 AS output_dimensionality)
);
```

### Example: Vector Similarity Search

```sql
-- Find similar documents using cosine distance
WITH query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM AI.GENERATE_EMBEDDING(
        MODEL `project.bq_demo.embedding_model`,
        (SELECT 'machine learning best practices' AS content),
        STRUCT('RETRIEVAL_QUERY' AS task_type)
    )
)
SELECT
    d.doc_id,
    d.title,
    ML.DISTANCE(d.embedding, q.embedding, 'COSINE') AS similarity
FROM `project.dataset.doc_embeddings` d
CROSS JOIN query_embedding q
ORDER BY similarity ASC
LIMIT 10;
```

---

## Complete Pipeline Example

Build a RAG (Retrieval Augmented Generation) pipeline:

```sql
-- Step 1: Find relevant documents using embeddings
WITH relevant_docs AS (
    SELECT title, content
    FROM AI.GENERATE_EMBEDDING(
        MODEL `project.bq_demo.embedding_model`,
        (SELECT 'What are the benefits of serverless?' AS content),
        STRUCT('RETRIEVAL_QUERY' AS task_type)
    ) query
    CROSS JOIN (
        SELECT doc_id, title, content, embedding
        FROM `project.dataset.doc_embeddings`
    ) docs
    ORDER BY ML.DISTANCE(docs.embedding, query.ml_generate_embedding_result, 'COSINE')
    LIMIT 3
)
-- Step 2: Generate response using retrieved context
SELECT ml_generate_text_result AS answer
FROM AI.GENERATE_TEXT(
    MODEL `project.bq_demo.gemini_model`,
    (SELECT CONCAT(
        'Based on these documents:\n',
        STRING_AGG(content, '\n\n'),
        '\n\nAnswer: What are the benefits of serverless?'
    ) AS prompt FROM relevant_docs),
    STRUCT(512 AS max_output_tokens, 0.3 AS temperature)
);
```

---

## Important Notes

1. **Connection Required**: All remote models need a Vertex AI connection
2. **Region Matching**: Dataset and connection must be in the same region
3. **Cost**: Remote model calls incur Vertex AI API costs
4. **Rate Limits**: Be mindful of Vertex AI quotas when processing large datasets
5. **Use LIMIT**: Always use LIMIT when testing to control costs
6. **Escape Single Quotes**: When using string literals with apostrophes, escape them by doubling:
   - WRONG: `'The surgeon who 'sees' inside patients'`
   - CORRECT: `'The surgeon who ''sees'' inside patients'`

## Troubleshooting

**Error: "connection not found"**
- Verify connection exists: `SELECT * FROM region-us.INFORMATION_SCHEMA.CONNECTIONS`
- Use correct format: `project.region.connection_id`

**Error: "model not found"**
- Check endpoint spelling matches exactly
- Verify model is available in your region

**Error: "permission denied"**
- Grant Vertex AI User role to connection's service account
