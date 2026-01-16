---
name: bigquery-ai
description: Execute generative AI operations in BigQuery - text generation, embeddings, vector search, and RAG workflows using Gemini, Claude, and other LLMs. Use when working with AI/ML inference, semantic search, or building RAG applications in BigQuery.
license: Apache-2.0
compatibility: BigQuery, Vertex AI, Gemini, Cloud AI APIs
metadata:
  author: Google Cloud
  version: "2.0"
  category: generative-ai
adk:
  config:
    timeout_seconds: 180
    max_parallel_calls: 10
    allow_network: true
  allowed_callers:
    - bigquery_agent
    - ai_agent
    - rag_agent
---

# BigQuery AI Skill

Execute generative AI operations directly in BigQuery using SQL. This skill covers text generation, embeddings, vector search, and retrieval-augmented generation (RAG) workflows.

## When to Use This Skill

Use this skill when you need to:
- Generate text using LLMs (Gemini, Claude, Llama, Mistral) on BigQuery data
- Create embeddings for semantic search and similarity matching
- Build vector search and RAG pipelines entirely in SQL
- Process documents, translate text, or analyze images at scale
- Connect BigQuery to Vertex AI models for inference

## Core Capabilities

| Capability | Function | Description |
|------------|----------|-------------|
| Text Generation | `AI.GENERATE_TEXT` | Generate text using remote LLM models |
| Embeddings | `ML.GENERATE_EMBEDDING` | Create vector embeddings from text/images |
| Vector Search | `VECTOR_SEARCH` | Find semantically similar items |
| Semantic Search | `AI.SEARCH` | Search with autonomous embeddings |
| Remote Models | `CREATE MODEL` | Connect to Vertex AI endpoints |

## Quick Start

### 1. Create a Remote Model Connection

```sql
-- Create connection to Gemini
CREATE OR REPLACE MODEL `project.dataset.gemini_model`
  REMOTE WITH CONNECTION `project.region.connection_id`
  OPTIONS (ENDPOINT = 'gemini-2.0-flash');
```

### 2. Generate Text

```sql
SELECT ml_generate_text_result
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_model`,
  (SELECT 'Summarize this text: ' || content AS prompt FROM my_table),
  STRUCT(256 AS max_output_tokens, 0.2 AS temperature)
);
```

### 3. Create Embeddings

```sql
-- Create embedding model
CREATE OR REPLACE MODEL `project.dataset.embedding_model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');

-- Generate embeddings
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT content FROM my_table)
);
```

### 4. Vector Search

```sql
SELECT base.id, base.content, distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.embeddings`, 'embedding',
  (SELECT embedding FROM query_embeddings),
  top_k => 10,
  distance_type => 'COSINE'
);
```

## AI Functions Reference

### AI.GENERATE_TEXT

Full control over text generation with model parameters:

```sql
SELECT * FROM AI.GENERATE_TEXT(
  MODEL `project.dataset.model`,
  (SELECT prompt FROM prompts_table),
  STRUCT(
    512 AS max_output_tokens,
    0.7 AS temperature,
    0.95 AS top_p,
    TRUE AS ground_with_google_search
  )
);
```

**Key Parameters:**
- `max_output_tokens`: 1-8192 (default: 128)
- `temperature`: 0.0-1.0 (default: 0, higher = more creative)
- `top_p`: 0.0-1.0 (default: 0.95)
- `ground_with_google_search`: Enable web grounding

### ML.GENERATE_EMBEDDING

Generate vector embeddings for semantic operations:

```sql
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, text_column AS content FROM source_table)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;  -- Filter errors
```

**Supported Models:**
- `text-embedding-005` (recommended)
- `text-embedding-004`
- `text-multilingual-embedding-002`
- `multimodalembedding@001` (text + images)

### VECTOR_SEARCH

Find nearest neighbors using embeddings:

```sql
SELECT query.id, base.id, base.content, distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.base_embeddings`, 'embedding',
  TABLE `project.dataset.query_embeddings`,
  top_k => 5,
  distance_type => 'COSINE',
  options => '{"fraction_lists_to_search": 0.01}'
);
```

**Distance Types:** `COSINE`, `EUCLIDEAN`, `DOT_PRODUCT`

## Supported Models

| Provider | Models | Use Case |
|----------|--------|----------|
| Google | Gemini 2.0, 1.5 Pro/Flash | Text generation, multimodal |
| Anthropic | Claude 3.5, 3 Opus/Sonnet | Complex reasoning |
| Meta | Llama 3.1, 3.2 | Open-source alternative |
| Mistral | Mistral Large, Medium | European compliance |

## Prerequisites

1. **BigQuery Connection**: Create a Cloud resource connection
2. **IAM Permissions**: Grant `bigquery.connectionUser` and `aiplatform.user`
3. **APIs Enabled**: BigQuery API, Vertex AI API, BigQuery Connection API

## References

Load detailed documentation as needed:

- `TEXT_GENERATION.md` - Complete AI.GENERATE_TEXT guide with all parameters
- `EMBEDDINGS.md` - Embedding models, multimodal embeddings, best practices
- `VECTOR_SEARCH.md` - Vector indexes, search optimization, recall tuning
- `REMOTE_MODELS.md` - CREATE MODEL syntax for all supported providers
- `RAG_WORKFLOW.md` - End-to-end RAG implementation patterns
- `CLOUD_AI_SERVICES.md` - Translation, NLP, document processing, vision

## Scripts

Helper scripts for common operations:

- `setup_remote_model.py` - Create remote model connections
- `generate_embeddings.py` - Batch embedding generation
- `semantic_search.py` - Build semantic search pipelines
- `rag_pipeline.py` - Complete RAG workflow setup

## Common Patterns

### Batch Text Classification

```sql
SELECT id, content,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS category
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini`,
  (SELECT id, content,
    CONCAT('Classify this text into one of: Tech, Sports, Politics\n\nText: ', content) AS prompt
   FROM articles)
);
```

### Semantic Similarity Search

```sql
-- Find similar documents to a query
WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT 'machine learning best practices' AS content)
  )
)
SELECT d.title, d.content, distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.doc_embeddings`, 'embedding',
  TABLE query_embedding,
  top_k => 10
)
JOIN `project.dataset.documents` d ON d.id = base.id;
```

### RAG with Context Injection

```sql
-- Retrieve relevant context and generate answer
WITH context AS (
  SELECT STRING_AGG(content, '\n\n') AS retrieved_context
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.knowledge_base`, 'embedding',
    (SELECT embedding FROM ML.GENERATE_EMBEDDING(MODEL m, (SELECT @query AS content))),
    top_k => 5
  )
)
SELECT ml_generate_text_result AS answer
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini`,
  (SELECT CONCAT(
    'Answer based on context:\n\n', retrieved_context,
    '\n\nQuestion: ', @query
  ) AS prompt FROM context)
);
```

## Error Handling

Check status columns for errors:

```sql
-- Text generation errors
SELECT * FROM ML.GENERATE_TEXT(...)
WHERE ml_generate_text_status != '';

-- Embedding errors
SELECT * FROM ML.GENERATE_EMBEDDING(...)
WHERE LENGTH(ml_generate_embedding_status) > 0;
```

## Performance Tips

1. **Use Vector Indexes**: Create indexes for tables with >100K embeddings
2. **Batch Requests**: Process multiple rows in single function calls
3. **Filter Before AI**: Apply WHERE clauses before expensive AI operations
4. **Cache Embeddings**: Store embeddings in tables, don't regenerate
5. **Tune Search**: Adjust `fraction_lists_to_search` for speed vs recall

## Limitations

- Max 10,000 rows per AI function call
- Embedding dimensions vary by model (768-3072)
- Rate limits apply based on Vertex AI quotas
- Some models require specific regions
