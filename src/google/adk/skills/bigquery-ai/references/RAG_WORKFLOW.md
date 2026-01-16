# RAG Workflows in BigQuery

Complete guide to building Retrieval-Augmented Generation (RAG) pipelines in BigQuery SQL.

## Overview

RAG combines semantic search with text generation to create responses grounded in your data:

1. **Embed** - Convert documents and queries to vectors
2. **Index** - Create vector indexes for fast retrieval
3. **Retrieve** - Find relevant documents via semantic search
4. **Generate** - Use retrieved context to generate accurate responses

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BigQuery RAG Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Documents   │───▶│  Embeddings  │───▶│  Vector Index        │   │
│  │  (text)      │    │  (vectors)   │    │  (fast search)       │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                   │                  │
│  ┌──────────────┐    ┌──────────────┐            │                  │
│  │  User Query  │───▶│  Query       │────────────┘                  │
│  │              │    │  Embedding   │                               │
│  └──────────────┘    └──────────────┘                               │
│                            │                                         │
│                            ▼                                         │
│                      ┌──────────────┐                               │
│                      │ VECTOR_SEARCH│                               │
│                      │ (retrieve    │                               │
│                      │  context)    │                               │
│                      └──────────────┘                               │
│                            │                                         │
│                            ▼                                         │
│                      ┌──────────────┐    ┌──────────────────────┐   │
│                      │ Context +    │───▶│  AI.GENERATE_TEXT    │   │
│                      │ Query        │    │  (generate answer)   │   │
│                      └──────────────┘    └──────────────────────┘   │
│                                                   │                  │
│                                                   ▼                  │
│                                          ┌──────────────────────┐   │
│                                          │  Grounded Response   │   │
│                                          └──────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step 1: Prepare Knowledge Base

### Create Document Table

```sql
CREATE OR REPLACE TABLE `project.dataset.knowledge_base` (
  doc_id STRING,
  title STRING,
  content STRING,
  source STRING,
  created_at TIMESTAMP,
  metadata JSON
);

-- Load documents
INSERT INTO `project.dataset.knowledge_base`
SELECT
  GENERATE_UUID() AS doc_id,
  title,
  body AS content,
  'internal_docs' AS source,
  CURRENT_TIMESTAMP() AS created_at,
  TO_JSON(STRUCT(author, category)) AS metadata
FROM `project.dataset.source_documents`;
```

### Chunk Long Documents

```sql
-- Split documents into chunks for better retrieval
CREATE OR REPLACE TABLE `project.dataset.document_chunks` AS
WITH chunks AS (
  SELECT
    doc_id,
    title,
    chunk_index,
    TRIM(chunk) AS chunk_text,
    LENGTH(chunk) AS chunk_length
  FROM `project.dataset.knowledge_base`,
    UNNEST(REGEXP_EXTRACT_ALL(content, r'.{1,1000}(?:\s|$)')) AS chunk WITH OFFSET AS chunk_index
)
SELECT
  CONCAT(doc_id, '_', chunk_index) AS chunk_id,
  doc_id,
  title,
  chunk_index,
  chunk_text,
  chunk_length
FROM chunks
WHERE chunk_length > 50;  -- Filter tiny chunks
```

## Step 2: Generate Embeddings

### Create Embedding Model

```sql
CREATE OR REPLACE MODEL `project.dataset.embedding_model`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');
```

### Embed Documents

```sql
CREATE OR REPLACE TABLE `project.dataset.kb_embeddings` AS
SELECT
  chunk_id,
  doc_id,
  title,
  chunk_text,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT chunk_id, doc_id, title, chunk_text AS content
   FROM `project.dataset.document_chunks`),
  STRUCT('RETRIEVAL_DOCUMENT' AS task_type)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;
```

## Step 3: Create Vector Index

```sql
CREATE OR REPLACE VECTOR INDEX kb_embedding_idx
ON `project.dataset.kb_embeddings`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 500}'
);
```

## Step 4: Create Generation Model

```sql
CREATE OR REPLACE MODEL `project.dataset.gemini_rag`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'gemini-2.0-flash');
```

## Step 5: Build RAG Query

### Basic RAG Query

```sql
DECLARE user_query STRING DEFAULT 'What is the refund policy?';

WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT user_query AS content),
    STRUCT('RETRIEVAL_QUERY' AS task_type)
  )
),
retrieved_context AS (
  SELECT
    base.chunk_id,
    base.title,
    base.chunk_text,
    distance
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.kb_embeddings`,
    'embedding',
    TABLE query_embedding,
    top_k => 5,
    distance_type => 'COSINE'
  )
  ORDER BY distance
),
context_string AS (
  SELECT STRING_AGG(
    CONCAT('Source: ', title, '\n', chunk_text),
    '\n\n---\n\n'
  ) AS context
  FROM retrieved_context
)
SELECT
  user_query AS question,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS answer,
  (SELECT ARRAY_AGG(STRUCT(title, chunk_text, distance))
   FROM retrieved_context) AS sources
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_rag`,
  (SELECT CONCAT(
    'Answer the question based ONLY on the following context. ',
    'If the answer is not in the context, say "I don\'t have information about that."\n\n',
    'Context:\n', context, '\n\n',
    'Question: ', user_query, '\n\n',
    'Answer:'
  ) AS prompt FROM context_string),
  STRUCT(512 AS max_output_tokens, 0.1 AS temperature)
);
```

### RAG as Stored Procedure

```sql
CREATE OR REPLACE PROCEDURE `project.dataset.ask_knowledge_base`(
  IN user_query STRING,
  IN num_sources INT64,
  OUT answer STRING,
  OUT sources ARRAY<STRUCT<title STRING, excerpt STRING, relevance FLOAT64>>
)
BEGIN
  DECLARE query_emb ARRAY<FLOAT64>;

  -- Get query embedding
  SET query_emb = (
    SELECT ml_generate_embedding_result
    FROM ML.GENERATE_EMBEDDING(
      MODEL `project.dataset.embedding_model`,
      (SELECT user_query AS content),
      STRUCT('RETRIEVAL_QUERY' AS task_type)
    )
  );

  -- Retrieve and generate
  SET (answer, sources) = (
    WITH retrieved AS (
      SELECT base.title, base.chunk_text, distance
      FROM VECTOR_SEARCH(
        TABLE `project.dataset.kb_embeddings`,
        'embedding',
        (SELECT query_emb AS embedding),
        top_k => num_sources
      )
    ),
    context AS (
      SELECT STRING_AGG(chunk_text, '\n\n') AS ctx,
             ARRAY_AGG(STRUCT(title, LEFT(chunk_text, 200) AS excerpt,
                              1.0 - distance AS relevance)) AS src
      FROM retrieved
    )
    SELECT
      JSON_VALUE(r.ml_generate_text_result, '$.predictions[0].content'),
      c.src
    FROM context c,
    ML.GENERATE_TEXT(
      MODEL `project.dataset.gemini_rag`,
      (SELECT CONCAT('Context:\n', c.ctx, '\n\nQuestion: ', user_query) AS prompt),
      STRUCT(512 AS max_output_tokens)
    ) r
  );
END;

-- Usage
DECLARE answer STRING;
DECLARE sources ARRAY<STRUCT<title STRING, excerpt STRING, relevance FLOAT64>>;
CALL `project.dataset.ask_knowledge_base`('What is the return policy?', 5, answer, sources);
SELECT answer, sources;
```

## Advanced Patterns

### Hybrid Search (Vector + Keyword)

```sql
WITH semantic_results AS (
  SELECT base.chunk_id, base.chunk_text, distance,
         1.0 / (1.0 + distance) AS semantic_score
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.kb_embeddings`,
    'embedding',
    TABLE query_embedding,
    top_k => 20
  )
),
keyword_results AS (
  SELECT chunk_id, chunk_text, search_score
  FROM `project.dataset.document_chunks`
  WHERE SEARCH(chunk_text, @query)
  ORDER BY search_score DESC
  LIMIT 20
),
combined AS (
  SELECT
    COALESCE(s.chunk_id, k.chunk_id) AS chunk_id,
    COALESCE(s.chunk_text, k.chunk_text) AS chunk_text,
    COALESCE(s.semantic_score, 0) * 0.7 +
      COALESCE(k.search_score, 0) * 0.3 AS combined_score
  FROM semantic_results s
  FULL OUTER JOIN keyword_results k ON s.chunk_id = k.chunk_id
  ORDER BY combined_score DESC
  LIMIT 5
)
SELECT STRING_AGG(chunk_text, '\n\n') AS context
FROM combined;
```

### Multi-Query RAG

```sql
-- Generate multiple search queries for better coverage
WITH expanded_queries AS (
  SELECT query
  FROM UNNEST([
    @original_query,
    -- LLM-generated query variations
    (SELECT JSON_VALUE(ml_generate_text_result, '$.predictions[0].content')
     FROM ML.GENERATE_TEXT(
       MODEL `project.dataset.gemini_rag`,
       (SELECT CONCAT('Rephrase this query: ', @original_query) AS prompt)
     ))
  ]) AS query
),
all_embeddings AS (
  SELECT query, ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT query AS content FROM expanded_queries)
  )
),
all_results AS (
  SELECT DISTINCT base.chunk_id, base.chunk_text, MIN(distance) AS best_distance
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.kb_embeddings`,
    'embedding',
    TABLE all_embeddings,
    top_k => 5
  )
  GROUP BY chunk_id, chunk_text
)
SELECT * FROM all_results ORDER BY best_distance LIMIT 5;
```

### Conversational RAG

```sql
CREATE OR REPLACE TABLE `project.dataset.chat_history` (
  session_id STRING,
  turn_id INT64,
  role STRING,  -- 'user' or 'assistant'
  content STRING,
  timestamp TIMESTAMP
);

-- Include chat history in context
WITH recent_history AS (
  SELECT STRING_AGG(
    CONCAT(role, ': ', content),
    '\n'
    ORDER BY turn_id
  ) AS history
  FROM `project.dataset.chat_history`
  WHERE session_id = @session_id
    AND turn_id >= (SELECT MAX(turn_id) - 5 FROM `project.dataset.chat_history`
                    WHERE session_id = @session_id)
),
-- ... rest of RAG pipeline
SELECT
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS response
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_rag`,
  (SELECT CONCAT(
    'Chat history:\n', history, '\n\n',
    'Context:\n', context, '\n\n',
    'User: ', @user_query, '\n',
    'Assistant:'
  ) AS prompt
  FROM recent_history, context_table)
);
```

### RAG with Source Citations

```sql
SELECT
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS answer
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_rag`,
  (SELECT CONCAT(
    'Answer the question using the sources below. ',
    'Cite sources using [1], [2], etc.\n\n',
    (SELECT STRING_AGG(
      CONCAT('[', chunk_index + 1, '] ', chunk_text),
      '\n\n'
    ) FROM retrieved_context),
    '\n\nQuestion: ', @query,
    '\n\nAnswer with citations:'
  ) AS prompt),
  STRUCT(512 AS max_output_tokens, 0.2 AS temperature)
);
```

## Incremental Updates

### Add New Documents

```sql
-- 1. Insert new documents
INSERT INTO `project.dataset.knowledge_base` ...;

-- 2. Chunk new documents
INSERT INTO `project.dataset.document_chunks`
SELECT ... FROM new_documents;

-- 3. Generate embeddings for new chunks
INSERT INTO `project.dataset.kb_embeddings`
SELECT chunk_id, doc_id, title, chunk_text, ml_generate_embedding_result
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT * FROM `project.dataset.document_chunks`
   WHERE chunk_id NOT IN (SELECT chunk_id FROM `project.dataset.kb_embeddings`))
);

-- Vector index updates automatically
```

### Delete Documents

```sql
-- Delete embeddings
DELETE FROM `project.dataset.kb_embeddings`
WHERE doc_id = @doc_to_delete;

-- Delete chunks
DELETE FROM `project.dataset.document_chunks`
WHERE doc_id = @doc_to_delete;

-- Delete source document
DELETE FROM `project.dataset.knowledge_base`
WHERE doc_id = @doc_to_delete;
```

## Performance Optimization

### 1. Pre-compute Common Queries

```sql
CREATE OR REPLACE TABLE `project.dataset.query_cache` AS
SELECT
  query_text,
  answer,
  sources,
  CURRENT_TIMESTAMP() AS cached_at
FROM common_queries_with_answers;
```

### 2. Filter Before Search

```sql
-- Restrict search to relevant category
FROM VECTOR_SEARCH(
  (SELECT * FROM `project.dataset.kb_embeddings`
   WHERE category = @user_category),
  'embedding',
  ...
)
```

### 3. Tune Retrieval Parameters

- Start with `top_k = 5`
- Increase if answers miss context
- Decrease if context is too noisy

### 4. Optimize Chunk Size

- 500-1000 characters typically optimal
- Larger for dense technical content
- Smaller for diverse Q&A

## Monitoring

### Track RAG Quality

```sql
CREATE OR REPLACE TABLE `project.dataset.rag_feedback` (
  query_id STRING,
  query_text STRING,
  answer STRING,
  sources ARRAY<STRING>,
  user_rating INT64,  -- 1-5
  feedback_text STRING,
  timestamp TIMESTAMP
);

-- Analyze low-rated responses
SELECT
  query_text,
  answer,
  user_rating,
  feedback_text
FROM `project.dataset.rag_feedback`
WHERE user_rating <= 2
ORDER BY timestamp DESC;
```

### Monitor Retrieval Quality

```sql
-- Check average retrieval distance
SELECT
  DATE(timestamp) AS date,
  AVG(min_distance) AS avg_top_distance,
  COUNT(*) AS query_count
FROM (
  SELECT timestamp, MIN(distance) AS min_distance
  FROM query_logs
  GROUP BY query_id, timestamp
)
GROUP BY date
ORDER BY date DESC;
```
