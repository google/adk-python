# Vector Embedding Search in BigQuery

## Overview

BigQuery supports vector search through embeddings, enabling semantic similarity search at scale.

## Setting Up Vector Search

### Step 1: Create Embeddings Table

```sql
CREATE TABLE `project.dataset.document_embeddings` (
  doc_id STRING,
  content STRING,
  embedding ARRAY<FLOAT64>,
  created_at TIMESTAMP
)
```

### Step 2: Generate and Store Embeddings

```sql
INSERT INTO `project.dataset.document_embeddings`
SELECT
  doc_id,
  content,
  ml_generate_embedding_result['embeddings'][0]['values'] AS embedding,
  CURRENT_TIMESTAMP() AS created_at
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT doc_id, content FROM `project.dataset.documents`),
  STRUCT(TRUE AS flatten_json_output)
)
```

### Step 3: Create Vector Index (Optional but Recommended)

```sql
CREATE VECTOR INDEX embedding_index
ON `project.dataset.document_embeddings`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 100}'
)
```

## Search Patterns

### Basic Similarity Search

```sql
-- Find top 10 most similar documents to a query
WITH query_embedding AS (
  SELECT ml_generate_embedding_result['embeddings'][0]['values'] AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT 'What is machine learning?' AS content),
    STRUCT(TRUE AS flatten_json_output)
  )
)
SELECT
  d.doc_id,
  d.content,
  ML.DISTANCE(d.embedding, q.embedding, 'COSINE') AS distance
FROM `project.dataset.document_embeddings` d
CROSS JOIN query_embedding q
ORDER BY distance ASC
LIMIT 10
```

### Using VECTOR_SEARCH Function

```sql
-- More efficient for large datasets with vector index
SELECT
  query.query_text,
  base.doc_id,
  base.content,
  distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.document_embeddings`,
  'embedding',
  (
    SELECT
      query_text,
      ml_generate_embedding_result['embeddings'][0]['values'] AS embedding
    FROM ML.GENERATE_EMBEDDING(
      MODEL `project.dataset.embedding_model`,
      (SELECT 'search query' AS content),
      STRUCT(TRUE AS flatten_json_output)
    ),
    UNNEST(['search query']) AS query_text
  ),
  'embedding',
  top_k => 10,
  distance_type => 'COSINE'
)
```

### Hybrid Search (Keywords + Semantic)

```sql
WITH semantic_results AS (
  SELECT doc_id, 0.7 * (1 - distance) AS semantic_score
  FROM VECTOR_SEARCH(...)
),
keyword_results AS (
  SELECT doc_id, 0.3 * SCORE() AS keyword_score
  FROM `project.dataset.documents`
  WHERE SEARCH(content, 'machine learning')
)
SELECT
  COALESCE(s.doc_id, k.doc_id) AS doc_id,
  COALESCE(s.semantic_score, 0) + COALESCE(k.keyword_score, 0) AS combined_score
FROM semantic_results s
FULL OUTER JOIN keyword_results k ON s.doc_id = k.doc_id
ORDER BY combined_score DESC
LIMIT 10
```

## RAG (Retrieval-Augmented Generation) Pattern

```sql
-- Step 1: Retrieve relevant context
WITH relevant_docs AS (
  SELECT content
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.knowledge_base`,
    'embedding',
    (
      SELECT ml_generate_embedding_result['embeddings'][0]['values'] AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `project.dataset.embedding_model`,
        (SELECT @user_question AS content),
        STRUCT(TRUE AS flatten_json_output)
      )
    ),
    'embedding',
    top_k => 5
  )
),
-- Step 2: Build prompt with context
context_prompt AS (
  SELECT CONCAT(
    'Answer the question based on the following context:\n\n',
    STRING_AGG(content, '\n\n'),
    '\n\nQuestion: ', @user_question,
    '\n\nAnswer:'
  ) AS prompt
  FROM relevant_docs
)
-- Step 3: Generate response
SELECT ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS answer
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_model`,
  (SELECT prompt FROM context_prompt),
  STRUCT(0.2 AS temperature, 1024 AS max_output_tokens)
)
```

## Performance Tips

1. **Vector Index**: Always create an index for tables > 10K rows
2. **Dimension Reduction**: Use `output_dimensionality` to reduce embedding size
3. **Batch Processing**: Generate embeddings in batches, not row-by-row
4. **Caching**: Store embeddings instead of regenerating
5. **Partitioning**: Partition large embedding tables by date or category

## Distance Types

| Type | Description | Use Case |
|------|-------------|----------|
| COSINE | Angle between vectors (0-2) | Text similarity |
| EUCLIDEAN | Straight-line distance | When magnitude matters |
| DOT_PRODUCT | Dot product (unnormalized) | Pre-normalized embeddings |

## Common Issues

### Empty Embeddings
Check for NULL content or API errors in `ml_generate_embedding_status`.

### Slow Queries
Create a vector index and use VECTOR_SEARCH instead of CROSS JOIN.

### Inconsistent Results
Ensure embeddings are generated with the same model and settings.
