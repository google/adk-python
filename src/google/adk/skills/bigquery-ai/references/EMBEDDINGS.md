# Embeddings in BigQuery

Complete guide to generating and using vector embeddings with ML.GENERATE_EMBEDDING.

## Overview

Embeddings are dense vector representations that capture semantic meaning. Use them for:
- Semantic search (find similar content)
- Clustering (group related items)
- Classification (as features for ML models)
- Recommendations (similarity-based suggestions)
- RAG (retrieve relevant context for LLMs)

## ML.GENERATE_EMBEDDING Syntax

```sql
SELECT *
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  { TABLE source_table | (SELECT query) },
  STRUCT(
    flatten_json_output AS flatten_json_output,
    task_type AS task_type,
    output_dimensionality AS output_dimensionality
  )
);
```

## Supported Embedding Models

| Model | Dimensions | Languages | Modalities | Use Case |
|-------|------------|-----------|------------|----------|
| `text-embedding-005` | 768 | 100+ | Text | General purpose (recommended) |
| `text-embedding-004` | 768 | 100+ | Text | Previous generation |
| `text-multilingual-embedding-002` | 768 | 100+ | Text | Multilingual focus |
| `textembedding-gecko@003` | 768 | English | Text | Legacy |
| `multimodalembedding@001` | 1408 | - | Text, Image, Video | Multimodal search |

## Creating an Embedding Model

```sql
-- Standard text embedding model
CREATE OR REPLACE MODEL `project.dataset.text_embeddings`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'text-embedding-005');

-- Multimodal embedding model
CREATE OR REPLACE MODEL `project.dataset.multimodal_embeddings`
  REMOTE WITH CONNECTION DEFAULT
  OPTIONS (ENDPOINT = 'multimodalembedding@001');

-- With specific connection
CREATE OR REPLACE MODEL `project.dataset.embeddings`
  REMOTE WITH CONNECTION `project.us.my_connection`
  OPTIONS (ENDPOINT = 'text-embedding-005');
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flatten_json_output` | BOOL | TRUE | Return embedding as ARRAY<FLOAT64> instead of JSON |
| `task_type` | STRING | - | Optimize for specific task (see below) |
| `output_dimensionality` | INT64 | Model default | Reduce embedding dimensions |

### Task Types

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `RETRIEVAL_QUERY` | Query for semantic search | Search queries |
| `RETRIEVAL_DOCUMENT` | Document to be searched | Indexing documents |
| `SEMANTIC_SIMILARITY` | General similarity | Clustering, deduplication |
| `CLASSIFICATION` | Text classification | ML features |
| `CLUSTERING` | Grouping similar items | Topic modeling |

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `ml_generate_embedding_result` | ARRAY<FLOAT64> | Vector embedding |
| `ml_generate_embedding_status` | STRING | Error (empty if success) |
| `ml_generate_embedding_statistics` | JSON | Token counts, truncation info |
| Original columns | Various | All input columns preserved |

## Examples

### Basic Text Embeddings

```sql
SELECT
  id,
  title,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, title, body AS content FROM articles)
)
WHERE ml_generate_embedding_status = '';
```

### Store Embeddings in Table

```sql
CREATE OR REPLACE TABLE `project.dataset.article_embeddings` AS
SELECT
  id,
  title,
  ml_generate_embedding_result AS embedding,
  CURRENT_TIMESTAMP() AS embedded_at
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, title, body AS content FROM articles)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;
```

### Query vs Document Embeddings

For best search results, use different task types for queries and documents:

```sql
-- Index documents with RETRIEVAL_DOCUMENT
INSERT INTO `project.dataset.doc_embeddings`
SELECT id, content, ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, content FROM new_documents),
  STRUCT('RETRIEVAL_DOCUMENT' AS task_type)
);

-- Embed queries with RETRIEVAL_QUERY
SELECT ml_generate_embedding_result AS query_embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT 'What is machine learning?' AS content),
  STRUCT('RETRIEVAL_QUERY' AS task_type)
);
```

### Multimodal Embeddings (Images)

```sql
-- Embed images from Cloud Storage
SELECT
  image_uri,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.multimodal_model`,
  (SELECT image_uri AS uri FROM image_catalog)
);

-- Embed text and images together
SELECT *
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.multimodal_model`,
  (SELECT
    product_name AS content,
    image_gcs_uri AS uri
   FROM products)
);
```

### Reduced Dimensionality

```sql
-- Generate smaller embeddings for efficiency
SELECT
  id,
  ml_generate_embedding_result AS embedding_256d
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, content FROM documents),
  STRUCT(256 AS output_dimensionality)
);
```

## Creating Vector Indexes

For tables with many embeddings, create indexes for faster search:

```sql
-- IVF index (recommended for most cases)
CREATE OR REPLACE VECTOR INDEX article_embedding_idx
ON `project.dataset.article_embeddings`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 1000}'
);

-- Tree-AH index (for very large datasets)
CREATE OR REPLACE VECTOR INDEX large_embedding_idx
ON `project.dataset.large_embeddings`(embedding)
OPTIONS (
  index_type = 'TREE_AH',
  distance_type = 'DOT_PRODUCT'
);
```

### Index Parameters

| Parameter | Options | Recommendation |
|-----------|---------|----------------|
| `index_type` | IVF, TREE_AH | IVF for <100M rows, TREE_AH for larger |
| `distance_type` | COSINE, EUCLIDEAN, DOT_PRODUCT | COSINE for normalized embeddings |
| `num_lists` | 100-10000 | sqrt(num_rows) as starting point |

## Batch Processing

### Incremental Embedding Generation

```sql
-- Only embed new documents
INSERT INTO `project.dataset.embeddings`
SELECT id, content, ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, content
   FROM documents d
   WHERE NOT EXISTS (
     SELECT 1 FROM `project.dataset.embeddings` e WHERE e.id = d.id
   ))
);
```

### Chunked Processing

```sql
-- Process in batches for large tables
DECLARE batch_size INT64 DEFAULT 10000;
DECLARE offset_val INT64 DEFAULT 0;

LOOP
  INSERT INTO `project.dataset.embeddings`
  SELECT id, ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT id, content FROM documents LIMIT batch_size OFFSET offset_val)
  )
  WHERE ml_generate_embedding_status = '';

  SET offset_val = offset_val + batch_size;
  IF offset_val >= (SELECT COUNT(*) FROM documents) THEN
    LEAVE;
  END IF;
END LOOP;
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `content too long` | Text exceeds model limit | Truncate or chunk text |
| `RESOURCE_EXHAUSTED` | Rate limit | Reduce batch size |
| `INVALID_ARGUMENT` | Missing content column | Ensure `content` column exists |

### Filter Errors

```sql
-- Get only successful embeddings
SELECT * FROM ML.GENERATE_EMBEDDING(...)
WHERE LENGTH(ml_generate_embedding_status) = 0;

-- Log errors separately
SELECT id, ml_generate_embedding_status AS error
FROM ML.GENERATE_EMBEDDING(...)
WHERE ml_generate_embedding_status != '';
```

### Handle Long Text

```sql
-- Truncate to first 5000 characters
SELECT *
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT id, LEFT(content, 5000) AS content FROM documents)
);

-- Or chunk into multiple embeddings
SELECT
  id,
  chunk_id,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT
    id,
    chunk_id,
    chunk_text AS content
   FROM document_chunks)
);
```

## Best Practices

### 1. Choose the Right Model
- Use `text-embedding-005` for general text
- Use `multimodalembedding@001` for images
- Match model to your language needs

### 2. Use Task Types
- `RETRIEVAL_DOCUMENT` for indexing
- `RETRIEVAL_QUERY` for search queries
- Improves search relevance significantly

### 3. Normalize Content
- Clean text before embedding
- Remove excessive whitespace
- Consider lowercasing for consistency

### 4. Manage Embedding Tables
- Add metadata columns (created_at, source)
- Create primary keys for updates
- Partition by date for large tables

### 5. Monitor Quality
- Sample and inspect embeddings
- Test search relevance
- Compare models if needed

## Cost Optimization

- Embeddings charged per 1000 characters
- Cache embeddings - don't regenerate
- Use reduced dimensionality when possible
- Filter before embedding (not after)
