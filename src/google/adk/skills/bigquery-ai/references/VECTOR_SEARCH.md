# Vector Search in BigQuery

Complete guide to semantic search using VECTOR_SEARCH function.

## Overview

VECTOR_SEARCH finds the most similar items to a query based on vector embeddings. It supports:
- Exact search (brute force)
- Approximate nearest neighbor (ANN) with vector indexes
- Multiple distance metrics
- Top-k retrieval

## VECTOR_SEARCH Syntax

```sql
SELECT *
FROM VECTOR_SEARCH(
  TABLE `project.dataset.base_table`,
  'embedding_column',
  { TABLE query_table | (SELECT query) },
  top_k => k,
  distance_type => 'COSINE',
  options => '{"option": value}'
);
```

## Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `TABLE` | Yes | Table reference | Table containing embeddings to search |
| `embedding_column` | Yes | STRING | Column name with vector embeddings |
| `query_table` | Yes | Table/Subquery | Query embeddings to find matches for |
| `top_k` | No | INT64 | Number of results per query (default: 10) |
| `distance_type` | No | STRING | Distance metric (default: COSINE) |
| `options` | No | JSON STRING | Additional configuration |

### Distance Types

| Type | Formula | Best For | Range |
|------|---------|----------|-------|
| `COSINE` | 1 - cos(a,b) | Normalized embeddings | [0, 2] |
| `EUCLIDEAN` | ||a-b||_2 | Absolute distances | [0, inf] |
| `DOT_PRODUCT` | -a·b | Pre-normalized, high performance | (-inf, inf) |

### Options

```sql
options => '{
  "fraction_lists_to_search": 0.01,
  "use_brute_force": false
}'
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fraction_lists_to_search` | FLOAT | Auto | Portion of index to search (ANN) |
| `use_brute_force` | BOOL | FALSE | Force exact search |

## Output Schema

The function returns a joined result with:

| Column | Type | Description |
|--------|------|-------------|
| `query.*` | Various | All columns from query table |
| `base.*` | Various | All columns from base table |
| `distance` | FLOAT64 | Distance between query and match |

## Examples

### Basic Semantic Search

```sql
-- Find 5 most similar documents to a query
SELECT
  base.id,
  base.title,
  base.content,
  distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.document_embeddings`,
  'embedding',
  (SELECT embedding
   FROM ML.GENERATE_EMBEDDING(
     MODEL `project.dataset.embedding_model`,
     (SELECT 'machine learning tutorial' AS content)
   )),
  top_k => 5,
  distance_type => 'COSINE'
);
```

### Search with Query Table

```sql
-- Find similar items for multiple queries
WITH query_embeddings AS (
  SELECT
    query_id,
    query_text,
    ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `project.dataset.embedding_model`,
    (SELECT query_id, query_text AS content FROM user_queries)
  )
)
SELECT
  query.query_id,
  query.query_text,
  base.document_id,
  base.title,
  distance
FROM VECTOR_SEARCH(
  TABLE `project.dataset.document_embeddings`,
  'embedding',
  TABLE query_embeddings,
  top_k => 3,
  distance_type => 'COSINE'
)
ORDER BY query.query_id, distance;
```

### Search with Filters

```sql
-- Combine vector search with WHERE clause
SELECT
  base.id,
  base.title,
  base.category,
  distance
FROM VECTOR_SEARCH(
  (SELECT * FROM `project.dataset.embeddings` WHERE category = 'technology'),
  'embedding',
  (SELECT embedding FROM query_embeddings),
  top_k => 10
);
```

### ANN Search with Index

```sql
-- Fast approximate search (requires vector index)
SELECT *
FROM VECTOR_SEARCH(
  TABLE `project.dataset.large_embeddings`,
  'embedding',
  TABLE query_embeddings,
  top_k => 100,
  distance_type => 'COSINE',
  options => '{"fraction_lists_to_search": 0.005}'
);
```

### Exact (Brute Force) Search

```sql
-- Force exact search for highest accuracy
SELECT *
FROM VECTOR_SEARCH(
  TABLE `project.dataset.embeddings`,
  'embedding',
  TABLE query_embeddings,
  top_k => 10,
  options => '{"use_brute_force": true}'
);
```

## Vector Indexes

### Creating Indexes

```sql
-- IVF index (most common)
CREATE OR REPLACE VECTOR INDEX my_index
ON `project.dataset.embeddings`(embedding)
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 1000}'
);

-- Check index status
SELECT
  table_name,
  index_name,
  index_status,
  coverage_percentage
FROM `project.dataset.INFORMATION_SCHEMA.VECTOR_INDEXES`;
```

### Index Types

| Type | Description | Best For |
|------|-------------|----------|
| `IVF` | Inverted file index | General purpose, <100M rows |
| `TREE_AH` | Tree-based with asymmetric hashing | Very large datasets |

### Index Parameters

```sql
-- IVF options
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 1000}'  -- sqrt(n) as starting point
)

-- TREE_AH options
OPTIONS (
  index_type = 'TREE_AH',
  distance_type = 'DOT_PRODUCT',
  tree_ah_options = '{"leaf_node_embedding_count": 1000}'
)
```

### Tuning Search Quality

The `fraction_lists_to_search` parameter trades speed for accuracy:

| Value | Speed | Recall | Use Case |
|-------|-------|--------|----------|
| 0.001 | Fastest | ~90% | Large-scale, speed critical |
| 0.01 | Fast | ~95% | Balanced (recommended) |
| 0.1 | Medium | ~99% | High accuracy needed |
| 1.0 | Slowest | 100% | Equivalent to brute force |

```sql
-- High speed, lower recall
options => '{"fraction_lists_to_search": 0.001}'

-- Balanced
options => '{"fraction_lists_to_search": 0.01}'

-- High recall
options => '{"fraction_lists_to_search": 0.1}'
```

## Common Patterns

### Similarity Threshold

```sql
-- Only return results above similarity threshold
SELECT *
FROM VECTOR_SEARCH(
  TABLE `project.dataset.embeddings`,
  'embedding',
  TABLE query_embeddings,
  top_k => 100,
  distance_type => 'COSINE'
)
WHERE distance < 0.5;  -- COSINE distance < 0.5 means high similarity
```

### Deduplicate Results

```sql
-- Find and group near-duplicates
WITH similarities AS (
  SELECT
    query.id AS id1,
    base.id AS id2,
    distance
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.embeddings`,
    'embedding',
    TABLE `project.dataset.embeddings`,
    top_k => 5,
    distance_type => 'COSINE'
  )
  WHERE query.id < base.id  -- Avoid self-matches and duplicates
    AND distance < 0.1      -- Very similar
)
SELECT * FROM similarities;
```

### Multi-Vector Search

```sql
-- Search across multiple embedding columns
WITH text_results AS (
  SELECT base.id, distance AS text_distance
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.items`,
    'text_embedding',
    TABLE query_embeddings,
    top_k => 50
  )
),
image_results AS (
  SELECT base.id, distance AS image_distance
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.items`,
    'image_embedding',
    TABLE query_image_embeddings,
    top_k => 50
  )
)
SELECT
  COALESCE(t.id, i.id) AS id,
  t.text_distance,
  i.image_distance,
  COALESCE(t.text_distance, 1) * 0.7 +
    COALESCE(i.image_distance, 1) * 0.3 AS combined_score
FROM text_results t
FULL OUTER JOIN image_results i ON t.id = i.id
ORDER BY combined_score;
```

### Hybrid Search (Vector + Keyword)

```sql
-- Combine semantic and keyword search
WITH semantic_results AS (
  SELECT base.id, distance, 1.0 / (1.0 + distance) AS semantic_score
  FROM VECTOR_SEARCH(
    TABLE `project.dataset.docs`,
    'embedding',
    TABLE query_embeddings,
    top_k => 100
  )
),
keyword_results AS (
  SELECT id, search_score
  FROM `project.dataset.docs`
  WHERE SEARCH(content, @query)
)
SELECT
  COALESCE(s.id, k.id) AS id,
  s.semantic_score,
  k.search_score,
  COALESCE(s.semantic_score, 0) * 0.6 +
    COALESCE(k.search_score, 0) * 0.4 AS hybrid_score
FROM semantic_results s
FULL OUTER JOIN keyword_results k ON s.id = k.id
ORDER BY hybrid_score DESC
LIMIT 20;
```

## Performance Tips

### 1. Use Vector Indexes
- Create indexes for tables > 10K rows
- Significant speedup for > 100K rows

### 2. Limit Base Table
- Filter base table before search when possible
- Reduces search space

### 3. Tune Recall vs Speed
- Start with `fraction_lists_to_search: 0.01`
- Increase if quality is insufficient

### 4. Batch Queries
- Process multiple queries in one call
- More efficient than individual queries

### 5. Monitor Costs
- Search costs scale with table size
- Index maintenance has ongoing costs

## Troubleshooting

### Slow Queries

```sql
-- Check if index exists
SELECT * FROM `project.dataset.INFORMATION_SCHEMA.VECTOR_INDEXES`;

-- Check index coverage
SELECT coverage_percentage FROM ...;

-- Ensure index is ACTIVE status
```

### Poor Results

1. Check embedding model consistency
2. Verify distance type matches index
3. Increase `fraction_lists_to_search`
4. Compare with brute force results

### Memory Errors

- Reduce `top_k`
- Filter base table
- Process queries in batches
