# VECTOR_SEARCH

BigQuery `VECTOR_SEARCH` is a table-valued function used to find semantically
similar entities by searching embeddings in a base table against embeddings in a
query table.

## Syntax

```sql
VECTOR_SEARCH(
  { TABLE base_table | (base_table_query) },
  column_to_search,
  { TABLE query_table | (query_table_query) },
  [, query_column_to_search => query_column_to_search_value]
  [, top_k => top_k_value ]
  [, distance_type => distance_type_value ]
  [, options => options_value ]
)
```

### Arguments
| Argument | Requirement | Type | Description |
| :--- | :--- | :--- | :--- |
| **`base_table`** | **One Of**  | TABLE | The table containing the embeddings to search. Mutually exclusive with base_table_query. |
| **`base_table_query`** | **One Of** | QUERY | A query that you can use to pre-filter the base table. Only SELECT, FROM, and WHERE clauses are allowed in this query. Don't apply any filters to the embedding column. Mutually exclusive with base_table. |
| **`column_to_search`** | **Required** | String | The name of the base table column containing embeddings (`ARRAY<FLOAT64>` or `STRING`). |
| **`query_table`** | **One Of**  | TABLE | The table that provides the embeddings for which to find nearest neighbors. All columns are passed through as output columns. Mutually exclusive with query_table_query |
| **`query_table_query`** | **One Of**  | QUERY | A query that provides the embeddings for which to find nearest neighbors. All columns are passed through as output columns. Mutually exclusive with query_table |
| **`query_column_to_search`** | Optional | String | The name of the column in the query table. Defaults to `column_to_search`. The value can be either `STRING` or `ARRAY<FLOAT64>`. If `STRING`, the base_table must have autonomous embedding generation enabled. The string values are embedded at runtime using the same connection and endpoint specified for the base table's embedding generation. If the value is an `ARRAY<FLOAT64>`, all elements in the array must be non-NULL and all values in the column must have the same array dimensions as the values in the column_to_search column.|
| **`top_k`** | Optional | Int64 | The number of nearest neighbors to return. Defaults to 10. |
| **`distance_type`** | Optional | String | The metric to use: `'EUCLIDEAN'`, `'COSINE'`, or `'DOT_PRODUCT'`. Defaults to `'EUCLIDEAN'`. |
| **`options`** | Optional | String | JSON string for search options (e.g., `use_brute_force`, `fraction_lists_to_search`). |

### Output Columns
| Column | Type | Description |
| :--- | :--- | :--- |
| **`base`** | STRUCT | A STRUCT value that contains all columns from base_table or a subset of the columns from base_table that you selected in the base_table_query query. |
| **`query`** | STRUCT | A STRUCT value that contains all selected columns from the query data. This column is only included in the output if you use the batch search syntax. For single vector searches, this column is omitted.|
| **`distance`** | FLOAT64 | A FLOAT64 value that represents the distance between the base data and the query data. |

---

## Example: Combined Vector Search with AI.GENERATE_EMBEDDING

```sql
SELECT
  query.content AS search_query,
  base.product_name,
  base.description,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `my_project.my_dataset.products`,
    'product_embedding',
    (
      SELECT
        content,
        embedding
      FROM
        AI.GENERATE_EMBEDDING(
          MODEL `my_project.my_dataset.embedding_model`,
          (SELECT "high-performance running shoes" AS content)
        )
    ),
    top_k => 5,
    distance_type => 'COSINE'
  );
```

---

## Example: Optimized Single Search

This version is optimized for finding neighbors for a single value.

```sql
SELECT
  base.product_name,
  base.description,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `my_project.my_dataset.products`,
    'product_embedding',
    query_value => (
      SELECT embedding
      FROM AI.GENERATE_EMBEDDING(
        MODEL `my_project.my_dataset.embedding_model`,
        (SELECT "high-performance running shoes" AS content)
      )
    ),
    top_k => 5
  );
```
