# AI.GENERATE_EMBEDDING
Used to generate high-dimensional numerical vectors (embeddings) for text or
visual content. These embeddings can be used for semantic search, clustering,
and recommendation systems.

## Syntax
```sql
SELECT
  *
FROM
  AI.GENERATE_EMBEDDING(
    MODEL `project.dataset.model`,
    { TABLE `project.dataset.table` | (QUERY_STATEMENT) }
    [, STRUCT(
      [TASK_TYPE AS task_type]
      [, OUTPUT_DIMENSIONALITY AS output_dimensionality]) ]
  )
```

## Input Arguments
| Argument | Requirement | Type | Description |
| :--- | :--- | :--- | :--- |
| **`model`** | **Required** | | The model resource (e.g., `project.dataset.model`). |
| **`input_data`** | **Required*** | | The source table or `SELECT` statement containing the data to embed. |
| **`task_type`** | Optional | String | The intended downstream application (e.g., `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`). |
| **`output_dimensionality`** | Optional | Int64 | The number of dimensions to use when generating embeddings. |

\* *`input_data` is optional for matrix factorization models.*

## Output Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| **`[Input Columns]`** | (As Input) | Every column included in your input data. |
| **`embedding`** | ARRAY<FLOAT64> | The generated numerical vector. |
| **`statistics`** | JSON | Metadata about the generation, such as token count. |
| **`status`** | STRING | Execution status; contains error messages on failure. |

## Example: Generating Text Embeddings
```sql
-- Generate embeddings for a literal string
SELECT *
FROM
  AI.GENERATE_EMBEDDING(
    MODEL `mydataset.text_embedding`,
    (SELECT "Example text to embed" AS content)
);
```
