# AI.GENERATE_TABLE
Uses Gemini models to extract information into a predefined SQL schema. It is a
"Table-Valued Function," meaning it is called in the `FROM` clause.

## Syntax
```sql
SELECT
  *
FROM
  AI.GENERATE_TABLE(
    MODEL `project.dataset.model`,
    { TABLE `project.dataset.table` | (QUERY_STATEMENT) },
  STRUCT(
    OUTPUT_SCHEMA AS output_schema
    [, MAX_OUTPUT_TOKENS AS max_output_tokens]
    [, TOP_P AS top_p]
    [, TEMPERATURE AS temperature]
    [, STOP_SEQUENCES AS stop_sequences]
    [, SAFETY_SETTINGS AS safety_settings]
    [, REQUEST_TYPE AS request_type])
  )
```

## Input Arguments
| Argument | Requirement | Type | Description |
| :--- | :--- | :--- | :--- |
| **`model`** | **Required** | | The remote model resource (e.g., `project.dataset.model`). |
| **`input_data`** | **Required** | | The source table or `SELECT` statement. Must contain a column named or aliased as `prompt`. |
| **`output_schema`** | **Required** | String | A `STRUCT` containing SQL-style column definitions (e.g., `"name STRING, qty INT64"`). |
| **`max_output_tokens`** | Optional | Int64 | [1, 8192]. Limits response length. |
| **`temperature`** | Optional | Float64 | [0.0, 2.0]. Degree of randomness (0.0 is recommended for deterministic responses). |
| **`top_p`** | Optional | Float64 | [0.0, 1.0]. Changes how the model selects tokens for output. |
| **`stop_sequences`** | Optional | Array<String> | Strings that halt model generation if matched. |
| **`safety_settings`** | Optional | Array<Struct> | Thresholds to filter hate speech, harassment, etc. |
| **`request_type`** | Optional | String | `SHARED`, `DEDICATED`, or `UNSPECIFIED` (defaults to `UNSPECIFIED`). |

### Prompt Construction (STRUCT Fields)
When using a `QUERY_STATEMENT`, you can build multimodal prompts by combining
types:

* **`STRING` / `ARRAY<STRING>`**: Literal text or column names.
* **`ObjectRefRuntime`**: Use `OBJ.GET_ACCESS_URL(col, 'r')` for images/PDFs.

## Output Schema

The output table is a join of your source data and the model's generated
content. It preserves all original input columns to maintain traceability.

| Column | Type | Description |
| :--- | :--- | :--- |
| **`[Input Columns]`** | (As Input) | Every column included in your input `TABLE` or `QUERY_STATEMENT`. |
| **`[Schema Columns]`**| (As Defined)| The typed columns you specified in the `OUTPUT_SCHEMA` argument. |
| **`full_response`** | JSON        | The raw, unparsed JSON response from the underlying Gemini model. |
| **`status`** | STRING      | Execution status; will contain error messages if a row fails extraction. |

## Example: Preserving Keys During LLM Summarization

```sql
-- The output includes 'ticket_id' from the input and 'priority' and 'issue_summary' from the AI schema
SELECT
  ticket_id,
  priority,
  issue_summary
FROM AI.GENERATE_TABLE(
  MODEL `prod.models.gemini_flash`,
  (SELECT ticket_id, body AS prompt FROM `support.tickets`),
  STRUCT("priority STRING, issue_summary STRING" AS output_schema)
)
```
