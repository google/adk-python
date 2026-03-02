# Error Handling for BigQuery

## Common Error Codes

### Syntax Errors

**Error**: `Syntax error in SQL query`

- Check for missing commas, unmatched parentheses, or incorrect keywords.
- Verify all table and column names are correctly spelled and backtick-quoted
  if they contain special characters.
- Ensure CTEs are separated by commas and the final CTE is followed by a
  main SELECT.

### Not Found Errors

**Error**: `Not found: Table project.dataset.table`

- Verify the project, dataset, and table names using `list_dataset_ids`
  and `list_table_ids`.
- Check that the fully-qualified path uses the correct project ID.
- Ensure the user has access to the specified project and dataset.

### Permission Errors

**Error**: `Access Denied: Table project.dataset.table`

- The service account or user lacks permissions for the resource.
- Required roles: `roles/bigquery.dataViewer` for reading,
  `roles/bigquery.dataEditor` for writing.
- Use `get_table_info` to confirm accessibility before querying.

### Column Not Found

**Error**: `Unrecognized name: column_name`

- Use `get_table_info` to verify exact column names and types.
- Column names are case-sensitive in some contexts.
- For nested fields, use dot notation: `struct_field.sub_field`.

## Quota and Resource Errors

### Query Too Large

**Error**: `Query exceeded resource limits`

- Add stricter WHERE clauses to reduce data scanned.
- Use partition pruning by filtering on partition columns.
- Break the query into smaller CTEs or intermediate tables.
- Use `LIMIT` to reduce output size.

### Concurrent Query Limit

**Error**: `Exceeded rate limits: too many concurrent queries`

- Wait briefly and retry.
- Consider combining multiple small queries into one.

### Bytes Billed Exceeded

**Error**: `Query exceeded maximum bytes billed`

- The query would scan more data than the configured limit.
- Add date range filters or partition filters.
- Use `LIMIT` for exploratory queries.
- Consider using approximate aggregation functions.

## Timeout Strategies

### Query Timeout

If a query runs longer than expected:

1. Check job status using `get_job_info` with the job ID.
2. Consider these optimizations:
   - Add partition filters to reduce data scanned.
   - Replace exact `COUNT(DISTINCT ...)` with `APPROX_COUNT_DISTINCT(...)`.
   - Reduce JOIN complexity by pre-filtering tables.
   - Use materialized views or intermediate tables for repeated patterns.

### Strategies for Large Datasets

- **Sampling**: Use `TABLESAMPLE SYSTEM (10 PERCENT)` for exploratory work.
- **Partitioning**: Always filter on partition columns first.
- **Clustering**: Filter on clustered columns for efficient scans.
- **Approximate functions**: Use `APPROX_COUNT_DISTINCT`,
  `APPROX_QUANTILES`, `APPROX_TOP_COUNT` for large cardinality data.

## Data Type Errors

### Type Mismatch

**Error**: `No matching signature for operator`

- Check column types with `get_table_info` before comparing or joining.
- Use explicit CAST: `CAST(column AS STRING)`, `CAST(column AS INT64)`.
- For dates: `PARSE_DATE('%Y-%m-%d', string_column)`.
- For timestamps: `TIMESTAMP(date_column)`.

### Division by Zero

**Error**: `division by zero`

- Use `SAFE_DIVIDE(numerator, denominator)` which returns NULL
  instead of an error.
- Or use `IF(denominator = 0, NULL, numerator / denominator)`.

### Overflow

**Error**: `Arithmetic overflow`

- Use `SAFE_MULTIPLY`, `SAFE_ADD`, `SAFE_SUBTRACT` for large numbers.
- Cast to NUMERIC or BIGNUMERIC for high-precision calculations.
