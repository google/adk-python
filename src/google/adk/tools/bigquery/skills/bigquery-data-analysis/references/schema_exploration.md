# Schema Exploration for BigQuery

## Using INFORMATION_SCHEMA

INFORMATION_SCHEMA views provide metadata about datasets, tables, and
columns without scanning actual data.

### List All Tables in a Dataset

```sql
SELECT
  table_name,
  table_type,
  creation_time,
  ROUND(size_bytes / POW(10, 9), 2) AS size_gb,
  row_count
FROM `project.dataset.INFORMATION_SCHEMA.TABLES`
ORDER BY size_bytes DESC
```

### Get Column Details for a Table

```sql
SELECT
  column_name,
  data_type,
  is_nullable,
  column_default
FROM `project.dataset.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'my_table'
ORDER BY ordinal_position
```

### Find Partitioning and Clustering Info

```sql
SELECT
  table_name,
  partition_expiration_days,
  ddl
FROM `project.dataset.INFORMATION_SCHEMA.TABLES`
WHERE table_name = 'my_table'
```

### Table Storage and Billing Info

```sql
SELECT
  table_name,
  total_rows,
  ROUND(total_logical_bytes / POW(10, 9), 2) AS logical_gb,
  ROUND(active_logical_bytes / POW(10, 9), 2) AS active_gb,
  ROUND(long_term_logical_bytes / POW(10, 9), 2) AS long_term_gb
FROM `project.dataset.INFORMATION_SCHEMA.TABLE_STORAGE`
ORDER BY total_logical_bytes DESC
```

## Column Profiling

Profile columns to understand data distribution before analysis.

### Numeric Column Profile

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(column_name IS NULL) AS null_count,
  MIN(column_name) AS min_value,
  MAX(column_name) AS max_value,
  AVG(column_name) AS mean_value,
  APPROX_QUANTILES(column_name, 4) AS quartiles
FROM `project.dataset.table`
```

### Categorical Column Profile

```sql
SELECT
  column_name,
  COUNT(*) AS frequency,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM `project.dataset.table`
WHERE column_name IS NOT NULL
GROUP BY column_name
ORDER BY frequency DESC
LIMIT 20
```

### Date Column Range

```sql
SELECT
  MIN(date_column) AS earliest,
  MAX(date_column) AS latest,
  DATE_DIFF(MAX(date_column), MIN(date_column), DAY) AS span_days,
  COUNT(DISTINCT date_column) AS distinct_dates,
  COUNTIF(date_column IS NULL) AS null_count
FROM `project.dataset.table`
```

## Relationship Discovery

### Find Potential Join Keys

Look for columns with matching names and compatible types across tables.

```sql
SELECT
  c1.table_name AS table_1,
  c1.column_name,
  c2.table_name AS table_2,
  c1.data_type
FROM `project.dataset.INFORMATION_SCHEMA.COLUMNS` c1
JOIN `project.dataset.INFORMATION_SCHEMA.COLUMNS` c2
  ON c1.column_name = c2.column_name
  AND c1.data_type = c2.data_type
  AND c1.table_name < c2.table_name
ORDER BY c1.column_name
```

### Validate Join Key Cardinality

Before joining, check if keys are unique or have duplicates.

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT key_column) AS distinct_keys,
  COUNT(*) - COUNT(DISTINCT key_column) AS duplicate_count
FROM `project.dataset.table`
```

### Check Referential Integrity

```sql
SELECT COUNT(*) AS orphaned_rows
FROM `project.dataset.child_table` c
LEFT JOIN `project.dataset.parent_table` p
  ON c.parent_id = p.id
WHERE p.id IS NULL
```

## Data Quality Checks

### NULL Rate by Column

```sql
SELECT
  COUNT(*) AS total,
  COUNTIF(col1 IS NULL) AS col1_nulls,
  COUNTIF(col2 IS NULL) AS col2_nulls,
  COUNTIF(col3 IS NULL) AS col3_nulls
FROM `project.dataset.table`
```

### Detect Duplicate Rows

```sql
SELECT
  key_column,
  COUNT(*) AS occurrences
FROM `project.dataset.table`
GROUP BY key_column
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 10
```
