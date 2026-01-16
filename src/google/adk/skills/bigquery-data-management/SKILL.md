---
name: bigquery-data-management
description: Load, transform, and manage data in BigQuery - batch/streaming ingestion, partitioning, clustering, external tables, and data formats. Use when importing data, optimizing table structures, or connecting to external data sources.
license: Apache-2.0
compatibility: BigQuery, Cloud Storage, BigLake
metadata:
  author: Google Cloud
  version: "1.0"
  category: data-management
adk:
  config:
    timeout_seconds: 900
    max_parallel_calls: 5
  allowed_callers:
    - bigquery_agent
    - data_engineer_agent
    - etl_agent
---

# BigQuery Data Management Skill

Comprehensive data loading, transformation, and table optimization in BigQuery. This skill covers ingestion patterns, table partitioning, clustering, and external data connections.

## When to Use This Skill

Use this skill when you need to:
- Load data from various sources (GCS, local files, Cloud SQL, etc.)
- Configure partitioned or clustered tables for performance
- Set up external tables or BigLake connections
- Transform data during loading
- Manage data formats (Parquet, Avro, ORC, CSV, JSON)
- Implement streaming ingestion patterns

**Note**: For ML model training, use the `bqml` skill. For AI/text generation, use the `bigquery-ai` skill.

## Data Loading Methods

| Method | Use Case | Throughput | Cost |
|--------|----------|------------|------|
| `LOAD DATA` | Batch from GCS/local | High | Free (slot usage) |
| `INSERT INTO` | Small inserts from query | Low | Query cost |
| `MERGE` | Upsert operations | Medium | Query cost |
| Streaming API | Real-time ingestion | Medium | Per-row cost |
| Storage Write API | High-throughput streaming | Very High | Per-byte cost |
| Data Transfer Service | Scheduled imports | Varies | Free + source cost |

## Quick Start

### 1. Load Data from Cloud Storage

```sql
-- Load CSV from GCS
LOAD DATA OVERWRITE `project.dataset.my_table`
FROM FILES (
  format = 'CSV',
  uris = ['gs://bucket/data/*.csv'],
  skip_leading_rows = 1
);
```

### 2. Create Partitioned Table

```sql
CREATE TABLE `project.dataset.events`
(
  event_id STRING,
  event_name STRING,
  event_timestamp TIMESTAMP,
  user_id STRING,
  event_data JSON
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, event_name;
```

### 3. Query External Data

```sql
CREATE EXTERNAL TABLE `project.dataset.external_logs`
WITH CONNECTION `project.region.connection_id`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/logs/*.parquet']
);
```

## Supported Data Formats

| Format | Extension | Compression | Best For |
|--------|-----------|-------------|----------|
| **Parquet** | .parquet | Snappy, GZIP | Analytics (recommended) |
| **Avro** | .avro | Deflate, Snappy | Schema evolution |
| **ORC** | .orc | Snappy, ZLIB | Hive compatibility |
| **CSV** | .csv | GZIP | Simple data |
| **JSON** | .json, .jsonl | GZIP | Semi-structured |
| **NEWLINE_DELIMITED_JSON** | .jsonl | GZIP | Streaming data |

## LOAD DATA Statement

### Full Syntax

```sql
LOAD DATA [OVERWRITE] target_table
[PARTITIONS (partition_clause)]
[CLUSTER BY column_list]
FROM FILES (
  format = 'FORMAT',
  uris = ['gs://bucket/path/*.ext'],
  -- Format-specific options
  skip_leading_rows = 1,          -- CSV
  field_delimiter = ',',           -- CSV
  quote = '"',                     -- CSV
  allow_quoted_newlines = TRUE,    -- CSV
  allow_jagged_rows = FALSE,       -- CSV
  null_marker = 'NULL',            -- CSV
  encoding = 'UTF-8',              -- CSV/JSON
  hive_partition_uri_prefix = 'gs://bucket/data/',  -- Hive
  require_hive_partition_filter = TRUE,
  projection_fields = ['field1', 'field2']  -- Specific columns
)
[WITH PARTITION COLUMNS]
[WITH CONNECTION `connection_id`];
```

### Load CSV with Schema

```sql
LOAD DATA OVERWRITE `project.dataset.sales`
(
  sale_id INT64,
  product_name STRING,
  amount NUMERIC,
  sale_date DATE
)
FROM FILES (
  format = 'CSV',
  uris = ['gs://bucket/sales/2024/*.csv'],
  skip_leading_rows = 1,
  allow_jagged_rows = TRUE,
  null_marker = ''
);
```

### Load Parquet with Partitions

```sql
LOAD DATA INTO `project.dataset.events`
FROM FILES (
  format = 'PARQUET',
  uris = ['gs://bucket/events/year=*/month=*/*.parquet'],
  hive_partition_uri_prefix = 'gs://bucket/events/'
)
WITH PARTITION COLUMNS (
  year INT64,
  month INT64
);
```

## Table Partitioning

### Partition Types

| Type | Syntax | Best For |
|------|--------|----------|
| **Time-unit (DATE)** | `PARTITION BY DATE(ts)` | Daily queries |
| **Time-unit (DATETIME)** | `PARTITION BY DATETIME_TRUNC(dt, MONTH)` | Monthly aggregations |
| **Time-unit (TIMESTAMP)** | `PARTITION BY TIMESTAMP_TRUNC(ts, HOUR)` | Hourly data |
| **Integer range** | `PARTITION BY RANGE_BUCKET(id, ...)` | Sequential IDs |
| **Ingestion time** | `PARTITION BY _PARTITIONDATE` | Append-only logs |

### Time-based Partitioning

```sql
-- Partition by date column
CREATE TABLE `project.dataset.user_events`
(
  user_id STRING,
  event_type STRING,
  event_time TIMESTAMP,
  properties JSON
)
PARTITION BY DATE(event_time)
OPTIONS (
  partition_expiration_days = 365,
  require_partition_filter = TRUE
);
```

### Integer Range Partitioning

```sql
CREATE TABLE `project.dataset.orders`
(
  order_id INT64,
  customer_id INT64,
  order_total NUMERIC
)
PARTITION BY RANGE_BUCKET(order_id, GENERATE_ARRAY(0, 100000000, 1000000));
```

### Partition Management

```sql
-- Delete old partitions
DELETE FROM `project.dataset.events`
WHERE DATE(event_time) < DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);

-- Copy partition
INSERT INTO `project.dataset.archive`
SELECT * FROM `project.dataset.events`
WHERE DATE(event_time) = '2024-01-01';
```

## Clustering

### Create Clustered Table

```sql
CREATE TABLE `project.dataset.logs`
(
  log_id STRING,
  log_level STRING,
  service_name STRING,
  message STRING,
  timestamp TIMESTAMP
)
PARTITION BY DATE(timestamp)
CLUSTER BY service_name, log_level;
```

### Clustering Guidelines

1. **Order matters**: Most frequently filtered column first
2. **Up to 4 columns**: Diminishing returns beyond 4
3. **Low cardinality first**: Put columns with fewer unique values first
4. **Combine with partitioning**: Cluster within partitions for best results

### Re-cluster Existing Table

```sql
-- Force re-clustering by overwriting
CREATE OR REPLACE TABLE `project.dataset.logs`
CLUSTER BY service_name, log_level
AS SELECT * FROM `project.dataset.logs`;
```

## External Tables

### BigLake Table (Managed)

```sql
CREATE EXTERNAL TABLE `project.dataset.biglake_sales`
WITH CONNECTION `project.us.my_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/sales/*.parquet'],
  metadata_cache_mode = 'AUTOMATIC'
);
```

### External Table with Hive Partitioning

```sql
CREATE EXTERNAL TABLE `project.dataset.partitioned_logs`
WITH PARTITION COLUMNS (
  year INT64,
  month INT64,
  day INT64
)
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/logs/*'],
  hive_partition_uri_prefix = 'gs://bucket/logs/',
  require_hive_partition_filter = TRUE
);
```

### Object Tables (Unstructured Data)

```sql
CREATE EXTERNAL TABLE `project.dataset.images`
WITH CONNECTION `project.us.my_connection`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://bucket/images/*']
);
```

## INSERT and MERGE Operations

### Insert from Query

```sql
INSERT INTO `project.dataset.summary`
SELECT
  DATE(event_time) AS date,
  COUNT(*) AS event_count,
  COUNT(DISTINCT user_id) AS unique_users
FROM `project.dataset.events`
WHERE DATE(event_time) = CURRENT_DATE()
GROUP BY 1;
```

### MERGE (Upsert)

```sql
MERGE INTO `project.dataset.customers` AS target
USING `project.dataset.customer_updates` AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN
  UPDATE SET
    email = source.email,
    updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
  INSERT (customer_id, email, created_at, updated_at)
  VALUES (source.customer_id, source.email, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
```

### Multi-statement Transaction

```sql
BEGIN TRANSACTION;

DELETE FROM `project.dataset.orders`
WHERE status = 'cancelled' AND created_at < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);

UPDATE `project.dataset.inventory`
SET last_cleaned = CURRENT_TIMESTAMP()
WHERE TRUE;

COMMIT TRANSACTION;
```

## Streaming Ingestion

### Storage Write API (Recommended)

```python
from google.cloud import bigquery_storage_v1
from google.cloud.bigquery_storage_v1 import types
from google.protobuf import descriptor_pb2

client = bigquery_storage_v1.BigQueryWriteClient()
parent = client.table_path("project", "dataset", "table")

write_stream = client.create_write_stream(
    parent=parent,
    write_stream=types.WriteStream(type_=types.WriteStream.Type.COMMITTED)
)

# Append rows
request = types.AppendRowsRequest(
    write_stream=write_stream.name,
    rows=types.AppendRowsRequest.ProtoData(
        rows=types.ProtoRows(serialized_rows=[...])
    )
)
```

### Legacy Streaming API

```python
from google.cloud import bigquery

client = bigquery.Client()
table_ref = client.dataset("dataset").table("table")

rows = [
    {"user_id": "123", "event": "click", "timestamp": "2024-01-15T10:30:00Z"},
    {"user_id": "456", "event": "view", "timestamp": "2024-01-15T10:30:01Z"},
]

errors = client.insert_rows_json(table_ref, rows)
if errors:
    print(f"Errors: {errors}")
```

## Data Transfer Service

### Scheduled Query

```sql
-- Create in BigQuery Console or via API
-- Runs daily at 6 AM UTC
SELECT
  DATE(event_time) AS date,
  COUNT(*) AS total_events
FROM `project.dataset.raw_events`
WHERE DATE(event_time) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
GROUP BY 1;
```

### Cross-region Copy

```sql
-- Copy dataset to another region
-- Use BigQuery Data Transfer Service API
-- or bq command: bq mk --transfer_config ...
```

## Best Practices

### Partitioning Strategy

1. **Choose the right granularity**: Match partition size to query patterns
2. **Require partition filters**: Prevent full-table scans
3. **Set expiration**: Auto-delete old partitions
4. **Avoid over-partitioning**: Aim for >1GB per partition

### Clustering Strategy

1. **Cluster on filter columns**: Most queried columns first
2. **Re-cluster periodically**: After many small inserts
3. **Monitor effectiveness**: Check bytes scanned reduction

### Loading Best Practices

1. **Use Parquet/Avro**: Better compression and performance
2. **Batch small files**: Combine files >1GB each
3. **Avoid streaming for bulk**: Use batch for large loads
4. **Parallel loads**: Load multiple files simultaneously

## References

Load detailed documentation as needed:

- `DATA_FORMATS.md` - Complete format specifications and options
- `PARTITIONING.md` - Advanced partitioning strategies
- `EXTERNAL_TABLES.md` - BigLake and external data connections
- `STREAMING.md` - Real-time ingestion patterns

## Scripts

Helper scripts for common operations:

- `validate_schema.py` - Validate data against table schema
- `partition_manager.py` - Manage partition lifecycle
- `load_monitor.py` - Monitor load job progress

## Limitations

- Maximum 10,000 partitions per table
- Clustering limited to 4 columns
- Streaming buffer not immediately queryable
- External table query performance varies
- Load jobs limited to 15TB per job
