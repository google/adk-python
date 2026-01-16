# BigQuery Partitioning Reference

Complete guide to table partitioning strategies and management.

## Partition Types

### Time-based Partitioning

Partition data by a TIMESTAMP, DATE, or DATETIME column.

```sql
-- Partition by DATE column (daily)
CREATE TABLE `project.dataset.events`
(
  event_id STRING,
  event_time TIMESTAMP,
  data JSON
)
PARTITION BY DATE(event_time);

-- Partition by DATETIME with monthly granularity
CREATE TABLE `project.dataset.monthly_summary`
(
  month_start DATETIME,
  total NUMERIC
)
PARTITION BY DATETIME_TRUNC(month_start, MONTH);

-- Partition by TIMESTAMP with hourly granularity
CREATE TABLE `project.dataset.hourly_logs`
(
  log_time TIMESTAMP,
  message STRING
)
PARTITION BY TIMESTAMP_TRUNC(log_time, HOUR);
```

### Granularity Options

| Granularity | Function | Partitions/Year | Use Case |
|-------------|----------|-----------------|----------|
| HOUR | `TIMESTAMP_TRUNC(col, HOUR)` | 8,760 | High-frequency data |
| DAY | `DATE(col)` | 365 | Most common |
| MONTH | `DATE_TRUNC(col, MONTH)` | 12 | Low-volume data |
| YEAR | `DATE_TRUNC(col, YEAR)` | 1 | Historical archives |

### Integer Range Partitioning

Partition by an integer column with defined ranges.

```sql
CREATE TABLE `project.dataset.orders`
(
  order_id INT64,
  customer_id INT64,
  amount NUMERIC
)
PARTITION BY RANGE_BUCKET(order_id, GENERATE_ARRAY(0, 1000000000, 10000000));

-- Creates partitions: [0, 10000000), [10000000, 20000000), ...
```

### Ingestion-time Partitioning

Partition by when data was loaded (system-managed).

```sql
CREATE TABLE `project.dataset.raw_logs`
(
  log_message STRING,
  source STRING
)
PARTITION BY _PARTITIONDATE;

-- Query specific partition
SELECT * FROM `project.dataset.raw_logs`
WHERE _PARTITIONDATE = '2024-01-15';
```

## Partition Options

### Table Options

```sql
CREATE TABLE `project.dataset.events`
(...)
PARTITION BY DATE(event_time)
OPTIONS (
  -- Automatically delete partitions older than N days
  partition_expiration_days = 365,

  -- Require WHERE clause to include partition column
  require_partition_filter = TRUE,

  -- Description
  description = 'User events partitioned by date'
);
```

### Partition Expiration

```sql
-- Set expiration on existing table
ALTER TABLE `project.dataset.events`
SET OPTIONS (partition_expiration_days = 90);

-- Remove expiration
ALTER TABLE `project.dataset.events`
SET OPTIONS (partition_expiration_days = NULL);
```

### Require Partition Filter

```sql
-- Enable filter requirement
ALTER TABLE `project.dataset.events`
SET OPTIONS (require_partition_filter = TRUE);

-- Query must include partition filter
SELECT * FROM `project.dataset.events`
WHERE DATE(event_time) = '2024-01-15';  -- Required

-- This will fail:
-- SELECT * FROM `project.dataset.events`;  -- Error!
```

## Partition Management

### View Partition Information

```sql
-- List all partitions
SELECT
  table_name,
  partition_id,
  total_rows,
  total_logical_bytes / 1024 / 1024 AS size_mb,
  last_modified_time
FROM `project.dataset.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = 'events'
ORDER BY partition_id DESC;
```

### Delete Specific Partition

```sql
-- Delete by partition column
DELETE FROM `project.dataset.events`
WHERE DATE(event_time) = '2024-01-01';

-- Delete using partition decorator (legacy)
DELETE FROM `project.dataset.events$20240101`
WHERE TRUE;
```

### Copy Partition

```sql
-- Copy partition to another table
INSERT INTO `project.dataset.archive`
SELECT * FROM `project.dataset.events`
WHERE DATE(event_time) = '2024-01-01';

-- Copy with partition decorator
INSERT INTO `project.dataset.archive$20240101`
SELECT * FROM `project.dataset.events$20240101`;
```

### Update Partition

```sql
-- Overwrite entire partition
MERGE INTO `project.dataset.events` AS target
USING (SELECT * FROM `project.dataset.staging` WHERE DATE(event_time) = '2024-01-15') AS source
ON FALSE  -- Always not matched for overwrite
WHEN NOT MATCHED BY SOURCE AND DATE(target.event_time) = '2024-01-15' THEN DELETE
WHEN NOT MATCHED THEN INSERT ROW;
```

## Partitioned External Tables

### Hive-style Partitioning

```sql
-- External table with Hive partitions
CREATE EXTERNAL TABLE `project.dataset.logs`
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

-- Query with partition filter
SELECT * FROM `project.dataset.logs`
WHERE year = 2024 AND month = 1 AND day = 15;
```

### Auto-detect Partitions

```sql
CREATE EXTERNAL TABLE `project.dataset.auto_partitioned`
WITH PARTITION COLUMNS
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/data/*'],
  hive_partition_uri_prefix = 'gs://bucket/data/'
);
```

## Performance Optimization

### Query Optimization

```sql
-- Good: Uses partition pruning
SELECT * FROM `project.dataset.events`
WHERE DATE(event_time) = '2024-01-15';

-- Good: Range filter uses pruning
SELECT * FROM `project.dataset.events`
WHERE event_time BETWEEN '2024-01-01' AND '2024-01-31';

-- Bad: Function prevents pruning
SELECT * FROM `project.dataset.events`
WHERE EXTRACT(YEAR FROM event_time) = 2024;

-- Bad: Cast prevents pruning
SELECT * FROM `project.dataset.events`
WHERE CAST(event_time AS DATE) = '2024-01-15';
```

### Partition Pruning Check

```sql
-- Check estimated bytes scanned
SELECT
  @bytes_billed_estimate
FROM (
  SELECT * FROM `project.dataset.events`
  WHERE DATE(event_time) = '2024-01-15'
);
```

## Design Patterns

### Daily Partitions (Most Common)

```sql
CREATE TABLE `project.dataset.web_events`
(
  session_id STRING,
  user_id STRING,
  page_url STRING,
  event_type STRING,
  event_time TIMESTAMP
)
PARTITION BY DATE(event_time)
CLUSTER BY user_id
OPTIONS (
  partition_expiration_days = 730,  -- 2 years
  require_partition_filter = TRUE
);
```

### Monthly Aggregates

```sql
CREATE TABLE `project.dataset.monthly_revenue`
(
  month_start DATE,
  product_category STRING,
  total_revenue NUMERIC,
  order_count INT64
)
PARTITION BY DATE_TRUNC(month_start, MONTH)
CLUSTER BY product_category;
```

### Real-time with Hourly Partitions

```sql
CREATE TABLE `project.dataset.realtime_metrics`
(
  metric_name STRING,
  metric_value FLOAT64,
  recorded_at TIMESTAMP
)
PARTITION BY TIMESTAMP_TRUNC(recorded_at, HOUR)
OPTIONS (
  partition_expiration_days = 7  -- Keep 1 week
);
```

### ID-based Sharding

```sql
CREATE TABLE `project.dataset.user_data`
(
  user_id INT64,
  user_name STRING,
  email STRING
)
PARTITION BY RANGE_BUCKET(user_id, GENERATE_ARRAY(0, 1000000000, 1000000));
```

## Limitations

| Limit | Value |
|-------|-------|
| Maximum partitions per table | 10,000 |
| Maximum partitions per load | 4,000 |
| Minimum partition size (recommended) | 1 GB |
| Partition expiration granularity | Days |

## Best Practices

1. **Choose appropriate granularity**: Match query patterns
2. **Avoid over-partitioning**: >1GB per partition ideal
3. **Use partition expiration**: Auto-cleanup old data
4. **Require partition filters**: Prevent full scans
5. **Combine with clustering**: Further optimize within partitions
6. **Monitor partition sizes**: Balance across partitions
7. **Test partition pruning**: Verify queries use pruning
