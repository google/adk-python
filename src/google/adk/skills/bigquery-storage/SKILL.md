---
name: bigquery-storage
description: Manage BigQuery storage architecture - create/modify tables, schema evolution, snapshots, clones, time travel, and storage optimization. Use when designing table structures, managing schema changes, or optimizing storage costs.
license: Apache-2.0
compatibility: BigQuery
metadata:
  author: Google Cloud
  version: "1.0"
  category: storage
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 5
  allowed_callers:
    - bigquery_agent
    - data_engineer_agent
    - dba_agent
---

# BigQuery Storage Skill

Manage BigQuery storage architecture including table creation, schema evolution, snapshots, clones, time travel, and storage optimization.

## When to Use This Skill

Use this skill when you need to:
- Create and manage tables, views, and materialized views
- Modify table schemas (add/drop columns, change types)
- Create table snapshots and clones
- Use time travel to query historical data
- Optimize storage costs and usage
- Manage datasets and data organization

**Note**: For data loading operations, use `bigquery-data-management` skill.

## Storage Architecture

| Object | Description | Use Case |
|--------|-------------|----------|
| **Table** | Structured data storage | Primary data storage |
| **View** | Virtual table from query | Abstraction layer |
| **Materialized View** | Pre-computed view | Query acceleration |
| **Snapshot** | Point-in-time backup | Data protection |
| **Clone** | Zero-copy table copy | Development/testing |

## Quick Start

### 1. Create a Table

```sql
CREATE TABLE `project.dataset.users`
(
  user_id STRING NOT NULL,
  email STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  profile STRUCT<name STRING, age INT64>,
  tags ARRAY<STRING>
)
OPTIONS (
  description = 'User profiles',
  labels = [('team', 'data'), ('env', 'prod')]
);
```

### 2. Create a View

```sql
CREATE VIEW `project.dataset.active_users` AS
SELECT * FROM `project.dataset.users`
WHERE last_login > DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);
```

### 3. Create a Materialized View

```sql
CREATE MATERIALIZED VIEW `project.dataset.daily_stats`
OPTIONS (enable_refresh = true, refresh_interval_minutes = 60)
AS
SELECT DATE(event_time) AS date, COUNT(*) AS events
FROM `project.dataset.events`
GROUP BY 1;
```

## Table Creation

### Standard Table

```sql
CREATE TABLE `project.dataset.table_name`
(
  -- Column definitions
  id INT64 NOT NULL,
  name STRING,
  created_at TIMESTAMP,
  -- Complex types
  address STRUCT<street STRING, city STRING, zip STRING>,
  phone_numbers ARRAY<STRING>,
  metadata JSON
)
-- Partitioning
PARTITION BY DATE(created_at)
-- Clustering
CLUSTER BY name
-- Table options
OPTIONS (
  description = 'Table description',
  labels = [('key', 'value')],
  expiration_timestamp = TIMESTAMP '2025-12-31',
  partition_expiration_days = 365,
  require_partition_filter = TRUE,
  friendly_name = 'My Table'
);
```

### Create Table As Select (CTAS)

```sql
CREATE TABLE `project.dataset.new_table`
PARTITION BY date_column
CLUSTER BY category
OPTIONS (description = 'Derived table')
AS
SELECT * FROM `project.dataset.source_table`
WHERE condition;
```

### Create If Not Exists

```sql
CREATE TABLE IF NOT EXISTS `project.dataset.table`
(id INT64, name STRING);
```

### Create Or Replace

```sql
CREATE OR REPLACE TABLE `project.dataset.table`
(id INT64, name STRING);
```

## Data Types

### Scalar Types

| Type | Description | Example |
|------|-------------|---------|
| `INT64` | 64-bit integer | `12345` |
| `FLOAT64` | 64-bit float | `3.14159` |
| `NUMERIC` | Exact decimal (38,9) | `123.456789` |
| `BIGNUMERIC` | Exact decimal (76,38) | Large precise numbers |
| `BOOL` | Boolean | `TRUE`, `FALSE` |
| `STRING` | UTF-8 text | `'Hello'` |
| `BYTES` | Binary data | `b'\\x00\\x01'` |
| `DATE` | Calendar date | `DATE '2024-01-15'` |
| `TIME` | Time of day | `TIME '10:30:00'` |
| `DATETIME` | Date and time | `DATETIME '2024-01-15 10:30:00'` |
| `TIMESTAMP` | Point in time (UTC) | `TIMESTAMP '2024-01-15 10:30:00 UTC'` |
| `GEOGRAPHY` | Geospatial | `ST_GEOGPOINT(-122, 37)` |
| `JSON` | JSON data | `JSON '{"key": "value"}'` |
| `INTERVAL` | Duration | `INTERVAL 1 DAY` |

### Complex Types

```sql
-- STRUCT (named fields)
STRUCT<
  name STRING,
  age INT64,
  address STRUCT<city STRING, zip STRING>
>

-- ARRAY
ARRAY<STRING>
ARRAY<STRUCT<id INT64, value FLOAT64>>

-- Nested example
CREATE TABLE example (
  id INT64,
  orders ARRAY<STRUCT<
    order_id STRING,
    items ARRAY<STRUCT<
      product_id STRING,
      quantity INT64,
      price NUMERIC
    >>,
    total NUMERIC
  >>
);
```

## Schema Evolution

### Add Columns

```sql
-- Add single column
ALTER TABLE `project.dataset.table`
ADD COLUMN new_column STRING;

-- Add column with default
ALTER TABLE `project.dataset.table`
ADD COLUMN status STRING DEFAULT 'active';

-- Add nested column
ALTER TABLE `project.dataset.table`
ADD COLUMN profile STRUCT<name STRING, bio STRING>;
```

### Drop Columns

```sql
-- Drop single column
ALTER TABLE `project.dataset.table`
DROP COLUMN column_name;

-- Drop multiple columns
ALTER TABLE `project.dataset.table`
DROP COLUMN col1,
DROP COLUMN col2;

-- Drop if exists
ALTER TABLE `project.dataset.table`
DROP COLUMN IF EXISTS maybe_column;
```

### Rename Columns

```sql
ALTER TABLE `project.dataset.table`
RENAME COLUMN old_name TO new_name;
```

### Change Column Type

```sql
-- Widen type (INT64 to FLOAT64)
ALTER TABLE `project.dataset.table`
ALTER COLUMN numeric_col SET DATA TYPE FLOAT64;

-- String to JSON
ALTER TABLE `project.dataset.table`
ALTER COLUMN json_string SET DATA TYPE JSON;
```

### Set Column Options

```sql
-- Set default value
ALTER TABLE `project.dataset.table`
ALTER COLUMN status SET DEFAULT 'pending';

-- Remove default
ALTER TABLE `project.dataset.table`
ALTER COLUMN status DROP DEFAULT;

-- Set NOT NULL (requires no NULL values)
ALTER TABLE `project.dataset.table`
ALTER COLUMN id SET NOT NULL;
```

## Views

### Standard View

```sql
CREATE VIEW `project.dataset.view_name` AS
SELECT
  user_id,
  COUNT(*) AS order_count,
  SUM(total) AS total_spent
FROM `project.dataset.orders`
GROUP BY user_id;
```

### Parameterized View (SQL UDF)

```sql
CREATE TABLE FUNCTION `project.dataset.orders_by_status`(status_param STRING)
AS (
  SELECT * FROM `project.dataset.orders`
  WHERE status = status_param
);

-- Usage
SELECT * FROM `project.dataset.orders_by_status`('completed');
```

### Authorized View

```sql
-- Grant access to underlying data through view
ALTER VIEW `project.dataset.view_name`
SET OPTIONS (
  description = 'Authorized view for limited access'
);

-- In dataset permissions, authorize the view
```

## Materialized Views

### Create Materialized View

```sql
CREATE MATERIALIZED VIEW `project.dataset.mv_daily_sales`
OPTIONS (
  enable_refresh = true,
  refresh_interval_minutes = 60,
  max_staleness = INTERVAL 4 HOUR
)
AS
SELECT
  DATE(sale_time) AS sale_date,
  product_category,
  SUM(amount) AS total_sales,
  COUNT(*) AS transaction_count
FROM `project.dataset.sales`
GROUP BY 1, 2;
```

### Refresh Options

```sql
-- Manual refresh
CALL BQ.REFRESH_MATERIALIZED_VIEW('project.dataset.mv_name');

-- Alter refresh settings
ALTER MATERIALIZED VIEW `project.dataset.mv_name`
SET OPTIONS (
  enable_refresh = true,
  refresh_interval_minutes = 30
);
```

### Supported Operations

- Aggregations: SUM, COUNT, AVG, MIN, MAX, etc.
- GROUP BY
- INNER JOIN (limited)
- Filters (WHERE)
- Window functions (limited)

## Snapshots

### Create Snapshot

```sql
CREATE SNAPSHOT TABLE `project.dataset.orders_snapshot_20240115`
CLONE `project.dataset.orders`
OPTIONS (
  expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
);
```

### Restore from Snapshot

```sql
-- Restore to new table
CREATE TABLE `project.dataset.orders_restored`
CLONE `project.dataset.orders_snapshot_20240115`;

-- Replace existing table
CREATE OR REPLACE TABLE `project.dataset.orders`
CLONE `project.dataset.orders_snapshot_20240115`;
```

## Clones

### Create Clone (Zero-Copy)

```sql
-- Table clone
CREATE TABLE `project.dataset.orders_clone`
CLONE `project.dataset.orders`;

-- Clone from point in time
CREATE TABLE `project.dataset.orders_yesterday`
CLONE `project.dataset.orders`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);
```

### Clone Use Cases

- Development/testing environments
- What-if analysis
- Quick backups before changes
- A/B testing datasets

## Time Travel

### Query Historical Data

```sql
-- Query as of specific time
SELECT * FROM `project.dataset.orders`
FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-15 10:00:00 UTC';

-- Query from N hours ago
SELECT * FROM `project.dataset.orders`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR);

-- Query from N days ago (up to 7 days)
SELECT * FROM `project.dataset.orders`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY);
```

### Restore Deleted Data

```sql
-- Recover accidentally deleted rows
INSERT INTO `project.dataset.orders`
SELECT * FROM `project.dataset.orders`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
WHERE order_id NOT IN (SELECT order_id FROM `project.dataset.orders`);
```

### Time Travel Window

- Default: 7 days
- Configurable per table: 2-7 days
- Storage charged for historical versions

```sql
-- Set time travel window
ALTER TABLE `project.dataset.table`
SET OPTIONS (max_time_travel_hours = 48);  -- 2 days
```

## Dataset Management

### Create Dataset

```sql
CREATE SCHEMA `project.dataset_name`
OPTIONS (
  location = 'US',
  default_table_expiration_days = 90,
  default_partition_expiration_days = 365,
  description = 'Dataset description',
  labels = [('team', 'analytics')]
);
```

### Alter Dataset

```sql
ALTER SCHEMA `project.dataset`
SET OPTIONS (
  default_table_expiration_days = 180,
  description = 'Updated description'
);
```

### Drop Dataset

```sql
-- Drop empty dataset
DROP SCHEMA `project.dataset`;

-- Drop with all contents
DROP SCHEMA `project.dataset` CASCADE;
```

## Table Operations

### Copy Table

```sql
-- Copy within project
CREATE TABLE `project.dataset.table_copy`
COPY `project.dataset.original_table`;

-- Copy across datasets
CREATE TABLE `project.other_dataset.table`
COPY `project.dataset.table`;
```

### Rename Table

```sql
ALTER TABLE `project.dataset.old_name`
RENAME TO `project.dataset.new_name`;
```

### Set Table Options

```sql
ALTER TABLE `project.dataset.table`
SET OPTIONS (
  description = 'New description',
  expiration_timestamp = TIMESTAMP '2025-12-31',
  labels = [('status', 'archive')]
);
```

### Drop Table

```sql
DROP TABLE `project.dataset.table`;
DROP TABLE IF EXISTS `project.dataset.table`;
```

## Storage Optimization

### Check Storage Usage

```sql
SELECT
  table_name,
  ROUND(total_logical_bytes / 1024 / 1024 / 1024, 2) AS logical_gb,
  ROUND(total_physical_bytes / 1024 / 1024 / 1024, 2) AS physical_gb,
  ROUND(time_travel_physical_bytes / 1024 / 1024 / 1024, 2) AS time_travel_gb
FROM `project.dataset.INFORMATION_SCHEMA.TABLE_STORAGE`;
```

### Long-term Storage

Tables not modified for 90 days automatically move to long-term storage (50% cheaper).

### Reduce Time Travel

```sql
-- Reduce to minimum (2 days)
ALTER TABLE `project.dataset.table`
SET OPTIONS (max_time_travel_hours = 48);
```

### Delete Old Data

```sql
-- Delete old partitions
DELETE FROM `project.dataset.events`
WHERE DATE(event_time) < DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY);
```

## References

- `DATA_TYPES.md` - Complete type reference
- `SCHEMA_EVOLUTION.md` - Schema change patterns
- `OPTIMIZATION.md` - Storage cost optimization

## Scripts

- `storage_report.py` - Generate storage usage report
- `schema_diff.py` - Compare table schemas

## Limitations

- Time travel: Maximum 7 days
- Snapshots: Count against storage quota
- Schema changes: Some type changes not allowed
- Clones: Base table changes propagate
