---
name: bigquery-admin
description: Administer BigQuery resources - slot reservations, BI Engine, job management, monitoring, quotas, and cost optimization. Use when managing BigQuery capacity, monitoring performance, or optimizing costs.
license: Apache-2.0
compatibility: BigQuery, Cloud Monitoring
metadata:
  author: Google Cloud
  version: "1.0"
  category: administration
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 5
  allowed_callers:
    - bigquery_agent
    - admin_agent
    - finops_agent
---

# BigQuery Admin Skill

Administer BigQuery resources including slot reservations, BI Engine, job management, monitoring, quotas, and cost optimization.

## When to Use This Skill

Use this skill when you need to:
- Manage slot reservations and capacity
- Configure BI Engine for acceleration
- Monitor and manage running jobs
- Set up quotas and cost controls
- Analyze query performance and costs
- Troubleshoot performance issues

## Administration Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Reservations** | Dedicated compute capacity | Predictable workloads |
| **BI Engine** | In-memory acceleration | Dashboard queries |
| **Jobs** | Query execution management | Monitoring, cancellation |
| **Quotas** | Usage limits | Cost control |
| **Monitoring** | Performance metrics | Optimization |

## Quick Start

### 1. View Running Jobs

```sql
SELECT
  job_id,
  user_email,
  state,
  total_bytes_processed,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), creation_time, SECOND) AS running_seconds
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE state = 'RUNNING'
ORDER BY creation_time;
```

### 2. Check Slot Usage

```sql
SELECT
  TIMESTAMP_TRUNC(period_start, HOUR) AS hour,
  AVG(period_slot_ms) / 1000 / 60 AS avg_slot_minutes
FROM `region-us.INFORMATION_SCHEMA.JOBS_TIMELINE_BY_PROJECT`
WHERE period_start > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY 1
ORDER BY 1;
```

### 3. Create Reservation

```sql
-- Using BigQuery Reservation API (not SQL)
-- See Reservations section below
```

## Slot Reservations

### Pricing Models

| Model | Description | Best For |
|-------|-------------|----------|
| **On-demand** | Pay per TB scanned | Variable workloads |
| **Editions** | Committed slots (Standard/Enterprise/Enterprise Plus) | Predictable workloads |
| **Autoscaling** | Automatic slot scaling | Variable with baseline |

### Create Reservation (API/gcloud)

```bash
# Create a reservation
gcloud bq reservations create my-reservation \
  --project=PROJECT_ID \
  --location=US \
  --slots=500 \
  --edition=ENTERPRISE

# Create an assignment
gcloud bq reservations assignments create \
  --project=PROJECT_ID \
  --location=US \
  --reservation=my-reservation \
  --assignee=projects/PROJECT_ID \
  --job-type=QUERY
```

### View Reservations

```sql
SELECT
  reservation_name,
  slot_capacity,
  target_job_concurrency
FROM `region-us.INFORMATION_SCHEMA.RESERVATIONS`;
```

### View Assignments

```sql
SELECT
  reservation_name,
  assignment_name,
  assignee_id,
  job_type
FROM `region-us.INFORMATION_SCHEMA.ASSIGNMENTS`;
```

### Autoscaling Configuration

```bash
# Enable autoscaling
gcloud bq reservations update my-reservation \
  --location=US \
  --autoscale-max-slots=1000
```

## BI Engine

### Enable BI Engine

```bash
# Create BI Engine reservation
gcloud bq reservations create bi-engine-reservation \
  --project=PROJECT_ID \
  --location=US \
  --bi-reservation-size=100  # GB of RAM
```

### Preferred Tables

```bash
# Configure preferred tables for BI Engine
gcloud bq reservations update bi-engine-reservation \
  --location=US \
  --preferred-tables="project.dataset.table1,project.dataset.table2"
```

### Check BI Engine Status

```sql
SELECT
  project_id,
  bi_engine_mode,
  bi_engine_reasons
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  AND bi_engine_mode IS NOT NULL;
```

### BI Engine Statistics

```sql
SELECT
  COUNT(*) AS total_queries,
  COUNTIF(bi_engine_mode = 'FULL') AS full_acceleration,
  COUNTIF(bi_engine_mode = 'PARTIAL') AS partial_acceleration,
  COUNTIF(bi_engine_mode = 'DISABLED') AS no_acceleration
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND statement_type = 'SELECT';
```

## Job Management

### List Running Jobs

```sql
SELECT
  job_id,
  user_email,
  creation_time,
  state,
  ROUND(total_bytes_processed / 1e9, 2) AS gb_processed,
  query
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE state = 'RUNNING'
ORDER BY creation_time DESC;
```

### Job History

```sql
SELECT
  job_id,
  user_email,
  creation_time,
  end_time,
  state,
  ROUND(total_bytes_billed / 1e9, 2) AS gb_billed,
  ROUND(total_slot_ms / 1000 / 60, 2) AS slot_minutes,
  error_result.message AS error_message
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY creation_time DESC
LIMIT 100;
```

### Cancel Job

```sql
-- Cancel by job ID
CALL BQ.JOBS.CANCEL('project:US.job_id_here');
```

```python
# Using Python client
from google.cloud import bigquery

client = bigquery.Client()
client.cancel_job("job_id", location="US")
```

### Job Performance Analysis

```sql
SELECT
  job_id,
  user_email,
  ROUND(total_bytes_processed / 1e9, 2) AS gb_processed,
  ROUND(total_slot_ms / 1000 / 60, 2) AS slot_minutes,
  TIMESTAMP_DIFF(end_time, start_time, SECOND) AS duration_seconds,
  cache_hit,
  ARRAY_LENGTH(referenced_tables) AS tables_referenced
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND state = 'DONE'
  AND total_bytes_processed > 1e9  -- > 1 GB
ORDER BY total_bytes_processed DESC
LIMIT 50;
```

## Monitoring

### Slot Utilization

```sql
WITH slot_usage AS (
  SELECT
    TIMESTAMP_TRUNC(period_start, MINUTE) AS minute,
    SUM(period_slot_ms) / 60000 AS slot_minutes
  FROM `region-us.INFORMATION_SCHEMA.JOBS_TIMELINE_BY_PROJECT`
  WHERE period_start > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  GROUP BY 1
)
SELECT
  minute,
  slot_minutes,
  AVG(slot_minutes) OVER (
    ORDER BY minute
    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
  ) AS avg_5min
FROM slot_usage
ORDER BY minute;
```

### Query Volume

```sql
SELECT
  TIMESTAMP_TRUNC(creation_time, HOUR) AS hour,
  COUNT(*) AS query_count,
  COUNT(DISTINCT user_email) AS unique_users,
  SUM(total_bytes_billed) / 1e12 AS tb_billed,
  SUM(total_slot_ms) / 1000 / 60 / 60 AS slot_hours
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND statement_type = 'SELECT'
GROUP BY 1
ORDER BY 1;
```

### Error Analysis

```sql
SELECT
  error_result.reason AS error_reason,
  error_result.message AS error_message,
  COUNT(*) AS occurrence_count,
  COUNT(DISTINCT user_email) AS affected_users
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND error_result IS NOT NULL
GROUP BY 1, 2
ORDER BY occurrence_count DESC;
```

### User Activity

```sql
SELECT
  user_email,
  COUNT(*) AS query_count,
  SUM(total_bytes_billed) / 1e9 AS gb_billed,
  SUM(total_slot_ms) / 1000 / 60 AS slot_minutes,
  ROUND(SUM(total_bytes_billed) / 1e12 * 5, 2) AS estimated_cost_usd
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND statement_type = 'SELECT'
GROUP BY user_email
ORDER BY gb_billed DESC
LIMIT 20;
```

## Quotas and Limits

### View Current Quotas

```sql
-- Quotas are managed via Cloud Console or gcloud
-- Common limits:
-- - On-demand: 2000 concurrent queries per project
-- - Reservations: Based on slot allocation
-- - Streaming: 1 GB/s per table
-- - Load jobs: 1000/table/day
```

### Set Custom Quotas

```bash
# Set query bytes limit per user per day
gcloud projects set-quota bigquery.googleapis.com/Query_Usage_per_day \
  --project=PROJECT_ID \
  --consumer-quota-limit=10737418240  # 10 TB
```

### Query Cost Control

```sql
-- Set maximum bytes billed for a query
-- In query settings or job configuration:
-- maximum_bytes_billed: 10737418240  -- 10 GB

-- Example using Python
# job_config = bigquery.QueryJobConfig(
#     maximum_bytes_billed=10 * 1024**3  # 10 GB
# )
```

## Cost Optimization

### Cost Analysis by User

```sql
SELECT
  user_email,
  COUNT(*) AS queries,
  ROUND(SUM(total_bytes_billed) / 1e12, 4) AS tb_scanned,
  ROUND(SUM(total_bytes_billed) / 1e12 * 5, 2) AS estimated_cost_usd
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY user_email
HAVING estimated_cost_usd > 10
ORDER BY estimated_cost_usd DESC;
```

### Cost Analysis by Table

```sql
SELECT
  CONCAT(ref.project_id, '.', ref.dataset_id, '.', ref.table_id) AS table_name,
  COUNT(DISTINCT j.job_id) AS query_count,
  SUM(j.total_bytes_billed) / 1e12 AS tb_scanned
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` j,
UNNEST(referenced_tables) AS ref
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY 1
ORDER BY tb_scanned DESC
LIMIT 20;
```

### Identify Expensive Queries

```sql
SELECT
  job_id,
  user_email,
  ROUND(total_bytes_billed / 1e12 * 5, 2) AS estimated_cost_usd,
  ROUND(total_bytes_billed / 1e9, 2) AS gb_billed,
  cache_hit,
  query
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND statement_type = 'SELECT'
ORDER BY total_bytes_billed DESC
LIMIT 10;
```

### Optimization Recommendations

```sql
-- Find queries that could benefit from partitioning
SELECT
  job_id,
  user_email,
  ROUND(total_bytes_billed / 1e9, 2) AS gb_billed,
  referenced_tables
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND total_bytes_billed > 10e9  -- > 10 GB
  AND NOT EXISTS (
    SELECT 1 FROM UNNEST(referenced_tables) t
    WHERE t.table_id LIKE '%$%'  -- Partition decorator
  )
ORDER BY total_bytes_billed DESC;
```

## Scheduled Queries

### View Scheduled Queries

```sql
SELECT
  name,
  schedule,
  state,
  destination_dataset_id,
  update_time
FROM `region-us.INFORMATION_SCHEMA.SCHEDULED_QUERIES`
ORDER BY update_time DESC;
```

### Create Scheduled Query

```python
from google.cloud import bigquery_datatransfer

client = bigquery_datatransfer.DataTransferServiceClient()

transfer_config = bigquery_datatransfer.TransferConfig(
    destination_dataset_id="destination_dataset",
    display_name="Daily Summary",
    data_source_id="scheduled_query",
    schedule="every 24 hours",
    params={
        "query": """
            SELECT DATE(timestamp) AS date, COUNT(*) AS events
            FROM `project.dataset.events`
            WHERE DATE(timestamp) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
            GROUP BY 1
        """
    }
)

client.create_transfer_config(
    parent=f"projects/{project_id}/locations/{location}",
    transfer_config=transfer_config
)
```

## Performance Troubleshooting

### Slow Query Analysis

```sql
SELECT
  job_id,
  query,
  TIMESTAMP_DIFF(end_time, start_time, SECOND) AS duration_sec,
  total_bytes_processed / 1e9 AS gb_processed,
  total_slot_ms / TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) AS avg_slots,
  cache_hit
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
  AND TIMESTAMP_DIFF(end_time, start_time, SECOND) > 60  -- > 1 minute
  AND state = 'DONE'
ORDER BY duration_sec DESC;
```

### Stage-Level Analysis

```sql
SELECT
  job_id,
  stage.name AS stage_name,
  stage.status,
  stage.records_read,
  stage.records_written,
  stage.shuffle_output_bytes / 1e9 AS shuffle_gb
FROM `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`,
UNNEST(job_stages) AS stage
WHERE job_id = 'your-job-id'
ORDER BY stage.start_ms;
```

## References

- `RESERVATIONS.md` - Detailed reservation management
- `MONITORING.md` - Cloud Monitoring integration
- `COST_OPTIMIZATION.md` - Cost reduction strategies

## Scripts

- `cost_report.py` - Generate cost analysis report
- `slot_monitor.py` - Real-time slot monitoring
- `job_killer.py` - Automated job cancellation

## Limitations

- INFORMATION_SCHEMA: 180-day retention
- Reservations: Minimum 100 slots
- BI Engine: Limited to specific regions
- Quotas: Some can't be customized
