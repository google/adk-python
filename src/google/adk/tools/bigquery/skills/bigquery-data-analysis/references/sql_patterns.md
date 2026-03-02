# SQL Patterns for BigQuery

## Window Functions

Window functions perform calculations across rows related to the current row.

### Running Totals and Averages

```sql
SELECT
  date,
  revenue,
  SUM(revenue) OVER (ORDER BY date) AS cumulative_revenue,
  AVG(revenue) OVER (
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7day_avg
FROM `project.dataset.daily_metrics`
ORDER BY date
```

### Ranking and Percentiles

```sql
SELECT
  product_name,
  revenue,
  RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
  PERCENT_RANK() OVER (ORDER BY revenue) AS percentile,
  NTILE(4) OVER (ORDER BY revenue DESC) AS quartile
FROM `project.dataset.products`
```

### Lead/Lag for Comparisons

```sql
SELECT
  date,
  metric_value,
  LAG(metric_value) OVER (ORDER BY date) AS prev_value,
  metric_value - LAG(metric_value) OVER (ORDER BY date) AS change
FROM `project.dataset.daily_metrics`
```

## Common Table Expressions (CTEs)

Use CTEs for readable, modular queries.

```sql
WITH
  daily_totals AS (
    SELECT
      DATE(created_at) AS date,
      COUNT(*) AS order_count,
      SUM(total_amount) AS revenue
    FROM `project.dataset.orders`
    WHERE created_at >= '2024-01-01'
    GROUP BY date
  ),
  weekly_summary AS (
    SELECT
      DATE_TRUNC(date, WEEK) AS week_start,
      SUM(order_count) AS weekly_orders,
      SUM(revenue) AS weekly_revenue,
      AVG(revenue) AS avg_daily_revenue
    FROM daily_totals
    GROUP BY week_start
  )
SELECT *
FROM weekly_summary
ORDER BY week_start
```

## STRUCT and ARRAY Handling

### Accessing Nested Fields

```sql
SELECT
  user_id,
  address.city,
  address.state
FROM `project.dataset.users`
WHERE address.country = 'US'
```

### Unnesting Arrays

```sql
SELECT
  order_id,
  item.product_name,
  item.quantity,
  item.price
FROM `project.dataset.orders`,
UNNEST(line_items) AS item
```

### Aggregating into Arrays

```sql
SELECT
  customer_id,
  ARRAY_AGG(STRUCT(product_name, quantity)) AS purchased_items
FROM `project.dataset.order_items`
GROUP BY customer_id
```

## Approximate Aggregations

Use approximate functions for large datasets where exact counts are
not required. These are significantly faster and use less resources.

```sql
SELECT
  category,
  APPROX_COUNT_DISTINCT(user_id) AS approx_unique_users,
  APPROX_QUANTILES(order_value, 100)[OFFSET(50)] AS median_value,
  APPROX_TOP_COUNT(product_name, 10) AS top_products
FROM `project.dataset.events`
GROUP BY category
```

## Partitioned Table Patterns

Always filter on partition columns to reduce data scanned and costs.

### Date-Partitioned Tables

```sql
-- Filter on the partition column to limit data scanned
SELECT *
FROM `project.dataset.events`
WHERE _PARTITIONDATE BETWEEN '2024-01-01' AND '2024-01-31'
```

### Column-Partitioned Tables

```sql
-- Use the partition column in WHERE to prune partitions
SELECT *
FROM `project.dataset.transactions`
WHERE transaction_date BETWEEN '2024-01-01' AND '2024-03-31'
  AND region = 'us-east1'
```

## Date and Timestamp Functions

```sql
SELECT
  CURRENT_TIMESTAMP() AS now,
  DATE_DIFF(end_date, start_date, DAY) AS duration_days,
  DATE_TRUNC(created_at, MONTH) AS month_start,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', created_at) AS formatted,
  EXTRACT(YEAR FROM created_at) AS year
FROM `project.dataset.events`
```

## String Functions

```sql
SELECT
  LOWER(email) AS normalized_email,
  REGEXP_EXTRACT(url, r'://([^/]+)') AS domain,
  SPLIT(tags, ',') AS tag_array,
  CONCAT(first_name, ' ', last_name) AS full_name
FROM `project.dataset.users`
```

## Conditional Aggregation

```sql
SELECT
  DATE(created_at) AS date,
  COUNTIF(status = 'completed') AS completed_count,
  COUNTIF(status = 'failed') AS failed_count,
  SAFE_DIVIDE(
    COUNTIF(status = 'completed'),
    COUNT(*)
  ) AS success_rate
FROM `project.dataset.jobs`
GROUP BY date
```
