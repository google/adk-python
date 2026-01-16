# BigQuery Window Functions Reference

Complete guide to analytical window functions in BigQuery.

## Window Function Syntax

```sql
function_name(expression) OVER (
  [PARTITION BY partition_expression [, ...]]
  [ORDER BY sort_expression [ASC|DESC] [NULLS {FIRST|LAST}] [, ...]]
  [window_frame_clause]
)
```

## Ranking Functions

### ROW_NUMBER

Assigns unique sequential integers to rows.

```sql
SELECT
  name,
  department,
  salary,
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees;
-- Result: 1, 2, 3, 4, 5... (no ties)
```

### RANK

Assigns rank with gaps for ties.

```sql
SELECT
  name,
  score,
  RANK() OVER (ORDER BY score DESC) AS rank
FROM players;
-- Scores: 100, 95, 95, 90 -> Ranks: 1, 2, 2, 4 (gap at 3)
```

### DENSE_RANK

Assigns rank without gaps for ties.

```sql
SELECT
  name,
  score,
  DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM players;
-- Scores: 100, 95, 95, 90 -> Ranks: 1, 2, 2, 3 (no gap)
```

### NTILE

Divides rows into N buckets.

```sql
SELECT
  customer_id,
  total_spent,
  NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
FROM customers;
-- Assigns 1, 2, 3, or 4 to each row
```

### PERCENT_RANK

Returns percentile rank (0 to 1).

```sql
SELECT
  name,
  salary,
  PERCENT_RANK() OVER (ORDER BY salary) AS percentile
FROM employees;
-- Returns values between 0 and 1
```

### CUME_DIST

Returns cumulative distribution.

```sql
SELECT
  name,
  salary,
  CUME_DIST() OVER (ORDER BY salary) AS cumulative_distribution
FROM employees;
-- Returns fraction of rows <= current row
```

## Navigation Functions

### LAG

Access value from previous row.

```sql
SELECT
  date,
  value,
  LAG(value, 1) OVER (ORDER BY date) AS prev_value,
  LAG(value, 7, 0) OVER (ORDER BY date) AS week_ago_value  -- with default
FROM metrics;
```

### LEAD

Access value from following row.

```sql
SELECT
  date,
  value,
  LEAD(value, 1) OVER (ORDER BY date) AS next_value,
  value - LEAD(value) OVER (ORDER BY date) AS change_to_next
FROM metrics;
```

### FIRST_VALUE

Get first value in window.

```sql
SELECT
  user_id,
  event_time,
  event_type,
  FIRST_VALUE(event_type) OVER (
    PARTITION BY user_id
    ORDER BY event_time
  ) AS first_event
FROM events;
```

### LAST_VALUE

Get last value in window (requires frame specification).

```sql
SELECT
  user_id,
  event_time,
  event_type,
  LAST_VALUE(event_type) OVER (
    PARTITION BY user_id
    ORDER BY event_time
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS last_event
FROM events;
```

### NTH_VALUE

Get Nth value in window.

```sql
SELECT
  user_id,
  NTH_VALUE(product_name, 2) OVER (
    PARTITION BY user_id
    ORDER BY purchase_date
  ) AS second_purchase
FROM purchases;
```

## Aggregate Window Functions

All aggregate functions can be used as window functions.

### Running Totals

```sql
SELECT
  date,
  amount,
  SUM(amount) OVER (ORDER BY date) AS running_total,
  COUNT(*) OVER (ORDER BY date) AS running_count,
  AVG(amount) OVER (ORDER BY date) AS running_avg
FROM transactions;
```

### Partition Totals

```sql
SELECT
  department,
  employee,
  salary,
  SUM(salary) OVER (PARTITION BY department) AS dept_total,
  salary / SUM(salary) OVER (PARTITION BY department) AS salary_share
FROM employees;
```

### Moving Averages

```sql
SELECT
  date,
  value,
  -- 7-day moving average
  AVG(value) OVER (
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS ma_7,
  -- Centered moving average
  AVG(value) OVER (
    ORDER BY date
    ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
  ) AS ma_centered_7
FROM daily_metrics;
```

## Window Frame Specifications

### ROWS vs RANGE

```sql
-- ROWS: Physical row offset
SELECT
  date,
  value,
  SUM(value) OVER (
    ORDER BY date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS sum_3_rows
FROM data;

-- RANGE: Logical value range
SELECT
  date,
  value,
  SUM(value) OVER (
    ORDER BY date
    RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND CURRENT ROW
  ) AS sum_3_days
FROM data;
```

### Frame Boundaries

| Boundary | Description |
|----------|-------------|
| `UNBOUNDED PRECEDING` | Start of partition |
| `n PRECEDING` | n rows/range before current |
| `CURRENT ROW` | Current row |
| `n FOLLOWING` | n rows/range after current |
| `UNBOUNDED FOLLOWING` | End of partition |

### Common Frame Patterns

```sql
-- Running total (default with ORDER BY)
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- Entire partition
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

-- 7-day rolling window
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW

-- Centered window
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING

-- Previous row only
ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING

-- Future rows only
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
```

## Named Windows

Define reusable window specifications.

```sql
SELECT
  date,
  value,
  SUM(value) OVER rolling_week AS weekly_sum,
  AVG(value) OVER rolling_week AS weekly_avg,
  MAX(value) OVER rolling_week AS weekly_max
FROM metrics
WINDOW rolling_week AS (
  ORDER BY date
  ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
);
```

## Practical Examples

### Gap and Island Detection

```sql
WITH numbered AS (
  SELECT
    user_id,
    login_date,
    login_date - INTERVAL ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY login_date
    ) DAY AS grp
  FROM logins
)
SELECT
  user_id,
  MIN(login_date) AS streak_start,
  MAX(login_date) AS streak_end,
  COUNT(*) AS streak_days
FROM numbered
GROUP BY user_id, grp
ORDER BY user_id, streak_start;
```

### Session Detection

```sql
WITH events_with_prev AS (
  SELECT
    user_id,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM events
),
session_starts AS (
  SELECT
    *,
    CASE
      WHEN prev_time IS NULL THEN 1
      WHEN TIMESTAMP_DIFF(event_time, prev_time, MINUTE) > 30 THEN 1
      ELSE 0
    END AS is_session_start
  FROM events_with_prev
)
SELECT
  *,
  SUM(is_session_start) OVER (
    PARTITION BY user_id ORDER BY event_time
  ) AS session_id
FROM session_starts;
```

### Top N per Group

```sql
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
  FROM products
)
SELECT * FROM ranked WHERE rn <= 3;
```

### Running Difference

```sql
SELECT
  date,
  value,
  value - LAG(value) OVER (ORDER BY date) AS daily_change,
  (value - LAG(value) OVER (ORDER BY date)) / NULLIF(LAG(value) OVER (ORDER BY date), 0) AS pct_change
FROM daily_metrics;
```

### Cumulative Distribution

```sql
SELECT
  product_id,
  revenue,
  SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue,
  SUM(revenue) OVER (ORDER BY revenue DESC) / SUM(revenue) OVER () AS cumulative_pct
FROM products;
```

## Performance Tips

1. **Minimize partitions**: Large partitions require more memory
2. **Use bounded frames**: Avoid UNBOUNDED when possible
3. **Pre-filter data**: Apply WHERE before window functions
4. **Index considerations**: ORDER BY columns benefit from clustering
5. **Avoid unnecessary ORDER BY**: Only include when needed
