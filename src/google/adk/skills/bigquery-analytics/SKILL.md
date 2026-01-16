---
name: bigquery-analytics
description: Execute advanced SQL analytics in BigQuery - window functions, aggregations, geospatial analysis, statistical functions, and BI integrations. Use when performing data analysis, building dashboards, or running complex SQL queries.
license: Apache-2.0
compatibility: BigQuery, Looker, Data Studio
metadata:
  author: Google Cloud
  version: "1.0"
  category: analytics
adk:
  config:
    timeout_seconds: 600
    max_parallel_calls: 10
  allowed_callers:
    - bigquery_agent
    - analytics_agent
    - bi_agent
---

# BigQuery Analytics Skill

Execute advanced SQL analytics in BigQuery. This skill covers window functions, aggregations, geospatial analysis, statistical functions, and BI tool integrations.

## When to Use This Skill

Use this skill when you need to:
- Write complex analytical SQL queries
- Use window functions for rankings, running totals, and time-series analysis
- Perform geospatial analysis on location data
- Calculate statistical metrics and distributions
- Build data for dashboards and BI tools
- Optimize query performance for analytics workloads

**Note**: For ML model training, use the `bqml` skill. For AI/text generation, use the `bigquery-ai` skill.

## SQL Analytics Functions

| Category | Functions | Use Cases |
|----------|-----------|-----------|
| **Aggregation** | SUM, AVG, COUNT, MIN, MAX | Basic metrics |
| **Window** | ROW_NUMBER, RANK, LAG, LEAD | Rankings, time series |
| **Statistical** | STDDEV, VARIANCE, CORR, PERCENTILE | Data distribution |
| **Geospatial** | ST_DISTANCE, ST_CONTAINS, ST_AREA | Location analysis |
| **Approximate** | APPROX_COUNT_DISTINCT, APPROX_QUANTILES | Large-scale estimates |

## Quick Start

### 1. Window Functions for Rankings

```sql
SELECT
  product_name,
  category,
  revenue,
  RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS category_rank,
  revenue / SUM(revenue) OVER (PARTITION BY category) AS category_share
FROM `project.dataset.sales`
WHERE sale_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);
```

### 2. Time Series Analysis

```sql
SELECT
  sale_date,
  daily_revenue,
  AVG(daily_revenue) OVER (
    ORDER BY sale_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7day_avg,
  daily_revenue - LAG(daily_revenue, 7) OVER (ORDER BY sale_date) AS wow_change
FROM (
  SELECT DATE(sale_time) AS sale_date, SUM(amount) AS daily_revenue
  FROM `project.dataset.transactions`
  GROUP BY 1
);
```

### 3. Geospatial Query

```sql
SELECT
  store_name,
  ST_DISTANCE(store_location, ST_GEOGPOINT(-122.4194, 37.7749)) / 1000 AS distance_km
FROM `project.dataset.stores`
WHERE ST_DWITHIN(store_location, ST_GEOGPOINT(-122.4194, 37.7749), 10000)
ORDER BY distance_km;
```

## Window Functions

### Ranking Functions

```sql
SELECT
  employee_id,
  department,
  salary,
  -- Dense rank (no gaps)
  DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rank,
  -- Rank (with gaps for ties)
  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank,
  -- Row number (unique)
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
  -- Percentile rank
  PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS percentile
FROM `project.dataset.employees`;
```

### Navigation Functions

```sql
SELECT
  user_id,
  event_time,
  event_type,
  -- Previous event
  LAG(event_type) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event,
  -- Next event
  LEAD(event_type) OVER (PARTITION BY user_id ORDER BY event_time) AS next_event,
  -- First event in session
  FIRST_VALUE(event_type) OVER (
    PARTITION BY user_id ORDER BY event_time
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS first_event,
  -- Time since last event
  TIMESTAMP_DIFF(
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
    SECOND
  ) AS seconds_since_last
FROM `project.dataset.events`;
```

### Running Aggregates

```sql
SELECT
  transaction_date,
  amount,
  -- Running total
  SUM(amount) OVER (ORDER BY transaction_date) AS running_total,
  -- Running average
  AVG(amount) OVER (ORDER BY transaction_date) AS running_avg,
  -- Running count
  COUNT(*) OVER (ORDER BY transaction_date) AS running_count,
  -- 30-day moving average
  AVG(amount) OVER (
    ORDER BY transaction_date
    RANGE BETWEEN INTERVAL 29 DAY PRECEDING AND CURRENT ROW
  ) AS moving_avg_30d
FROM `project.dataset.transactions`;
```

### Frame Specifications

```sql
-- ROWS vs RANGE
SELECT
  date,
  value,
  -- ROWS: exact number of rows
  AVG(value) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS rows_avg,
  -- RANGE: by value range (for dates/timestamps)
  AVG(value) OVER (
    ORDER BY date
    RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND INTERVAL 2 DAY FOLLOWING
  ) AS range_avg
FROM table;

-- Common frame patterns
-- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- Running total
-- ROWS BETWEEN 6 PRECEDING AND CURRENT ROW          -- 7-day window
-- ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING          -- 3-point smoothing
-- ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- Entire partition
```

## Aggregation Functions

### Basic Aggregations

```sql
SELECT
  category,
  COUNT(*) AS total_count,
  COUNT(DISTINCT customer_id) AS unique_customers,
  SUM(amount) AS total_amount,
  AVG(amount) AS avg_amount,
  MIN(amount) AS min_amount,
  MAX(amount) AS max_amount,
  -- Conditional aggregation
  COUNTIF(amount > 100) AS high_value_count,
  SUMIF(amount, status = 'completed') AS completed_amount
FROM `project.dataset.orders`
GROUP BY category;
```

### GROUPING SETS, ROLLUP, CUBE

```sql
-- ROLLUP: Hierarchical totals
SELECT
  COALESCE(region, 'ALL REGIONS') AS region,
  COALESCE(product_category, 'ALL CATEGORIES') AS category,
  SUM(revenue) AS total_revenue
FROM `project.dataset.sales`
GROUP BY ROLLUP(region, product_category);

-- CUBE: All combinations
SELECT
  COALESCE(region, 'ALL') AS region,
  COALESCE(year, 'ALL') AS year,
  SUM(revenue) AS total_revenue
FROM `project.dataset.sales`
GROUP BY CUBE(region, year);

-- GROUPING SETS: Custom combinations
SELECT
  region,
  product_category,
  SUM(revenue) AS total_revenue
FROM `project.dataset.sales`
GROUP BY GROUPING SETS (
  (region, product_category),
  (region),
  (product_category),
  ()
);
```

### Array Aggregations

```sql
SELECT
  user_id,
  -- Collect values into array
  ARRAY_AGG(product_name) AS purchased_products,
  -- Collect distinct values
  ARRAY_AGG(DISTINCT category) AS categories,
  -- Collect ordered values
  ARRAY_AGG(product_name ORDER BY purchase_date DESC LIMIT 5) AS recent_products,
  -- String aggregation
  STRING_AGG(product_name, ', ') AS products_list
FROM `project.dataset.purchases`
GROUP BY user_id;
```

## Statistical Functions

### Descriptive Statistics

```sql
SELECT
  category,
  COUNT(*) AS n,
  AVG(value) AS mean,
  STDDEV(value) AS std_dev,
  VARIANCE(value) AS variance,
  -- Coefficient of variation
  STDDEV(value) / NULLIF(AVG(value), 0) AS cv,
  -- Min/Max
  MIN(value) AS min_val,
  MAX(value) AS max_val,
  -- Percentiles
  APPROX_QUANTILES(value, 4)[OFFSET(2)] AS median,
  APPROX_QUANTILES(value, 100)[OFFSET(25)] AS p25,
  APPROX_QUANTILES(value, 100)[OFFSET(75)] AS p75
FROM `project.dataset.metrics`
GROUP BY category;
```

### Correlation and Covariance

```sql
SELECT
  CORR(price, quantity) AS price_quantity_corr,
  COVAR_POP(price, quantity) AS covariance_pop,
  COVAR_SAMP(price, quantity) AS covariance_samp
FROM `project.dataset.sales`;

-- Correlation matrix
WITH metrics AS (
  SELECT metric_a, metric_b, metric_c FROM `project.dataset.data`
)
SELECT
  'metric_a' AS metric,
  CORR(metric_a, metric_a) AS corr_a,
  CORR(metric_a, metric_b) AS corr_b,
  CORR(metric_a, metric_c) AS corr_c
FROM metrics
UNION ALL
SELECT
  'metric_b',
  CORR(metric_b, metric_a),
  CORR(metric_b, metric_b),
  CORR(metric_b, metric_c)
FROM metrics;
```

### Distribution Analysis

```sql
-- Histogram buckets
SELECT
  FLOOR(value / 10) * 10 AS bucket,
  COUNT(*) AS frequency,
  REPEAT('*', CAST(COUNT(*) / 100 AS INT64)) AS histogram
FROM `project.dataset.data`
GROUP BY bucket
ORDER BY bucket;

-- Z-scores
WITH stats AS (
  SELECT AVG(value) AS mean, STDDEV(value) AS stddev
  FROM `project.dataset.data`
)
SELECT
  id,
  value,
  (value - mean) / NULLIF(stddev, 0) AS z_score
FROM `project.dataset.data`, stats;
```

## Geospatial Analysis

### Creating Geography Objects

```sql
-- Point from coordinates
SELECT ST_GEOGPOINT(longitude, latitude) AS location
FROM `project.dataset.places`;

-- Well-Known Text (WKT)
SELECT ST_GEOGFROMTEXT('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))') AS polygon;

-- GeoJSON
SELECT ST_GEOGFROMGEOJSON('{"type":"Point","coordinates":[-122.4194,37.7749]}');
```

### Distance and Area

```sql
SELECT
  store_a.name AS store_a,
  store_b.name AS store_b,
  -- Distance in meters
  ST_DISTANCE(store_a.location, store_b.location) AS distance_m,
  -- Distance in kilometers
  ST_DISTANCE(store_a.location, store_b.location) / 1000 AS distance_km
FROM `project.dataset.stores` store_a
CROSS JOIN `project.dataset.stores` store_b
WHERE store_a.id < store_b.id;

-- Area of polygons
SELECT
  region_name,
  ST_AREA(boundary) / 1000000 AS area_sq_km
FROM `project.dataset.regions`;
```

### Spatial Queries

```sql
-- Find points within distance
SELECT store_name
FROM `project.dataset.stores`
WHERE ST_DWITHIN(
  location,
  ST_GEOGPOINT(-122.4194, 37.7749),  -- San Francisco
  5000  -- 5km radius
);

-- Find points within polygon
SELECT customer_id
FROM `project.dataset.customers`
WHERE ST_CONTAINS(
  (SELECT boundary FROM `project.dataset.regions` WHERE name = 'Bay Area'),
  customer_location
);

-- Nearest neighbor
SELECT
  customer_id,
  ARRAY_AGG(
    store_name
    ORDER BY ST_DISTANCE(customer_location, store_location)
    LIMIT 3
  ) AS nearest_stores
FROM `project.dataset.customers`
CROSS JOIN `project.dataset.stores`
GROUP BY customer_id;
```

### Geospatial Joins

```sql
-- Assign customers to regions
SELECT
  c.customer_id,
  r.region_name
FROM `project.dataset.customers` c
JOIN `project.dataset.regions` r
  ON ST_CONTAINS(r.boundary, c.location);

-- Find overlapping areas
SELECT
  a.name AS area_a,
  b.name AS area_b,
  ST_AREA(ST_INTERSECTION(a.boundary, b.boundary)) AS overlap_area
FROM `project.dataset.zones` a
JOIN `project.dataset.zones` b
  ON ST_INTERSECTS(a.boundary, b.boundary)
  AND a.id < b.id;
```

## Approximate Aggregations

For large-scale analytics where exact results aren't required:

```sql
SELECT
  -- Approximate count distinct (HyperLogLog++)
  APPROX_COUNT_DISTINCT(user_id) AS approx_unique_users,
  -- Exact for comparison
  COUNT(DISTINCT user_id) AS exact_unique_users,
  -- Approximate quantiles
  APPROX_QUANTILES(amount, 100)[OFFSET(50)] AS approx_median,
  -- Approximate top count
  APPROX_TOP_COUNT(category, 10) AS top_categories,
  -- Approximate top sum
  APPROX_TOP_SUM(product_name, revenue, 10) AS top_products_by_revenue
FROM `project.dataset.transactions`;
```

## Common Analytical Patterns

### Cohort Analysis

```sql
WITH user_cohorts AS (
  SELECT
    user_id,
    DATE_TRUNC(first_purchase_date, MONTH) AS cohort_month
  FROM `project.dataset.users`
),
monthly_activity AS (
  SELECT
    user_id,
    DATE_TRUNC(activity_date, MONTH) AS activity_month
  FROM `project.dataset.activity`
)
SELECT
  c.cohort_month,
  DATE_DIFF(a.activity_month, c.cohort_month, MONTH) AS months_since_cohort,
  COUNT(DISTINCT a.user_id) AS active_users,
  COUNT(DISTINCT a.user_id) / (
    SELECT COUNT(DISTINCT user_id)
    FROM user_cohorts
    WHERE cohort_month = c.cohort_month
  ) AS retention_rate
FROM user_cohorts c
JOIN monthly_activity a ON c.user_id = a.user_id
GROUP BY 1, 2
ORDER BY 1, 2;
```

### Funnel Analysis

```sql
WITH funnel AS (
  SELECT
    user_id,
    MAX(IF(event_name = 'page_view', 1, 0)) AS step_1_view,
    MAX(IF(event_name = 'add_to_cart', 1, 0)) AS step_2_cart,
    MAX(IF(event_name = 'checkout', 1, 0)) AS step_3_checkout,
    MAX(IF(event_name = 'purchase', 1, 0)) AS step_4_purchase
  FROM `project.dataset.events`
  WHERE event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY user_id
)
SELECT
  COUNT(*) AS total_users,
  SUM(step_1_view) AS viewed,
  SUM(step_2_cart) AS added_to_cart,
  SUM(step_3_checkout) AS checked_out,
  SUM(step_4_purchase) AS purchased,
  SAFE_DIVIDE(SUM(step_2_cart), SUM(step_1_view)) AS view_to_cart_rate,
  SAFE_DIVIDE(SUM(step_4_purchase), SUM(step_1_view)) AS conversion_rate
FROM funnel;
```

### Year-over-Year Comparison

```sql
SELECT
  FORMAT_DATE('%Y-%m', date) AS month,
  SUM(revenue) AS revenue,
  SUM(IF(EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE()), revenue, 0)) AS current_year,
  SUM(IF(EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE()) - 1, revenue, 0)) AS prior_year,
  SAFE_DIVIDE(
    SUM(IF(EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE()), revenue, 0)),
    SUM(IF(EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE()) - 1, revenue, 0))
  ) - 1 AS yoy_growth
FROM `project.dataset.sales`
GROUP BY 1
ORDER BY 1;
```

## Query Optimization

### Best Practices

1. **Filter early**: Apply WHERE clauses as early as possible
2. **Select only needed columns**: Avoid SELECT *
3. **Use approximate functions**: For large-scale analytics
4. **Partition pruning**: Always filter on partition column
5. **Avoid CROSS JOINs**: Unless necessary

### Analyzing Query Performance

```sql
-- Check bytes processed
SELECT @@project_id;

-- Use EXPLAIN to understand query plan
EXPLAIN
SELECT * FROM `project.dataset.table`
WHERE date_column = CURRENT_DATE();
```

## References

Load detailed documentation as needed:

- `WINDOW_FUNCTIONS.md` - Complete window function reference
- `GEOSPATIAL.md` - Advanced geospatial operations
- `OPTIMIZATION.md` - Query performance tuning

## Scripts

Helper scripts for common operations:

- `query_analyzer.py` - Analyze query performance
- `data_profiler.py` - Generate data profiling reports

## Limitations

- Window functions process all rows before returning
- Geospatial functions have precision limits
- Approximate functions have error margins
- Large CROSS JOINs can be expensive
