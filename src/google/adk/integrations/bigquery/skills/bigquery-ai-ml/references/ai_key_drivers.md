# BigQuery AI.Key_Drivers

`AI.KEY_DRIVERS` automatically identifies the key dimensional segments most
responsible for driving changes in a specified metric between a defined interest
group and a reference group.

## Syntax Reference

```sql
SELECT
  *
FROM
  AI.KEY_DRIVERS(
    { TABLE TABLE | (QUERY_STATEMENT) },
    metric_col => 'METRIC_COL',
    dimension_cols => DIMENSION_COLS,
    interest_label_col => 'INTEREST_LABEL_COL'
    [, min_apriori_support => MIN_APRIORI_SUPPORT]
    [, top_k => TOP_K]
    [, enable_pruning => ENABLE_PRUNING]
  )
```

### Input Arguments

| Argument                  | Requirement  | Type            | Description                |
| :------------------------ | :----------- | :-------------- | :------------------------- |
| **`input_data`**          | **Required** |                 | The source table or        |
:                           :              :                 : subquery containing the    :
:                           :              :                 : data to analyze.           :
| **`metric_col`**          | **Required** | `STRING`        | Metric column name. Must   |
:                           :              :                 : be of type\: INT64,        :
:                           :              :                 : NUMERIC, BIGNUMERIC, or    :
:                           :              :                 : FLOAT64.                   :
| **`interest_label_col`**  | **Required** | `STRING`        | Boolean column name:       |
:                           :              :                 : `TRUE` for interest group, :
:                           :              :                 : `FALSE` for reference      :
:                           :              :                 : group.                     :
| **`dimension_cols`**      | **Required** | `ARRAY<STRING>` | 1-12 dimension columns     |
:                           :              :                 : (INT64, BOOL, STRING);     :
:                           :              :                 : cannot be `metric_col` or  :
:                           :              :                 : `interest_label_col`.      :
| **`min_apriori_support`** | Optional     | `FLOAT64`       | Minimum apriori support    |
:                           :              :                 : threshold [0, 1] for       :
:                           :              :                 : output segments. Default\: :
:                           :              :                 : 0.1. Cannot be used with   :
:                           :              :                 : `top_k`.                   :
| **`top_k`**               | Optional     | `INT64`         | Return top k insights [1,  |
:                           :              :                 : 1M] by apriori support. If :
:                           :              :                 : unset, uses                :
:                           :              :                 : `min_apriori_support=0.1`. :
:                           :              :                 : Cannot be used with        :
:                           :              :                 : `min_apriori_support`.     :
| **`enable_pruning`**      | Optional     | `BOOL`          | If `TRUE` (default),       |
:                           :              :                 : redundant insights are     :
:                           :              :                 : pruned. If `FALSE`, all    :
:                           :              :                 : insights meeting           :
:                           :              :                 : thresholds are returned.   :
:                           :              :                 : Two segments are redundant :
:                           :              :                 : if two conditions are      :
:                           :              :                 : met\: 1) their metric      :
:                           :              :                 : values are equal 2) The    :
:                           :              :                 : dimensions and             :
:                           :              :                 : corresponding values of    :
:                           :              :                 : one row are a subset of    :
:                           :              :                 : the dimensions and         :
:                           :              :                 : corresponding values of    :
:                           :              :                 : the other. In this case,   :
:                           :              :                 : the row with more          :
:                           :              :                 : dimensions (the more       :
:                           :              :                 : descriptive row) is kept.  :

### Output Schema

Returns a `STRUCT` with the following fields:

Column Name                          | Type            | Description
:----------------------------------- | :-------------- | :----------
**`drivers`**                        | `ARRAY<STRING>` | Provides a list of drivers, or dimension values of interest, which describes each of the segments.
**`metric_interest`**                | `NUMERIC`       | The sum of the metric_column for the data in the interest segment.
**`metric_reference`**               | `NUMERIC`       | The sum of the metric_column for data in the reference segment.
**`difference`**                     | `NUMERIC`       | The difference between the interest and reference metric values for a segment.
**`relative_difference`**            | `NUMERIC`       | The relative change of a segment, calculated as the difference divided by the reference metric value.
**`unexpected_difference`**          | `NUMERIC`       | Measures deviation of segment from the rest of the population's growth. Calculated as: (segment relative_difference - complement relative_difference) * segment reference metric.
**`relative_unexpected_difference`** | `NUMERIC`       | The unexpected_difference divided by the expected interest metric value for a segment.
**`contribution`**                   | `NUMERIC`       | Contains the absolute value of the difference value: `ABS(difference)`.
**`apriori_support`**                | `NUMERIC`       | Segment size relative to the total population (filters small segments).

## Examples

### Identifying Key Drivers in 2024 H2 Liquor Sales

```sql
WITH InputData AS (
  SELECT
    sale_dollars,
    city,
    category_name,
    vendor_name,
    (date > '2024-07-01') AS IS_H2
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE EXTRACT(YEAR FROM DATE) = 2024
)
SELECT *
FROM AI.KEY_DRIVERS(
  TABLE InputData,
  metric_col => 'sale_dollars',
  dimension_cols => ['city', 'vendor_name', 'category_name'],
  interest_label_col => 'IS_H2',
  min_apriori_support => 0
);
```

### Finding Top 5 Key Drivers in Bottle Costs

```sql
SELECT *
FROM AI.KEY_DRIVERS(
  (
    SELECT
      state_bottle_cost,
      city,
      category_name,
      (EXTRACT(MONTH FROM date) >= 7) AS is_h2
    FROM `bigquery-public-data.iowa_liquor_sales.sales`
    WHERE EXTRACT(YEAR FROM date) = 2024
  ),
  metric_col => 'state_bottle_cost',
  dimension_cols => ['city', 'category_name'],
  interest_label_col => 'is_h2',
  top_k => 5
);
```
