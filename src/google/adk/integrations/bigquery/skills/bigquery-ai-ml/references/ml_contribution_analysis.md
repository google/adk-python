# Contribution Analysis

Contribution analysis, also called key driver analysis, is a method used to
generate insights about changes to key metrics in your multi-dimensional data.

## Syntax

To use contribution analysis in BigQuery ML, create a contribution analysis
model with the CREATE MODEL statement.

After you have created a contribution analysis model, you can use the
ML.GET_INSIGHTS function to retrieve the metric information calculated by the
model.

### CREATE MODEL Syntax

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS (
  MODEL_TYPE = 'CONTRIBUTION_ANALYSIS',
  CONTRIBUTION_METRIC = 'contribution_metric',
  IS_TEST_COL = 'is_test_col'
  [, DIMENSION_ID_COLS = dimension_column_array]
  [, MIN_APRIORI_SUPPORT = min_apriori_support]
  [, TOP_K_INSIGHTS_BY_APRIORI_SUPPORT = top_k_insights_by_apriori_support]
  [, PRUNING_METHOD = {'NO_PRUNING', 'PRUNE_REDUNDANT_INSIGHTS'}]
)
AS query_statement;
```

### Options
| Option | Requirement | Type | Description |
| :--- | :--- | :--- | :--- |
| **`MODEL_TYPE`** | **Required** | String | Must be `'CONTRIBUTION_ANALYSIS'`. |
| **`CONTRIBUTION_METRIC`** | **Required** | String | The metric to analyze. Supported metric is `SUM(metric_column_name)`. |
| **`IS_TEST_COL`** | **Required** | String | The name of the boolean column that distinguishes test data (TRUE) from control data (FALSE). |
| **`DIMENSION_ID_COLS`** | Optional | Array<String> | The columns to use as dimensions for segmenting the data. Defaults to all columns in the input data that aren't specified in other options. Restricted to columns of type `INT64`, `BOOL`, or `STRING`. Limited to a max of 50 columns.|
| **`MIN_APRIORI_SUPPORT`** | Optional | Float64 | Minimum support threshold to prune small segments. Cannot be used with `TOP_K_INSIGHTS_BY_APRIORI_SUPPORT`. Defaults to 0.1 if neither is specified. The value is in the range of [0, 1]. |
| **`TOP_K_INSIGHTS_BY_APRIORI_SUPPORT`** | Optional | Int64 | The maximum number of insights to output with the highest apriori support value. Cannot be used with `MIN_APRIORI_SUPPORT`. This value should be within the interval [1, 1,000,000]|
| **`PRUNING_METHOD`** | Optional | String | The pruning method used to filter insights. Supported: `'NO_PRUNING'`, `'PRUNE_REDUNDANT_INSIGHTS'`. Defaults to `'NO_PRUNING'`. `PRUNE_REDUNDANT_INSIGHTS` prunes redundant segments, excluding the `all` segment. Two segments are redundant if two conditions are met: 1) their metric values are equal 2) The dimension IDs and corresponding values of one row are a subset of the dimension IDs and corresponding values of the other. In this case, the row with more dimension IDs (the more descriptive row) is kept.|

---

## ML.GET_INSIGHTS Syntax

Use `ML.GET_INSIGHTS` to retrieve the results from the model:

```sql
ML.GET_INSIGHTS(
  MODEL `PROJECT_ID.DATASET.MODEL_NAME`
)
```

### Output Schema
| Column | Description |
| :--- | :--- |
| **`contributors`** | The dimension values for a given segment (e.g., `['dim1=val1', 'dim2=val2']`). |
| **`metric_test`** | A numeric value that contains the sum of the value of the metric column in the test dataset for the given segment. |
| **`metric_control`** | A numeric value that contains the sum of the value of the metric column in the control dataset for the given segment. |
| **`difference`** | A numeric value that contains the difference between the metric_test and metric_control values: `metric_test - metric_control`. |
| **`relative_difference`** | A numeric value that contains the relative change in the segment value between the test and control datasets. |
| **`unexpected_difference`** | A numeric value that contains the unexpected difference between the segment's actual metric_test value and the segment's expected metric_test value, which is determined by comparing the ratio of change for this segment against the complement ratio of change. |
| **`relative_unexpected_difference`** | A numeric value that contains the ratio between the unexpected_difference value and the expected_metric_test value. |
| **`apriori_support`** | The proportion of the total metric in the test and control sets that this segment represents. |
| **`contribution`** | A numeric value that contains the absolute value of the difference value. |
---

## Example: Analyzing Revenue Change

```sql
-- 1. Create the model
CREATE OR REPLACE MODEL `my_project.my_dataset.revenue_analysis`
OPTIONS (
  MODEL_TYPE = 'CONTRIBUTION_ANALYSIS',
  is_test_col = 'is_current_year',
  dimension_id_cols = ['region', 'product_category', 'customer_segment'],
  contribution_metric = 'SUM(revenue)'
) AS
SELECT
  region,
  product_category,
  customer_segment,
  revenue,
  IF(date >= '2024-01-01', TRUE, FALSE) AS is_current_year
FROM
  `my_project.my_dataset.sales_data`;

-- 2. Get insights
SELECT
  *
FROM
  ML.GET_INSIGHTS(MODEL `my_project.my_dataset.revenue_analysis`)
ORDER BY
  unexpected_difference DESC;
```
