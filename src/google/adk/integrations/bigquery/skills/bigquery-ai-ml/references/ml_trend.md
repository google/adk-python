# BigQuery ML.Trend

`ML.TREND` is a time-series decomposition function that extracts the underlying
long-term direction of data, isolating the persistent growth or decline from
seasonal fluctuations and random noise without requiring a pre-trained model.

## Syntax Reference

```sql
SELECT
  *
FROM
  ML.TREND(
    { TABLE `project.dataset.table` | (QUERY_STATEMENT) },
    data_col => 'DATA_COL',
    timestamp_col => 'TIMESTAMP_COL'
    [, id_cols => ID_COLS]
    [, horizon => HORIZON]
    [, smoothing_window_size => SMOOTHING_WINDOW_SIZE]
    [, adjust_step_changes => ADJUST_STEP_CHANGES]
  )
```

### Input Arguments

Argument                    | Requirement  | Type          | Description
:-------------------------- | :----------- | :------------ | :----------
**`input_data`**            | **Required** |               | The source table or query containing historical time series data.
**`data_col`**              | **Required** | String        | The name of the numeric column in `input_data`. Supported types: `INT64`, `FLOAT64`, `NUMERIC`, `BIGNUMERIC`.
**`timestamp_col`**         | **Required** | String        | The name of the date/timestamp column in `input_data`. Supported types: `DATE`, `DATETIME`, `TIMESTAMP`.
**`id_cols`**               | Optional     | Array<String> | The names of the grouping columns for multiple series (e.g., `['store_id']`). Supported types: `STRING`, `INT64`, `ARRAY<STRING>`, `ARRAY<INT64>`.
**`horizon`**               | Optional     | Int64         | Number of future time points to forecast. Default is `0` (historical trend only). Range `[1, 10,000]`.
**`smoothing_window_size`** | Optional     | Int64         | Size of the center moving average smoothing window used to smooth out noise before trend extraction. Default is `5`. Must be positive.
**`adjust_step_changes`**   | Optional     | Bool          | Whether to perform automatic step change detection and adjustment. When true, identifies abrupt level shifts and blends their impact into the surrounding data before extracting the trend, generating a smoother, more continuous result.

### Output Schema

| Column                 | Type       | Description                            |
| :--------------------- | :--------- | :------------------------------------- |
| **`id_cols`**          | (As Input) | Original identifiers for the series.   |
| **`[timestamp_col]`**  | (As Input) | The column name and type match the     |
:                        :            : input column specified in              :
:                        :            : *TIMESTAMP_COL*.                       :
| **`[data_col]`**       | FLOAT64    | The column name matches the input      |
:                        :            : column specified in *DATA_COL*. For    :
:                        :            : 'history' rows, this contains the      :
:                        :            : training data (automatically           :
:                        :            : interpolated to resolve any missing    :
:                        :            : points). For 'forecast' rows, it       :
:                        :            : contains the forecast value.           :
| **`time_series_type`** | STRING     | Label: `'history'` or `'forecast'`.    |
| **`trend`**            | FLOAT64    | The calculated trend component for the |
:                        :            : time point.                            :
| **`status`**           | STRING     | Error messages or empty string on      |
:                        :            : success. A minimum of 3 data points is :
:                        :            : required.                              :

## Examples

### Analyzing Trends in Daily Visits

This example analyzes the daily visit trend from Google Analytics sample data:

```sql
WITH DailyVisits AS (
  SELECT
    PARSE_TIMESTAMP('%Y%m%d', date) AS visit_timestamp,
    SUM(totals.visits) AS total_visits
  FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  GROUP BY
    visit_timestamp
)
SELECT
  *
FROM
  ML.TREND(
    TABLE DailyVisits,
    data_col => 'total_visits',
    timestamp_col => 'visit_timestamp'
  );
```
