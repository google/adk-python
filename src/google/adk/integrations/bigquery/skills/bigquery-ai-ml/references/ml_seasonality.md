# BigQuery ML.Seasonality

`ML.SEASONALITY` is a time-series decomposition function that extracts seasonal
components to decouple recurring patterns from the underlying trend and noise.

## Syntax Reference

```sql
SELECT
  *
FROM
  ML.SEASONALITY(
    { TABLE `project.dataset.table` | (QUERY_STATEMENT) },
    data_col => 'DATA_COL',
    timestamp_col => 'TIMESTAMP_COL'
    [, id_cols => ID_COLS]
    [, seasonalities => SEASONALITIES]
    [, horizon => HORIZON]
  )
```

### Input Arguments

| Argument            | Requirement  | Type          | Description             |
| :------------------ | :----------- | :------------ | :---------------------- |
| **`input_data`**    | **Required** |               | The source table or     |
:                     :              :               : query containing        :
:                     :              :               : historical time series  :
:                     :              :               : data.                   :
| **`data_col`**      | **Required** | String        | The name of the numeric |
:                     :              :               : column in `input_data`. :
:                     :              :               : Supported types\:       :
:                     :              :               : `INT64`, `FLOAT64`,     :
:                     :              :               : `NUMERIC`,              :
:                     :              :               : `BIGNUMERIC`.           :
| **`timestamp_col`** | **Required** | String        | The name of the         |
:                     :              :               : date/timestamp column   :
:                     :              :               : in `input_data`.        :
:                     :              :               : Supported types\:       :
:                     :              :               : `DATE`, `DATETIME`,     :
:                     :              :               : `TIMESTAMP`.            :
| **`id_cols`**       | Optional     | Array<String> | The names of the        |
:                     :              :               : grouping columns for    :
:                     :              :               : multiple series (e.g.,  :
:                     :              :               : `['store_id']`).        :
:                     :              :               : Supported types\:       :
:                     :              :               : `STRING`, `INT64`,      :
:                     :              :               : `ARRAY<STRING>`,        :
:                     :              :               : `ARRAY<INT64>`.         :
| **`seasonalities`** | Optional     | Array<String> | Seasonality types to    |
:                     :              :               : extract. Valid values\: :
:                     :              :               : `Yearly`, `Quarterly`,  :
:                     :              :               : `Monthly`, `Weekly`,    :
:                     :              :               : `Daily`. If omitted,    :
:                     :              :               : the function            :
:                     :              :               : automatically detects   :
:                     :              :               : all seasonalities.      :
| **`horizon`**       | Optional     | Int64         | Number of future time   |
:                     :              :               : points to forecast.     :
:                     :              :               : Default is `0` (history :
:                     :              :               : only). Range `[1,       :
:                     :              :               : 10,000]`.               :

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
| **`yearly`**           | FLOAT64    | Calculated yearly seasonal component.  |
:                        :            : A NULL value means no yearly           :
:                        :            : seasonality detected.                  :
| **`quarterly`**        | FLOAT64    | Calculated quarterly seasonal          |
:                        :            : component. A NULL value means no       :
:                        :            : quarterly seasonality detected.        :
| **`monthly`**          | FLOAT64    | Calculated monthly seasonal component. |
:                        :            : A NULL value means no monthly          :
:                        :            : seasonality detected.                  :
| **`weekly`**           | FLOAT64    | Calculated weekly seasonal component.  |
:                        :            : A NULL value means no weekly           :
:                        :            : seasonality detected.                  :
| **`daily`**            | FLOAT64    | Calculated daily seasonal component. A |
:                        :            : NULL value means no daily seasonality  :
:                        :            : detected.                              :
| **`status`**           | STRING     | Error messages or empty string on      |
:                        :            : success. A minimum of 3 data points is :
:                        :            : required.                              :

## Examples

### Analyzing Seasonality in Daily Visits

This example analyzes the daily website visits from Google Analytics session
data:

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
  ML.SEASONALITY(
    TABLE DailyVisits,
    data_col => 'total_visits',
    timestamp_col => 'visit_timestamp'
  );
```
