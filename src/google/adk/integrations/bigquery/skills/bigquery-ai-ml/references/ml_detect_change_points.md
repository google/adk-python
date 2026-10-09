# BigQuery ML.Detect_Change_Points

`ML.DETECT_CHANGE_POINTS` is a one-step Table-Valued Function (TVF) that
identifies intervals where the statistical behavior of time series data have
shifted. It distinguishes structural breaks (e.g. "slow bleed" anomalies,
sustained structural shifts) from transient or isolated events. All identified
change points are represented as time windows defined by both a beginning and an
ending timestamp.

## Syntax Reference

```sql
SELECT
  *
FROM
  ML.DETECT_CHANGE_POINTS(
    { TABLE `project.dataset.table` | (QUERY_STATEMENT) },
    data_col => 'DATA_COL',
    timestamp_col => 'TIMESTAMP_COL'
    [, id_cols => ID_COLS]
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
:                     :              :               : column in `input_data`  :
:                     :              :               : to analyze. The data    :
:                     :              :               : column must use one of  :
:                     :              :               : the following data      :
:                     :              :               : types\: `INT64`,        :
:                     :              :               : `FLOAT64`, `NUMERIC`,   :
:                     :              :               : `BIGNUMERIC`.           :
| **`timestamp_col`** | **Required** | String        | The name of the         |
:                     :              :               : date/timestamp column   :
:                     :              :               : in `input_data`. The    :
:                     :              :               : timestamp column must   :
:                     :              :               : use one of the          :
:                     :              :               : following data types\:  :
:                     :              :               : `DATE`, `DATETIME`,     :
:                     :              :               : `TIMESTAMP`.            :
| **`id_cols`**       | Optional     | Array<String> | The names of the        |
:                     :              :               : grouping columns for    :
:                     :              :               : multiple time series    :
:                     :              :               : (e.g., `['store_id']`). :
:                     :              :               : The columns that you    :
:                     :              :               : specify must use one of :
:                     :              :               : the following data      :
:                     :              :               : types\: `STRING`,       :
:                     :              :               : `INT64`,                :
:                     :              :               : `ARRAY<STRING>`,        :
:                     :              :               : `ARRAY<INT64>`.         :

### Output Schema

| Column                | Type       | Description                             |
| :-------------------- | :--------- | :-------------------------------------- |
| **`id_cols`**         | (As Input) | Original identifiers for the time       |
:                       :            : series.                                 :
| **`begin_timestamp`** | TIMESTAMP  | Timestamp corresponding to the start of |
:                       :            : the change point.                       :
| **`end_timestamp`**   | TIMESTAMP  | Timestamp corresponding to the end of   |
:                       :            : the change point.                       :
| **`metrics`**         | STRUCT     | A set of metrics describing the         |
:                       :            : statistical behavior.                   :
| **`metrics.avg`**     | FLOAT64    | Calculated average metric value within  |
:                       :            : this change point.                      :
| **`metrics.min`**     | FLOAT64    | Minimum metric value observed.          |
| **`metrics.max`**     | FLOAT64    | Maximum metric value observed.          |
| **`metrics.stddev`**  | FLOAT64    | Standard deviation within this change   |
:                       :            : point.                                  :
| **`metrics.count`**   | INT64      | Total count of data points in the       |
:                       :            : identified change point.                :
| **`status`**          | STRING     | Error messages or empty string on       |
:                       :            : success.                                :

## Examples

### Detecting Change Points in Taxi Trips

This example detects structural shifts in the number of New York taxi trips
spanning 2019 to 2020:

```sql
WITH daily_trips AS (
  SELECT
    EXTRACT(DATE FROM pickup_datetime) AS trip_date,
    COUNT(*) AS total_trips
  FROM
    `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_20*`
  WHERE
    _TABLE_SUFFIX BETWEEN '19' AND '20'
    AND EXTRACT(DATE FROM pickup_datetime) BETWEEN '2019-01-01' AND '2020-12-31'
  GROUP BY
    trip_date
)
SELECT
  *
FROM
  ML.DETECT_CHANGE_POINTS(
    TABLE daily_trips,
    data_col => 'total_trips',
    timestamp_col => 'trip_date'
  );
```
