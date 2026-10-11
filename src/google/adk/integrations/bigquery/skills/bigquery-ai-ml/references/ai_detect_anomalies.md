# BigQuery AI.Detect_Anomalies

`AI.DETECT_ANOMALIES` uses the pre-trained **TimesFM** model to identify
deviations in time series data without needing to train a custom model.

## Syntax Reference

This function compares a target dataset against a historical dataset to identify
anomalies.

```sql
SELECT *
FROM AI.DETECT_ANOMALIES(
  { TABLE `project.dataset.history_table` | (SELECT * FROM history_query) },
  { TABLE `project.dataset.target_table` | (SELECT * FROM target_query) },
  data_col => 'DATA_COL',
  timestamp_col => 'TIMESTAMP_COL'
  [, model => 'MODEL']
  [, id_cols => ID_COLS]
  [, anomaly_prob_threshold => ANOMALY_PROB_THRESHOLD]
)

```

### Input Arguments

Argument                     | Requirement  | Type          | Description
:--------------------------- | :----------- | :------------ | :----------
**`historical_data`**        | **Required** | Table/Query   | The source table or subquery containing historical data for training context.
**`target_data`**            | **Required** | Table/Query   | The source table or subquery containing data to analyze for anomalies.
**`data_col`**               | **Required** | String        | The numeric column to analyze.
**`timestamp_col`**          | **Required** | String        | The column containing dates/timestamps.
**`id_cols`**                | Optional     | Array<String> | Grouping columns for multiple series (e.g., `['store_id']`).
**`anomaly_prob_threshold`** | Optional     | Float64       | Threshold for anomaly detection (0 to 1). Defaults to 0.95.
**`model`**                  | Optional     | String        | Model version. Defaults to `'TimesFM 2.0'`.

### Output Schema

| Column                           | Type       | Description                  |
| :------------------------------- | :--------- | :--------------------------- |
| **`id_cols`**                    | (As Input) | Original identifiers for the |
:                                  :            : series.                      :
| **`time_series_timestamp`**      | TIMESTAMP  | Timestamp for the analyzed   |
:                                  :            : points.                      :
| **`time_series_data`**           | FLOAT64    | The original data value.     |
| **`is_anomaly`**                 | BOOL       | TRUE if the point is         |
:                                  :            : identified as an anomaly.    :
| **`lower_bound`**                | FLOAT64    | Lower bound of the expected  |
:                                  :            : range.                       :
| **`upper_bound`**                | FLOAT64    | Upper bound of the expected  |
:                                  :            : range.                       :
| **`anomaly_probability`**        | FLOAT64    | Probability that the point   |
:                                  :            : is an anomaly.               :
| **`ai_detect_anomalies_status`** | STRING     | Error messages or empty      |
:                                  :            : string on success. A minimum :
:                                  :            : of 3 data points is          :
:                                  :            : required.                    :

## Examples

### Basic Anomaly Detection

Detect anomalies in daily bike trips for a specific 2-month window based on
prior history.

```sql
WITH bike_trips AS (
  SELECT EXTRACT(DATE FROM starttime) AS date, COUNT(*) AS num_trips
  FROM `bigquery-public-data.new_york.citibike_trips`
  GROUP BY date
)
SELECT *
FROM AI.DETECT_ANOMALIES(
  -- Historical context (Training data equivalent)
  (SELECT * FROM bike_trips WHERE date <= DATE('2016-06-30')),
  -- Target range (Data to inspect for anomalies)
  (SELECT * FROM bike_trips WHERE date BETWEEN '2016-07-01' AND '2016-09-01'),
  data_col => 'num_trips',
  timestamp_col => 'date'
);

```

### Multivariate Detection (Multiple Series)

Use `id_cols` to detect anomalies separately for different user types (e.g.,
Subscriber vs. Customer) in the same query.

```sql
WITH bike_trips AS (
    SELECT
      EXTRACT(DATE FROM starttime) AS date, usertype, gender,
      COUNT(*) AS num_trips
    FROM `bigquery-public-data.new_york.citibike_trips`
    GROUP BY date, usertype, gender
  )
SELECT *
FROM
  AI.DETECT_ANOMALIES(
    # Historical data from a query
    (SELECT * FROM bike_trips WHERE date <= DATE('2016-06-30')),
    # Target data from a query
    (SELECT * FROM bike_trips WHERE date BETWEEN '2016-07-01' AND '2016-09-01'),
    data_col => 'num_trips',
    timestamp_col => 'date',
    id_cols => ['usertype', 'gender'],
    model => "TimesFM 2.5",
    anomaly_prob_threshold => 0.8);

```
