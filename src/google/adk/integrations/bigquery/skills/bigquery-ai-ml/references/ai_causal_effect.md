# BigQuery AI.CAUSAL_EFFECT

`AI.CAUSAL_EFFECT` quantifies the impact of specific interventions on time
series data by comparing observed post-intervention values against an
`ARIMA_PLUS` counterfactual baseline. It returns aggregated effect statistics or
detailed pointwise time series evaluations.

## Syntax Reference

```sql
SELECT
  *
FROM
  AI.CAUSAL_EFFECT(
    { TABLE TABLE | (QUERY_STATEMENT) },
    data_col => 'DATA_COL',
    timestamp_col => 'TIMESTAMP_COL',
    intervention_timestamp => INTERVENTION_TIMESTAMP
    [, id_cols => ID_COLS]
    [, confidence_level => CONFIDENCE_LEVEL]
    [, output_time_series => OUTPUT_TIME_SERIES]
    [, num_post_intervention_points => NUM_POST_INTERVENTION_POINTS]
  )
```

### Input Arguments

| Argument                           | Requirement  | Type          | Description              |
| :--------------------------------- | :----------- | :------------ | :----------------------- |
| **`input_data`**                   | **Required** | Table / Query | The source table or      |
:                                    :              :               : GoogleSQL query          :
:                                    :              :               : containing time series   :
:                                    :              :               : data from before and     :
:                                    :              :               : after the intervention.  :
| **`data_col`**                     | **Required** | String        | The name of the column   |
:                                    :              :               : with time series values  :
:                                    :              :               : to analyze. The target   :
:                                    :              :               : column must be of type   :
:                                    :              :               : `INT64`, `NUMERIC`,      :
:                                    :              :               : `BIGNUMERIC`, or         :
:                                    :              :               : `FLOAT64`.               :
| **`timestamp_col`**                | **Required** | String        | The name of the column   |
:                                    :              :               : that contains the        :
:                                    :              :               : timestamps for the time  :
:                                    :              :               : series. The target       :
:                                    :              :               : column must be of type   :
:                                    :              :               : `TIMESTAMP`, `DATE`, or  :
:                                    :              :               : `DATETIME`.              :
| **`intervention_timestamp`**       | **Required** | Timestamp /   | A timestamp value that   |
:                                    :              : Date          : indicates when the       :
:                                    :              :               : intervention occurred,   :
:                                    :              :               : dividing the data into   :
:                                    :              :               : pre-intervention and     :
:                                    :              :               : post-intervention        :
:                                    :              :               : periods.                 :
| **`id_cols`**                      | Optional     | Array<String> | The names of columns     |
:                                    :              :               : that identify individual :
:                                    :              :               : time series for parallel :
:                                    :              :               : processing. The target   :
:                                    :              :               : columns must be of type  :
:                                    :              :               : `STRING` or `INT64`.     :
| **`confidence_level`**             | Optional     | Float64       | A `FLOAT64` value in the |
:                                    :              :               : range `[0, 1)`           :
:                                    :              :               : specifying the           :
:                                    :              :               : percentage of future     :
:                                    :              :               : values expected to fall  :
:                                    :              :               : within the prediction    :
:                                    :              :               : interval (default\:      :
:                                    :              :               : `0.95`).                 :
| **`output_time_series`**           | Optional     | Bool          | Determines the level of  |
:                                    :              :               : detail in the output.    :
:                                    :              :               : `FALSE` (default)        :
:                                    :              :               : returns a summary view;  :
:                                    :              :               : `TRUE` returns the       :
:                                    :              :               : detailed time series     :
:                                    :              :               : view including all       :
:                                    :              :               : pointwise data.          :
| **`num_post_intervention_points`** | Optional     | Int64         | Specifies the number of  |
:                                    :              :               : time series points after :
:                                    :              :               : the                      :
:                                    :              :               : `intervention_timestamp` :
:                                    :              :               : to include in the causal :
:                                    :              :               : effect analysis. If you  :
:                                    :              :               : don't specify a value,   :
:                                    :              :               : the analysis includes    :
:                                    :              :               : all data points from the :
:                                    :              :               : `intervention_timestamp` :
:                                    :              :               : to the end of the time   :
:                                    :              :               : series.                  :

### Output Schema

The output includes all columns specified in the `id_cols` argument in addition
to the following columns:

#### Summary View (`output_time_series => FALSE`, default)

| Column Name              | Type      | Description                          |
| :----------------------- | :-------- | :----------------------------------- |
| **`p_value`**            | `FLOAT64` | Two-tailed p-value for the null      |
:                          :           : hypothesis across the entire         :
:                          :           : post-intervention period.            :
| **`prob_causal_effect`** | `FLOAT64` | Probability of a causal effect,      |
:                          :           : calculated as `(1 - p_value)`.       :
| **`absolute_effect`**    | `FLOAT64` | Total cumulative difference between  |
:                          :           : observed and predicted values        :
:                          :           : post-intervention\:                  :
:                          :           : `SUM(actual_value -                  :
:                          :           : expected_value)`.                    :
| **`relative_effect`**    | `FLOAT64` | Relative change of observed values:  |
:                          :           : `SUM(actual_value -                  :
:                          :           : expected_value)/SUM(expected_value)` :
:                          :           : across the post-intervention period. :
| **`status`**             | `STRING`  | Forecast status; empty string on     |
:                          :           : success, or an error string.         :

#### Detailed Pointwise View (`output_time_series => TRUE`)

Includes the summary statistics columns above, repeated across every row, along
with:

| Column Name                | Type        | Description               |
| :------------------------- | :---------- | :------------------------ |
| **`<timestamp_col name>`** | `TIMESTAMP` | Timestamp of the data     |
:                            :             : point from the            :
:                            :             : `timestamp_col` input     :
:                            :             : (both pre- and            :
:                            :             : post-intervention).       :
| **`is_post_intervention`** | `BOOL`      | `TRUE` for timestamps     |
:                            :             : greater than or equal to  :
:                            :             : `intervention_timestamp`, :
:                            :             : `FALSE` for timestamps    :
:                            :             : prior to the              :
:                            :             : intervention.             :
| **`<data_col name>`**      | `FLOAT64`   | Observed value from       |
:                            :             : `data_col` at the         :
:                            :             : specified timestamp.      :
| **`predicted_<data_col     | `FLOAT64`   | Forecasted                |
: name>`**                   :             : (counterfactual) value at :
:                            :             : the specified timestamp   :
:                            :             : (`NULL` for               :
:                            :             : pre-intervention          :
:                            :             : timestamps).              :
| **`lower_bound`**          | `FLOAT64`   | Lower bound of the        |
:                            :             : prediction result (`NULL` :
:                            :             : for pre-intervention      :
:                            :             : timestamps).              :
| **`upper_bound`**          | `FLOAT64`   | Upper bound of the        |
:                            :             : prediction result (`NULL` :
:                            :             : for pre-intervention      :
:                            :             : timestamps).              :

## Limitations

-   A minimum of 3 historical data points in the pre-intervention period is
    required to generate a forecast.

## Examples

### Impact of the COVID-19 Pandemic on NYC Taxi Trips (Detailed Pointwise View)

```sql
SELECT * FROM AI.CAUSAL_EFFECT(
  (
    SELECT
      DATE(pickup_datetime) AS pickup_date,
      COUNT(*) AS trip_count
    FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2020`
    WHERE EXTRACT(YEAR FROM pickup_datetime) = 2020
    GROUP BY pickup_date
  ),
  data_col => 'trip_count',
  timestamp_col => 'pickup_date',
  intervention_timestamp => '2020-03-11', -- WHO declares COVID-19 a pandemic
  num_post_intervention_points => 120,
  output_time_series => TRUE
);
```
