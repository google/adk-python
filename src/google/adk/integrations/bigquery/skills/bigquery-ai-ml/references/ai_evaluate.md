# AI.EVALUATE
Evaluates the accuracy of a forecasting model by comparing predicted values
against actual historical data. It computes standard regression metrics like
MAE, MSE, and RMSE.

## Syntax
```sql
SELECT
  *
FROM
  AI.EVALUATE(
    { TABLE `project.dataset.history_table` | (HISTORY_QUERY) },
    { TABLE `project.dataset.actual_table` | (ACTUAL_QUERY) },
    data_col => 'DATA_COL',
    timestamp_col => 'TIMESTAMP_COL'
    [, model => 'MODEL']
    [, id_cols => ID_COLS]
    [, horizon => HORIZON]
  )
```

## Input Arguments
| Argument | Requirement | Type | Description |
| :--- | :--- | :--- | :--- |
| **`history_data`** | **Required** | | The table or query containing historical data used for the forecast. |
| **`actual_data`** | **Required** | | The table or query containing the actual "ground truth" data. |
| **`data_col`** | **Required** | String | The numeric column to evaluate. |
| **`timestamp_col`** | **Required** | String | The column containing dates/timestamps. |
| **`id_cols`** | Optional | Array<String> | Grouping columns for multiple series. |
| **`horizon`** | Optional | Int64 | Number of points to evaluate. Defaults to 1024. |
| **`model`** | Optional | String | Model version (e.g., `'TimesFM 2.0'`). |

## Output Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| **`id_cols`** | (As Input) | Original identifiers for the series (if provided). |
| **`mean_absolute_error`** | FLOAT64 | Average magnitude of errors in the predictions. |
| **`mean_squared_error`** | FLOAT64 | Average of the squares of the errors. |
| **`root_mean_squared_error`**| FLOAT64 | Square root of the mean squared error. |
| **`mean_absolute_percentage_error`** | FLOAT64 | Mean absolute percentage error for the series. |
| **`symmetric_mean_absolute_percentage_error`** | FLOAT64 | Symmetric mean absolute percentage error for the series. |
| **`ai_evaluate_status`** | STRING | Error messages or empty string on success. |

## Example: Evaluating Forecast Quality
```sql
SELECT * FROM AI.EVALUATE(
  (SELECT * FROM `sales` WHERE date < '2024-01-01'),
  (SELECT * FROM `sales` WHERE date >= '2024-01-01'),
  data_col => 'total_sales',
  timestamp_col => 'date',
  id_cols => ['store_id']
);
```
