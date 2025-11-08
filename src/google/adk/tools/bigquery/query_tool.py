# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import json
import types
from typing import Callable
from typing import Optional
import uuid

from google.auth.credentials import Credentials
from google.cloud import bigquery

from . import client
from ..tool_context import ToolContext
from .config import BigQueryToolConfig
from .config import WriteMode

BIGQUERY_SESSION_INFO_KEY = "bigquery_session_info"


def _execute_sql(
    project_id: str,
    query: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
    dry_run: bool = False,
    caller_id: Optional[str] = None,
) -> dict:
  try:
    # Validate compute project if applicable
    if (
        settings.compute_project_id
        and project_id != settings.compute_project_id
    ):
      return {
          "status": "ERROR",
          "error_details": (
              f"Cannot execute query in the project {project_id}, as the tool"
              " is restricted to execute queries only in the project"
              f" {settings.compute_project_id}."
          ),
      }

    # Get BigQuery client
    bq_client = client.get_bigquery_client(
        project=project_id,
        credentials=credentials,
        location=settings.location,
        user_agent=[settings.application_name, caller_id],
    )

    # BigQuery connection properties where applicable
    bq_connection_properties = []

    # BigQuery job labels if applicable
    bq_job_labels = {}
    if caller_id:
      bq_job_labels["adk-bigquery-tool"] = caller_id

    if not settings or settings.write_mode == WriteMode.BLOCKED:
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(
              dry_run=True, labels=bq_job_labels
          ),
      )
      if dry_run_query_job.statement_type != "SELECT":
        return {
            "status": "ERROR",
            "error_details": "Read-only mode only supports SELECT statements.",
        }
    elif settings.write_mode == WriteMode.PROTECTED:
      # In protected write mode, write operation only to a temporary artifact is
      # allowed. This artifact must have been created in a BigQuery session. In
      # such a scenario, the session info (session id and the anonymous dataset
      # containing the artifact) is persisted in the tool context.
      bq_session_info = tool_context.state.get(BIGQUERY_SESSION_INFO_KEY, None)
      if bq_session_info:
        bq_session_id, bq_session_dataset_id = bq_session_info
      else:
        session_creator_job = bq_client.query(
            "SELECT 1",
            project=project_id,
            job_config=bigquery.QueryJobConfig(
                dry_run=True, create_session=True, labels=bq_job_labels
            ),
        )
        bq_session_id = session_creator_job.session_info.session_id
        bq_session_dataset_id = session_creator_job.destination.dataset_id

        # Remember the BigQuery session info for subsequent queries
        tool_context.state[BIGQUERY_SESSION_INFO_KEY] = (
            bq_session_id,
            bq_session_dataset_id,
        )

      # Session connection property will be set in the query execution
      bq_connection_properties.append(
          bigquery.ConnectionProperty("session_id", bq_session_id)
      )

      # Check the query type w.r.t. the BigQuery session
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(
              dry_run=True,
              connection_properties=bq_connection_properties,
              labels=bq_job_labels,
          ),
      )
      if (
          dry_run_query_job.statement_type != "SELECT"
          and dry_run_query_job.destination
          and dry_run_query_job.destination.dataset_id != bq_session_dataset_id
      ):
        return {
            "status": "ERROR",
            "error_details": (
                "Protected write mode only supports SELECT statements, or write"
                " operations in the anonymous dataset of a BigQuery session."
            ),
        }

    # Return the dry run characteristics of the query if requested
    if dry_run:
      dry_run_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(
              dry_run=True,
              connection_properties=bq_connection_properties,
              labels=bq_job_labels,
          ),
      )
      return {"status": "SUCCESS", "dry_run_info": dry_run_job.to_api_repr()}

    # Finally execute the query, fetch the result, and return it
    row_iterator = bq_client.query_and_wait(
        query,
        job_config=bigquery.QueryJobConfig(
            connection_properties=bq_connection_properties,
            labels=bq_job_labels,
        ),
        project=project_id,
        max_results=settings.max_query_result_rows,
    )
    rows = []
    for row in row_iterator:
      row_values = {}
      for key, val in row.items():
        try:
          # if the json serialization of the value succeeds, use it as is
          json.dumps(val)
        except:
          val = str(val)
        row_values[key] = val
      rows.append(row_values)

    result = {"status": "SUCCESS", "rows": rows}
    if (
        settings.max_query_result_rows is not None
        and len(rows) == settings.max_query_result_rows
    ):
      result["result_is_likely_truncated"] = True
    return result
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def execute_sql(
    project_id: str,
    query: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
    dry_run: bool = False,
) -> dict:
  """Run a BigQuery or BigQuery ML SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.
      tool_context (ToolContext): The context for the tool.
      dry_run (bool, default False): If True, the query will not be executed.
        Instead, the query will be validated and information about the query
        will be returned. Defaults to False.

  Returns:
      dict: If `dry_run` is False, dictionary representing the result of the
            query. If the result contains the key "result_is_likely_truncated"
            with value True, it means that there may be additional rows matching
            the query not returned in the result.
            If `dry_run` is True, dictionary with "dry_run_info" field
            containing query information returned by BigQuery.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project",
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": [
                {
                    "island": "Dream",
                    "population": 124
                },
                {
                    "island": "Biscoe",
                    "population": 168
                },
                {
                    "island": "Torgersen",
                    "population": 52
                }
            ]
          }

      Validate a query and estimate costs without executing it:

          >>> execute_sql(
          ...     "my_project",
          ...     "SELECT island FROM "
          ...     "bigquery-public-data.ml_datasets.penguins",
          ...     dry_run=True
          ... )
          {
            "status": "SUCCESS",
            "dry_run_info": {
              "configuration": {
                "dryRun": True,
                "jobType": "QUERY",
                "query": {
                  "destinationTable": {
                    "datasetId": "_...",
                    "projectId": "my_project",
                    "tableId": "anon..."
                  },
                  "priority": "INTERACTIVE",
                  "query": "SELECT island FROM bigquery-public-data.ml_datasets.penguins",
                  "useLegacySql": False,
                  "writeDisposition": "WRITE_TRUNCATE"
                }
              },
              "jobReference": {
                "location": "US",
                "projectId": "my_project"
              }
            }
          }
  """
  return _execute_sql(
      project_id=project_id,
      query=query,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
      dry_run=dry_run,
      caller_id="execute_sql",
  )


def _execute_sql_write_mode(*args, **kwargs) -> dict:
  """Run a BigQuery or BigQuery ML SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.
      tool_context (ToolContext): The context for the tool.
      dry_run (bool, default False): If True, the query will not be executed.
        Instead, the query will be validated and information about the query
        will be returned. Defaults to False.

  Returns:
      dict: If `dry_run` is False, dictionary representing the result of the
            query. If the result contains the key "result_is_likely_truncated"
            with value True, it means that there may be additional rows matching
            the query not returned in the result.
            If `dry_run` is True, dictionary with "dry_run_info" field
            containing query information returned by BigQuery.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project",
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": [
                {
                    "island": "Dream",
                    "population": 124
                },
                {
                    "island": "Biscoe",
                    "population": 168
                },
                {
                    "island": "Torgersen",
                    "population": 52
                }
            ]
          }

      Validate a query and estimate costs without executing it:

          >>> execute_sql(
          ...     "my_project",
          ...     "SELECT island FROM "
          ...     "bigquery-public-data.ml_datasets.penguins",
          ...     dry_run=True
          ... )
          {
            "status": "SUCCESS",
            "dry_run_info": {
              "configuration": {
                "dryRun": True,
                "jobType": "QUERY",
                "query": {
                  "destinationTable": {
                    "datasetId": "_...",
                    "projectId": "my_project",
                    "tableId": "anon..."
                  },
                  "priority": "INTERACTIVE",
                  "query": "SELECT island FROM bigquery-public-data.ml_datasets.penguins",
                  "useLegacySql": False,
                  "writeDisposition": "WRITE_TRUNCATE"
                }
              },
              "jobReference": {
                "location": "US",
                "projectId": "my_project"
              }
            }
          }

      Create a table with schema prescribed:

          >>> execute_sql("my_project",
          ... "CREATE TABLE my_project.my_dataset.my_table "
          ... "(island STRING, population INT64)")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Insert data into an existing table:

          >>> execute_sql("my_project",
          ... "INSERT INTO my_project.my_dataset.my_table (island, population) "
          ... "VALUES ('Dream', 124), ('Biscoe', 168)")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a table from the result of a query:

          >>> execute_sql("my_project",
          ... "CREATE TABLE my_project.my_dataset.my_table AS "
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Delete a table:

          >>> execute_sql("my_project",
          ... "DROP TABLE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Copy a table to another table:

          >>> execute_sql("my_project",
          ... "CREATE TABLE my_project.my_dataset.my_table_clone "
          ... "CLONE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a snapshot (a lightweight, read-optimized copy) of en existing
      table:

          >>> execute_sql("my_project",
          ... "CREATE SNAPSHOT TABLE my_project.my_dataset.my_table_snapshot "
          ... "CLONE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a BigQuery ML linear regression model:

          >>> execute_sql("my_project",
          ... "CREATE MODEL `my_dataset.my_model` "
          ... "OPTIONS (model_type='linear_reg', input_label_cols=['body_mass_g']) AS "
          ... "SELECT * FROM `bigquery-public-data.ml_datasets.penguins` "
          ... "WHERE body_mass_g IS NOT NULL")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Evaluate BigQuery ML model:

          >>> execute_sql("my_project",
          ... "SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`)")
          {
            "status": "SUCCESS",
            "rows": [{'mean_absolute_error': 227.01223667447218,
                      'mean_squared_error': 81838.15989216768,
                      'mean_squared_log_error': 0.0050704473735013,
                      'median_absolute_error': 173.08081641661738,
                      'r2_score': 0.8723772534253441,
                      'explained_variance': 0.8723772534253442}]
          }

      Evaluate BigQuery ML model on custom data:

          >>> execute_sql("my_project",
          ... "SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`, "
          ... "(SELECT * FROM `my_dataset.my_table`))")
          {
            "status": "SUCCESS",
            "rows": [{'mean_absolute_error': 227.01223667447218,
                      'mean_squared_error': 81838.15989216768,
                      'mean_squared_log_error': 0.0050704473735013,
                      'median_absolute_error': 173.08081641661738,
                      'r2_score': 0.8723772534253441,
                      'explained_variance': 0.8723772534253442}]
          }

      Predict using BigQuery ML model:

          >>> execute_sql("my_project",
          ... "SELECT * FROM ML.PREDICT(MODEL `my_dataset.my_model`, "
          ... "(SELECT * FROM `my_dataset.my_table`))")
          {
            "status": "SUCCESS",
            "rows": [
                {
                  "predicted_body_mass_g": "3380.9271650847013",
                  ...
                }, {
                  "predicted_body_mass_g": "3873.6072435386004",
                  ...
                },
                ...
            ]
          }

      Delete a BigQuery ML model:

          >>> execute_sql("my_project", "DROP MODEL `my_dataset.my_model`")
          {
            "status": "SUCCESS",
            "rows": []
          }

  Notes:
      - If a destination table already exists, there are a few ways to overwrite
      it:
          - Use "CREATE OR REPLACE TABLE" instead of "CREATE TABLE".
          - First run "DROP TABLE", followed by "CREATE TABLE".
      - If a model already exists, there are a few ways to overwrite it:
          - Use "CREATE OR REPLACE MODEL" instead of "CREATE MODEL".
          - First run "DROP MODEL", followed by "CREATE MODEL".
  """
  return execute_sql(*args, **kwargs)


def _execute_sql_protected_write_mode(*args, **kwargs) -> dict:
  """Run a BigQuery or BigQuery ML SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.
      tool_context (ToolContext): The context for the tool.
      dry_run (bool, default False): If True, the query will not be executed.
        Instead, the query will be validated and information about the query
        will be returned. Defaults to False.

  Returns:
      dict: If `dry_run` is False, dictionary representing the result of the
            query. If the result contains the key "result_is_likely_truncated"
            with value True, it means that there may be additional rows matching
            the query not returned in the result.
            If `dry_run` is True, dictionary with "dry_run_info" field
            containing query information returned by BigQuery.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project",
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": [
                {
                    "island": "Dream",
                    "population": 124
                },
                {
                    "island": "Biscoe",
                    "population": 168
                },
                {
                    "island": "Torgersen",
                    "population": 52
                }
            ]
          }

      Validate a query and estimate costs without executing it:

          >>> execute_sql(
          ...     "my_project",
          ...     "SELECT island FROM "
          ...     "bigquery-public-data.ml_datasets.penguins",
          ...     dry_run=True
          ... )
          {
            "status": "SUCCESS",
            "dry_run_info": {
              "configuration": {
                "dryRun": True,
                "jobType": "QUERY",
                "query": {
                  "destinationTable": {
                    "datasetId": "_...",
                    "projectId": "my_project",
                    "tableId": "anon..."
                  },
                  "priority": "INTERACTIVE",
                  "query": "SELECT island FROM bigquery-public-data.ml_datasets.penguins",
                  "useLegacySql": False,
                  "writeDisposition": "WRITE_TRUNCATE"
                }
              },
              "jobReference": {
                "location": "US",
                "projectId": "my_project"
              }
            }
          }

      Create a temporary table with schema prescribed:

          >>> execute_sql("my_project",
          ... "CREATE TEMP TABLE my_table (island STRING, population INT64)")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Insert data into an existing temporary table:

          >>> execute_sql("my_project",
          ... "INSERT INTO my_table (island, population) "
          ... "VALUES ('Dream', 124), ('Biscoe', 168)")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a temporary table from the result of a query:

          >>> execute_sql("my_project",
          ... "CREATE TEMP TABLE my_table AS "
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Delete a temporary table:

          >>> execute_sql("my_project", "DROP TABLE my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Copy a temporary table to another temporary table:

          >>> execute_sql("my_project",
          ... "CREATE TEMP TABLE my_table_clone CLONE my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a temporary BigQuery ML linear regression model:

          >>> execute_sql("my_project",
          ... "CREATE TEMP MODEL my_model "
          ... "OPTIONS (model_type='linear_reg', input_label_cols=['body_mass_g']) AS"
          ... "SELECT * FROM `bigquery-public-data.ml_datasets.penguins` "
          ... "WHERE body_mass_g IS NOT NULL")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Evaluate BigQuery ML model:

          >>> execute_sql("my_project", "SELECT * FROM ML.EVALUATE(MODEL my_model)")
          {
            "status": "SUCCESS",
            "rows": [{'mean_absolute_error': 227.01223667447218,
                      'mean_squared_error': 81838.15989216768,
                      'mean_squared_log_error': 0.0050704473735013,
                      'median_absolute_error': 173.08081641661738,
                      'r2_score': 0.8723772534253441,
                      'explained_variance': 0.8723772534253442}]
          }

      Evaluate BigQuery ML model on custom data:

          >>> execute_sql("my_project",
          ... "SELECT * FROM ML.EVALUATE(MODEL my_model, "
          ... "(SELECT * FROM `my_dataset.my_table`))")
          {
            "status": "SUCCESS",
            "rows": [{'mean_absolute_error': 227.01223667447218,
                      'mean_squared_error': 81838.15989216768,
                      'mean_squared_log_error': 0.0050704473735013,
                      'median_absolute_error': 173.08081641661738,
                      'r2_score': 0.8723772534253441,
                      'explained_variance': 0.8723772534253442}]
          }

      Predict using BigQuery ML model:

          >>> execute_sql("my_project",
          ... "SELECT * FROM ML.PREDICT(MODEL my_model, "
          ... "(SELECT * FROM `my_dataset.my_table`))")
          {
            "status": "SUCCESS",
            "rows": [
                {
                  "predicted_body_mass_g": "3380.9271650847013",
                  ...
                }, {
                  "predicted_body_mass_g": "3873.6072435386004",
                  ...
                },
                ...
            ]
          }

      Delete a BigQuery ML model:

          >>> execute_sql("my_project", "DROP MODEL my_model")
          {
            "status": "SUCCESS",
            "rows": []
          }

  Notes:
      - If a destination table already exists, there are a few ways to overwrite
      it:
          - Use "CREATE OR REPLACE TEMP TABLE" instead of "CREATE TEMP TABLE".
          - First run "DROP TABLE", followed by "CREATE TEMP TABLE".
      - Only temporary tables can be created, inserted into or deleted. Please
      do not try creating a permanent table (non-TEMP table), inserting into or
      deleting one.
      - If a destination model already exists, there are a few ways to overwrite
      it:
          - Use "CREATE OR REPLACE TEMP MODEL" instead of "CREATE TEMP MODEL".
          - First run "DROP MODEL", followed by "CREATE TEMP MODEL".
      - Only temporary models can be created or deleted. Please do not try
      creating a permanent model (non-TEMP model) or deleting one.
  """
  return execute_sql(*args, **kwargs)


def get_execute_sql(settings: BigQueryToolConfig) -> Callable[..., dict]:
  """Get the execute_sql tool customized as per the given tool settings.

  Args:
      settings: BigQuery tool settings indicating the behavior of the
        execute_sql tool.

  Returns:
      callable[..., dict]: A version of the execute_sql tool respecting the tool
      settings.
  """

  if not settings or settings.write_mode == WriteMode.BLOCKED:
    return execute_sql

  # Create a new function object using the original function's code and globals.
  # We pass the original code, globals, name, defaults, and closure.
  # This creates a raw function object without copying other metadata yet.
  execute_sql_wrapper = types.FunctionType(
      execute_sql.__code__,
      execute_sql.__globals__,
      execute_sql.__name__,
      execute_sql.__defaults__,
      execute_sql.__closure__,
  )

  # Use functools.update_wrapper to copy over other essential attributes
  # from the original function to the new one.
  # This includes __name__, __qualname__, __module__, __annotations__, etc.
  # It specifically allows us to then set __doc__ separately.
  functools.update_wrapper(execute_sql_wrapper, execute_sql)

  # Now, set the new docstring
  if settings.write_mode == WriteMode.PROTECTED:
    execute_sql_wrapper.__doc__ = _execute_sql_protected_write_mode.__doc__
  else:
    execute_sql_wrapper.__doc__ = _execute_sql_write_mode.__doc__

  return execute_sql_wrapper


def forecast(
    project_id: str,
    history_data: str,
    timestamp_col: str,
    data_col: str,
    horizon: int = 10,
    id_cols: Optional[list[str]] = None,
    *,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict:
  """Run a BigQuery AI time series forecast using AI.FORECAST.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      history_data (str): The table id of the BigQuery table containing the
        history time series data or a query statement that select the history
        data.
      timestamp_col (str): The name of the column containing the timestamp for
        each data point.
      data_col (str): The name of the column containing the numerical values to
        be forecasted.
      horizon (int, optional): The number of time steps to forecast into the
        future. Defaults to 10.
      id_cols (list, optional): The column names of the id columns to indicate
        each time series when there are multiple time series in the table. All
        elements must be strings. Defaults to None.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.
      tool_context (ToolContext): The context for the tool.

  Returns:
      dict: Dictionary representing the result of the forecast. The result
            contains the forecasted values along with prediction intervals.

  Examples:
      Forecast daily sales for the next 7 days based on historical data from
      a BigQuery table:

          >>> forecast(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.my-sales-table",
          ...     timestamp_col="sale_date",
          ...     data_col="daily_sales",
          ...     horizon=7
          ... )
          {
            "status": "SUCCESS",
            "rows": [
              {
                "forecast_timestamp": "2025-01-08T00:00:00",
                "forecast_value": 12345.67,
                "confidence_level": 0.95,
                "prediction_interval_lower_bound": 11000.0,
                "prediction_interval_upper_bound": 13691.34,
                "ai_forecast_status": ""
              },
              ...
            ]
          }

      Forecast multiple time series using a SQL query as input:

          >>> history_query = (
          ...     "SELECT unique_id, timestamp, value "
          ...     "FROM `my-project.my-dataset.my-timeseries-table` "
          ...     "WHERE timestamp > '1980-01-01'"
          ... )
          >>> forecast(
          ...     project_id="my-gcp-project",
          ...     history_data=history_query,
          ...     timestamp_col="timestamp",
          ...     data_col="value",
          ...     id_cols=["unique_id"],
          ...     horizon=14
          ... )
          {
            "status": "SUCCESS",
            "rows": [
              {
                "unique_id": "T1",
                "forecast_timestamp": "1980-08-28T00:00:00",
                "forecast_value": 1253218.75,
                "confidence_level": 0.95,
                "prediction_interval_lower_bound": 274252.51,
                "prediction_interval_upper_bound": 2232184.99,
                "ai_forecast_status": ""
              },
              ...
            ]
          }

      Error Scenarios:
          When an element in `id_cols` is not a string:

          >>> forecast(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.my-sales-table",
          ...     timestamp_col="sale_date",
          ...     data_col="daily_sales",
          ...     id_cols=["store_id", 123]
          ... )
          {
            "status": "ERROR",
            "error_details": "All elements in id_cols must be strings."
          }

          When `history_data` refers to a table that does not exist:

          >>> forecast(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.nonexistent-table",
          ...     timestamp_col="sale_date",
          ...     data_col="daily_sales"
          ... )
          {
            "status": "ERROR",
            "error_details": "Not found: Table
            my-gcp-project:my-dataset.nonexistent-table was not found in
            location US"
          }
  """
  model = "TimesFM 2.0"
  confidence_level = 0.95
  trimmed_upper_history_data = history_data.strip().upper()
  if trimmed_upper_history_data.startswith(
      "SELECT"
  ) or trimmed_upper_history_data.startswith("WITH"):
    history_data_source = f"({history_data})"
  else:
    history_data_source = f"TABLE `{history_data}`"

  if id_cols:
    if not all(isinstance(item, str) for item in id_cols):
      return {
          "status": "ERROR",
          "error_details": "All elements in id_cols must be strings.",
      }
    id_cols_str = "[" + ", ".join([f"'{col}'" for col in id_cols]) + "]"

    query = f"""
  SELECT * FROM AI.FORECAST(
    {history_data_source},
    data_col => '{data_col}',
    timestamp_col => '{timestamp_col}',
    model => '{model}',
    id_cols => {id_cols_str},
    horizon => {horizon},
    confidence_level => {confidence_level}
  )
  """
  else:
    query = f"""
  SELECT * FROM AI.FORECAST(
    {history_data_source},
    data_col => '{data_col}',
    timestamp_col => '{timestamp_col}',
    model => '{model}',
    horizon => {horizon},
    confidence_level => {confidence_level}
  )
  """
  return _execute_sql(
      project_id=project_id,
      query=query,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
      caller_id="forecast",
  )


def analyze_contribution(
    project_id: str,
    input_data: str,
    contribution_metric: str,
    dimension_id_cols: list[str],
    is_test_col: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
    top_k_insights: int = 30,
    pruning_method: str = "PRUNE_REDUNDANT_INSIGHTS",
) -> dict:
  """Run a BigQuery ML contribution analysis using ML.CREATE_MODEL and ML.GET_INSIGHTS.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      input_data (str): The data that contain the test and control data to
        analyze. Can be a fully qualified BigQuery table ID or a SQL query.
      dimension_id_cols (list[str]): The column names of the dimension columns.
      contribution_metric (str): The name of the column that contains the metric
        to analyze. Provides the expression to use to calculate the metric you
        are analyzing. To calculate a summable metric, the expression must be in
        the form SUM(metric_column_name), where metric_column_name is a numeric
        data type.  To calculate a summable ratio metric, the expression must be
        in the form
        SUM(numerator_metric_column_name)/SUM(denominator_metric_column_name),
        where numerator_metric_column_name and denominator_metric_column_name
        are numeric data types.  To calculate a summable by category metric, the
        expression must be in the form
        SUM(metric_sum_column_name)/COUNT(DISTINCT categorical_column_name). The
        summed column must be a numeric data type. The categorical column must
        have type BOOL, DATE, DATETIME, TIME, TIMESTAMP, STRING, or INT64.
      is_test_col (str): The name of the column to use to determine whether a
        given row is test data or control data. The column must have a BOOL data
        type.
      credentials: The credentials to use for the request.
      settings: The settings for the tool.
      tool_context: The context for the tool.
      top_k_insights (int, optional): The number of top insights to return,
        ranked by apriori support. Defaults to 30.
      pruning_method (str, optional): The method to use for pruning redundant
        insights. Can be 'NO_PRUNING' or 'PRUNE_REDUNDANT_INSIGHTS'. Defaults to
        "PRUNE_REDUNDANT_INSIGHTS".

  Returns:
      dict: Dictionary representing the result of the contribution analysis.

  Examples:
      Analyze the contribution of different dimensions to the total sales:

          >>> analyze_contribution(
          ...     project_id="my-gcp-project",
          ...     input_data="my-dataset.my-sales-table",
          ...     dimension_id_cols=["store_id", "product_category"],
          ...     contribution_metric="SUM(total_sales)",
          ...     is_test_col="is_test"
          ... )
          The return is:
          {
            "status": "SUCCESS",
            "rows": [
              {
                "store_id": "S1",
                "product_category": "Electronics",
                "contributors": ["S1", "Electronics"],
                "metric_test": 120,
                "metric_control": 100,
                "difference": 20,
                "relative_difference": 0.2,
                "unexpected_difference": 5,
                "relative_unexpected_difference": 0.043,
                "apriori_support": 0.15
              },
              ...
            ]
          }

      Analyze the contribution of different dimensions to the total sales using
      a SQL query as input:

          >>> analyze_contribution(
          ...     project_id="my-gcp-project",
          ...     input_data="SELECT store_id, product_category, total_sales, "
          ...     "is_test FROM `my-project.my-dataset.my-sales-table` "
          ...     "WHERE transaction_date > '2025-01-01'"
          ...     dimension_id_cols=["store_id", "product_category"],
          ...     contribution_metric="SUM(total_sales)",
          ...     is_test_col="is_test"
          ... )
          The return is:
          {
            "status": "SUCCESS",
            "rows": [
              {
                "store_id": "S2",
                "product_category": "Groceries",
                "contributors": ["S2", "Groceries"],
                "metric_test": 250,
                "metric_control": 200,
                "difference": 50,
                "relative_difference": 0.25,
                "unexpected_difference": 10,
                "relative_unexpected_difference": 0.041,
                "apriori_support": 0.22
              },
              ...
            ]
          }
  """
  if not all(isinstance(item, str) for item in dimension_id_cols):
    return {
        "status": "ERROR",
        "error_details": "All elements in dimension_id_cols must be strings.",
    }

  # Generate a unique temporary model name
  model_name = (
      f"contribution_analysis_model_{str(uuid.uuid4()).replace('-', '_')}"
  )

  id_cols_str = "[" + ", ".join([f"'{col}'" for col in dimension_id_cols]) + "]"
  options = [
      "MODEL_TYPE = 'CONTRIBUTION_ANALYSIS'",
      f"CONTRIBUTION_METRIC = '{contribution_metric}'",
      f"IS_TEST_COL = '{is_test_col}'",
      f"DIMENSION_ID_COLS = {id_cols_str}",
  ]

  options.append(f"TOP_K_INSIGHTS_BY_APRIORI_SUPPORT = {top_k_insights}")

  upper_pruning = pruning_method.upper()
  if upper_pruning not in ["NO_PRUNING", "PRUNE_REDUNDANT_INSIGHTS"]:
    return {
        "status": "ERROR",
        "error_details": f"Invalid pruning_method: {pruning_method}",
    }
  options.append(f"PRUNING_METHOD = '{upper_pruning}'")

  options_str = ", ".join(options)

  trimmed_upper_input_data = input_data.strip().upper()
  if trimmed_upper_input_data.startswith(
      "SELECT"
  ) or trimmed_upper_input_data.startswith("WITH"):
    input_data_source = f"({input_data})"
  else:
    input_data_source = f"SELECT * FROM `{input_data}`"

  create_model_query = f"""
  CREATE TEMP MODEL {model_name}
    OPTIONS ({options_str})
  AS {input_data_source}
  """

  get_insights_query = f"""
  SELECT * FROM ML.GET_INSIGHTS(MODEL {model_name})
  """

  # Create a session and run the create model query.
  original_write_mode = settings.write_mode
  try:
    if original_write_mode == WriteMode.BLOCKED:
      raise ValueError("analyze_contribution is not allowed in this session.")
    elif original_write_mode != WriteMode.PROTECTED:
      # Running create temp model requires a session. So we set the write mode
      # to PROTECTED to run the create model query and job query in the same
      # session.
      settings.write_mode = WriteMode.PROTECTED

    result = _execute_sql(
        project_id=project_id,
        query=create_model_query,
        credentials=credentials,
        settings=settings,
        tool_context=tool_context,
        caller_id="analyze_contribution",
    )
    if result["status"] != "SUCCESS":
      return result

    result = _execute_sql(
        project_id=project_id,
        query=get_insights_query,
        credentials=credentials,
        settings=settings,
        tool_context=tool_context,
        caller_id="analyze_contribution",
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": f"Error during analyze_contribution: {str(ex)}",
    }
  finally:
    # Restore the original write mode.
    settings.write_mode == original_write_mode

  return result


def detect_anomalies(
    project_id: str,
    history_data: str,
    times_series_timestamp_col: str,
    times_series_data_col: str,
    horizon: Optional[int] = 1000,
    target_data: Optional[str] = None,
    times_series_id_cols: Optional[list[str]] = None,
    anomaly_prob_threshold: Optional[float] = 0.95,
    *,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict:
  """Run a BigQuery time series ARIMA_PLUS model training and anomaly detection using CREATE MODEL and ML.DETECT_ANOMALIES clauses.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      history_data (str): The table id of the BigQuery table containing the
        history time series data or a query statement that select the history
        data.
      times_series_timestamp_col (str): The name of the column containing the
        timestamp for each data point.
      times_series_data_col (str): The name of the column containing the
        numerical values to be forecasted and anomaly detected.
      horizon (int, optional): The number of time steps to forecast into the
        future. Defaults to 1000.
      target_data (str, optional): The table id of the BigQuery table containing
        the target time series data or a query statement that select the target
        data.
      times_series_id_cols (list, optional): The column names of the id columns
        to indicate each time series when there are multiple time series in the
        table. All elements must be strings. Defaults to None.
      anomaly_prob_threshold (float, optional): The probability threshold to
        determine if a data point is an anomaly. Defaults to 0.95.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.
      tool_context (ToolContext): The context for the tool.

  Returns:
      dict: Dictionary representing the result of the anomaly detection. The
            result contains the boolean value if the data point is anomaly or
            not, lower bound, upper bound and anomaly probability for each data
            point and also the probability of whether the data point is anomaly
            or not.

  Examples:
      Detect Anomalies daily sales based on historical data from a BigQuery
      table:

          >>> detect_anomalies(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.my-sales-table",
          ...     times_series_timestamp_col="sale_date",
          ...     times_series_data_col="daily_sales"
          ... )
          {
            "status": "SUCCESS",
            "rows": [
              {
                "ts_timestamp": "2021-01-01 00:00:01 UTC",
                "ts_data": 125.3,
                "is_anomaly": TRUE,
                "lower_bound": 129.5,
                "upper_bound": 133.6 ,
                "anomaly_probability": 0.93
              },
              ...
            ]
          }

      Detect Anomalies on multiple time series using a SQL query as input:

          >>> history_query = (
          ...     "SELECT unique_id, timestamp, value "
          ...     "FROM `my-project.my-dataset.my-timeseries-table` "
          ...     "WHERE timestamp > '1980-01-01'"
          ... )
          >>> detect_anomalies(
          ...     project_id="my-gcp-project",
          ...     history_data=history_query,
          ...     times_series_timestamp_col="timestamp",
          ...     times_series_data_col="value",
          ...     times_series_id_cols=["unique_id"]
          ... )
          {
            "status": "SUCCESS",
            "rows": [
              {
                "unique_id": "T1",
                "ts_timestamp": "2021-01-01 00:00:01 UTC",
                "ts_data": 125.3,
                "is_anomaly": TRUE,
                "lower_bound": 129.5,
                "upper_bound": 133.6 ,
                "anomaly_probability": 0.93
              },
              ...
            ]
          }

      Error Scenarios:
          When an element in `times_series_id_cols` is not a string:

          >>> detect_anomalies(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.my-sales-table",
          ...     times_series_timestamp_col="sale_date",
          ...     times_series_data_col="daily_sales",
          ...     times_series_id_cols=["store_id", 123]
          ... )
          {
            "status": "ERROR",
            "error_details": "All elements in times_series_id_cols must be
            strings."
          }

          When `history_data` refers to a table that does not exist:

          >>> detect_anomalies(
          ...     project_id="my-gcp-project",
          ...     history_data="my-dataset.nonexistent-table",
          ...     times_series_timestamp_col="sale_date",
          ...     times_series_data_col="daily_sales"
          ... )
          {
            "status": "ERROR",
            "error_details": "Not found: Table
            my-gcp-project:my-dataset.nonexistent-table was not found in
            location US"
          }
  """
  trimmed_upper_history_data = history_data.strip().upper()
  if trimmed_upper_history_data.startswith(
      "SELECT"
  ) or trimmed_upper_history_data.startswith("WITH"):
    history_data_source = f"({history_data})"
  else:
    history_data_source = f"SELECT * FROM `{history_data}`"

  options = [
      "MODEL_TYPE = 'ARIMA_PLUS'",
      f"TIME_SERIES_TIMESTAMP_COL = '{times_series_timestamp_col}'",
      f"TIME_SERIES_DATA_COL = '{times_series_data_col}'",
      f"HORIZON = {horizon}",
  ]

  if times_series_id_cols:
    if not all(isinstance(item, str) for item in times_series_id_cols):
      return {
          "status": "ERROR",
          "error_details": (
              "All elements in times_series_id_cols must be strings."
          ),
      }
    times_series_id_cols_str = (
        "[" + ", ".join([f"'{col}'" for col in times_series_id_cols]) + "]"
    )
    options.append(f"TIME_SERIES_ID_COL = {times_series_id_cols_str}")

  options_str = ", ".join(options)

  model_name = f"detect_anomalies_model_{str(uuid.uuid4()).replace('-', '_')}"

  create_model_query = f"""
  CREATE TEMP MODEL {model_name}
    OPTIONS ({options_str})
  AS {history_data_source}
  """
  order_by_id_cols = (
      ", ".join(col for col in times_series_id_cols) + ", "
      if times_series_id_cols
      else ""
  )

  anomaly_detection_query = f"""
  SELECT * FROM ML.DETECT_ANOMALIES(MODEL {model_name}, STRUCT({anomaly_prob_threshold} AS anomaly_prob_threshold)) ORDER BY {order_by_id_cols}{times_series_timestamp_col}
  """
  if target_data:
    trimmed_upper_target_data = target_data.strip().upper()
    if trimmed_upper_target_data.startswith(
        "SELECT"
    ) or trimmed_upper_target_data.startswith("WITH"):
      target_data_source = f"({target_data})"
    else:
      target_data_source = f"(SELECT * FROM `{target_data}`)"

    anomaly_detection_query = f"""
    SELECT * FROM ML.DETECT_ANOMALIES(MODEL {model_name}, STRUCT({anomaly_prob_threshold} AS anomaly_prob_threshold), {target_data_source}) ORDER BY {order_by_id_cols}{times_series_timestamp_col}
    """

  # Create a session and run the create model query.
  original_write_mode = settings.write_mode
  try:
    if settings.write_mode == WriteMode.BLOCKED:
      raise ValueError("anomaly detection is not allowed in this session.")
    elif original_write_mode != WriteMode.PROTECTED:
      # Running create temp model requires a session. So we set the write mode
      # to PROTECTED to run the create model query and job query in the same
      # session.
      settings.write_mode = WriteMode.PROTECTED

    result = _execute_sql(
        project_id=project_id,
        query=create_model_query,
        credentials=credentials,
        settings=settings,
        tool_context=tool_context,
        caller_id="detect_anomalies",
    )
    if result["status"] != "SUCCESS":
      return result

    result = _execute_sql(
        project_id=project_id,
        query=anomaly_detection_query,
        credentials=credentials,
        settings=settings,
        tool_context=tool_context,
        caller_id="detect_anomalies",
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": f"Error during anomaly detection: {str(ex)}",
    }
  finally:
    # Restore the original write mode.
    settings.write_mode == original_write_mode

  return result
