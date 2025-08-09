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

from google.auth.credentials import Credentials
from google.cloud import bigquery

from . import client
from ..tool_context import ToolContext
from .config import BigQueryToolConfig
from .config import WriteMode

BIGQUERY_SESSION_INFO_KEY = "bigquery_session_info"


def execute_sql(
    project_id: str,
    query: str,
    credentials: Credentials,
    config: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict:
  """Run a BigQuery or BigQuery ML SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      config (BigQueryToolConfig): The configuration for the tool.
      tool_context (ToolContext): The context for the tool.

  Returns:
      dict: Dictionary representing the result of the query.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.

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
  """

  try:
    # Get BigQuery client
    bq_client = client.get_bigquery_client(
        project=project_id, credentials=credentials
    )

    # BigQuery connection properties where applicable
    bq_connection_properties = None

    if not config or config.write_mode == WriteMode.BLOCKED:
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(dry_run=True),
      )
      if dry_run_query_job.statement_type != "SELECT":
        return {
            "status": "ERROR",
            "error_details": "Read-only mode only supports SELECT statements.",
        }
    elif config.write_mode == WriteMode.PROTECTED:
      # In protected write mode, write operation only to a temporary artifact is
      # allowed. This artifact must have been created in a BigQuery session. In
      # such a scenario the session info (session id and the anonymous dataset
      # containing the artifact) is persisted in the tool context.
      bq_session_info = tool_context.state.get(BIGQUERY_SESSION_INFO_KEY, None)
      if bq_session_info:
        bq_session_id, bq_session_dataset_id = bq_session_info
      else:
        session_creator_job = bq_client.query(
            "SELECT 1",
            project=project_id,
            job_config=bigquery.QueryJobConfig(
                dry_run=True, create_session=True
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
      bq_connection_properties = [
          bigquery.ConnectionProperty("session_id", bq_session_id)
      ]

      # Check the query type w.r.t. the BigQuery session
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(
              dry_run=True,
              connection_properties=bq_connection_properties,
          ),
      )
      if (
          dry_run_query_job.statement_type != "SELECT"
          and dry_run_query_job.destination.dataset_id != bq_session_dataset_id
      ):
        return {
            "status": "ERROR",
            "error_details": (
                "Protected write mode only supports SELECT statements, or write"
                " operations in the anonymous dataset of a BigQuery session."
            ),
        }

    # Finally execute the query and fetch the result
    job_config = (
        bigquery.QueryJobConfig(connection_properties=bq_connection_properties)
        if bq_connection_properties
        else None
    )
    row_iterator = bq_client.query_and_wait(
        query,
        job_config=job_config,
        project=project_id,
        max_results=config.max_query_result_rows,
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
        config.max_query_result_rows is not None
        and len(rows) == config.max_query_result_rows
    ):
      result["result_is_likely_truncated"] = True
    return result
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


_execute_sql_write_examples = """
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


_execute_sql_protecetd_write_examples = """
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


def get_execute_sql(config: BigQueryToolConfig) -> Callable[..., dict]:
  """Get the execute_sql tool customized as per the given tool config.

  Args:
      config: BigQuery tool configuration indicating the behavior of the
        execute_sql tool.

  Returns:
      callable[..., dict]: A version of the execute_sql tool respecting the tool
      config.
  """

  if not config or config.write_mode == WriteMode.BLOCKED:
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
  if config.write_mode == WriteMode.PROTECTED:
    execute_sql_wrapper.__doc__ += _execute_sql_protecetd_write_examples
  else:
    execute_sql_wrapper.__doc__ += _execute_sql_write_examples

  return execute_sql_wrapper
