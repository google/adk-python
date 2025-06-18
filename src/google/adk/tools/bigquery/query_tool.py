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

import functools
import types
from typing import Callable

from google.cloud import bigquery
from google.oauth2.credentials import Credentials

from . import client
from .config import BigQueryToolConfig
from .config import WriteMode

MAX_DOWNLOADED_QUERY_RESULT_ROWS = 50


def execute_sql(
    project_id: str,
    query: str,
    credentials: Credentials,
    config: BigQueryToolConfig,
) -> dict:
  """Run a BigQuery SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the result of the query.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("bigframes-dev",
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "ERROR",
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
    bq_client = client.get_bigquery_client(
        project=project_id, credentials=credentials
    )
    if not config or config.write_mode == WriteMode.BLOCKED:
      query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(dry_run=True),
      )
      if query_job.statement_type != "SELECT":
        return {
            "status": "ERROR",
            "error_details": "Read-only mode only supports SELECT statements.",
        }

    row_iterator = bq_client.query_and_wait(
        query, project=project_id, max_results=MAX_DOWNLOADED_QUERY_RESULT_ROWS
    )
    rows = [{key: val for key, val in row.items()} for row in row_iterator]
    result = {"status": "SUCCESS", "rows": rows}
    if (
        MAX_DOWNLOADED_QUERY_RESULT_ROWS is not None
        and len(rows) == MAX_DOWNLOADED_QUERY_RESULT_ROWS
    ):
      result["result_is_likely_truncated"] = True
    return result
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


_execute_sql_write_examples = """
      Create a table from the result of a query:

          >>> execute_sql("bigframes-dev",
          ... "CREATE TABLE my_project.my_dataset.my_table AS "
          ... "SELECT island, COUNT(*) AS population "
          ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Delete a table:

          >>> execute_sql("bigframes-dev",
          ... "DROP TABLE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Copy a table to another table:

          >>> execute_sql("bigframes-dev",
          ... "CREATE TABLE my_project.my_dataset.my_table_clone "
          ... "CLONE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

      Create a snapshot (a lightweight, read-optimized copy) of en existing
      table:

          >>> execute_sql("bigframes-dev",
          ... "CREATE SNAPSHOT TABLE my_project.my_dataset.my_table_snapshot "
          ... "CLONE my_project.my_dataset.my_table")
          {
            "status": "SUCCESS",
            "rows": []
          }

  Notes:
      - If a destination table already exists, there are a few ways to overwrite
      it:
          - Use "CREATE OR REPLACE TABLE" instead of "CREATE TABLE".
          - First run "DROP TABLE", followed by "CREATE TABLE".
      - To insert data into a table, use "INSERT INTO" statement.
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
  execute_sql_wrapper.__doc__ += _execute_sql_write_examples

  return execute_sql_wrapper
