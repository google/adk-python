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

import datetime
import decimal
import os
import textwrap
from typing import Optional
from unittest import mock

import dateutil
import dateutil.relativedelta
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
from google.adk.tools.bigquery.query_tool import analyze_contribution
from google.adk.tools.bigquery.query_tool import detect_anomalies
from google.adk.tools.bigquery.query_tool import execute_sql
from google.adk.tools.bigquery.query_tool import forecast
from google.adk.tools.tool_context import ToolContext
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import pytest


async def get_tool(
    name: str, tool_settings: Optional[BigQueryToolConfig] = None
) -> BaseTool:
  """Get a tool from BigQuery toolset.

  This method gets the tool view that an Agent using the BigQuery toolset would
  see.

  Returns:
    The tool.
  """
  credentials_config = BigQueryCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = BigQueryToolset(
      credentials_config=credentials_config,
      tool_filter=[name],
      bigquery_tool_config=tool_settings,
  )

  tools = await toolset.get_tools()
  assert tools is not None
  assert len(tools) == 1
  return tools[0]


@pytest.mark.parametrize(
    ("tool_settings",),
    [
        pytest.param(None, id="no-config"),
        pytest.param(BigQueryToolConfig(), id="default-config"),
        pytest.param(
            BigQueryToolConfig(write_mode=WriteMode.BLOCKED),
            id="explicit-no-write",
        ),
    ],
)
@pytest.mark.asyncio
async def test_execute_sql_declaration_read_only(tool_settings):
  """Test BigQuery execute_sql tool declaration in read-only mode.

  This test verifies that the execute_sql tool declaration reflects the
  read-only capability.
  """
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, tool_settings)
  assert tool.name == tool_name
  assert tool.description == textwrap.dedent("""\
    Run a BigQuery or BigQuery ML SQL query in the project and return the result.

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
            }""")


@pytest.mark.parametrize(
    ("tool_settings",),
    [
        pytest.param(
            BigQueryToolConfig(write_mode=WriteMode.ALLOWED),
            id="explicit-all-write",
        ),
    ],
)
@pytest.mark.asyncio
async def test_execute_sql_declaration_write(tool_settings):
  """Test BigQuery execute_sql tool declaration with all writes enabled.

  This test verifies that the execute_sql tool declaration reflects the write
  capability.
  """
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, tool_settings)
  assert tool.name == tool_name
  assert tool.description == textwrap.dedent("""\
    Run a BigQuery or BigQuery ML SQL query in the project and return the result.

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
            - First run "DROP MODEL", followed by "CREATE MODEL".""")


@pytest.mark.parametrize(
    ("tool_settings",),
    [
        pytest.param(
            BigQueryToolConfig(write_mode=WriteMode.PROTECTED),
            id="explicit-protected-write",
        ),
    ],
)
@pytest.mark.asyncio
async def test_execute_sql_declaration_protected_write(tool_settings):
  """Test BigQuery execute_sql tool declaration with protected writes enabled.

  This test verifies that the execute_sql tool declaration reflects the
  protected write capability.
  """
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, tool_settings)
  assert tool.name == tool_name
  assert tool.description == textwrap.dedent("""\
    Run a BigQuery or BigQuery ML SQL query in the project and return the result.

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
        creating a permanent model (non-TEMP model) or deleting one.""")


@pytest.mark.parametrize(
    ("write_mode",),
    [
        pytest.param(WriteMode.BLOCKED, id="blocked"),
        pytest.param(WriteMode.PROTECTED, id="protected"),
        pytest.param(WriteMode.ALLOWED, id="allowed"),
    ],
)
def test_execute_sql_select_stmt(write_mode):
  """Test execute_sql tool for SELECT query when writes are blocked."""
  project = "my_project"
  query = "SELECT 123 AS num"
  statement_type = "SELECT"
  query_result = [{"num": 123}]
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=write_mode)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.state.get.return_value = (
      "test-bq-session-id",
      "_anonymous_dataset",
  )

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    # The mock instance
    bq_client = Client.return_value

    # Simulate the result of query API
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    bq_client.query.return_value = query_job

    # Simulate the result of query_and_wait API
    bq_client.query_and_wait.return_value = query_result

    # Test the tool
    result = execute_sql(
        project, query, credentials, tool_settings, tool_context
    )
    assert result == {"status": "SUCCESS", "rows": query_result}


@pytest.mark.parametrize(
    ("query", "statement_type"),
    [
        pytest.param(
            "CREATE TABLE my_dataset.my_table AS SELECT 123 AS num",
            "CREATE_AS_SELECT",
            id="create-as-select",
        ),
        pytest.param(
            "DROP TABLE my_dataset.my_table",
            "DROP_TABLE",
            id="drop-table",
        ),
        pytest.param(
            "CREATE MODEL my_dataset.my_model (model_type='linear_reg',"
            " input_label_cols=['label_col']) AS SELECT * FROM"
            " my_dataset.my_table",
            "CREATE_MODEL",
            id="create-model",
        ),
        pytest.param(
            "DROP MODEL my_dataset.my_model",
            "DROP_MODEL",
            id="drop-model",
        ),
    ],
)
def test_execute_sql_non_select_stmt_write_allowed(query, statement_type):
  """Test execute_sql tool for non-SELECT query when writes are blocked."""
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    # The mock instance
    bq_client = Client.return_value

    # Simulate the result of query API
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    bq_client.query.return_value = query_job

    # Simulate the result of query_and_wait API
    bq_client.query_and_wait.return_value = query_result

    # Test the tool
    result = execute_sql(
        project, query, credentials, tool_settings, tool_context
    )
    assert result == {"status": "SUCCESS", "rows": query_result}


@pytest.mark.parametrize(
    ("query", "statement_type"),
    [
        pytest.param(
            "CREATE TABLE my_dataset.my_table AS SELECT 123 AS num",
            "CREATE_AS_SELECT",
            id="create-as-select",
        ),
        pytest.param(
            "DROP TABLE my_dataset.my_table",
            "DROP_TABLE",
            id="drop-table",
        ),
        pytest.param(
            "CREATE MODEL my_dataset.my_model (model_type='linear_reg',"
            " input_label_cols=['label_col']) AS SELECT * FROM"
            " my_dataset.my_table",
            "CREATE_MODEL",
            id="create-model",
        ),
        pytest.param(
            "DROP MODEL my_dataset.my_model",
            "DROP_MODEL",
            id="drop-model",
        ),
    ],
)
def test_execute_sql_non_select_stmt_write_blocked(query, statement_type):
  """Test execute_sql tool for non-SELECT query when writes are blocked."""
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    # The mock instance
    bq_client = Client.return_value

    # Simulate the result of query API
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    bq_client.query.return_value = query_job

    # Simulate the result of query_and_wait API
    bq_client.query_and_wait.return_value = query_result

    # Test the tool
    result = execute_sql(
        project, query, credentials, tool_settings, tool_context
    )
    assert result == {
        "status": "ERROR",
        "error_details": "Read-only mode only supports SELECT statements.",
    }


@pytest.mark.parametrize(
    ("query", "statement_type"),
    [
        pytest.param(
            "CREATE TEMP TABLE my_table AS SELECT 123 AS num",
            "CREATE_AS_SELECT",
            id="create-as-select",
        ),
        pytest.param(
            "DROP TABLE my_table",
            "DROP_TABLE",
            id="drop-table",
        ),
        pytest.param(
            "CREATE TEMP MODEL my_model (model_type='linear_reg',"
            " input_label_cols=['label_col']) AS SELECT * FROM"
            " my_dataset.my_table",
            "CREATE_MODEL",
            id="create-model",
        ),
        pytest.param(
            "DROP MODEL my_model",
            "DROP_MODEL",
            id="drop-model",
        ),
    ],
)
def test_execute_sql_non_select_stmt_write_protected(query, statement_type):
  """Test execute_sql tool for non-SELECT query when writes are protected."""
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.state.get.return_value = (
      "test-bq-session-id",
      "_anonymous_dataset",
  )

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    # The mock instance
    bq_client = Client.return_value

    # Simulate the result of query API
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    query_job.destination.dataset_id = "_anonymous_dataset"
    bq_client.query.return_value = query_job

    # Simulate the result of query_and_wait API
    bq_client.query_and_wait.return_value = query_result

    # Test the tool
    result = execute_sql(
        project, query, credentials, tool_settings, tool_context
    )
    assert result == {"status": "SUCCESS", "rows": query_result}


@pytest.mark.parametrize(
    ("query", "statement_type"),
    [
        pytest.param(
            "CREATE TABLE my_dataset.my_table AS SELECT 123 AS num",
            "CREATE_AS_SELECT",
            id="create-as-select",
        ),
        pytest.param(
            "DROP TABLE my_dataset.my_table",
            "DROP_TABLE",
            id="drop-table",
        ),
        pytest.param(
            "CREATE MODEL my_dataset.my_model (model_type='linear_reg',"
            " input_label_cols=['label_col']) AS SELECT * FROM"
            " my_dataset.my_table",
            "CREATE_MODEL",
            id="create-model",
        ),
        pytest.param(
            "DROP MODEL my_dataset.my_model",
            "DROP_MODEL",
            id="drop-model",
        ),
    ],
)
def test_execute_sql_non_select_stmt_write_protected_persistent_target(
    query, statement_type
):
  """Test execute_sql tool for non-SELECT query when writes are protected.

  This is a special case when the destination table is a persistent/permanent
  one and the protected write is enabled. In this case the operation should
  fail.
  """
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.state.get.return_value = (
      "test-bq-session-id",
      "_anonymous_dataset",
  )

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    # The mock instance
    bq_client = Client.return_value

    # Simulate the result of query API
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    query_job.destination.dataset_id = "my_dataset"
    bq_client.query.return_value = query_job

    # Simulate the result of query_and_wait API
    bq_client.query_and_wait.return_value = query_result

    # Test the tool
    result = execute_sql(
        project, query, credentials, tool_settings, tool_context
    )
    assert result == {
        "status": "ERROR",
        "error_details": (
            "Protected write mode only supports SELECT statements, or write"
            " operations in the anonymous dataset of a BigQuery session."
        ),
    }


def test_execute_sql_dry_run_true():
  """Test execute_sql tool with dry_run=True."""
  project = "my_project"
  query = "SELECT 123 AS num"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  api_repr = {
      "configuration": {"dryRun": True, "query": {"query": query}},
      "jobReference": {"projectId": project, "location": "US"},
  }

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    bq_client = Client.return_value

    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.to_api_repr.return_value = api_repr
    bq_client.query.return_value = query_job

    result = execute_sql(
        project, query, credentials, tool_settings, tool_context, dry_run=True
    )
    assert result == {"status": "SUCCESS", "dry_run_info": api_repr}
    bq_client.query.assert_called_once()
    _, mock_kwargs = bq_client.query.call_args
    assert mock_kwargs["job_config"].dry_run == True
    bq_client.query_and_wait.assert_not_called()


@pytest.mark.parametrize(
    ("write_mode",),
    [
        pytest.param(WriteMode.BLOCKED, id="blocked"),
        pytest.param(WriteMode.PROTECTED, id="protected"),
        pytest.param(WriteMode.ALLOWED, id="allowed"),
    ],
)
@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.query_and_wait", autospec=True)
@mock.patch("google.cloud.bigquery.Client.query", autospec=True)
@mock.patch("google.auth.default", autospec=True)
def test_execute_sql_no_default_auth(
    mock_default_auth, mock_query, mock_query_and_wait, write_mode
):
  """Test execute_sql tool invocation does not involve calling default auth."""
  project = "my_project"
  query = "SELECT 123 AS num"
  statement_type = "SELECT"
  query_result = [{"num": 123}]
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=write_mode)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.state.get.return_value = (
      "test-bq-session-id",
      "_anonymous_dataset",
  )

  # Simulate the behavior of default auth - on purpose throw exception when
  # the default auth is called
  mock_default_auth.side_effect = DefaultCredentialsError(
      "Your default credentials were not found"
  )

  # Simulate the result of query API
  query_job = mock.create_autospec(bigquery.QueryJob)
  query_job.statement_type = statement_type
  mock_query.return_value = query_job

  # Simulate the result of query_and_wait API
  mock_query_and_wait.return_value = query_result

  # Test the tool worked without invoking default auth
  result = execute_sql(project, query, credentials, tool_settings, tool_context)
  assert result == {"status": "SUCCESS", "rows": query_result}
  mock_default_auth.assert_not_called()


@pytest.mark.parametrize(
    ("query", "query_result", "tool_result_rows"),
    [
        pytest.param(
            "SELECT [1,2,3] AS x",
            [{"x": [1, 2, 3]}],
            [{"x": [1, 2, 3]}],
            id="ARRAY",
        ),
        pytest.param(
            "SELECT TRUE AS x", [{"x": True}], [{"x": True}], id="BOOL"
        ),
        pytest.param(
            "SELECT b'Hello World!' AS x",
            [{"x": b"Hello World!"}],
            [{"x": "b'Hello World!'"}],
            id="BYTES",
        ),
        pytest.param(
            "SELECT DATE '2025-07-21' AS x",
            [{"x": datetime.date(2025, 7, 21)}],
            [{"x": "2025-07-21"}],
            id="DATE",
        ),
        pytest.param(
            "SELECT DATETIME '2025-07-21 14:30:45' AS x",
            [{"x": datetime.datetime(2025, 7, 21, 14, 30, 45)}],
            [{"x": "2025-07-21 14:30:45"}],
            id="DATETIME",
        ),
        pytest.param(
            "SELECT ST_GEOGFROMTEXT('POINT(-122.21 47.48)') as x",
            [{"x": "POINT(-122.21 47.48)"}],
            [{"x": "POINT(-122.21 47.48)"}],
            id="GEOGRAPHY",
        ),
        pytest.param(
            "SELECT INTERVAL 10 DAY as x",
            [{"x": dateutil.relativedelta.relativedelta(days=10)}],
            [{"x": "relativedelta(days=+10)"}],
            id="INTERVAL",
        ),
        pytest.param(
            "SELECT JSON_OBJECT('name', 'Alice', 'age', 30) AS x",
            [{"x": {"age": 30, "name": "Alice"}}],
            [{"x": {"age": 30, "name": "Alice"}}],
            id="JSON",
        ),
        pytest.param("SELECT 1 AS x", [{"x": 1}], [{"x": 1}], id="INT64"),
        pytest.param(
            "SELECT CAST(1.2 AS NUMERIC) AS x",
            [{"x": decimal.Decimal("1.2")}],
            [{"x": "1.2"}],
            id="NUMERIC",
        ),
        pytest.param(
            "SELECT CAST(1.2 AS BIGNUMERIC) AS x",
            [{"x": decimal.Decimal("1.2")}],
            [{"x": "1.2"}],
            id="BIGNUMERIC",
        ),
        pytest.param(
            "SELECT 1.23 AS x", [{"x": 1.23}], [{"x": 1.23}], id="FLOAT64"
        ),
        pytest.param(
            "SELECT RANGE(DATE '2023-01-01', DATE '2023-01-31') as x",
            [{
                "x": {
                    "start": datetime.date(2023, 1, 1),
                    "end": datetime.date(2023, 1, 31),
                }
            }],
            [{
                "x": (
                    "{'start': datetime.date(2023, 1, 1), 'end':"
                    " datetime.date(2023, 1, 31)}"
                )
            }],
            id="RANGE",
        ),
        pytest.param(
            "SELECT 'abc' AS x", [{"x": "abc"}], [{"x": "abc"}], id="STRING"
        ),
        pytest.param(
            "SELECT STRUCT('Alice' AS name, 30 AS age) as x",
            [{"x": {"name": "Alice", "age": 30}}],
            [{"x": {"name": "Alice", "age": 30}}],
            id="STRUCT",
        ),
        pytest.param(
            "SELECT TIME '10:30:45' as x",
            [{"x": datetime.time(10, 30, 45)}],
            [{"x": "10:30:45"}],
            id="TIME",
        ),
        pytest.param(
            "SELECT TIMESTAMP '2025-07-21 10:30:45-07:00' as x",
            [{
                "x": datetime.datetime(
                    2025, 7, 21, 17, 30, 45, tzinfo=datetime.timezone.utc
                )
            }],
            [{"x": "2025-07-21 17:30:45+00:00"}],
            id="TIMESTAMP",
        ),
        pytest.param(
            "SELECT NULL AS x", [{"x": None}], [{"x": None}], id="NULL"
        ),
    ],
)
@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.query_and_wait", autospec=True)
@mock.patch("google.cloud.bigquery.Client.query", autospec=True)
def test_execute_sql_result_dtype(
    mock_query, mock_query_and_wait, query, query_result, tool_result_rows
):
  """Test execute_sql tool invocation for various BigQuery data types.

  See all the supported BigQuery data types at
  https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#data_type_list.
  """
  project = "my_project"
  statement_type = "SELECT"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig()
  tool_context = mock.create_autospec(ToolContext, instance=True)

  # Simulate the result of query API
  query_job = mock.create_autospec(bigquery.QueryJob)
  query_job.statement_type = statement_type
  mock_query.return_value = query_job

  # Simulate the result of query_and_wait API
  mock_query_and_wait.return_value = query_result

  # Test the tool worked without invoking default auth
  result = execute_sql(project, query, credentials, tool_settings, tool_context)
  assert result == {"status": "SUCCESS", "rows": tool_result_rows}


@mock.patch(
    "google.adk.tools.bigquery.client.get_bigquery_client", autospec=True
)
def test_execute_sql_bq_client_creation(mock_get_bigquery_client):
  """Test BigQuery client creation params during execute_sql tool invocation."""
  project = "my_project_id"
  query = "SELECT 1"
  credentials = mock.create_autospec(Credentials, instance=True)
  application_name = "my-agent"
  tool_settings = BigQueryToolConfig(application_name=application_name)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  execute_sql(project, query, credentials, tool_settings, tool_context)
  mock_get_bigquery_client.assert_called_once()
  assert len(mock_get_bigquery_client.call_args.kwargs) == 4
  assert mock_get_bigquery_client.call_args.kwargs["project"] == project
  assert mock_get_bigquery_client.call_args.kwargs["credentials"] == credentials
  assert mock_get_bigquery_client.call_args.kwargs["user_agent"] == [
      application_name,
      "execute_sql",
  ]


def test_execute_sql_unexpected_project_id():
  """Test execute_sql tool invocation with unexpected project id."""
  compute_project_id = "compute_project_id"
  tool_call_project_id = "project_id"
  query = "SELECT 1"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(compute_project_id=compute_project_id)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  result = execute_sql(
      tool_call_project_id, query, credentials, tool_settings, tool_context
  )
  assert result == {
      "status": "ERROR",
      "error_details": (
          f"Cannot execute query in the project {tool_call_project_id}, as the"
          " tool is restricted to execute queries only in the project"
          f" {compute_project_id}."
      ),
  }


# AI.Forecast calls _execute_sql with a specific query statement. We need to
# test that the query is properly constructed and call _execute_sql with the
# correct parameters exactly once.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
def test_forecast_with_table_id(mock_execute_sql):
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig()
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)

  forecast(
      project_id="test-project",
      history_data="test-dataset.test-table",
      timestamp_col="ts_col",
      data_col="data_col",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      horizon=20,
      id_cols=["id1", "id2"],
  )

  expected_query = """
  SELECT * FROM AI.FORECAST(
    TABLE `test-dataset.test-table`,
    data_col => 'data_col',
    timestamp_col => 'ts_col',
    model => 'TimesFM 2.0',
    id_cols => ['id1', 'id2'],
    horizon => 20,
    confidence_level => 0.95
  )
  """
  mock_execute_sql.assert_called_once_with(
      project_id="test-project",
      query=expected_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="forecast",
  )


# AI.Forecast calls _execute_sql with a specific query statement. We need to
# test that the query is properly constructed and call _execute_sql with the
# correct parameters exactly once.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
def test_forecast_with_query_statement(mock_execute_sql):
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig()
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)

  history_data_query = "SELECT * FROM `test-dataset.test-table`"
  forecast(
      project_id="test-project",
      history_data=history_data_query,
      timestamp_col="ts_col",
      data_col="data_col",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_query = f"""
  SELECT * FROM AI.FORECAST(
    ({history_data_query}),
    data_col => 'data_col',
    timestamp_col => 'ts_col',
    model => 'TimesFM 2.0',
    horizon => 10,
    confidence_level => 0.95
  )
  """
  mock_execute_sql.assert_called_once_with(
      project_id="test-project",
      query=expected_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="forecast",
  )


def test_forecast_with_invalid_id_cols():
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig()
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)

  result = forecast(
      project_id="test-project",
      history_data="test-dataset.test-table",
      timestamp_col="ts_col",
      data_col="data_col",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      id_cols=["id1", 123],
  )

  assert result["status"] == "ERROR"
  assert "All elements in id_cols must be strings." in result["error_details"]


# analyze_contribution calls _execute_sql twice. We need to test that the
# queries are properly constructed and call _execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_analyze_contribution_with_table_id(mock_uuid, mock_execute_sql):
  """Test analyze_contribution tool invocation with a table id."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  analyze_contribution(
      project_id="test-project",
      input_data="test-dataset.test-table",
      dimension_id_cols=["dim1", "dim2"],
      contribution_metric="SUM(metric)",
      is_test_col="is_test",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = """
  CREATE TEMP MODEL contribution_analysis_model_test_uuid
    OPTIONS (MODEL_TYPE = 'CONTRIBUTION_ANALYSIS', CONTRIBUTION_METRIC = 'SUM(metric)', IS_TEST_COL = 'is_test', DIMENSION_ID_COLS = ['dim1', 'dim2'], TOP_K_INSIGHTS_BY_APRIORI_SUPPORT = 30, PRUNING_METHOD = 'PRUNE_REDUNDANT_INSIGHTS')
  AS SELECT * FROM `test-dataset.test-table`
  """

  expected_get_insights_query = """
  SELECT * FROM ML.GET_INSIGHTS(MODEL contribution_analysis_model_test_uuid)
  """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="analyze_contribution",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_get_insights_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="analyze_contribution",
  )


# analyze_contribution calls _execute_sql twice. We need to test that the
# queries are properly constructed and call _execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_analyze_contribution_with_query_statement(mock_uuid, mock_execute_sql):
  """Test analyze_contribution tool invocation with a query statement."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  input_data_query = "SELECT * FROM `test-dataset.test-table`"
  analyze_contribution(
      project_id="test-project",
      input_data=input_data_query,
      dimension_id_cols=["dim1", "dim2"],
      contribution_metric="SUM(metric)",
      is_test_col="is_test",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = f"""
  CREATE TEMP MODEL contribution_analysis_model_test_uuid
    OPTIONS (MODEL_TYPE = 'CONTRIBUTION_ANALYSIS', CONTRIBUTION_METRIC = 'SUM(metric)', IS_TEST_COL = 'is_test', DIMENSION_ID_COLS = ['dim1', 'dim2'], TOP_K_INSIGHTS_BY_APRIORI_SUPPORT = 30, PRUNING_METHOD = 'PRUNE_REDUNDANT_INSIGHTS')
  AS ({input_data_query})
  """

  expected_get_insights_query = """
  SELECT * FROM ML.GET_INSIGHTS(MODEL contribution_analysis_model_test_uuid)
  """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="analyze_contribution",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_get_insights_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="analyze_contribution",
  )


def test_analyze_contribution_with_invalid_dimension_id_cols():
  """Test analyze_contribution tool invocation with invalid dimension_id_cols."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig()
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)

  result = analyze_contribution(
      project_id="test-project",
      input_data="test-dataset.test-table",
      dimension_id_cols=["dim1", 123],
      contribution_metric="metric",
      is_test_col="is_test",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  assert result["status"] == "ERROR"
  assert (
      "All elements in dimension_id_cols must be strings."
      in result["error_details"]
  )


# detect_anomalies calls _execute_sql twice. We need to test that
# the queries are properly constructed and call _execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_detect_anomalies_with_table_id(mock_uuid, mock_execute_sql):
  """Test time series anomaly detection tool invocation with a table id."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  history_data_query = "SELECT * FROM `test-dataset.test-table`"
  detect_anomalies(
      project_id="test-project",
      history_data=history_data_query,
      times_series_timestamp_col="ts_timestamp",
      times_series_data_col="ts_data",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = """
  CREATE TEMP MODEL detect_anomalies_model_test_uuid
    OPTIONS (MODEL_TYPE = 'ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL = 'ts_timestamp', TIME_SERIES_DATA_COL = 'ts_data', HORIZON = 1000)
  AS (SELECT * FROM `test-dataset.test-table`)
  """

  expected_anomaly_detection_query = """
  SELECT * FROM ML.DETECT_ANOMALIES(MODEL detect_anomalies_model_test_uuid, STRUCT(0.95 AS anomaly_prob_threshold)) ORDER BY ts_timestamp
  """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_anomaly_detection_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )


# detect_anomalies calls _execute_sql twice. We need to test that
# the queries are properly constructed and call _execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_detect_anomalies_with_custom_params(mock_uuid, mock_execute_sql):
  """Test time series anomaly detection tool invocation with a table id."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  history_data_query = "SELECT * FROM `test-dataset.test-table`"
  detect_anomalies(
      project_id="test-project",
      history_data=history_data_query,
      times_series_timestamp_col="ts_timestamp",
      times_series_data_col="ts_data",
      times_series_id_cols=["dim1", "dim2"],
      horizon=20,
      anomaly_prob_threshold=0.8,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = """
  CREATE TEMP MODEL detect_anomalies_model_test_uuid
    OPTIONS (MODEL_TYPE = 'ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL = 'ts_timestamp', TIME_SERIES_DATA_COL = 'ts_data', HORIZON = 20, TIME_SERIES_ID_COL = ['dim1', 'dim2'])
  AS (SELECT * FROM `test-dataset.test-table`)
  """

  expected_anomaly_detection_query = """
  SELECT * FROM ML.DETECT_ANOMALIES(MODEL detect_anomalies_model_test_uuid, STRUCT(0.8 AS anomaly_prob_threshold)) ORDER BY dim1, dim2, ts_timestamp
  """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_anomaly_detection_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )


# detect_anomalies calls _execute_sql twice. We need to test that
# the queries are properly constructed and call _execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_detect_anomalies_on_target_table(mock_uuid, mock_execute_sql):
  """Test time series anomaly detection tool with target data is provided."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  history_data_query = "SELECT * FROM `test-dataset.history-table`"
  target_data_query = "SELECT * FROM `test-dataset.target-table`"
  detect_anomalies(
      project_id="test-project",
      history_data=history_data_query,
      times_series_timestamp_col="ts_timestamp",
      times_series_data_col="ts_data",
      times_series_id_cols=["dim1", "dim2"],
      horizon=20,
      target_data=target_data_query,
      anomaly_prob_threshold=0.8,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = """
  CREATE TEMP MODEL detect_anomalies_model_test_uuid
    OPTIONS (MODEL_TYPE = 'ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL = 'ts_timestamp', TIME_SERIES_DATA_COL = 'ts_data', HORIZON = 20, TIME_SERIES_ID_COL = ['dim1', 'dim2'])
  AS (SELECT * FROM `test-dataset.history-table`)
  """

  expected_anomaly_detection_query = """
    SELECT * FROM ML.DETECT_ANOMALIES(MODEL detect_anomalies_model_test_uuid, STRUCT(0.8 AS anomaly_prob_threshold), (SELECT * FROM `test-dataset.target-table`)) ORDER BY dim1, dim2, ts_timestamp
    """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_anomaly_detection_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )


# detect_anomalies calls execute_sql twice. We need to test that
# the queries are properly constructed and call execute_sql with the correct
# parameters exactly twice.
@mock.patch("google.adk.tools.bigquery.query_tool._execute_sql", autospec=True)
@mock.patch("uuid.uuid4", autospec=True)
def test_detect_anomalies_with_str_table_id(mock_uuid, mock_execute_sql):
  """Test time series anomaly detection tool invocation with a table id."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig(write_mode=WriteMode.PROTECTED)
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_uuid.return_value = "test_uuid"
  mock_execute_sql.return_value = {"status": "SUCCESS"}

  history_data_query = "SELECT * FROM `test-dataset.test-table`"
  detect_anomalies(
      project_id="test-project",
      history_data=history_data_query,
      times_series_timestamp_col="ts_timestamp",
      times_series_data_col="ts_data",
      target_data="test-dataset.target-table",
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  expected_create_model_query = """
  CREATE TEMP MODEL detect_anomalies_model_test_uuid
    OPTIONS (MODEL_TYPE = 'ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL = 'ts_timestamp', TIME_SERIES_DATA_COL = 'ts_data', HORIZON = 1000)
  AS (SELECT * FROM `test-dataset.test-table`)
  """

  expected_anomaly_detection_query = """
    SELECT * FROM ML.DETECT_ANOMALIES(MODEL detect_anomalies_model_test_uuid, STRUCT(0.95 AS anomaly_prob_threshold), (SELECT * FROM `test-dataset.target-table`)) ORDER BY ts_timestamp
    """

  assert mock_execute_sql.call_count == 2
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_create_model_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )
  mock_execute_sql.assert_any_call(
      project_id="test-project",
      query=expected_anomaly_detection_query,
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
      caller_id="detect_anomalies",
  )


def test_detect_anomalies_with_invalid_id_cols():
  """Test time series anomaly detection tool invocation with invalid times_series_id_cols."""
  mock_credentials = mock.MagicMock(spec=Credentials)
  mock_settings = BigQueryToolConfig()
  mock_tool_context = mock.create_autospec(ToolContext, instance=True)

  result = detect_anomalies(
      project_id="test-project",
      history_data="test-dataset.test-table",
      times_series_timestamp_col="ts_timestamp",
      times_series_data_col="ts_data",
      times_series_id_cols=["dim1", 123],
      credentials=mock_credentials,
      settings=mock_settings,
      tool_context=mock_tool_context,
  )

  assert result["status"] == "ERROR"
  assert (
      "All elements in times_series_id_cols must be strings."
      in result["error_details"]
  )


@pytest.mark.parametrize(
    ("write_mode", "dry_run", "query_call_count", "query_and_wait_call_count"),
    [
        pytest.param(WriteMode.ALLOWED, False, 0, 1, id="write-allowed"),
        pytest.param(WriteMode.ALLOWED, True, 1, 0, id="write-allowed-dry-run"),
        pytest.param(WriteMode.BLOCKED, False, 1, 1, id="write-blocked"),
        pytest.param(WriteMode.BLOCKED, True, 2, 0, id="write-blocked-dry-run"),
        pytest.param(WriteMode.PROTECTED, False, 2, 1, id="write-protected"),
        pytest.param(
            WriteMode.PROTECTED, True, 3, 0, id="write-protected-dry-run"
        ),
    ],
)
def test_execute_sql_job_labels(
    write_mode, dry_run, query_call_count, query_and_wait_call_count
):
  """Test execute_sql tool for job label."""
  project = "my_project"
  query = "SELECT 123 AS num"
  statement_type = "SELECT"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigQueryToolConfig(write_mode=write_mode)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.state.get.return_value = None

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    bq_client = Client.return_value

    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = statement_type
    bq_client.query.return_value = query_job

    execute_sql(
        project,
        query,
        credentials,
        tool_settings,
        tool_context,
        dry_run=dry_run,
    )

    assert bq_client.query.call_count == query_call_count
    assert bq_client.query_and_wait.call_count == query_and_wait_call_count
    for call_args_list in [
        bq_client.query.call_args_list,
        bq_client.query_and_wait.call_args_list,
    ]:
      for call_args in call_args_list:
        _, mock_kwargs = call_args
        assert mock_kwargs["job_config"].labels == {
            "adk-bigquery-tool": "execute_sql"
        }


@pytest.mark.parametrize(
    ("tool_call", "expected_label"),
    [
        pytest.param(
            lambda tool_context: forecast(
                project_id="test-project",
                history_data="SELECT * FROM `test-dataset.test-table`",
                timestamp_col="ts_col",
                data_col="data_col",
                credentials=mock.create_autospec(Credentials, instance=True),
                settings=BigQueryToolConfig(write_mode=WriteMode.ALLOWED),
                tool_context=tool_context,
            ),
            "forecast",
            id="forecast",
        ),
        pytest.param(
            lambda tool_context: analyze_contribution(
                project_id="test-project",
                input_data="test-dataset.test-table",
                dimension_id_cols=["dim1", "dim2"],
                contribution_metric="SUM(metric)",
                is_test_col="is_test",
                credentials=mock.create_autospec(Credentials, instance=True),
                settings=BigQueryToolConfig(write_mode=WriteMode.ALLOWED),
                tool_context=tool_context,
            ),
            "analyze_contribution",
            id="analyze-contribution",
        ),
        pytest.param(
            lambda tool_context: detect_anomalies(
                project_id="test-project",
                history_data="SELECT * FROM `test-dataset.test-table`",
                times_series_timestamp_col="ts_timestamp",
                times_series_data_col="ts_data",
                credentials=mock.create_autospec(Credentials, instance=True),
                settings=BigQueryToolConfig(write_mode=WriteMode.ALLOWED),
                tool_context=tool_context,
            ),
            "detect_anomalies",
            id="detect-anomalies",
        ),
    ],
)
def test_ml_tool_job_labels(tool_call, expected_label):
  """Test ML tools for job label."""

  with mock.patch("google.cloud.bigquery.Client", autospec=False) as Client:
    bq_client = Client.return_value

    tool_context = mock.create_autospec(ToolContext, instance=True)
    tool_context.state.get.return_value = None
    tool_call(tool_context)

    for call_args_list in [
        bq_client.query.call_args_list,
        bq_client.query_and_wait.call_args_list,
    ]:
      for call_args in call_args_list:
        _, mock_kwargs = call_args
        assert mock_kwargs["job_config"].labels == {
            "adk-bigquery-tool": expected_label
        }
