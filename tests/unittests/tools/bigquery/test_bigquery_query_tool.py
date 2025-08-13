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
from google.adk.tools.bigquery.query_tool import execute_sql
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

  This is a special case when the destination table is a persistent/permananent
  one and the protected write is enabled. In this case the operation should fail.
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
