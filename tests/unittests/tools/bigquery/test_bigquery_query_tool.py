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

import textwrap
from typing import Optional
from unittest import mock

from google.adk.tools import BaseTool
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
from google.adk.tools.bigquery.query_tool import execute_sql
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import pytest


async def get_tool(
    name: str, tool_config: Optional[BigQueryToolConfig] = None
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
      bigquery_tool_config=tool_config,
  )

  tools = await toolset.get_tools()
  assert tools is not None
  assert len(tools) == 1
  return tools[0]


@pytest.mark.parametrize(
    ("tool_config",),
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
async def test_execute_sql_declaration_read_only(tool_config):
  """Test BigQuery execute_sql tool declaration in read-only mode.

  This test verifies that the execute_sql tool declaration reflects the
  read-only capability.
  """
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, tool_config)
  assert tool.name == tool_name
  assert tool.description == textwrap.dedent("""\
    Run a BigQuery SQL query in the project and return the result.

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
            }""")


@pytest.mark.parametrize(
    ("tool_config",),
    [
        pytest.param(
            BigQueryToolConfig(write_mode=WriteMode.ALLOWED),
            id="explicit-all-write",
        ),
    ],
)
@pytest.mark.asyncio
async def test_execute_sql_declaration_write(tool_config):
  """Test BigQuery execute_sql tool declaration with all writes enabled.

  This test verifies that the execute_sql tool declaration reflects the write
  capability.
  """
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, tool_config)
  assert tool.name == tool_name
  assert tool.description == textwrap.dedent("""\
    Run a BigQuery SQL query in the project and return the result.

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
        - To insert data into a table, use "INSERT INTO" statement.""")


@pytest.mark.parametrize(
    ("write_mode",),
    [
        pytest.param(
            WriteMode.BLOCKED,
            id="blocked",
        ),
        pytest.param(
            WriteMode.ALLOWED,
            id="allowed",
        ),
    ],
)
def test_execute_sql_select_stmt(write_mode):
  """Test execute_sql tool for SELECT query when writes are blocked."""
  project = "my_project"
  query = "SELECT 123 AS num"
  statement_type = "SELECT"
  query_result = [{"num": 123}]
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_config = BigQueryToolConfig(write_mode=write_mode)

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
    result = execute_sql(project, query, credentials, tool_config)
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
    ],
)
def test_execute_sql_non_select_stmt_write_allowed(query, statement_type):
  """Test execute_sql tool for SELECT query when writes are blocked."""
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)

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
    result = execute_sql(project, query, credentials, tool_config)
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
    ],
)
def test_execute_sql_non_select_stmt_write_blocked(query, statement_type):
  """Test execute_sql tool for SELECT query when writes are blocked."""
  project = "my_project"
  query_result = []
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_config = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)

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
    result = execute_sql(project, query, credentials, tool_config)
    assert result == {
        "status": "ERROR",
        "error_details": "Read-only mode only supports SELECT statements.",
    }
