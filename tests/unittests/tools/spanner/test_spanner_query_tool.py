# Copyright 2026 Google LLC
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
from unittest import mock

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.spanner import query_tool
from google.adk.tools.spanner import settings
from google.adk.tools.spanner.settings import QueryResultMode
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
import pytest


async def get_tool(
    name: str, tool_settings: SpannerToolSettings | None = None
) -> BaseTool:
  """Get a tool from Spanner toolset."""
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = SpannerToolset(
      credentials_config=credentials_config,
      tool_filter=[name],
      spanner_tool_settings=tool_settings,
  )

  tools = await toolset.get_tools()
  assert tools is not None
  assert len(tools) == 1
  return tools[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_result_mode, expected_description",
    [
        (
            QueryResultMode.DEFAULT,
            textwrap.dedent(
                """\
    Run a Spanner Read-Only query in the spanner database and return the result.

    Args:
        project_id (str): The GCP project id in which the spanner database
          resides.
        instance_id (str): The instance id of the spanner database.
        database_id (str): The database id of the spanner database.
        query (str): The Spanner SQL query to be executed.
        credentials (Credentials): The credentials to use for the request.
        settings (SpannerToolSettings): The settings for the tool.
        tool_context (ToolContext): The context for the tool.

    Returns:
        dict: Dictionary with the result of the query.
              If the result contains the key "result_is_likely_truncated" with
              value True, it means that there may be additional rows matching the
              query not returned in the result.

    Examples:
        <Example>
          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT COUNT(*) AS count FROM my_table")
          {
            "status": "SUCCESS",
            "rows": [
              [100]
            ]
          }
        </Example>

        <Example>
          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT name, rating, description FROM hotels_table")
          {
            "status": "SUCCESS",
            "rows": [
              ["The Hotel", 4.1, "Modern hotel."],
              ["Park Inn", 4.5, "Cozy hotel."],
              ...
            ]
          }
        </Example>

    Note:
      This is running with Read-Only Transaction for query that only read data."""
            ),
        ),
        (
            QueryResultMode.DICT_LIST,
            textwrap.dedent(
                """\
    Run a Spanner Read-Only query in the spanner database and return the result.

    Args:
        project_id (str): The GCP project id in which the spanner database
          resides.
        instance_id (str): The instance id of the spanner database.
        database_id (str): The database id of the spanner database.
        query (str): The Spanner SQL query to be executed.
        credentials (Credentials): The credentials to use for the request.
        settings (SpannerToolSettings): The settings for the tool.
        tool_context (ToolContext): The context for the tool.

    Returns:
        dict: Dictionary with the result of the query.
              If the result contains the key "result_is_likely_truncated" with
              value True, it means that there may be additional rows matching the
              query not returned in the result.

    Examples:
        <Example>
          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT COUNT(*) AS count FROM my_table")
          {
            "status": "SUCCESS",
            "rows": [
              {
                "count": 100
              }
            ]
          }
        </Example>

        <Example>
          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT COUNT(*) FROM my_table")
          {
            "status": "SUCCESS",
            "rows": [
              {
                "": 100
              }
            ]
          }
        </Example>

        <Example>
          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT name, rating, description FROM hotels_table")
          {
            "status": "SUCCESS",
            "rows": [
              {
                "name": "The Hotel",
                "rating": 4.1,
                "description": "Modern hotel."
              },
              {
                "name": "Park Inn",
                "rating": 4.5,
                "description": "Cozy hotel."
              },
              ...
            ]
          }
        </Example>

    Note:
      This is running with Read-Only Transaction for query that only read data."""
            ),
        ),
    ],
)
async def test_execute_sql_query_result(
    query_result_mode, expected_description
):
  """Test Spanner execute_sql tool query result in different modes."""
  tool_name = "execute_sql"
  tool_settings = SpannerToolSettings(query_result_mode=query_result_mode)
  tool = await get_tool(tool_name, tool_settings)
  assert tool.name == tool_name
  assert tool.description == expected_description


@mock.patch.object(query_tool.utils, "execute_sql", spec_set=True)
def test_execute_sql(mock_utils_execute_sql):
  """Test execute_sql function in query result default mode."""
  mock_credentials = mock.create_autospec(
      Credentials, instance=True, spec_set=True
  )
  mock_tool_context = mock.create_autospec(
      ToolContext, instance=True, spec_set=True
  )
  mock_utils_execute_sql.return_value = {"status": "SUCCESS", "rows": [[1]]}

  result = query_tool.execute_sql(
      project_id="test-project",
      instance_id="test-instance",
      database_id="test-database",
      query="SELECT 1",
      credentials=mock_credentials,
      settings=settings.SpannerToolSettings(),
      tool_context=mock_tool_context,
  )

  mock_utils_execute_sql.assert_called_once_with(
      "test-project",
      "test-instance",
      "test-database",
      "SELECT 1",
      mock_credentials,
      settings.SpannerToolSettings(),
      mock_tool_context,
  )
  assert result == {"status": "SUCCESS", "rows": [[1]]}
