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

from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.google_tool import GoogleTool
import pytest


@pytest.mark.asyncio
async def test_bigquery_toolset_tools_default():
  """Test default BigQuery toolset.

  This test verifies the behavior of the BigQuery toolset when no filter is
  specified.
  """
  credentials_config = BigQueryCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigQueryToolset(
      credentials_config=credentials_config, bigquery_tool_config=None
  )
  # Verify that the tool config is initialized to default values.
  assert isinstance(toolset._tool_settings, BigQueryToolConfig)  # pylint: disable=protected-access
  assert toolset._tool_settings.__dict__ == BigQueryToolConfig().__dict__  # pylint: disable=protected-access

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 5
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_dataset_ids",
      "get_dataset_info",
      "list_table_ids",
      "get_table_info",
      "execute_sql",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    "selected_tools",
    [
        pytest.param([], id="None"),
        pytest.param(
            ["list_dataset_ids", "get_dataset_info"], id="dataset-metadata"
        ),
        pytest.param(["list_table_ids", "get_table_info"], id="table-metadata"),
        pytest.param(["execute_sql"], id="query"),
    ],
)
@pytest.mark.asyncio
async def test_bigquery_toolset_tools_selective(selected_tools):
  """Test BigQuery toolset with filter.

  This test verifies the behavior of the BigQuery toolset when filter is
  specified. A use case for this would be when the agent builder wants to
  use only a subset of the tools provided by the toolset.
  """
  credentials_config = BigQueryCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigQueryToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )
  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(selected_tools)
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set(selected_tools)
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    ("selected_tools", "returned_tools"),
    [
        pytest.param(["unknown"], [], id="all-unknown"),
        pytest.param(
            ["unknown", "execute_sql"],
            ["execute_sql"],
            id="mixed-known-unknown",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bigquery_toolset_unknown_tool(selected_tools, returned_tools):
  """Test BigQuery toolset with filter.

  This test verifies the behavior of the BigQuery toolset when filter is
  specified with an unknown tool.
  """
  credentials_config = BigQueryCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = BigQueryToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(returned_tools)
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set(returned_tools)
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names
