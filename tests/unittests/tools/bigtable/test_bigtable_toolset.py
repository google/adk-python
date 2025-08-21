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

from unittest import mock

from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable import metadata_tool
from google.adk.tools.bigtable import query_tool
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset
from google.adk.tools.bigtable.bigtable_toolset import DEFAULT_BIGTABLE_TOOL_NAME_PREFIX
from google.adk.tools.google_tool import GoogleTool
import pytest


def test_bigtable_toolset_name_prefix():
  """Test Bigtable toolset name prefix."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(credentials_config=credentials_config)
  assert toolset.tool_name_prefix == DEFAULT_BIGTABLE_TOOL_NAME_PREFIX


@pytest.mark.asyncio
async def test_bigtable_toolset_tools_default():
  """Test default Bigtable toolset."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(credentials_config=credentials_config)

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 5
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_instances",
      "get_instance_info",
      "list_tables",
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
            ["list_instances", "get_instance_info"], id="instance-metadata"
        ),
        pytest.param(["list_tables", "get_table_info"], id="table-metadata"),
        pytest.param(["execute_sql"], id="query"),
    ],
)
@pytest.mark.asyncio
async def test_bigtable_toolset_tools_selective(selected_tools):
  """Test Bigtable toolset with filter.

  This test verifies the behavior of the Bigtable toolset when filter is
  specified. A use case for this would be when the agent builder wants to
  use only a subset of the tools provided by the toolset.
  """
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(
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
async def test_bigtable_toolset_unknown_tool(selected_tools, returned_tools):
  """Test Bigtable toolset with filter.

  This test verifies the behavior of the Bigtable toolset when filter is
  specified with an unknown tool.
  """
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = BigtableToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(returned_tools)
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set(returned_tools)
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names
