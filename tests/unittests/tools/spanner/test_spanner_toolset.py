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

from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.spanner import SpannerCredentialsConfig
from google.adk.tools.spanner import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings
import pytest


@pytest.mark.asyncio
async def test_spanner_toolset_tools_default():
  """Test default Spanner toolset.

  This test verifies the behavior of the Spanner toolset when no filter is
  specified.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = SpannerToolset(credentials_config=credentials_config)
  assert isinstance(toolset._tool_settings, SpannerToolSettings)  # pylint: disable=protected-access
  assert toolset._tool_settings.__dict__ == SpannerToolSettings().__dict__  # pylint: disable=protected-access
  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 6
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_table_names",
      "list_table_indexes",
      "list_table_index_columns",
      "list_named_schemas",
      "get_table_schema",
      "execute_sql",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    "selected_tools",
    [
        pytest.param([], id="None"),
        pytest.param(
            ["list_table_names", "get_table_schema"],
            id="table-metadata",
        ),
        pytest.param(["execute_sql"], id="query"),
    ],
)
@pytest.mark.asyncio
async def test_spanner_toolset_selective(selected_tools):
  """Test selective Spanner toolset.

  This test verifies the behavior of the Spanner toolset when a filter is
  specified.

  Args:
      selected_tools: A list of tool names to filter.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = SpannerToolset(
      credentials_config=credentials_config,
      tool_filter=selected_tools,
      spanner_tool_settings=SpannerToolSettings(),
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
async def test_spanner_toolset_unknown_tool(selected_tools, returned_tools):
  """Test Spanner toolset with unknown tools.

  This test verifies the behavior of the Spanner toolset when unknown tools are
  specified in the filter.

  Args:
      selected_tools: A list of tool names to filter, including unknown ones.
      returned_tools: A list of tool names that are expected to be returned.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = SpannerToolset(
      credentials_config=credentials_config,
      tool_filter=selected_tools,
      spanner_tool_settings=SpannerToolSettings(),
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(returned_tools)
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set(returned_tools)
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    ("selected_tools", "returned_tools"),
    [
        pytest.param(
            ["execute_sql", "list_table_names"],
            ["list_table_names"],
            id="read-not-added",
        ),
        pytest.param(
            ["list_table_names", "list_table_indexes"],
            ["list_table_names", "list_table_indexes"],
            id="no-effect",
        ),
    ],
)
@pytest.mark.asyncio
async def test_spanner_toolset_without_read_capability(
    selected_tools, returned_tools
):
  """Test Spanner toolset without read capability.

  This test verifies the behavior of the Spanner toolset when read capability is
  not enabled.

  Args:
      selected_tools: A list of tool names to filter.
      returned_tools: A list of tool names that are expected to be returned.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  spanner_tool_settings = SpannerToolSettings(capabilities=[])
  toolset = SpannerToolset(
      credentials_config=credentials_config,
      tool_filter=selected_tools,
      spanner_tool_settings=spanner_tool_settings,
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(returned_tools)
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set(returned_tools)
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names
