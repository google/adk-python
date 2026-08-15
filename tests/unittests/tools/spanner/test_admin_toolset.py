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

from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
import pytest


@pytest.mark.asyncio
async def test_spanner_toolset_tools_default():
  """Test Admin Spanner toolset.

  This test verifies the behavior of the Spanner admin toolset when no filter is
  specified.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = SpannerAdminToolset(credentials_config=credentials_config)
  assert isinstance(toolset._tool_settings, SpannerToolSettings)  # pylint: disable=protected-access
  assert toolset._tool_settings.__dict__ == SpannerToolSettings().__dict__  # pylint: disable=protected-access
  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 7
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_instances",
      "get_instance",
      "list_databases",
      "create_instance",
      "create_database",
      "list_instance_configs",
      "get_instance_config",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    "selected_tools",
    [
        pytest.param(
            ["list_instances"],
            id="list-instances",
        )
    ],
)
@pytest.mark.asyncio
async def test_spanner_admin_toolset_selective(selected_tools):
  """Test selective Admin Spanner toolset.

  This test verifies the behavior of the Spanner admin toolset when a filter is
  specified.

  Args:
      selected_tools: A list of tool names to filter.
  """
  credentials_config = SpannerCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = SpannerAdminToolset(
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
