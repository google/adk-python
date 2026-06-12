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

from google.adk.integrations.gcs import GCSCredentialsConfig
from google.adk.integrations.gcs.admin_toolset import GCSAdminToolset
from google.adk.integrations.gcs.storage_toolset import DEFAULT_GCS_TOOL_NAME_PREFIX
from google.adk.integrations.gcs.storage_toolset import GCSToolset
from google.adk.tools.google_tool import GoogleTool
import pytest


def test_gcs_toolset_name_prefix():
  """Test GCS toolset name prefix."""
  credentials_config = GCSCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = GCSToolset(credentials_config=credentials_config)
  assert toolset.tool_name_prefix == DEFAULT_GCS_TOOL_NAME_PREFIX

  admin_toolset = GCSAdminToolset(credentials_config=credentials_config)
  assert admin_toolset.tool_name_prefix == DEFAULT_GCS_TOOL_NAME_PREFIX


@pytest.mark.asyncio
async def test_gcs_toolset_tools_default():
  """Test default GCS toolset."""
  credentials_config = GCSCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = GCSToolset(credentials_config=credentials_config)

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 4
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "get_bucket",
      "get_object_data",
      "get_object_metadata",
      "list_objects",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.asyncio
async def test_gcs_admin_toolset_tools_default():
  """Test default GCS admin toolset."""
  credentials_config = GCSCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = GCSAdminToolset(credentials_config=credentials_config)

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 1
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_buckets",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    "selected_tools, expected_count",
    [
        pytest.param(None, 4, id="None"),
        pytest.param(["get_bucket"], 1, id="bucket-get"),
        pytest.param(
            ["list_objects", "get_object_metadata"], 2, id="object-metadata"
        ),
    ],
)
@pytest.mark.asyncio
async def test_gcs_toolset_tools_selective(selected_tools, expected_count):
  """Test GCS toolset with filter."""
  credentials_config = GCSCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = GCSToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == expected_count
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  if selected_tools is not None:
    expected_tool_names = set(selected_tools)
    actual_tool_names = set([tool.name for tool in tools])
    assert actual_tool_names == expected_tool_names
