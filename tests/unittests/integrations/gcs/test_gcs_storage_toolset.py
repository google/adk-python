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

from unittest import mock

from google.adk.integrations.gcs import client
from google.adk.integrations.gcs import GCSCredentialsConfig
from google.adk.integrations.gcs.admin_toolset import GCSAdminToolset
from google.adk.integrations.gcs.settings import GCSToolSettings
from google.adk.integrations.gcs.storage_toolset import DEFAULT_GCS_TOOL_NAME_PREFIX
from google.adk.integrations.gcs.storage_toolset import GCSToolset
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.tool_context import ToolContext
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

  assert len(tools) == 3
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
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

  assert len(tools) == 2
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "get_bucket",
      "list_buckets",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.asyncio
async def test_gcs_toolset_hides_settings_from_declaration():
  """Test the injected tool settings are not offered to the model."""
  credentials_config = GCSCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = GCSToolset(
      credentials_config=credentials_config, tool_filter=["get_object_data"]
  )

  (tool,) = await toolset.get_tools()

  declaration = tool._get_declaration()
  assert declaration is not None
  if declaration.parameters_json_schema:
    parameter_names = set(declaration.parameters_json_schema["properties"])
  else:
    parameter_names = set(declaration.parameters.properties)

  assert "settings" not in parameter_names
  assert "destination_file_path" in parameter_names


async def _run_get_object_data(*, local_file_root, destination_file_path):
  """Runs get_object_data through GCSToolset against a mocked GCS client."""
  toolset = GCSToolset(
      gcs_tool_settings=GCSToolSettings(local_file_root=local_file_root),
      tool_filter=["get_object_data"],
  )
  (tool,) = await toolset.get_tools()

  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    result = await tool.run_async(
        args={
            "bucket_name": "test-bucket",
            "object_name": "test-object",
            "destination_file_path": destination_file_path,
        },
        tool_context=mock.Mock(spec=ToolContext),
    )
  return result, mock_blob


@pytest.mark.asyncio
async def test_gcs_toolset_run_async_injects_settings(tmp_path):
  """Test running a tool from the toolset resolves paths under the root."""
  result, mock_blob = await _run_get_object_data(
      local_file_root=str(tmp_path),
      destination_file_path="path/to/download.txt",
  )

  assert result["status"] == "SUCCESS"
  mock_blob.download_to_filename.assert_called_once_with(
      str(tmp_path.resolve() / "path" / "to" / "download.txt")
  )


@pytest.mark.asyncio
async def test_gcs_toolset_run_async_enforces_local_file_root(tmp_path):
  """Test running a tool from the toolset refuses a path outside the root."""
  root = tmp_path / "root"
  root.mkdir()

  result, mock_blob = await _run_get_object_data(
      local_file_root=str(root), destination_file_path="../outside.txt"
  )

  assert result["status"] == "ERROR"
  assert "escapes the configured root" in result["error_details"]
  mock_blob.download_to_filename.assert_not_called()


@pytest.mark.parametrize(
    "selected_tools, expected_count",
    [
        pytest.param(None, 3, id="None"),
        pytest.param(["get_object_data"], 1, id="object-data-get"),
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
