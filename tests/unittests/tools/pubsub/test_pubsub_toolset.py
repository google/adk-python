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
from google.adk.tools.pubsub import PubSubCredentialsConfig
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
import pytest


@pytest.mark.asyncio
async def test_pubsub_toolset_tools_default():
  """Test default PubSub toolset.

  This test verifies the behavior of the PubSub toolset when no filter is
  specified.
  """
  credentials_config = PubSubCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = PubSubToolset(
      credentials_config=credentials_config, pubsub_tool_config=None
  )
  # Verify that the tool config is initialized to default values.
  assert isinstance(toolset._tool_settings, PubSubToolConfig)  # pylint: disable=protected-access
  assert toolset._tool_settings.__dict__ == PubSubToolConfig().__dict__  # pylint: disable=protected-access

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == 3
  assert all(isinstance(tool, GoogleTool) for tool in tools)

  expected_tool_names = set([
      "publish_message",
      "pull_messages",
      "acknowledge_messages",
  ])
  actual_tool_names = {tool.name for tool in tools}
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    "selected_tools",
    [
        pytest.param([], id="None"),
        pytest.param(["publish_message"], id="publish"),
        pytest.param(["pull_messages"], id="pull"),
        pytest.param(["acknowledge_messages"], id="ack"),
    ],
)
@pytest.mark.asyncio
async def test_pubsub_toolset_tools_selective(selected_tools):
  """Test PubSub toolset with filter.

  This test verifies the behavior of the PubSub toolset when filter is
  specified. A use case for this would be when the agent builder wants to
  use only a subset of the tools provided by the toolset.

  Args:
    selected_tools: The list of tools to select from the toolset.
  """
  credentials_config = PubSubCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = PubSubToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )
  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(selected_tools)
  assert all(isinstance(tool, GoogleTool) for tool in tools)

  expected_tool_names = set(selected_tools)
  actual_tool_names = {tool.name for tool in tools}
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize(
    ("selected_tools", "returned_tools"),
    [
        pytest.param(["unknown"], [], id="all-unknown"),
        pytest.param(
            ["unknown", "publish_message"],
            ["publish_message"],
            id="mixed-known-unknown",
        ),
    ],
)
@pytest.mark.asyncio
async def test_pubsub_toolset_unknown_tool(selected_tools, returned_tools):
  """Test PubSub toolset with filter.

  This test verifies the behavior of the PubSub toolset when filter is
  specified with an unknown tool.

  Args:
    selected_tools: The list of tools to select from the toolset.
    returned_tools: The list of tools that are expected to be returned.
  """
  credentials_config = PubSubCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = PubSubToolset(
      credentials_config=credentials_config, tool_filter=selected_tools
  )

  tools = await toolset.get_tools()
  assert tools is not None

  assert len(tools) == len(returned_tools)
  assert all(isinstance(tool, GoogleTool) for tool in tools)

  expected_tool_names = set(returned_tools)
  actual_tool_names = {tool.name for tool in tools}
  assert actual_tool_names == expected_tool_names
