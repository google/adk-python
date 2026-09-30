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

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable import metadata_tool
from google.adk.tools.bigtable import query_tool
from google.adk.tools.bigtable.bigtable_toolset import BigtableParameterizedViewTool
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset
from google.adk.tools.bigtable.bigtable_toolset import DEFAULT_BIGTABLE_TOOL_NAME_PREFIX
from google.adk.tools.bigtable.settings import BigtableToolSettings
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
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

  assert len(tools) == 7
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_instances",
      "get_instance_info",
      "list_tables",
      "get_table_info",
      "execute_sql",
      "list_clusters",
      "get_cluster_info",
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


@pytest.mark.asyncio
async def test_bigtable_toolset_query_tool_wrapped():
  """Test that execute_sql is wrapped in BigtableQueryTool."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(credentials_config=credentials_config)

  tools = await toolset.get_tools()
  query_tools = [tool for tool in tools if tool.name == "execute_sql"]
  assert len(query_tools) == 1

  parameterized_tools = [
      tool for tool in tools if tool.name == "execute_sql_parameterized"
  ]
  assert len(parameterized_tools) == 0


@pytest.mark.asyncio
async def test_bigtable_toolset_query_tool_wrapped_custom_mapping():
  """Test that BigtableParameterizedViewTool accepts custom mapping."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(
      credentials_config=credentials_config,
      view_parameter_names=["user_id"],
  )

  tools = await toolset.get_tools()
  parameterized_tools = [
      tool for tool in tools if tool.name == "execute_sql_parameterized"
  ]
  assert len(parameterized_tools) == 1
  tool = parameterized_tools[0]
  assert isinstance(tool, BigtableParameterizedViewTool)
  assert tool.view_parameter_names == ["user_id"]


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_execution():
  """Test that BigtableParameterizedViewTool maps attributes from tool_context to view_parameters."""

  # Define a dummy function to wrap that has '_view_parameters' parameter
  def mock_execute_sql(_view_parameters=None):
    return {"status": "SUCCESS", "_view_parameters": _view_parameters}

  credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = BigtableToolSettings()
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.user_id = "test-user-123"

  # Create tool with custom mapping
  tool = BigtableParameterizedViewTool(
      func=mock_execute_sql,
      view_parameter_names=["user_id"],
  )

  # Run the tool
  res = await tool._run_async_with_credential(
      credentials=credentials,
      tool_settings=tool_settings,
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {"user_id": "test-user-123"},
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_login_flow():
  """Test showing how user_id is updated in the session/context after login."""

  def mock_execute_sql(_view_parameters=None):
    return {"status": "SUCCESS", "_view_parameters": _view_parameters}

  session = Session(id="session-1", app_name="test-app", user_id="anonymous")

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.session = session
  type(invocation_context).user_id = property(lambda self: self.session.user_id)

  tool_context = Context(invocation_context=invocation_context)
  assert tool_context.user_id == "anonymous"

  # Simulate login by updating user_id on session
  session.user_id = "authenticated-user-999"

  credentials = mock.create_autospec(Credentials, instance=True)
  tool = BigtableParameterizedViewTool(
      func=mock_execute_sql,
      view_parameter_names=["user_id"],
  )

  res = await tool._run_async_with_credential(
      credentials=credentials,
      tool_settings=BigtableToolSettings(),
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {"user_id": "authenticated-user-999"},
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_execution_session_state_fallback():
  """Test that BigtableParameterizedViewTool falls back to tool_context.state for custom parameters."""

  def mock_execute_sql(_view_parameters=None):
    return {"status": "SUCCESS", "_view_parameters": _view_parameters}

  # Create session with application-level state
  session = Session(
      id="session-1",
      app_name="test-app",
      user_id="user-123",
      state={"tenant_id": "tenant-xyz"},
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.session = session
  type(invocation_context).user_id = property(lambda self: self.session.user_id)

  tool_context = Context(invocation_context=invocation_context)

  # Ensure 'tenant_id' is NOT a top-level property or attribute on tool_context
  assert not hasattr(tool_context, "tenant_id")
  assert "tenant_id" in tool_context.state

  credentials = mock.create_autospec(Credentials, instance=True)
  tool = BigtableParameterizedViewTool(
      func=mock_execute_sql,
      view_parameter_names=["tenant_id"],
  )

  res = await tool._run_async_with_credential(
      credentials=credentials,
      tool_settings=BigtableToolSettings(),
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {"tenant_id": "tenant-xyz"},
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_execution_multiple_parameters():
  """Test that BigtableParameterizedViewTool maps multiple attributes to view_parameters."""

  def mock_execute_sql(_view_parameters=None):
    return {"status": "SUCCESS", "_view_parameters": _view_parameters}

  session = Session(
      id="session-1",
      app_name="test-app",
      user_id="user-123",
      state={"tenant_id": "tenant-xyz", "agent_id": "agent-123"},
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.session = session
  type(invocation_context).user_id = property(lambda self: self.session.user_id)

  tool_context = Context(invocation_context=invocation_context)

  credentials = mock.create_autospec(Credentials, instance=True)
  # Pass list of parameter names to be matched
  tool = BigtableParameterizedViewTool(
      func=mock_execute_sql,
      view_parameter_names=["user_id", "tenant_id", "agent_id"],
  )

  res = await tool._run_async_with_credential(
      credentials=credentials,
      tool_settings=BigtableToolSettings(),
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
          "tenant_id": "tenant-xyz",
          "agent_id": "agent-123",
      },
  }
