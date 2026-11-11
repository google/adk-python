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

import logging
from typing import Any
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.sessions.session import Session
from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable import metadata_tool
from google.adk.tools.bigtable import query_tool
from google.adk.tools.bigtable.bigtable_toolset import _BigtableQueryTool
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
async def test_bigtable_toolset_parameterized_view_replaces_query_tool():
  """Test that the parameterized tool replaces the plain query tool."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(
      credentials_config=credentials_config,
      view_parameter_names=["user_id"],
  )

  tools = await toolset.get_tools()

  assert len(tools) == 7
  expected_tool_names = set([
      "list_instances",
      "get_instance_info",
      "list_tables",
      "get_table_info",
      "execute_sql_parameterized",
      "list_clusters",
      "get_cluster_info",
  ])
  actual_tool_names = set([tool.name for tool in tools])
  assert actual_tool_names == expected_tool_names


@pytest.mark.parametrize("json_schema", [True, False])
@pytest.mark.asyncio
async def test_bigtable_toolset_plain_query_tool_hides_view_parameters(
    json_schema,
):
  """Test that the plain query tool does not declare _view_parameters."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(credentials_config=credentials_config)

  tools = await toolset.get_tools()
  tool = next(tool for tool in tools if tool.name == "execute_sql")

  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema
  ):
    declaration = tool._get_declaration()

  properties = (
      declaration.parameters_json_schema["properties"]
      if declaration.parameters_json_schema
      else declaration.parameters.properties
  )
  assert "_view_parameters" not in properties


@pytest.mark.asyncio
async def test_bigtable_plain_query_tool_drops_model_view_parameters():
  """Test that the plain query tool strips a model-supplied _view_parameters."""

  async def mock_execute_sql(query: str, _view_parameters=None):
    return {"status": "SUCCESS", "_view_parameters": _view_parameters}

  with mock.patch.object(query_tool, "execute_sql", mock_execute_sql):
    tool = _BigtableQueryTool()

  res = await tool._run_async_with_credential(
      credentials=mock.create_autospec(Credentials, instance=True),
      tool_settings=BigtableToolSettings(),
      args={"query": "SELECT 1", "_view_parameters": {"user_id": "attacker"}},
      tool_context=mock.create_autospec(ToolContext, instance=True),
  )

  assert res == {"status": "SUCCESS", "_view_parameters": None}


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"view_parameter_names": ["user_id"]},
            id="view_parameter_names",
        ),
        pytest.param(
            {"view_parameters": {"tenant_id": "tenant-xyz"}},
            id="view_parameters",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bigtable_toolset_warns_when_tool_filter_drops_query_tool(
    caplog: pytest.LogCaptureFixture,
    kwargs: dict[str, Any],
) -> None:
  """Test that filtering on execute_sql with view parameters warns."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  toolset = BigtableToolset(
      credentials_config=credentials_config,
      tool_filter=["execute_sql"],
      **kwargs,
  )

  with caplog.at_level(logging.WARNING):
    tools = await toolset.get_tools()

  assert tools == []
  assert (
      "which view_parameter_names or view_parameters replaces with"
      " execute_sql_parameterized"
      in caplog.text
  )


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


def _make_tool_context(
    state: dict[str, object] | None = None,
    *,
    custom_metadata: dict[str, object] | None = None,
) -> Context:
  session = Session(
      id="session-1",
      app_name="test-app",
      user_id="user-123",
      state=(
          state
          if state is not None
          else {"tenant_id": "attacker-chosen-tenant"}
      ),
  )
  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.session = session
  type(invocation_context).user_id = property(lambda self: self.session.user_id)
  if custom_metadata is not None:
    invocation_context._custom_metadata = custom_metadata
  return Context(invocation_context=invocation_context)


def _mock_execute_sql(
    _view_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
  return {"status": "SUCCESS", "_view_parameters": _view_parameters}


@pytest.mark.asyncio
async def test_bigtable_view_tool_ignores_state_for_custom_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Test that view parameters are never resolved from session state."""
  tool_context = _make_tool_context()

  assert not hasattr(tool_context, "tenant_id")
  assert "tenant_id" in tool_context.state

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["tenant_id"],
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {},
  }
  assert (
      "View parameter 'tenant_id' is no longer read from session state; pass"
      " it via view_parameters instead."
      in caplog.text
  )


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_execution_multiple_parameters():
  """Test that BigtableParameterizedViewTool only resolves attributes."""
  tool_context = _make_tool_context(
      state={"tenant_id": "tenant-xyz", "agent_id": "agent-123"}
  )

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id", "tenant_id", "agent_id"],
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
      },
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_with_trusted_view_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Test that BigtableParameterizedViewTool passes trusted view_parameters."""
  tool_context = _make_tool_context()

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id", "tenant_id"],
      view_parameters={"tenant_id": "trusted-tenant-xyz"},
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
          "tenant_id": "trusted-tenant-xyz",
      },
  }
  assert "is no longer read from session state" not in caplog.text


@pytest.mark.asyncio
async def test_bigtable_toolset_passes_view_parameters() -> None:
  """Test that BigtableToolset forwards view_parameters to the view tool."""
  toolset = BigtableToolset(
      view_parameter_names=["user_id"],
      view_parameters={"tenant_id": "tenant-xyz"},
  )
  tools = await toolset.get_tools()
  param_tools = [
      t for t in tools if isinstance(t, BigtableParameterizedViewTool)
  ]
  assert len(param_tools) == 1
  assert param_tools[0].view_parameter_names == ["user_id"]
  assert param_tools[0].view_parameters == {"tenant_id": "tenant-xyz"}


@pytest.mark.asyncio
async def test_bigtable_toolset_with_view_parameters_only() -> None:
  """Test that BigtableToolset registers view tool with view_parameters."""
  toolset = BigtableToolset(
      view_parameters={"tenant_id": "tenant-xyz"},
  )
  tools = await toolset.get_tools()
  param_tools = [
      t for t in tools if isinstance(t, BigtableParameterizedViewTool)
  ]
  assert len(param_tools) == 1
  assert param_tools[0].view_parameter_names is None
  assert param_tools[0].view_parameters == {"tenant_id": "tenant-xyz"}


@pytest.mark.asyncio
async def test_bigtable_view_tool_with_callable_view_parameters() -> None:
  """Test that BigtableParameterizedViewTool resolves a sync callable."""
  tool_context = _make_tool_context()

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id"],
      view_parameters=lambda ctx: {"tenant_id": f"tenant-for-{ctx.user_id}"},
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
          "tenant_id": "tenant-for-user-123",
      },
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_with_async_callable() -> None:
  """Test that BigtableParameterizedViewTool resolves an async callable."""
  tool_context = _make_tool_context()

  async def get_params(ctx: ToolContext) -> dict[str, str]:
    return {"tenant_id": f"async-tenant-for-{ctx.user_id}"}

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id"],
      view_parameters=get_params,
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
          "tenant_id": "async-tenant-for-user-123",
      },
  }


@pytest.mark.asyncio
async def test_bigtable_view_tool_with_callable_dict_values() -> None:
  """Test that BigtableParameterizedViewTool resolves dict callables."""

  async def get_region(ctx: ToolContext) -> str:
    return f"region-for-{ctx.user_id}"

  tool_context = _make_tool_context()

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameters={
          "static_param": "val1",
          "tenant_id": lambda ctx: f"tenant-for-{ctx.user_id}",
          "region": get_region,
          "skipped_param": lambda ctx: None,
      },
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "static_param": "val1",
          "tenant_id": "tenant-for-user-123",
          "region": "region-for-user-123",
      },
  }


@pytest.mark.asyncio
async def test_bigtable_view_tool_view_parameters_precedence() -> None:
  """Test that view_parameters overwrites keys from view_parameter_names."""
  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id"],
      view_parameters={"user_id": "override-user-999", "tenant_id": "t-1"},
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "override-user-999",
          "tenant_id": "t-1",
      },
  }


@pytest.mark.parametrize(
    "view_parameters",
    [
        pytest.param(
            {"user_id": None, "extra": None, "tenant_id": "t-1"},
            id="static-mapping",
        ),
        pytest.param(
            {
                "user_id": lambda ctx: None,
                "extra": lambda ctx: None,
                "tenant_id": lambda ctx: "t-1",
            },
            id="per-value-callable",
        ),
        pytest.param(
            lambda ctx: {"user_id": None, "extra": None, "tenant_id": "t-1"},
            id="top-level-callable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_omits_none_parameter_values(
    view_parameters: Any,
) -> None:
  """Test that None parameter values are omitted across all forms."""
  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id"],
      view_parameters=view_parameters,
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {"tenant_id": "t-1"},
  }


@pytest.mark.asyncio
async def test_bigtable_parameterized_view_tool_skips_none_resolver() -> None:
  """Test that a top-level resolver returning None skips view_parameters."""

  def resolve_none(ctx: ToolContext) -> dict[str, object] | None:
    del ctx
    return None

  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id"],
      view_parameters=resolve_none,
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {"user_id": "user-123"},
  }


@pytest.mark.asyncio
async def test_bigtable_view_tool_rejects_non_mapping_resolver_return() -> None:
  """Test that a non-mapping resolver return value raises TypeError."""
  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameters=mock.Mock(return_value="ab"),
  )

  with pytest.raises(
      TypeError, match="view_parameters callable must return a Mapping"
  ):
    await tool.run_async(
        args={},
        tool_context=tool_context,
    )


@pytest.mark.asyncio
async def test_bigtable_view_tool_propagates_resolver_exception() -> None:
  """Test that an exception raised by a resolver propagates from run_async."""

  def failing_resolver(ctx: ToolContext) -> dict[str, object]:
    del ctx
    raise RuntimeError("tenant lookup failed")

  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameters=failing_resolver,
  )

  with pytest.raises(RuntimeError, match="tenant lookup failed"):
    await tool.run_async(
        args={},
        tool_context=tool_context,
    )


@pytest.mark.asyncio
async def test_bigtable_view_tool_returns_error_dict_on_func_exception() -> (
    None
):
  """Test that an exception raised by func is returned as an ERROR dict."""

  def failing_execute_sql(
      _view_parameters: dict[str, object] | None = None,
  ) -> dict[str, object]:
    del _view_parameters
    raise RuntimeError("query execution failed")

  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(
      func=failing_execute_sql,
      view_parameter_names=["user_id"],
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "ERROR",
      "error_details": "query execution failed",
  }


def test_bigtable_view_tool_rejects_invalid_view_parameters() -> None:
  """Test BigtableParameterizedViewTool rejects invalid view_parameters."""
  invalid_params: Any = ["invalid"]
  with pytest.raises(
      TypeError, match="view_parameters must be a Mapping, callable, or None"
  ):
    BigtableParameterizedViewTool(
        func=_mock_execute_sql,
        view_parameters=invalid_params,
    )


def test_bigtable_toolset_rejects_invalid_view_parameters() -> None:
  """Test that BigtableToolset rejects invalid view_parameters."""
  invalid_params: Any = ["invalid"]
  with pytest.raises(
      TypeError, match="view_parameters must be a Mapping, callable, or None"
  ):
    BigtableToolset(
        view_parameters=invalid_params,
    )


@pytest.mark.parametrize("invalid_value", [["invalid"], [], ""])
@pytest.mark.asyncio
async def test_bigtable_view_tool_runtime_rejects_invalid_view_parameters(
    invalid_value: Any,
) -> None:
  """Test runtime rejection of invalid mutated view_parameters."""
  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(func=_mock_execute_sql)
  tool.view_parameters = invalid_value

  with pytest.raises(
      TypeError, match="view_parameters must be a Mapping, callable, or None"
  ):
    await tool.run_async(
        args={},
        tool_context=tool_context,
    )


@pytest.mark.asyncio
async def test_bigtable_view_tool_strips_caller_view_parameters() -> None:
  """Test that caller-supplied _view_parameters in args is stripped."""
  tool_context = _make_tool_context()
  tool = BigtableParameterizedViewTool(func=_mock_execute_sql)

  res = await tool.run_async(
      args={"_view_parameters": {"user_id": "attacker"}},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": None,
  }


@pytest.mark.asyncio
async def test_bigtable_view_tool_ignores_custom_metadata() -> None:
  """Test that BigtableParameterizedViewTool ignores custom_metadata."""
  tool_context = _make_tool_context(
      custom_metadata={"tenant_id": "attacker-chosen-metadata-tenant"}
  )

  tool = BigtableParameterizedViewTool(
      func=_mock_execute_sql,
      view_parameter_names=["user_id", "tenant_id"],
  )

  res = await tool.run_async(
      args={},
      tool_context=tool_context,
  )

  assert res == {
      "status": "SUCCESS",
      "_view_parameters": {
          "user_id": "user-123",
      },
  }
