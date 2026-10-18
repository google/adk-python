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

import asyncio
import inspect
import threading
from typing import Any
from unittest import mock

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations.bigquery import BigQueryCredentialsConfig
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
from google.cloud import bigquery
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

  assert len(tools) == 11
  assert all([isinstance(tool, GoogleTool) for tool in tools])

  expected_tool_names = set([
      "list_dataset_ids",
      "get_dataset_info",
      "list_table_ids",
      "get_table_info",
      "get_job_info",
      "execute_sql",
      "ask_data_insights",
      "forecast",
      "analyze_contribution",
      "detect_anomalies",
      "search_catalog",
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


@pytest.mark.asyncio
async def test_bigquery_toolset_tools_run_off_event_loop() -> None:
  """Test that BigQueryToolset tools run off the event loop."""
  credentials = mock.create_autospec(Credentials, instance=True)
  toolset = BigQueryToolset()
  tools = await toolset.get_tools()
  assert len(tools) == 11

  for tool in tools:
    assert isinstance(tool, FunctionTool)
    assert inspect.iscoroutinefunction(tool.func)

  execute_sql_tool = next(t for t in tools if t.name == "execute_sql")

  query_started = threading.Event()
  query_may_return = threading.Event()
  ticks = 0
  loop_ticked = False

  async def count_ticks() -> None:
    nonlocal ticks
    while not query_started.is_set() or ticks < 3:
      ticks += 1
      await asyncio.sleep(0)
    query_may_return.set()

  def blocking_query_and_wait(
      *args: Any, **kwargs: Any
  ) -> list[dict[str, int]]:
    nonlocal loop_ticked
    query_started.set()
    loop_ticked = query_may_return.wait(timeout=10)
    return [{"num": 123}]

  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch.object(bigquery, "Client", autospec=True) as client_cls:
    bq_client = client_cls.return_value
    query_job = mock.create_autospec(bigquery.QueryJob)
    query_job.statement_type = "SELECT"
    bq_client.query.return_value = query_job
    bq_client.query_and_wait.side_effect = blocking_query_and_wait

    result, _ = await asyncio.gather(
        execute_sql_tool.run_async(
            args={
                "project_id": "my_project",
                "query": "SELECT 123 AS num",
                "credentials": credentials,
            },
            tool_context=tool_context,
        ),
        count_ticks(),
    )

  assert loop_ticked, "the event loop was blocked for the whole query"
  assert ticks >= 3
  assert result == {"status": "SUCCESS", "rows": [{"num": 123}]}


@pytest.mark.asyncio
async def test_bigquery_toolset_declarations_schema_object() -> None:
  """Test that tool declarations build when JSON schema feature is disabled."""
  with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
    toolset = BigQueryToolset()
    tools = await toolset.get_tools()
    for tool in tools:
      declaration = tool._get_declaration()
      assert declaration is not None
      assert declaration.name == tool.name
      assert declaration.parameters is not None
