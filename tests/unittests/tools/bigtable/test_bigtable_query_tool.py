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

from google.adk.tools.bigtable import client
from google.adk.tools.bigtable.query_tool import execute_sql
from google.adk.tools.bigtable.settings import BigtableToolSettings
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
from google.cloud.bigtable.data.execute_query import ExecuteQueryIterator
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "query",
        "settings",
        "parameters",
        "parameter_types",
        "execute_query_side_effect",
        "iterator_yield_values",
        "expected_result",
    ),
    [
        pytest.param(
            "SELECT * FROM my_table",
            BigtableToolSettings(),
            None,
            None,
            None,
            [{"col1": "val1", "col2": 123}],
            {"status": "SUCCESS", "rows": [{"col1": "val1", "col2": 123}]},
            id="basic",
        ),
        pytest.param(
            "SELECT * FROM my_table",
            BigtableToolSettings(max_query_result_rows=1),
            None,
            None,
            None,
            [{"col1": "val1"}, {"col1": "val2"}],
            {
                "status": "SUCCESS",
                "rows": [{"col1": "val1"}],
                "result_is_likely_truncated": True,
            },
            id="truncated",
        ),
        pytest.param(
            "SELECT * FROM my_table",
            BigtableToolSettings(),
            None,
            None,
            Exception("Test error"),
            None,
            {"status": "ERROR", "error_details": "Test error"},
            id="error",
        ),
        pytest.param(
            "SELECT * FROM my_table WHERE col1 = @param1",
            BigtableToolSettings(),
            {"param1": "val1"},
            {"param1": "string"},
            None,
            [{"col1": "val1"}],
            {"status": "SUCCESS", "rows": [{"col1": "val1"}]},
            id="with_parameters",
        ),
        pytest.param(
            "SELECT * FROM my_table WHERE 1=0",
            BigtableToolSettings(),
            None,
            None,
            None,
            [],
            {"status": "SUCCESS", "rows": []},
            id="empty_results",
        ),
        pytest.param(
            "SELECT * FROM my_table",
            BigtableToolSettings(max_query_result_rows=10),
            None,
            None,
            None,
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            {
                "status": "SUCCESS",
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            },
            id="multiple_rows",
        ),
        pytest.param(
            "SELECT * FROM my_table",
            None,
            None,
            None,
            None,
            [{"id": i} for i in range(51)],
            {
                "status": "SUCCESS",
                "rows": [{"id": i} for i in range(50)],
                "result_is_likely_truncated": True,
            },
            id="settings_none_uses_default",
        ),
        pytest.param(
            "SELECT * FROM my_table",
            BigtableToolSettings(),
            None,
            None,
            None,
            Exception("Iteration failed"),
            {"status": "ERROR", "error_details": "Iteration failed"},
            id="iteration_error_calls_close",
        ),
    ],
)
async def test_execute_sql(
    query,
    settings,
    parameters,
    parameter_types,
    execute_query_side_effect,
    iterator_yield_values,
    expected_result,
):
  """Test execute_sql tool functionality."""
  project = "my_project"
  instance_id = "my_instance"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch.object(client, "get_bigtable_data_client") as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client

    if execute_query_side_effect:
      mock_client.execute_query.side_effect = execute_query_side_effect
    else:
      mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
      mock_client.execute_query.return_value = mock_iterator

      if isinstance(iterator_yield_values, Exception):

        def raise_error():
          yield mock.MagicMock()
          raise iterator_yield_values

        mock_iterator.__iter__.side_effect = raise_error
      else:
        mock_rows = []
        for fields in iterator_yield_values:
          mock_row = mock.MagicMock()
          mock_row.fields = fields
          mock_rows.append(mock_row)
        mock_iterator.__iter__.return_value = mock_rows

    result = await execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=settings,
        tool_context=tool_context,
        parameters=parameters,
        parameter_types=parameter_types,
    )

    if expected_result["status"] == "ERROR":
      assert result["status"] == "ERROR"
      assert expected_result["error_details"] in result["error_details"]
    else:
      assert result == expected_result

    if not execute_query_side_effect:
      mock_client.execute_query.assert_called_once_with(
          query=query,
          instance_id=instance_id,
          parameters=parameters,
          parameter_types=parameter_types,
      )
      mock_iterator.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_sql_row_value_circular_reference_fallback():
  """Test execute_sql converts circular row values to strings."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch.object(client, "get_bigtable_data_client") as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator
    circular_value = []
    circular_value.append(circular_value)
    mock_row = mock.MagicMock()
    mock_row.fields = {"col1": circular_value}
    mock_iterator.__iter__.return_value = [mock_row]

    result = await execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(),
        tool_context=tool_context,
    )

  assert result["status"] == "SUCCESS"
  assert result["rows"][0]["col1"] == str(circular_value)
