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

from typing import Optional
from unittest import mock

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable.bigtable_toolset import BigtableToolset
from google.adk.tools.bigtable.query_tool import execute_sql
from google.adk.tools.bigtable.settings import BigtableToolSettings
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
from google.cloud import bigtable
from google.cloud.bigtable.data.execute_query import ExecuteQueryIterator
import pytest


def test_execute_sql_basic():
  """Test execute_sql tool basic functionality."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_data_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator

    # Mock row data
    mock_row = mock.MagicMock()
    mock_row.fields = {"col1": "val1", "col2": 123}
    mock_iterator.__iter__.return_value = [mock_row]

    result = execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(),
        tool_context=tool_context,
    )

    expected_rows = [{"col1": "val1", "col2": 123}]
    assert result == {"status": "SUCCESS", "rows": expected_rows}
    mock_client.execute_query.assert_called_once_with(
        query=query, instance_id=instance_id
    )
    mock_iterator.close.assert_called_once()


def test_execute_sql_truncated():
  """Test execute_sql tool truncation functionality."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_data_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator

    # Mock row data
    mock_row1 = mock.MagicMock()
    mock_row1.fields = {"col1": "val1"}
    mock_row2 = mock.MagicMock()
    mock_row2.fields = {"col1": "val2"}
    mock_iterator.__iter__.return_value = [mock_row1, mock_row2]

    result = execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(max_query_result_rows=1),
        tool_context=tool_context,
    )

    expected_rows = [{"col1": "val1"}]
    assert result == {
        "status": "SUCCESS",
        "rows": expected_rows,
        "result_is_likely_truncated": True,
    }
    mock_client.execute_query.assert_called_once_with(
        query=query, instance_id=instance_id
    )
    mock_iterator.close.assert_called_once()


def test_execute_sql_error():
  """Test execute_sql tool error handling."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_data_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.execute_query.side_effect = Exception("Test error")

    result = execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(),
        tool_context=tool_context,
    )
    assert result == {"status": "ERROR", "error_details": "Test error"}
