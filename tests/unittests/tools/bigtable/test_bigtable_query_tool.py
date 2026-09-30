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
from unittest import mock

from google.adk.tools.bigtable import client
from google.adk.tools.bigtable.query_tool import execute_sql
from google.adk.tools.bigtable.settings import BigtableToolSettings
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import AnonymousCredentials
from google.auth.credentials import Credentials
from google.cloud.bigtable import data
from google.cloud.bigtable.data.execute_query import ExecuteQueryIterator
import pytest


def _mock_iterator(rows):
  """Builds a query iterator that yields ``rows``, or raises if given an error."""
  iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
  if isinstance(rows, Exception):

    def raise_error():
      yield mock.MagicMock()
      raise rows

    iterator.__iter__.side_effect = raise_error
  else:
    mock_rows = []
    for fields in rows:
      mock_row = mock.MagicMock()
      mock_row.fields = fields
      mock_rows.append(mock_row)
    iterator.__iter__.return_value = mock_rows
  return iterator


def _unreachable_data_client() -> data.BigtableDataClient:
  """Builds a real SDK client that can only ever dial a dead loopback port.

  The client, its transport and its background channel-refresh worker are all
  genuine, so ``close()`` is exercised exactly as in production; only the
  endpoint is redirected, which keeps the test off the network.
  """
  return data.BigtableDataClient(
      project="my_project",
      credentials=AnonymousCredentials(),
      client_options={"api_endpoint": "localhost:1"},
  )


def _channel_is_shut_down(bt_client: data.BigtableDataClient) -> bool:
  """Reports whether the client's gRPC channel refuses new RPCs.

  A shut-down channel rejects the probe locally, before any connection is
  attempted. A live one dials its endpoint, which ``_unreachable_data_client``
  has pinned to a closed loopback port, so it is refused rather than served.
  """
  try:
    bt_client.transport.grpc_channel.unary_unary("/probe/Probe")(
        b"", timeout=0.01
    )
  except ValueError:
    # Raised by gRPC only for a closed channel.
    return True
  except Exception:
    return False
  return False


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
          view_parameters=None,
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


@pytest.mark.asyncio
async def test_execute_sql_with_view_parameters():
  """Test execute_sql with _view_parameters passed."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  view_parameters = {"user_id": "test-user-123"}

  with mock.patch.object(client, "get_bigtable_data_client") as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator
    mock_iterator.__iter__.return_value = []

    result = await execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(),
        tool_context=tool_context,
        _view_parameters=view_parameters,
    )

  assert result["status"] == "SUCCESS"
  mock_client.execute_query.assert_called_once_with(
      query=query,
      instance_id=instance_id,
      parameters=None,
      parameter_types=None,
      view_parameters=view_parameters,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "execute_query_error", "expected_status"),
    [
        pytest.param([{"col1": "val1"}], None, "SUCCESS", id="success"),
        pytest.param(
            [{"col1": i} for i in range(51)], None, "SUCCESS", id="truncated"
        ),
        pytest.param(
            Exception("Iteration failed"), None, "ERROR", id="iteration_error"
        ),
        pytest.param(None, Exception("SDK failure"), "ERROR", id="sdk_error"),
    ],
)
async def test_execute_sql_releases_the_client(
    rows, execute_query_error, expected_status
):
  """Every exit path shuts down the per-call client's gRPC channel.

  A real SDK client is used so that the assertion observes the channel
  actually being released, not merely that ``close`` was called.
  """
  created = []

  def _new_client(*, project, credentials):
    del project, credentials
    bt_client = _unreachable_data_client()
    if execute_query_error is not None:
      bt_client.execute_query = mock.Mock(side_effect=execute_query_error)
    else:
      bt_client.execute_query = mock.Mock(return_value=_mock_iterator(rows))
    created.append(bt_client)
    return bt_client

  with mock.patch.object(
      client, "get_bigtable_data_client", side_effect=_new_client
  ):
    result = await execute_sql(
        project_id="my_project",
        instance_id="my_instance",
        query="SELECT * FROM my_table",
        credentials=mock.create_autospec(Credentials, instance=True),
        settings=BigtableToolSettings(),
        tool_context=mock.create_autospec(ToolContext, instance=True),
    )

  try:
    assert result["status"] == expected_status
    assert len(created) == 1
    assert _channel_is_shut_down(created[0])
  finally:
    # Should the release regress, the client is still live and its worker
    # thread would block interpreter exit; drop it so the regression surfaces
    # as this assertion rather than as a suite-wide hang. Closing twice is a
    # no-op on the passing path.
    for bt_client in created:
      data.BigtableDataClient.close(bt_client)


@pytest.mark.asyncio
async def test_execute_sql_returns_result_when_release_fails(caplog):
  """A failure while releasing the client is logged, not raised."""
  bt_client = _unreachable_data_client()
  bt_client.execute_query = mock.Mock(
      return_value=_mock_iterator([{"col1": "val1"}])
  )
  bt_client.close = mock.Mock(side_effect=RuntimeError("close failed"))

  caplog.set_level(logging.ERROR)
  try:
    with mock.patch.object(
        client, "get_bigtable_data_client", return_value=bt_client
    ):
      result = await execute_sql(
          project_id="my_project",
          instance_id="my_instance",
          query="SELECT * FROM my_table",
          credentials=mock.create_autospec(Credentials, instance=True),
          settings=BigtableToolSettings(),
          tool_context=mock.create_autospec(ToolContext, instance=True),
      )
  finally:
    # ``close`` is stubbed above, so release the real client explicitly.
    data.BigtableDataClient.close(bt_client)

  assert result == {"status": "SUCCESS", "rows": [{"col1": "val1"}]}
  bt_client.close.assert_called_once()
  assert [
      record
      for record in caplog.records
      if record.levelno == logging.ERROR
      and record.exc_info
      and isinstance(record.exc_info[1], RuntimeError)
  ]


@pytest.mark.asyncio
async def test_execute_sql_with_multiple_view_parameters():
  """Test execute_sql with multiple view_parameters of different names."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  view_parameters = {
      "user_id": "test-user-123",
      "tenant_id": "tenant-xyz",
      "role": "admin",
  }

  with mock.patch.object(client, "get_bigtable_data_client") as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator
    mock_iterator.__iter__.return_value = []

    result = await execute_sql(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        settings=BigtableToolSettings(),
        tool_context=tool_context,
        _view_parameters=view_parameters,
    )

  assert result["status"] == "SUCCESS"
  mock_client.execute_query.assert_called_once_with(
      query=query,
      instance_id=instance_id,
      parameters=None,
      parameter_types=None,
      view_parameters=view_parameters,
  )
