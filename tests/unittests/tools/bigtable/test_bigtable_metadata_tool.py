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

import logging
from unittest import mock

from google.adk.tools.bigtable import metadata_tool
from google.auth.credentials import Credentials


def test_list_instances():
  """Test list_instances function."""
  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_admin_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_instance = mock.MagicMock()
    mock_instance.instance_id = "test-instance"
    mock_client.list_instances.return_value = ([mock_instance], [])

    creds = mock.create_autospec(Credentials, instance=True)
    result = metadata_tool.list_instances("test-project", creds)
    assert result == {"status": "SUCCESS", "results": ["test-instance"]}


def test_list_instances_failed_locations():
  """Test list_instances function when some locations fail."""
  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_admin_client"
  ) as mock_get_client:
    with mock.patch.object(logging, "warning") as mock_warning:
      mock_client = mock.MagicMock()
      mock_get_client.return_value = mock_client
      mock_instance = mock.MagicMock()
      mock_instance.instance_id = "test-instance"
      failed_locations = ["us-west1-a"]
      mock_client.list_instances.return_value = (
          [mock_instance],
          failed_locations,
      )

      creds = mock.create_autospec(Credentials, instance=True)
      result = metadata_tool.list_instances("test-project", creds)
      assert result == {"status": "SUCCESS", "results": ["test-instance"]}
      mock_warning.assert_called_once_with(
          "Failed to list instances from the following locations: %s",
          failed_locations,
      )


def test_get_instance_info():
  """Test get_instance_info function."""
  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_admin_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_instance = mock.MagicMock()
    mock_client.instance.return_value = mock_instance
    mock_instance.instance_id = "test-instance"
    mock_instance.display_name = "Test Instance"
    mock_instance.state = "READY"
    mock_instance.type_ = "PRODUCTION"
    mock_instance.labels = {"env": "test"}

    creds = mock.create_autospec(Credentials, instance=True)
    result = metadata_tool.get_instance_info(
        "test-project", "test-instance", creds
    )
    expected_result = {
        "project_id": "test-project",
        "instance_id": "test-instance",
        "display_name": "Test Instance",
        "state": "READY",
        "type": "PRODUCTION",
        "labels": {"env": "test"},
    }
    assert result == {"status": "SUCCESS", "results": expected_result}
    mock_instance.reload.assert_called_once()


def test_list_tables():
  """Test list_tables function."""
  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_admin_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_instance = mock.MagicMock()
    mock_client.instance.return_value = mock_instance
    mock_table = mock.MagicMock()
    mock_table.table_id = "test-table"
    mock_instance.list_tables.return_value = [mock_table]

    creds = mock.create_autospec(Credentials, instance=True)
    result = metadata_tool.list_tables("test-project", "test-instance", creds)
    assert result == {"status": "SUCCESS", "results": ["test-table"]}


def test_get_table_info():
  """Test get_table_info function."""
  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_admin_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_instance = mock.MagicMock()
    mock_client.instance.return_value = mock_instance
    mock_table = mock.MagicMock()
    mock_instance.table.return_value = mock_table
    mock_table.table_id = "test-table"
    mock_instance.instance_id = "test-instance"
    mock_table.list_column_families.return_value = {"cf1": mock.MagicMock()}

    creds = mock.create_autospec(Credentials, instance=True)
    result = metadata_tool.get_table_info(
        "test-project", "test-instance", "test-table", creds
    )
    expected_result = {
        "project_id": "test-project",
        "instance_id": "test-instance",
        "table_id": "test-table",
        "column_families": ["cf1"],
    }
    assert result == {"status": "SUCCESS", "results": expected_result}
