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

import logging
from unittest import mock

from google.adk.tools.bigtable import client
from google.adk.tools.bigtable import metadata_tool
from google.auth.credentials import Credentials
from google.cloud.bigtable import enums
import pytest


@pytest.fixture
def mock_get_client():
  with mock.patch.object(
      client, "get_bigtable_admin_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    yield mock_get_client


def test_list_instances(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_instance.instance_id = "test-instance"
  mock_get_client.return_value.list_instances.return_value = (
      [mock_instance],
      [],
  )

  mock_instance.display_name = "Test Instance"
  mock_instance.state = enums.Instance.State.READY
  mock_instance.type_ = enums.Instance.Type.PRODUCTION
  mock_instance.labels = {"env": "test"}

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.list_instances(
      project_id="test-project", credentials=creds
  )
  expected_result = {
      "project_id": "test-project",
      "instance_id": "test-instance",
      "display_name": "Test Instance",
      "state": "READY",
      "type": "PRODUCTION",
      "labels": {"env": "test"},
  }
  assert result == {"status": "SUCCESS", "results": [expected_result]}


def test_list_instances_failed_locations(mock_get_client):
  with mock.patch.object(logging, "warning") as mock_warning:
    mock_instance = mock.MagicMock()
    mock_instance.instance_id = "test-instance"
    failed_locations = ["us-west1-a"]
    mock_get_client.return_value.list_instances.return_value = (
        [mock_instance],
        failed_locations,
    )

    mock_instance.display_name = "Test Instance"
    mock_instance.state = enums.Instance.State.READY
    mock_instance.type_ = enums.Instance.Type.PRODUCTION
    mock_instance.labels = {"env": "test"}

    creds = mock.create_autospec(Credentials, instance=True)
    result = metadata_tool.list_instances(
        project_id="test-project", credentials=creds
    )
    expected_result = {
        "project_id": "test-project",
        "instance_id": "test-instance",
        "display_name": "Test Instance",
        "state": "READY",
        "type": "PRODUCTION",
        "labels": {"env": "test"},
    }
    assert result == {"status": "SUCCESS", "results": [expected_result]}
    mock_warning.assert_called_once_with(
        "Failed to list instances from the following locations: %s",
        failed_locations,
    )


def test_get_instance_info(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_get_client.return_value.instance.return_value = mock_instance
  mock_instance.instance_id = "test-instance"
  mock_instance.display_name = "Test Instance"
  mock_instance.state = enums.Instance.State.READY
  mock_instance.type_ = enums.Instance.Type.PRODUCTION
  mock_instance.labels = {"env": "test"}

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.get_instance_info(
      project_id="test-project",
      instance_id="test-instance",
      credentials=creds,
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


def test_list_tables(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_get_client.return_value.instance.return_value = mock_instance
  mock_table = mock.MagicMock()
  mock_table.table_id = "test-table"
  mock_table.name = (
      "projects/test-project/instances/test-instance/tables/test-table"
  )
  mock_instance.list_tables.return_value = [mock_table]

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.list_tables(
      project_id="test-project",
      instance_id="test-instance",
      credentials=creds,
  )
  expected_result = [{
      "project_id": "test-project",
      "instance_id": "test-instance",
      "table_id": "test-table",
      "table_name": (
          "projects/test-project/instances/test-instance/tables/test-table"
      ),
  }]
  assert result == {"status": "SUCCESS", "results": expected_result}


def test_get_table_info(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_instance.instance_id = "test-instance"
  mock_get_client.return_value.instance.return_value = mock_instance
  mock_table = mock.MagicMock()
  mock_instance.table.return_value = mock_table
  mock_table.table_id = "test-table"
  mock_table.list_column_families.return_value = {"cf1": mock.MagicMock()}

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.get_table_info(
      project_id="test-project",
      instance_id="test-instance",
      table_id="test-table",
      credentials=creds,
  )
  expected_result = {
      "project_id": "test-project",
      "instance_id": "test-instance",
      "table_id": "test-table",
      "column_families": ["cf1"],
  }
  assert result == {"status": "SUCCESS", "results": expected_result}


def test_list_clusters(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_get_client.return_value.instance.return_value = mock_instance
  mock_cluster = mock.MagicMock()
  mock_cluster.cluster_id = "test-cluster"
  mock_cluster.name = (
      "projects/test-project/instances/test-instance/clusters/test-cluster"
  )
  mock_cluster.state = enums.Cluster.State.READY
  mock_cluster.serve_nodes = 3
  mock_cluster.default_storage_type = enums.StorageType.SSD
  mock_cluster.location_id = "us-central1-a"
  mock_instance.list_clusters.return_value = ([mock_cluster], [])

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.list_clusters(
      project_id="test-project",
      instance_id="test-instance",
      credentials=creds,
  )
  expected_result = [{
      "project_id": "test-project",
      "instance_id": "test-instance",
      "cluster_id": "test-cluster",
      "cluster_name": mock_cluster.name,
      "state": "READY",
      "serve_nodes": 3,
      "default_storage_type": "SSD",
      "location_id": "us-central1-a",
  }]
  assert result == {"status": "SUCCESS", "results": expected_result}


def test_list_clusters_error(mock_get_client):
  mock_get_client.side_effect = Exception("test-error")
  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.list_clusters(
      project_id="test-project",
      instance_id="test-instance",
      credentials=creds,
  )
  assert result == {
      "status": "ERROR",
      "error_details": "Exception('test-error')",
  }


def test_get_cluster_info(mock_get_client):
  mock_instance = mock.MagicMock()
  mock_get_client.return_value.instance.return_value = mock_instance
  mock_cluster = mock.MagicMock()
  mock_instance.cluster.return_value = mock_cluster
  mock_cluster.cluster_id = "test-cluster"
  mock_cluster.state = enums.Cluster.State.READY
  mock_cluster.serve_nodes = 3
  mock_cluster.default_storage_type = enums.StorageType.SSD
  mock_cluster.location_id = "us-central1-a"
  mock_cluster.min_serve_nodes = 3
  mock_cluster.max_serve_nodes = 10
  mock_cluster.cpu_utilization_percent = 50

  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.get_cluster_info(
      project_id="test-project",
      instance_id="test-instance",
      cluster_id="test-cluster",
      credentials=creds,
  )
  expected_results = {
      "project_id": "test-project",
      "instance_id": "test-instance",
      "cluster_id": "test-cluster",
      "state": "READY",
      "serve_nodes": 3,
      "default_storage_type": "SSD",
      "location_id": "us-central1-a",
      "min_serve_nodes": 3,
      "max_serve_nodes": 10,
      "cpu_utilization_percent": 50,
  }
  assert result == {"status": "SUCCESS", "results": expected_results}
  mock_cluster.reload.assert_called_once()


def test_get_cluster_info_error(mock_get_client):
  mock_get_client.side_effect = Exception("test-error")
  creds = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.get_cluster_info(
      project_id="test-project",
      instance_id="test-instance",
      cluster_id="test-cluster",
      credentials=creds,
  )
  assert result == {
      "status": "ERROR",
      "error_details": "Exception('test-error')",
  }
