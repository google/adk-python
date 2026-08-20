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

from unittest.mock import create_autospec
from unittest.mock import patch

from google.adk.tools.spanner import admin_tool
from google.api_core.operation_async import AsyncOperation
from google.auth.credentials import Credentials
from google.cloud import spanner_admin_database_v1
from google.cloud import spanner_admin_instance_v1
import pytest


class AsyncListIterator:
  """Asynchronous iterator for a list."""

  def __init__(self, list_):
    self._iter = iter(list_)

  def __aiter__(self):
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration as exc:
      raise StopAsyncIteration from exc


@pytest.fixture
def mock_credentials():
  return create_autospec(Credentials, instance=True)


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_list_instances_success(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the list_instances function in admin_tool."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance1 = create_autospec(
      spanner_admin_instance_v1.types.Instance, instance=True
  )
  mock_instance1.name = "projects/test-project/instances/test-instance-1"
  mock_instance2 = create_autospec(
      spanner_admin_instance_v1.types.Instance, instance=True
  )
  mock_instance2.name = "projects/test-project/instances/test-instance-2"
  mock_instance_admin_client.list_instances.return_value = AsyncListIterator([
      mock_instance1,
      mock_instance2,
  ])

  result = await admin_tool.list_instances("test-project", mock_credentials)

  assert result == {
      "status": "SUCCESS",
      "results": ["test-instance-1", "test-instance-2"],
  }
  mock_instance_admin_client.list_instances.assert_called_once()


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_list_instances_error(
    mock_instance_admin_client_cls, mock_credentials
):
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.list_instances.side_effect = Exception(
      "test error"
  )
  result = await admin_tool.list_instances("test-project", mock_credentials)
  assert result == {
      "status": "ERROR",
      "error_details": "Exception('test error')",
  }


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_get_instance_success(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the get_instance function in admin_tool."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance = create_autospec(
      spanner_admin_instance_v1.types.Instance, instance=True
  )
  mock_instance.display_name = "Test Instance"
  mock_instance.config = (
      "projects/test-project/instanceConfigs/regional-us-central1"
  )
  mock_instance.node_count = 1
  mock_instance.processing_units = 1000
  mock_instance.labels = {"env": "test"}
  mock_instance_admin_client.get_instance.return_value = mock_instance

  result = await admin_tool.get_instance(
      project_id="test-project",
      instance_id="test-instance",
      credentials=mock_credentials,
  )

  assert result == {
      "status": "SUCCESS",
      "results": {
          "instance_id": "test-instance",
          "display_name": "Test Instance",
          "config": (
              "projects/test-project/instanceConfigs/regional-us-central1"
          ),
          "node_count": 1,
          "processing_units": 1000,
          "labels": {"env": "test"},
      },
  }
  mock_instance_admin_client.instance_path.assert_called_once_with(
      "test-project", "test-instance"
  )
  mock_instance_admin_client.get_instance.assert_called_once()


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_get_instance_error(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the get_instance function in admin_tool when an error occurs."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.get_instance.side_effect = Exception("test error")
  result = await admin_tool.get_instance(
      project_id="test-project",
      instance_id="test-instance",
      credentials=mock_credentials,
  )
  assert result == {
      "status": "ERROR",
      "error_details": "Exception('test error')",
  }


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_list_instance_configs_success(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the list_instance_configs function in admin_tool."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.common_project_path.return_value = (
      "projects/test-project"
  )
  mock_config1 = create_autospec(
      spanner_admin_instance_v1.types.InstanceConfig, instance=True
  )
  mock_config1.name = "projects/test-project/instanceConfigs/config-1"
  mock_config2 = create_autospec(
      spanner_admin_instance_v1.types.InstanceConfig, instance=True
  )
  mock_config2.name = "projects/test-project/instanceConfigs/config-2"
  mock_instance_admin_client.list_instance_configs.return_value = (
      AsyncListIterator([
          mock_config1,
          mock_config2,
      ])
  )

  result = await admin_tool.list_instance_configs(
      "test-project", mock_credentials
  )

  assert result == {"status": "SUCCESS", "results": ["config-1", "config-2"]}
  mock_instance_admin_client.common_project_path.assert_called_once_with(
      "test-project"
  )
  mock_instance_admin_client.list_instance_configs.assert_called_once_with(
      parent="projects/test-project"
  )


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_get_instance_config_success(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the get_instance_config function in admin_tool."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.instance_config_path.return_value = (
      "projects/test-project/instanceConfigs/config-1"
  )
  mock_config = create_autospec(
      spanner_admin_instance_v1.types.InstanceConfig, instance=True
  )
  mock_config.name = "projects/test-project/instanceConfigs/config-1"
  mock_config.display_name = "Config 1"
  mock_config.labels = {"env": "test"}
  mock_replica = create_autospec(
      spanner_admin_instance_v1.types.ReplicaInfo, instance=True
  )
  mock_replica.location = "us-central1"
  mock_replica.type = 1  # READ_WRITE
  mock_replica.default_leader_location = True
  mock_config.replicas = [mock_replica]
  mock_instance_admin_client.get_instance_config.return_value = mock_config

  result = await admin_tool.get_instance_config(
      project_id="test-project",
      config_id="config-1",
      credentials=mock_credentials,
  )

  assert result == {
      "status": "SUCCESS",
      "results": {
          "name": "projects/test-project/instanceConfigs/config-1",
          "display_name": "Config 1",
          "replicas": [{
              "location": "us-central1",
              "type": "READ_WRITE",
              "default_leader_location": True,
          }],
          "labels": {"env": "test"},
      },
  }
  mock_instance_admin_client.instance_config_path.assert_called_once_with(
      "test-project", "config-1"
  )
  mock_instance_admin_client.get_instance_config.assert_called_once_with(
      name="projects/test-project/instanceConfigs/config-1"
  )


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_get_instance_config_error(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the get_instance_config function when an error occurs."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.get_instance_config.side_effect = Exception(
      "test error"
  )
  result = await admin_tool.get_instance_config(
      project_id="test-project",
      config_id="config-1",
      credentials=mock_credentials,
  )
  assert result == {
      "status": "ERROR",
      "error_details": "Exception('test error')",
  }


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.InstanceAdminAsyncClient",
    autospec=True,
)
async def test_create_instance_success(
    mock_instance_admin_client_cls, mock_credentials
):
  """Tests the create_instance function in admin_tool."""
  mock_instance_admin_client = mock_instance_admin_client_cls.return_value
  mock_instance_admin_client.instance_config_path.return_value = (
      "projects/test-project/instanceConfigs/config-1"
  )
  mock_instance_admin_client.common_project_path.return_value = (
      "projects/test-project"
  )
  mock_op = create_autospec(AsyncOperation, instance=True)
  mock_instance_admin_client.create_instance.return_value = mock_op
  result = await admin_tool.create_instance(
      project_id="test-project",
      instance_id="test-instance",
      config_id="config-1",
      display_name="Test Instance",
      credentials=mock_credentials,
  )
  assert result == {
      "status": "SUCCESS",
      "results": "Instance test-instance created successfully.",
  }
  mock_instance_admin_client.create_instance.assert_called_once()


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.DatabaseAdminAsyncClient",
    autospec=True,
)
async def test_list_databases_success(
    mock_db_admin_client_cls, mock_credentials
):
  """Tests the list_databases function in admin_tool."""
  mock_db_admin_client = mock_db_admin_client_cls.return_value
  mock_db_admin_client.instance_path.return_value = (
      "projects/test-project/instances/test-instance"
  )
  mock_db1 = create_autospec(
      spanner_admin_database_v1.types.Database, instance=True
  )
  mock_db1.name = "projects/test-project/instances/test-instance/databases/db-1"
  mock_db2 = create_autospec(
      spanner_admin_database_v1.types.Database, instance=True
  )
  mock_db2.name = "projects/test-project/instances/test-instance/databases/db-2"
  mock_db_admin_client.list_databases.return_value = AsyncListIterator([
      mock_db1,
      mock_db2,
  ])

  result = await admin_tool.list_databases(
      project_id="test-project",
      instance_id="test-instance",
      credentials=mock_credentials,
  )

  assert result == {"status": "SUCCESS", "results": ["db-1", "db-2"]}
  mock_db_admin_client.instance_path.assert_called_once_with(
      "test-project", "test-instance"
  )
  mock_db_admin_client.list_databases.assert_called_once_with(
      parent="projects/test-project/instances/test-instance"
  )


@pytest.mark.asyncio
@patch(
    "google.adk.tools.spanner.admin_tool.DatabaseAdminAsyncClient",
    autospec=True,
)
async def test_create_database_success(
    mock_db_admin_client_cls, mock_credentials
):
  """Tests the create_database function in admin_tool."""
  mock_db_admin_client = mock_db_admin_client_cls.return_value
  mock_db_admin_client.instance_path.return_value = (
      "projects/test-project/instances/test-instance"
  )
  mock_op = create_autospec(AsyncOperation, instance=True)
  mock_db_admin_client.create_database.return_value = mock_op
  result = await admin_tool.create_database(
      project_id="test-project",
      instance_id="test-instance",
      database_id="db-1",
      credentials=mock_credentials,
  )
  assert result == {
      "status": "SUCCESS",
  }
  mock_db_admin_client.create_database.assert_called_once()
