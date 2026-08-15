# Copyright 2026 Google LLC
#
"""Spanner Admin Tool."""

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

from typing import Any

from google.auth.credentials import Credentials
from google.cloud import spanner_admin_instance_v1
from google.cloud.spanner_admin_database_v1 import DatabaseAdminAsyncClient
from google.cloud.spanner_admin_instance_v1 import InstanceAdminAsyncClient


async def list_instances(
    project_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """List Spanner instances within a project.

  Args:
      project_id: The Google Cloud project id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with the status and a list of the Spanner instance IDs.

  Examples:
      >>> await list_instances("my_project", credentials)
      {
        "status": "SUCCESS",
        "results": [
          "instance_1",
          "instance_2"
        ]
      }
  """
  try:
    instance_admin_api = InstanceAdminAsyncClient(credentials=credentials)
    instances = []
    async for instance in await instance_admin_api.list_instances(
        parent=f"projects/{project_id}"
    ):
      instances.append(instance.name.split("/")[-1])

    return {"status": "SUCCESS", "results": instances}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def get_instance(
    project_id: str,
    *,
    instance_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """Get details of a Spanner instance.

  Args:
      project_id: The Google Cloud project id.
      instance_id: The Spanner instance id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with the status and the Spanner instance details.

  Examples:
      >>> await get_instance(project_id="my_project", instance_id="my_instance",
      ... credentials=credentials)
      {
        "status": "SUCCESS",
        "results": {
          "instance_id": "my_instance",
          "display_name": "My Instance",
          "config": "projects/my_project/instanceConfigs/regional-us-central1",
          "node_count": 1,
          "processing_units": 1000,
          "labels": {"env": "prod"}
        }
      }
  """
  try:
    instance_admin_api = InstanceAdminAsyncClient(credentials=credentials)
    instance_path = instance_admin_api.instance_path(project_id, instance_id)
    instance = await instance_admin_api.get_instance(name=instance_path)

    return {
        "status": "SUCCESS",
        "results": {
            "instance_id": instance_id,
            "display_name": instance.display_name,
            "config": instance.config,
            "node_count": instance.node_count,
            "processing_units": instance.processing_units,
            "labels": dict(instance.labels),
        },
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def list_instance_configs(
    project_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """List Spanner instance configs available for a project.

  Args:
      project_id: The Google Cloud project id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of Spanner instance config IDs.

  Examples:
      >>> await list_instance_configs("my_project", credentials)
      {
        "status": "SUCCESS",
        "results": [
          "regional-us-central1",
          "nam3"
        ]
      }
  """
  try:
    instance_admin_api = InstanceAdminAsyncClient(credentials=credentials)
    configs = await instance_admin_api.list_instance_configs(
        parent=instance_admin_api.common_project_path(project_id)
    )
    config_ids = [config.name.split("/")[-1] async for config in configs]

    return {"status": "SUCCESS", "results": config_ids}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def get_instance_config(
    project_id: str,
    *,
    config_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """Get details of a Spanner instance config.

  Args:
      project_id: The Google Cloud project id.
      config_id: The Spanner instance config id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with the status and the Spanner instance config details.

  Examples:
      >>> await get_instance_config(project_id="my_project",
      ... config_id="regional-us-central1", credentials=credentials)
      {
        "status": "SUCCESS",
        "results": {
          "name": "projects/my_project/instanceConfigs/regional-us-central1",
          "display_name": "us-central1",
          "replicas": [
              {'location': 'us-central1', 'type': 'READ_WRITE',
              'default_leader_location': True}
          ],
          "labels": {},
        }
      }
  """
  try:
    instance_admin_api = InstanceAdminAsyncClient(credentials=credentials)
    config_name = instance_admin_api.instance_config_path(project_id, config_id)
    config = await instance_admin_api.get_instance_config(name=config_name)

    replicas = [
        {
            "location": r.location,
            "type": (
                spanner_admin_instance_v1.types.ReplicaInfo.ReplicaType(
                    r.type
                ).name
            ),
            "default_leader_location": r.default_leader_location,
        }
        for r in config.replicas
    ]

    return {
        "status": "SUCCESS",
        "results": {
            "name": config.name,
            "display_name": config.display_name,
            "replicas": replicas,
            "labels": dict(config.labels),
        },
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def create_instance(
    project_id: str,
    *,
    instance_id: str,
    config_id: str,
    display_name: str,
    credentials: Credentials,
    nodes: int = 1,
) -> dict[str, Any]:
  """Create a Spanner instance.

  Args:
      project_id: The Google Cloud project id.
      instance_id: The Spanner instance id to create.
      config_id: The instance config id, e.g. regional-us-central1.
      display_name: The display name for the instance.
      credentials: The credentials to use for the request.
      nodes: Number of nodes for the instance. Defaults to 1.

  Returns:
      dict: Dictionary with the status and result of instance creation.

  Examples:
      >>> await create_instance(project_id="my_project",
      instance_id="my_instance",
      ... config_id="regional-us-central1", display_name="My Instance",
      credentials=credentials)
      {
        "status": "SUCCESS",
        "results": "Instance my_instance created successfully."
      }
  """
  try:
    instance_admin_api = InstanceAdminAsyncClient(credentials=credentials)
    instance_config = instance_admin_api.instance_config_path(
        project_id, config_id
    )
    instance = spanner_admin_instance_v1.types.Instance(
        display_name=display_name,
        config=instance_config,
        node_count=nodes,
    )
    operation = await instance_admin_api.create_instance(
        parent=instance_admin_api.common_project_path(project_id),
        instance_id=instance_id,
        instance=instance,
    )
    await operation.result(timeout=300)  # waits for completion

    return {
        "status": "SUCCESS",
        "results": f"Instance {instance_id} created successfully.",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def list_databases(
    project_id: str,
    *,
    instance_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """List Spanner databases within an instance.

  Args:
      project_id: The Google Cloud project id.
      instance_id: The Spanner instance id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with the status and a list of the Spanner database IDs.

  Examples:
      >>> await list_databases(project_id="my_project",
      ... instance_id="my_instance", credentials=credentials)
      {
        "status": "SUCCESS",
        "results": [
          "database_1",
          "database_2"
        ]
      }
  """
  try:
    database_admin_api = DatabaseAdminAsyncClient(credentials=credentials)
    databases = await database_admin_api.list_databases(
        parent=database_admin_api.instance_path(project_id, instance_id)
    )
    database_ids = [
        database.name.split("/")[-1] async for database in databases
    ]

    return {"status": "SUCCESS", "results": database_ids}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def create_database(
    project_id: str,
    *,
    instance_id: str,
    database_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """Create a Spanner database.

  Args:
      project_id: The Google Cloud project id.
      instance_id: The Spanner instance id.
      database_id: The Spanner database id.
      credentials: The credentials to use for the request.

  Returns:
      dict: Dictionary with result of database creation.

  Examples:
      >>> await create_database(project_id="my_project",
      instance_id="my_instance",
      ... database_id="my_database", credentials=credentials)
      {
        "status": "SUCCESS",
      }
  """
  try:
    database_admin_api = DatabaseAdminAsyncClient(credentials=credentials)
    operation = await database_admin_api.create_database(
        parent=database_admin_api.instance_path(project_id, instance_id),
        create_statement=f"CREATE DATABASE `{database_id}`",
    )
    # Wait for the operation to complete (default timeout 5 minutes).
    # Result on success is
    # google.cloud.spanner_admin_database_v1.types.Database
    await operation.result(timeout=300)

    return {
        "status": "SUCCESS",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }
