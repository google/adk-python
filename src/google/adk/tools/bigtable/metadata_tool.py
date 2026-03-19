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

import enum
import logging

from google.auth.credentials import Credentials
from google.cloud.bigtable import enums

from . import client

logger = logging.getLogger(f"google_adk.{__name__}")


def list_instances(project_id: str, credentials: Credentials) -> dict:
  """List Bigtable instance ids in a Google Cloud project.

  Args:
      project_id (str): The Google Cloud project id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of dictionaries, each representing a Bigtable instance.

      Example:
        {
          "status": "SUCCESS",
          "results": [
              {
                  "project_id": "test-project",
                  "instance_id": "test-instance",
                  "display_name": "Test Instance",
                  "state": "READY",
                  "type": "PRODUCTION",
                  "labels": {"env": "test"},
              }
          ],
        }
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instances_list, failed_locations_list = bt_client.list_instances()
    if failed_locations_list:
      logging.warning(
          "Failed to list instances from the following locations: %s",
          failed_locations_list,
      )
    result = [
        {
            "project_id": project_id,
            "instance_id": instance.instance_id,
            "display_name": instance.display_name,
            "state": _enum_name_from_value(
                enums.Instance.State, instance.state, "UNKNOWN_STATE"
            ),
            "type": _enum_name_from_value(
                enums.Instance.Type, instance.type_, "UNKNOWN_TYPE"
            ),
            "labels": instance.labels,
        }
        for instance in instances_list
    ]
    return {"status": "SUCCESS", "results": result}
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def get_instance_info(
    project_id: str, instance_id: str, credentials: Credentials
) -> dict:
  """Get metadata information about a Bigtable instance.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the instance.
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    instance.reload()
    return {
        "status": "SUCCESS",
        "results": {
            "project_id": project_id,
            "instance_id": instance.instance_id,
            "display_name": instance.display_name,
            "state": _enum_name_from_value(
                enums.Instance.State, instance.state, "UNKNOWN_STATE"
            ),
            "type": _enum_name_from_value(
                enums.Instance.Type, instance.type_, "UNKNOWN_TYPE"
            ),
            "labels": instance.labels,
        },
    }
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def list_tables(
    project_id: str, instance_id: str, credentials: Credentials
) -> dict:
  """List tables and their metadata in a Bigtable instance.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: A dictionary with status and results, where results is a list of
      table properties.

      Example:
        {
          "status": "SUCCESS",
          "results": [
              {
                  "project_id": "test-project",
                  "instance_id": "test-instance",
                  "table_id": "test-table",
                  "table_name": "fake-table-name",
              }
          ],
        }
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    tables = instance.list_tables()
    result = [
        {
            "project_id": project_id,
            "instance_id": instance_id,
            "table_id": table.table_id,
            "table_name": table.name,
        }
        for table in tables
    ]
    return {"status": "SUCCESS", "results": result}
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def get_table_info(
    project_id: str,
    instance_id: str,
    table_id: str,
    credentials: Credentials,
) -> dict:
  """Get metadata information about a Bigtable table.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id containing the table.
      table_id (str): The Bigtable table id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the table.

      Example:
        {
          "status": "SUCCESS",
          "results": {
              "project_id": "test-project",
              "instance_id": "test-instance",
              "table_id": "test-table",
              "column_families": ["cf1", "cf2"],
          },
        }
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    table = instance.table(table_id)
    column_families = table.list_column_families()
    return {
        "status": "SUCCESS",
        "results": {
            "project_id": project_id,
            "instance_id": instance.instance_id,
            "table_id": table.table_id,
            "column_families": list(column_families.keys()),
        },
    }
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def _enum_name_from_value(
    enum_class: type[enum.Enum], value: int, prefix: str = "UNKNOWN"
) -> str:
  for attr_name in dir(enum_class):
    if not attr_name.startswith("_"):
      if getattr(enum_class, attr_name) == value:
        return attr_name
  return f"{prefix}_{value}"


def list_clusters(
    project_id: str, instance_id: str, credentials: Credentials
) -> dict:
  """List clusters and their metadata in a Bigtable instance.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the cluster.

      Example:
        {
          "status": "SUCCESS",
          "results": [
              {
                  "project_id": "test-project",
                  "instance_id": "test-instance",
                  "cluster_id": "test-cluster",
                  "cluster_name": "fake-cluster-name",
                  "state": "READY",
                  "serve_nodes": 3,
                  "default_storage_type": "SSD",
                  "location_id": "us-central1-a",
              }
          ],
        }
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    instance.reload()
    clusters_list, failed_locations = instance.list_clusters()
    if failed_locations:
      logging.warning(
          "Failed to list clusters from the following locations: %s",
          failed_locations,
      )

    result = [
        {
            "project_id": project_id,
            "instance_id": instance_id,
            "cluster_id": cluster.cluster_id,
            "cluster_name": cluster.name,
            "state": _enum_name_from_value(
                enums.Cluster.State, cluster.state, "UNKNOWN_STATE"
            ),
            "serve_nodes": cluster.serve_nodes,
            "default_storage_type": _enum_name_from_value(
                enums.StorageType,
                cluster.default_storage_type,
                "UNKNOWN_STORAGE_TYPE",
            ),
            "location_id": cluster.location_id,
        }
        for cluster in clusters_list
    ]
    return {"status": "SUCCESS", "results": result}
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def get_cluster_info(
    project_id: str,
    instance_id: str,
    cluster_id: str,
    credentials: Credentials,
) -> dict:
  """Get detailed metadata information about a Bigtable cluster.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id containing the cluster.
      cluster_id (str): The Bigtable cluster id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the cluster.

      Example:
        {
          "status": "SUCCESS",
          "results": {
              "project_id": "test-project",
              "instance_id": "test-instance",
              "cluster_id": "test-cluster",
              "state": "READY",
              "serve_nodes": 3,
              "default_storage_type": "SSD",
              "location_id": "us-central1-a",
              "min_serve_nodes": 1,
              "max_serve_nodes": 10,
              "cpu_utilization_percent": 80,
          },
        }
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    instance.reload()
    cluster = instance.cluster(cluster_id)
    cluster.reload()
    return {
        "status": "SUCCESS",
        "results": {
            "project_id": project_id,
            "instance_id": instance_id,
            "cluster_id": cluster.cluster_id,
            "state": _enum_name_from_value(
                enums.Cluster.State, cluster.state, "UNKNOWN_STATE"
            ),
            "serve_nodes": cluster.serve_nodes,
            "default_storage_type": _enum_name_from_value(
                enums.StorageType,
                cluster.default_storage_type,
                "UNKNOWN_STORAGE_TYPE",
            ),
            "location_id": cluster.location_id,
            "min_serve_nodes": cluster.min_serve_nodes,
            "max_serve_nodes": cluster.max_serve_nodes,
            "cpu_utilization_percent": cluster.cpu_utilization_percent,
        },
    }
  except Exception as ex:
    logger.exception("Bigtable metadata tool failed: %s", ex)
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }
