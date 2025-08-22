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

import logging

from google.auth.credentials import Credentials

from . import client


def list_instances(project_id: str, credentials: Credentials) -> dict:
  """List Bigtable instance ids in a Google Cloud project.

  Args:
      project_id (str): The Google Cloud project id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of the Bigtable instance ids present in the project.
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    (instances_list, failed_locations_list) = bt_client.list_instances()
    if failed_locations_list:
      logging.warning(
          "Failed to list instances from the following locations: %s",
          failed_locations_list,
      )
    instance_ids = [instance.instance_id for instance in instances_list]
    return {"status": "SUCCESS", "results": instance_ids}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
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
    instance_info = {
        "project_id": project_id,
        "instance_id": instance.instance_id,
        "display_name": instance.display_name,
        "state": instance.state,
        "type": instance.type_,
        "labels": instance.labels,
    }
    return {"status": "SUCCESS", "results": instance_info}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_tables(
    project_id: str, instance_id: str, credentials: Credentials
) -> dict:
  """List table ids in a Bigtable instance.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of the tables ids present in the instance.
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    tables = instance.list_tables()
    table_ids = [table.table_id for table in tables]
    return {"status": "SUCCESS", "results": table_ids}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_table_info(
    project_id: str, instance_id: str, table_id: str, credentials: Credentials
) -> dict:
  """Get metadata information about a Bigtable table.

  Args:
      project_id (str): The Google Cloud project id containing the instance.
      instance_id (str): The Bigtable instance id containing the table.
      table_id (str): The Bigtable table id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the table.
  """
  try:
    bt_client = client.get_bigtable_admin_client(
        project=project_id, credentials=credentials
    )
    instance = bt_client.instance(instance_id)
    table = instance.table(table_id)
    column_families = table.list_column_families()
    table_info = {
        "project_id": project_id,
        "instance_id": instance.instance_id,
        "table_id": table.table_id,
        "column_families": list(column_families.keys()),
    }
    return {"status": "SUCCESS", "results": table_info}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
