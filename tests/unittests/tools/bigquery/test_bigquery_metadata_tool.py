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

import os
from unittest import mock

from google.adk.tools.bigquery import metadata_tool
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import pytest


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.list_datasets", autospec=True)
@mock.patch("google.auth.default", autospec=True)
def test_list_dataset_ids(mock_default_auth, mock_list_datasets):
  """Test list_dataset_ids tool invocation."""
  project = "my_project_id"
  mock_credentials = mock.create_autospec(Credentials, instance=True)

  # Simulate the behavior of default auth - on purpose throw exception when
  # the default auth is called
  mock_default_auth.side_effect = DefaultCredentialsError(
      "Your default credentials were not found"
  )

  mock_list_datasets.return_value = [
      bigquery.DatasetReference(project, "dataset1"),
      bigquery.DatasetReference(project, "dataset2"),
  ]
  result = metadata_tool.list_dataset_ids(project, mock_credentials)
  assert result == ["dataset1", "dataset2"]
  mock_default_auth.assert_not_called()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.get_dataset", autospec=True)
@mock.patch("google.auth.default", autospec=True)
def test_get_dataset_info(mock_default_auth, mock_get_dataset):
  """Test get_dataset_info tool invocation."""
  mock_credentials = mock.create_autospec(Credentials, instance=True)

  # Simulate the behavior of default auth - on purpose throw exception when
  # the default auth is called
  mock_default_auth.side_effect = DefaultCredentialsError(
      "Your default credentials were not found"
  )

  mock_get_dataset.return_value = mock.create_autospec(
      Credentials, instance=True
  )
  result = metadata_tool.get_dataset_info(
      "my_project_id", "my_dataset_id", mock_credentials
  )
  assert result != {
      "status": "ERROR",
      "error_details": "Your default credentials were not found",
  }
  mock_default_auth.assert_not_called()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.list_tables", autospec=True)
@mock.patch("google.auth.default", autospec=True)
def test_list_table_ids(mock_default_auth, mock_list_tables):
  """Test list_table_ids tool invocation."""
  project = "my_project_id"
  dataset = "my_dataset_id"
  dataset_ref = bigquery.DatasetReference(project, dataset)
  mock_credentials = mock.create_autospec(Credentials, instance=True)

  # Simulate the behavior of default auth - on purpose throw exception when
  # the default auth is called
  mock_default_auth.side_effect = DefaultCredentialsError(
      "Your default credentials were not found"
  )

  mock_list_tables.return_value = [
      bigquery.TableReference(dataset_ref, "table1"),
      bigquery.TableReference(dataset_ref, "table2"),
  ]
  result = metadata_tool.list_table_ids(project, dataset, mock_credentials)
  assert result == ["table1", "table2"]
  mock_default_auth.assert_not_called()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("google.cloud.bigquery.Client.get_table", autospec=True)
@mock.patch("google.auth.default", autospec=True)
def test_get_table_info(mock_default_auth, mock_get_table):
  """Test get_table_info tool invocation."""
  mock_credentials = mock.create_autospec(Credentials, instance=True)

  # Simulate the behavior of default auth - on purpose throw exception when
  # the default auth is called
  mock_default_auth.side_effect = DefaultCredentialsError(
      "Your default credentials were not found"
  )

  mock_get_table.return_value = mock.create_autospec(Credentials, instance=True)
  result = metadata_tool.get_table_info(
      "my_project_id", "my_dataset_id", "my_table_id", mock_credentials
  )
  assert result != {
      "status": "ERROR",
      "error_details": "Your default credentials were not found",
  }
  mock_default_auth.assert_not_called()
