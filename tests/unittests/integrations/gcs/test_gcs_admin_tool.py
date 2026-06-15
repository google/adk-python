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

from unittest import mock

from google.adk.integrations.gcs import admin_tool
from google.adk.integrations.gcs import client
from google.auth.credentials import Credentials


def test_list_buckets():
  """Test list_buckets function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_bucket.name = "test-bucket"
    mock_client.list_buckets.return_value = [mock_bucket]

    creds = mock.create_autospec(Credentials, instance=True)
    result = admin_tool.list_buckets(
        project_id="test-project", credentials=creds
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-bucket"],
    }


def test_list_buckets_pagination():
  """Test list_buckets function with pagination."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_bucket.name = "test-bucket"
    mock_buckets = mock.MagicMock()
    mock_buckets.pages = iter([[mock_bucket]])
    mock_buckets.next_page_token = "next-page-token"
    mock_client.list_buckets.return_value = mock_buckets

    creds = mock.create_autospec(Credentials, instance=True)
    result = admin_tool.list_buckets(
        project_id="test-project",
        credentials=creds,
        page_size=1,
        page_token="token",
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-bucket"],
        "next_page_token": "next-page-token",
    }
    mock_client.list_buckets.assert_called_once_with(
        max_results=1, page_token="token"
    )


def test_create_bucket():
  """Test create_bucket function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket_obj = mock.MagicMock()
    mock_client.bucket.return_value = mock_bucket_obj
    mock_new_bucket = mock.MagicMock()
    mock_new_bucket.name = "test-bucket"
    mock_client.create_bucket.return_value = mock_new_bucket

    creds = mock.create_autospec(Credentials, instance=True)
    result = admin_tool.create_bucket(
        project_id="test-project", bucket_name="test-bucket", credentials=creds
    )
    assert result["status"] == "SUCCESS"
    mock_client.create_bucket.assert_called_once_with(
        mock_bucket_obj, location=None
    )


def test_update_bucket():
  """Test update_bucket function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_bucket.name = "test-bucket"
    mock_bucket.iam_configuration = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    creds = mock.create_autospec(Credentials, instance=True)
    result = admin_tool.update_bucket(
        bucket_name="test-bucket",
        credentials=creds,
        versioning_enabled=True,
        uniform_bucket_level_access_enabled=True,
    )
    assert result["status"] == "SUCCESS"
    assert mock_bucket.versioning_enabled is True
    assert (
        mock_bucket.iam_configuration.uniform_bucket_level_access_enabled
        is True
    )
    mock_bucket.patch.assert_called_once()


def test_delete_bucket():
  """Test delete_bucket function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    creds = mock.create_autospec(Credentials, instance=True)
    result = admin_tool.delete_bucket(
        bucket_name="test-bucket", credentials=creds
    )
    assert result["status"] == "SUCCESS"
    mock_bucket.delete.assert_called_once()
