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

from google.adk.integrations.gcs import client
from google.adk.integrations.gcs import storage_tool
from google.auth.credentials import Credentials


def test_get_bucket():
  """Test get_bucket function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    setattr(
        mock_bucket,
        "_properties",
        {
            "bucket_id": "test-bucket-id",
            "bucket_name": "test-bucket",
            "location": "US",
            "storage_class": "STANDARD",
            "time_created": "2024-01-01",
            "updated": "2024-01-02",
            "labels": {"env": "test"},
        },
    )

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_bucket(
        bucket_name="test-bucket", credentials=creds
    )
    expected_result = getattr(mock_bucket, "_properties", {}).copy()
    assert result == {"status": "SUCCESS", "results": expected_result}


def test_get_bucket_with_properties():
  """Test get_bucket function when bucket has raw _properties populated."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    setattr(
        mock_bucket,
        "_properties",
        {
            "kind": "storage#bucket",
            "id": "test-bucket-id",
            "name": "test-bucket",
            "location": "US",
            "storageClass": "STANDARD",
            "timeCreated": "2024-01-01",
            "updated": "2024-01-02",
            "labels": {"env": "test"},
            "locationType": "region",
            "etag": "etag-val",
            "metageneration": 2,
            "versioning": {"enabled": True},
            "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}},
        },
    )

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_bucket(
        bucket_name="test-bucket", credentials=creds
    )
    expected_result = getattr(mock_bucket, "_properties", {}).copy()
    assert result == {"status": "SUCCESS", "results": expected_result}


def test_list_objects():
  """Test list_objects function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_blob.name = "test-object"
    mock_bucket.list_blobs.return_value = [mock_blob]

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.list_objects(
        bucket_name="test-bucket", credentials=creds
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-object"],
    }


def test_list_objects_pagination():
  """Test list_objects function with pagination."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_blob.name = "test-object"
    mock_blobs = mock.MagicMock()
    mock_blobs.pages = iter([[mock_blob]])
    mock_blobs.next_page_token = "next-page-token"
    mock_bucket.list_blobs.return_value = mock_blobs

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.list_objects(
        bucket_name="test-bucket",
        credentials=creds,
        page_size=1,
        page_token="token",
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-object"],
        "next_page_token": "next-page-token",
    }
    mock_bucket.list_blobs.assert_called_once_with(
        max_results=1, page_token="token"
    )


def test_get_object_metadata():
  """Test get_object_metadata function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    setattr(
        mock_blob,
        "_properties",
        {
            "kind": "storage#object",
            "id": "test-bucket/test-object/1",
            "name": "test-object",
            "bucket": "test-bucket",
            "size": "1024",
            "contentType": "text/plain",
            "timeCreated": "2024-01-01",
            "updated": "2024-01-02",
            "md5Hash": "hash",
            "metadata": {"key": "value"},
        },
    )

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_metadata(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
        generation=1,
    )
    expected_result = getattr(mock_blob, "_properties", {}).copy()
    assert result == {"status": "SUCCESS", "results": expected_result}
    mock_bucket.get_blob.assert_called_once_with("test-object", generation=1)


def test_get_object_metadata_not_found():
  """Test get_object_metadata function when object is not found."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_bucket.get_blob.return_value = None

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_metadata(
        bucket_name="test-bucket",
        object_name="non-existent",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "not found" in result["error_details"]
    mock_bucket.get_blob.assert_called_once_with("non-existent")


def test_create_object():
  """Test create_object function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        data="data",
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_blob.upload_from_string.assert_called_once_with("data")


def test_create_object_from_file():
  """Test create_object function using source_file_path."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        source_file_path="path/to/file.txt",
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_blob.upload_from_filename.assert_called_once_with("path/to/file.txt")


def test_create_object_no_data():
  """Test create_object function when neither data nor source_file_path is provided."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "must be provided" in result["error_details"]


def test_get_object_data():
  """Test get_object_data function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    mock_blob.download_as_bytes.return_value = b"content"

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
        generation=1,
    )
    assert result == {
        "status": "SUCCESS",
        "results": "content",
        "encoding": "text",
    }
    mock_bucket.get_blob.assert_called_once_with("test-object", generation=1)


def test_get_object_data_no_generation():
  """Test get_object_data function without generation parameter."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    mock_blob.download_as_bytes.return_value = b"\xff\xff"

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
    )
    assert result == {
        "status": "SUCCESS",
        "results": "//8=",
        "encoding": "base64",
    }
    mock_bucket.get_blob.assert_called_once_with("test-object")


def test_get_object_data_to_file():
  """Test get_object_data function downloading directly to destination_file_path."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="path/to/download.txt",
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_blob.download_to_filename.assert_called_once_with(
        "path/to/download.txt"
    )


def test_delete_objects():
  """Test delete_objects function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.delete_objects(
        bucket_name="test-bucket",
        object_names=["test-object"],
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_bucket.delete_blobs.assert_called_once_with(blobs=["test-object"])
